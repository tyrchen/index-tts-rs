//! Unix Domain Socket warm-model daemon (`~/.cache/index-tts-rs/daemon.sock`).
//!
//! Keeps `IndexTtsEngine` resident in memory across CLI invocations to eliminate model loading
//! overhead, enforces `0600` permissions on the socket file, and automatically exits after
//! `idle_timeout_secs` of inactivity.

use std::{
    env, fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use index_tts_rs_core::{
    config::{EngineConfig, SynthesisRequest, validate_safe_file_path},
    engine::{AudioChunk, IndexTtsEngine, StageTimings},
};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{UnixListener, UnixStream},
    sync::Notify,
    time::timeout,
};
use tracing::{info, warn};

/// Default relative socket path under `$HOME`.
pub const DEFAULT_DAEMON_SOCKET_SUBPATH: &str = ".cache/index-tts-rs/daemon.sock";
/// Default daemon idle timeout in seconds (`600` seconds = 10 minutes).
pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 600;
/// Maximum allowed JSONL request line length in bytes (`64 KiB`).
const MAX_REQUEST_LINE_BYTES: usize = 65_536;

/// Daemon runtime status snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DaemonStatusInfo {
    /// Process ID of the running daemon.
    pub pid: u32,
    /// Socket path on disk.
    pub socket_path: PathBuf,
    /// Resolved model directory.
    pub model_dir: PathBuf,
    /// Daemon uptime in seconds.
    pub uptime_secs: u64,
    /// Seconds since the last synthesis or status command.
    pub idle_secs: u64,
    /// Configured idle auto-shutdown timeout in seconds.
    pub idle_timeout_secs: u64,
    /// Number of registered voices in the store.
    pub voices_count: usize,
}

/// Streaming chunk payload transmitted over the Unix Domain Socket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DaemonChunkPayload {
    /// Zero-based sentence segment index.
    pub segment_index: usize,
    /// Total sentence segments.
    pub total_segments: usize,
    /// Base64-encoded little-endian `i16` PCM bytes.
    pub pcm_base64: String,
    /// Sample rate in Hz (`22050`).
    pub sample_rate: u32,
    /// Generated Mel code count for this segment.
    pub mel_tokens: usize,
    /// Per-stage timing breakdown.
    pub timings: StageTimings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DaemonRequest {
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    synth_request: Option<SynthesisRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DaemonResponseLine {
    id: u64,
    event: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<DaemonStatusInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunk: Option<DaemonChunkPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Resolves the default daemon Unix Domain Socket path (`~/.cache/index-tts-rs/daemon.sock`).
#[must_use]
pub fn default_daemon_socket_path() -> PathBuf {
    if let Ok(home) = env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home).join(DEFAULT_DAEMON_SOCKET_SUBPATH);
    }
    PathBuf::from(DEFAULT_DAEMON_SOCKET_SUBPATH)
}

/// Starts the Unix Domain Socket daemon server in the current process and runs until stopped or
/// idle timeout expires.
///
/// # Errors
///
/// Returns an error if binding the socket or initializing `IndexTtsEngine` fails.
#[allow(clippy::disallowed_methods, clippy::too_many_lines)]
pub async fn run_daemon_server(
    config: EngineConfig,
    socket_path: &Path,
    idle_timeout_secs: u64,
) -> Result<()> {
    validate_safe_file_path(socket_path, "socket_path")?;
    if let Some(parent) = socket_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create socket directory `{}`", parent.display()))?;
    }

    if socket_path.exists() {
        if timeout(Duration::from_millis(300), UnixStream::connect(socket_path))
            .await
            .is_ok_and(|res| res.is_ok())
        {
            return Err(anyhow!(
                "daemon is already running at `{}`",
                socket_path.display()
            ));
        }
        let _ = fs::remove_file(socket_path);
    }

    let engine = Arc::new(
        IndexTtsEngine::start(config)
            .context("failed to start IndexTtsEngine for daemon server")?,
    );

    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("failed to bind Unix socket at `{}`", socket_path.display()))?;
    fs::set_permissions(socket_path, fs::Permissions::from_mode(0o600)).with_context(|| {
        format!(
            "failed to set 0600 permissions on `{}`",
            socket_path.display()
        )
    })?;

    let started_at = Instant::now();
    let last_active_ms = Arc::new(AtomicU64::new(now_epoch_ms()));
    let shutdown_notify = Arc::new(Notify::new());

    if idle_timeout_secs > 0 {
        let active_clone = Arc::clone(&last_active_ms);
        let notify_clone = Arc::clone(&shutdown_notify);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                let elapsed_ms =
                    now_epoch_ms().saturating_sub(active_clone.load(Ordering::Relaxed));
                if elapsed_ms >= idle_timeout_secs.saturating_mul(1000) {
                    info!(
                        idle_secs = elapsed_ms / 1000,
                        "Daemon idle timeout reached; shutting down"
                    );
                    notify_clone.notify_waiters();
                    break;
                }
            }
        });
    }

    info!(
        socket = %socket_path.display(),
        pid = std::process::id(),
        idle_timeout_secs,
        "IndexTTS-2.5 daemon listening"
    );

    loop {
        tokio::select! {
            accept_res = listener.accept() => {
                match accept_res {
                    Ok((stream, _)) => {
                        last_active_ms.store(now_epoch_ms(), Ordering::Relaxed);
                        let eng = Arc::clone(&engine);
                        let active = Arc::clone(&last_active_ms);
                        let notify = Arc::clone(&shutdown_notify);
                        let sock_path = socket_path.to_path_buf();
                        tokio::spawn(async move {
                            if let Err(err) = handle_client_stream(
                                stream,
                                eng,
                                started_at,
                                active,
                                idle_timeout_secs,
                                sock_path,
                                notify,
                            ).await {
                                warn!(error = %err, "Daemon client handler error");
                            }
                        });
                    }
                    Err(err) => {
                        warn!(error = %err, "UnixListener accept error");
                    }
                }
            }
            () = shutdown_notify.notified() => {
                break;
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Received SIGINT; stopping daemon");
                break;
            }
        }
    }

    let _ = fs::remove_file(socket_path);
    engine.stop().context("failed to stop IndexTtsEngine")?;
    Ok(())
}

async fn handle_client_stream(
    stream: UnixStream,
    engine: Arc<IndexTtsEngine>,
    started_at: Instant,
    last_active_ms: Arc<AtomicU64>,
    idle_timeout_secs: u64,
    socket_path: PathBuf,
    shutdown_notify: Arc<Notify>,
) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        if line.len() > MAX_REQUEST_LINE_BYTES {
            let resp = DaemonResponseLine {
                id: 0,
                event: "error".to_owned(),
                status: None,
                chunk: None,
                error: Some("request line exceeds 64 KiB limit".to_owned()),
            };
            write_response_line(&mut writer, &resp).await?;
            break;
        }

        let req: DaemonRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(err) => {
                let resp = DaemonResponseLine {
                    id: 0,
                    event: "error".to_owned(),
                    status: None,
                    chunk: None,
                    error: Some(format!("invalid JSON request: {err}")),
                };
                write_response_line(&mut writer, &resp).await?;
                continue;
            }
        };

        match req.method.as_str() {
            "status" => {
                let idle_ms = now_epoch_ms().saturating_sub(last_active_ms.load(Ordering::Relaxed));
                let voices_count = engine.voice_store().list().map_or(0, |v| v.len());
                let status = DaemonStatusInfo {
                    pid: std::process::id(),
                    socket_path: socket_path.clone(),
                    model_dir: engine.model_dir().to_path_buf(),
                    uptime_secs: started_at.elapsed().as_secs(),
                    idle_secs: idle_ms / 1000,
                    idle_timeout_secs,
                    voices_count,
                };
                let resp = DaemonResponseLine {
                    id: req.id,
                    event: "status".to_owned(),
                    status: Some(status),
                    chunk: None,
                    error: None,
                };
                write_response_line(&mut writer, &resp).await?;
            }
            "stop" => {
                let resp = DaemonResponseLine {
                    id: req.id,
                    event: "stopped".to_owned(),
                    status: None,
                    chunk: None,
                    error: None,
                };
                write_response_line(&mut writer, &resp).await?;
                shutdown_notify.notify_waiters();
                break;
            }
            "synth" => {
                last_active_ms.store(now_epoch_ms(), Ordering::Relaxed);
                let Some(synth_req) = req.synth_request else {
                    let resp = DaemonResponseLine {
                        id: req.id,
                        event: "error".to_owned(),
                        status: None,
                        chunk: None,
                        error: Some("missing synthRequest payload".to_owned()),
                    };
                    write_response_line(&mut writer, &resp).await?;
                    continue;
                };
                stream_synthesis_to_client(&engine, req.id, synth_req, &mut writer).await?;
                last_active_ms.store(now_epoch_ms(), Ordering::Relaxed);
            }
            other => {
                let resp = DaemonResponseLine {
                    id: req.id,
                    event: "error".to_owned(),
                    status: None,
                    chunk: None,
                    error: Some(format!("unknown daemon method `{other}`")),
                };
                write_response_line(&mut writer, &resp).await?;
            }
        }
    }

    Ok(())
}

async fn stream_synthesis_to_client(
    engine: &IndexTtsEngine,
    req_id: u64,
    synth_req: SynthesisRequest,
    writer: &mut tokio::net::unix::OwnedWriteHalf,
) -> Result<()> {
    let stream = match engine.synthesize_stream(synth_req) {
        Ok(s) => s,
        Err(err) => {
            let resp = DaemonResponseLine {
                id: req_id,
                event: "error".to_owned(),
                status: None,
                chunk: None,
                error: Some(err.to_string()),
            };
            write_response_line(writer, &resp).await?;
            return Ok(());
        }
    };

    while let Some(chunk_res) = stream.next_chunk().await {
        match chunk_res {
            Ok(chunk) => {
                let pcm_bytes = i16_slice_to_le_bytes(&chunk.pcm_i16);
                let payload = DaemonChunkPayload {
                    segment_index: chunk.segment_index,
                    total_segments: chunk.total_segments,
                    pcm_base64: BASE64.encode(&pcm_bytes),
                    sample_rate: chunk.sample_rate,
                    mel_tokens: chunk.mel_tokens,
                    timings: chunk.timings,
                };
                let resp = DaemonResponseLine {
                    id: req_id,
                    event: "chunk".to_owned(),
                    status: None,
                    chunk: Some(payload),
                    error: None,
                };
                if write_response_line(writer, &resp).await.is_err() {
                    break;
                }
            }
            Err(err) => {
                let resp = DaemonResponseLine {
                    id: req_id,
                    event: "error".to_owned(),
                    status: None,
                    chunk: None,
                    error: Some(err.to_string()),
                };
                let _ = write_response_line(writer, &resp).await;
                return Ok(());
            }
        }
    }

    let done = DaemonResponseLine {
        id: req_id,
        event: "done".to_owned(),
        status: None,
        chunk: None,
        error: None,
    };
    let _ = write_response_line(writer, &done).await;
    Ok(())
}

async fn write_response_line(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    resp: &DaemonResponseLine,
) -> Result<()> {
    let mut encoded = serde_json::to_vec(resp)?;
    encoded.push(b'\n');
    writer.write_all(&encoded).await?;
    writer.flush().await?;
    Ok(())
}

/// Queries the status of a running daemon at `socket_path`, returning `Ok(None)` if no daemon is
/// listening.
///
/// # Errors
///
/// Returns an error if the daemon returns a malformed payload.
pub async fn query_daemon_status(socket_path: &Path) -> Result<Option<DaemonStatusInfo>> {
    if !socket_path.exists() {
        return Ok(None);
    }
    let Ok(Ok(stream)) =
        timeout(Duration::from_millis(400), UnixStream::connect(socket_path)).await
    else {
        return Ok(None);
    };

    let (reader, mut writer) = stream.into_split();
    let req = DaemonRequest {
        id: 1,
        method: "status".to_owned(),
        synth_request: None,
    };
    let mut raw = serde_json::to_vec(&req)?;
    raw.push(b'\n');
    writer.write_all(&raw).await?;
    writer.flush().await?;

    let mut lines = BufReader::new(reader).lines();
    if let Some(line) = lines.next_line().await? {
        let resp: DaemonResponseLine = serde_json::from_str(&line)?;
        return Ok(resp.status);
    }
    Ok(None)
}

/// Sends a graceful shutdown command to the daemon at `socket_path`.
///
/// # Errors
///
/// Returns an error if communication with an active socket fails.
#[allow(clippy::disallowed_methods)]
pub async fn stop_daemon(socket_path: &Path) -> Result<bool> {
    if !socket_path.exists() {
        return Ok(false);
    }
    let Ok(Ok(stream)) =
        timeout(Duration::from_millis(400), UnixStream::connect(socket_path)).await
    else {
        let _ = fs::remove_file(socket_path);
        return Ok(false);
    };

    let (reader, mut writer) = stream.into_split();
    let req = DaemonRequest {
        id: 1,
        method: "stop".to_owned(),
        synth_request: None,
    };
    let mut raw = serde_json::to_vec(&req)?;
    raw.push(b'\n');
    writer.write_all(&raw).await?;
    writer.flush().await?;

    let mut lines = BufReader::new(reader).lines();
    if let Some(line) = lines.next_line().await? {
        let resp: DaemonResponseLine = serde_json::from_str(&line)?;
        return Ok(resp.event == "stopped");
    }
    Ok(true)
}

/// Attempts to stream synthesis via an active daemon at `socket_path`.
///
/// Returns `Ok(false)` if no daemon is listening so the caller can fall back to in-process
/// synthesis; returns `Ok(true)` if the daemon handled the request.
///
/// # Errors
///
/// Returns an error if the daemon reports a synthesis failure or `on_chunk` returns an error.
pub async fn try_synthesize_via_daemon<F>(
    socket_path: &Path,
    synth_req: &SynthesisRequest,
    mut on_chunk: F,
) -> Result<bool>
where
    F: FnMut(AudioChunk) -> Result<()>,
{
    if !socket_path.exists() {
        return Ok(false);
    }
    let Ok(Ok(stream)) =
        timeout(Duration::from_millis(400), UnixStream::connect(socket_path)).await
    else {
        return Ok(false);
    };

    let (reader, mut writer) = stream.into_split();
    let req = DaemonRequest {
        id: 1,
        method: "synth".to_owned(),
        synth_request: Some(synth_req.clone()),
    };
    let mut raw = serde_json::to_vec(&req)?;
    raw.push(b'\n');
    writer.write_all(&raw).await?;
    writer.flush().await?;

    let mut lines = BufReader::new(reader).lines();
    while let Some(line) = lines.next_line().await? {
        let resp: DaemonResponseLine = serde_json::from_str(&line)?;
        match resp.event.as_str() {
            "chunk" => {
                if let Some(payload) = resp.chunk {
                    let bytes = BASE64
                        .decode(&payload.pcm_base64)
                        .context("failed to decode base64 PCM from daemon")?;
                    let pcm_i16 = le_bytes_to_i16_vec(&bytes);
                    let chunk = AudioChunk::new(
                        payload.segment_index,
                        payload.total_segments,
                        pcm_i16,
                        payload.sample_rate,
                        payload.mel_tokens,
                        payload.timings,
                    );
                    on_chunk(chunk)?;
                }
            }
            "done" => return Ok(true),
            "error" => {
                let msg = resp
                    .error
                    .unwrap_or_else(|| "daemon synthesis error".to_owned());
                return Err(anyhow!("{msg}"));
            }
            _ => {}
        }
    }
    Ok(true)
}

#[allow(clippy::cast_possible_truncation)]
fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn i16_slice_to_le_bytes(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn le_bytes_to_i16_vec(bytes: &[u8]) -> Vec<i16> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for &pair in bytes.as_chunks::<2>().0 {
        out.push(i16::from_le_bytes(pair));
    }
    out
}

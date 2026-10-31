//! HTTP and OpenAI-compatible (`/v1/audio/speech`) speech synthesis server for `index-tts-rs`.
//!
//! Exposes:
//! - `GET /health`: Health and model snapshot status JSON.
//! - `POST /v1/audio/speech`: Synthesizes speech from `{ "input": "...", "voice": "...", "lang":
//!   "zh", "speed": 1.0 }` and returns `22,050 Hz` 16-bit mono `audio/wav`.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

use std::{net::SocketAddr, path::PathBuf, str::FromStr, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use clap::Parser;
use index_tts_rs_core::{
    config::{
        CfmParams, DurationFactor, EngineConfig, Language, SpeakerSource, SynthesisRequest,
        SynthesisText, VoiceId,
    },
    engine::IndexTtsEngine,
};
use serde::Deserialize;
use serde_json::json;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Semaphore,
    time::timeout,
};
use tracing::{info, warn};

/// Hard cap on HTTP request size (headers + body = 32 KiB).
const MAX_HTTP_REQUEST_BYTES: usize = 32_768;
/// Maximum concurrent HTTP connections.
const MAX_CONCURRENT_CONNECTIONS: usize = 8;
/// Per-request synthesis timeout.
const REQUEST_TIMEOUT_SECS: u64 = 120;

#[derive(Debug, Parser)]
#[command(
    name = "index-tts-rs-server",
    version,
    about = "OpenAI-compatible /v1/audio/speech HTTP server for IndexTTS-2.5 on Apple Silicon"
)]
struct ServerArgs {
    /// Bind address (e.g. `127.0.0.1:8080`).
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Optional YAML configuration file path.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Explicit local model directory override.
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SpeechPayload {
    input: String,
    voice: String,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = ServerArgs::parse();

    let mut cfg = match &args.config {
        Some(path) => EngineConfig::from_yaml_file(path)?,
        None => EngineConfig::default(),
    };
    if args.model_dir.is_some() {
        cfg.model_dir = args.model_dir;
    }
    cfg.validate_config()?;

    let engine = Arc::new(
        IndexTtsEngine::start(cfg)
            .context("failed to initialize IndexTtsEngine for HTTP server")?,
    );
    let listener = TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("failed to bind TCP listener on {}", args.bind))?;
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_CONNECTIONS));

    info!(
        bind = %args.bind,
        model_dir = %engine.model_dir().display(),
        "index-tts-rs-server listening"
    );

    loop {
        tokio::select! {
            accept_res = listener.accept() => {
                let (stream, peer) = match accept_res {
                    Ok(pair) => pair,
                    Err(err) => {
                        warn!(error = %err, "TCP accept error");
                        continue;
                    }
                };
                let Ok(permit) = Arc::clone(&semaphore).try_acquire_owned() else {
                    warn!(%peer, "Rejecting connection: concurrency limit reached");
                    continue;
                };
                let eng = Arc::clone(&engine);
                tokio::spawn(async move {
                    let _permit = permit;
                    let _ = timeout(
                        Duration::from_secs(REQUEST_TIMEOUT_SECS),
                        handle_http_connection(stream, eng),
                    )
                    .await;
                });
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Shutting down index-tts-rs-server");
                break;
            }
        }
    }

    engine.stop()?;
    Ok(())
}

async fn handle_http_connection(mut stream: TcpStream, engine: Arc<IndexTtsEngine>) -> Result<()> {
    let mut buf = vec![0_u8; MAX_HTTP_REQUEST_BYTES];
    let n = stream.read(&mut buf).await?;
    if n == 0 {
        return Ok(());
    }
    let raw = String::from_utf8_lossy(buf.get(..n).unwrap_or(&[]));
    let (header_part, body_part) = raw.split_once("\r\n\r\n").unwrap_or((&raw, ""));
    let first_line = header_part.lines().next().unwrap_or_default();
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();

    match (method, path) {
        ("GET", "/health") => {
            let body = serde_json::to_vec(&json!({
                "status": "ok",
                "running": engine.is_running(),
                "sampleRate": engine.config().sample_rate,
            }))?;
            write_http_response(&mut stream, 200, "OK", "application/json", &body).await
        }
        ("POST", "/v1/audio/speech") => {
            let parsed: SpeechPayload = match serde_json::from_str(body_part) {
                Ok(p) => p,
                Err(err) => {
                    let body = serde_json::to_vec(&json!({
                        "error": format!("invalid request JSON: {err}")
                    }))?;
                    return write_http_response(
                        &mut stream,
                        400,
                        "Bad Request",
                        "application/json",
                        &body,
                    )
                    .await;
                }
            };
            match synthesize_speech_payload(&engine, parsed).await {
                Ok(wav_bytes) => {
                    write_http_response(&mut stream, 200, "OK", "audio/wav", &wav_bytes).await
                }
                Err(err) => {
                    let body = serde_json::to_vec(&json!({
                        "error": err.to_string()
                    }))?;
                    write_http_response(
                        &mut stream,
                        422,
                        "Unprocessable Entity",
                        "application/json",
                        &body,
                    )
                    .await
                }
            }
        }
        _ => {
            let body = serde_json::to_vec(&json!({
                "error": "not found"
            }))?;
            write_http_response(&mut stream, 404, "Not Found", "application/json", &body).await
        }
    }
}

async fn synthesize_speech_payload(
    engine: &IndexTtsEngine,
    payload: SpeechPayload,
) -> Result<Vec<u8>> {
    let text = SynthesisText::new(payload.input)?;
    let v_path = PathBuf::from(&payload.voice);
    let speaker = if v_path.extension().and_then(|e| e.to_str()) == Some("itts_voice") {
        SpeakerSource::VoiceFile(v_path)
    } else if v_path.is_file() {
        SpeakerSource::ReferenceAudio(v_path)
    } else {
        SpeakerSource::VoiceId(VoiceId::new(payload.voice)?)
    };
    let language = match payload.lang {
        Some(l) => Language::from_str(&l)?,
        None => Language::Zh,
    };
    let duration_factor = match payload.speed {
        Some(s) if s > 0.0 => DurationFactor::new(1.0 / s)?,
        _ => DurationFactor::default(),
    };
    let cfm = CfmParams::builder()
        .duration_factor(duration_factor)
        .build();
    let req = SynthesisRequest::builder()
        .text(text)
        .speaker(speaker)
        .language(language)
        .cfm(cfm)
        .build();

    let res = engine.synthesize(req).await?;
    Ok(res.to_wav_bytes()?)
}

async fn write_http_response(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let header = format!(
        "HTTP/1.1 {status_code} {status_text}\r\nContent-Type: {content_type}\r\nContent-Length: \
         {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    Ok(())
}

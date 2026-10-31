//! Daemon lifecycle: detached start with readiness reporting, stop, and log access.
//!
//! `itts daemon start` re-executes the current binary as `itts daemon run` in its own process
//! group with stdin closed, stdout piped back for startup events, and stderr appended to the
//! daemon log. The parent renders progress and exits once the child reports `ready` (exit 0) or
//! `failed` (the error's exit code), so agents get a definitive answer from a single command
//! without holding a background shell job.

use std::{
    env,
    io::{self, SeekFrom, Write},
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use index_tts_rs_service::{ErrorCode, ServiceError};
use serde_json::Value;
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader},
    net::UnixStream,
    process::Command,
    time::{Instant, sleep, timeout},
};

use crate::daemon::{
    DaemonPaths,
    client::DaemonClient,
    protocol::{ReadyInfo, StartupLine},
    server::StartupReporter,
    transport::read_json,
};

/// Maximum time to wait for a detached daemon to become ready.
const START_TIMEOUT: Duration = Duration::from_secs(180);
/// Maximum time to wait for a stopping daemon to release its socket.
const STOP_TIMEOUT: Duration = Duration::from_secs(20);
/// Log size that triggers rotation to `<log>.1` at startup.
const MAX_LOG_BYTES: u64 = 10 * 1024 * 1024;
/// Log lines shown when a detached start fails.
const FAILURE_LOG_LINES: usize = 20;

/// Everything needed to launch `itts daemon run`.
#[derive(Debug, Clone)]
pub struct LaunchSpec {
    /// Global `--config` (absolute).
    pub config: Option<PathBuf>,
    /// Global `--model-dir` (absolute).
    pub model_dir: Option<PathBuf>,
    /// Global `--voices-dir` (absolute).
    pub voices_dir: Option<PathBuf>,
    /// Daemon files.
    pub paths: DaemonPaths,
    /// Idle timeout in seconds.
    pub idle_timeout_secs: u64,
    /// Maximum concurrent jobs.
    pub max_in_flight: usize,
}

/// Result of a start request.
#[derive(Debug, Clone)]
pub enum StartOutcome {
    /// A new daemon is ready.
    Started(Box<ReadyInfo>),
    /// A daemon was already running.
    AlreadyRunning {
        /// Its pid, when known.
        pid: Option<u32>,
    },
}

/// Spawns a detached daemon and waits until it is ready or fails.
///
/// # Errors
///
/// Returns the daemon's startup failure, or [`ErrorCode::Internal`] with the log tail if the
/// process exits without reporting.
pub async fn spawn_detached(
    spec: &LaunchSpec,
    mut on_progress: impl FnMut(&StartupLine),
) -> Result<StartOutcome, ServiceError> {
    let paths = &spec.paths;
    if let Some(parent) = paths.log.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .await
            .map_err(|err| io_error(parent, &err))?;
    }
    rotate_log(&paths.log).await;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.log)
        .await
        .map_err(|err| io_error(&paths.log, &err))?
        .into_std()
        .await;
    let exe = env::current_exe()
        .map_err(|err| ServiceError::internal(format!("cannot locate itts binary: {err}")))?;

    let mut command = Command::new(exe);
    for (flag, value) in [
        ("--config", &spec.config),
        ("--model-dir", &spec.model_dir),
        ("--voices-dir", &spec.voices_dir),
    ] {
        if let Some(value) = value {
            command.arg(flag).arg(value);
        }
    }
    command
        .args(["daemon", "run", "--socket"])
        .arg(&paths.socket)
        .arg("--idle-timeout-secs")
        .arg(spec.idle_timeout_secs.to_string())
        .arg("--max-in-flight")
        .arg(spec.max_in_flight.to_string())
        .current_dir("/")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(log))
        .process_group(0);
    let mut child = command
        .spawn()
        .map_err(|err| ServiceError::internal(format!("failed to spawn daemon: {err}")))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ServiceError::internal("daemon stdout was not captured"))?;

    let mut reader = BufReader::new(stdout);
    let mut buf = Vec::new();
    let deadline = Instant::now() + START_TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let line: Option<StartupLine> = timeout(remaining, read_json(&mut reader, &mut buf))
            .await
            .map_err(|_| {
                ServiceError::new(ErrorCode::Timeout, "daemon did not become ready in 180 s")
                    .with_hint(log_hint(&paths.log))
            })?
            .map_err(|err| {
                ServiceError::internal(format!("invalid daemon startup output: {err}"))
            })?;
        let Some(line) = line else {
            let status = child.wait().await.ok();
            let tail = tail_log(&paths.log, FAILURE_LOG_LINES).await.join("\n");
            return Err(ServiceError::internal(format!(
                "daemon exited during startup ({}):\n{tail}",
                status.map_or_else(|| "unknown status".to_owned(), |s| s.to_string())
            ))
            .with_hint(log_hint(&paths.log)));
        };
        on_progress(&line);
        match line {
            StartupLine::Progress { .. } => {}
            StartupLine::Ready(info) => return Ok(StartOutcome::Started(info)),
            StartupLine::AlreadyRunning { pid, .. } => {
                return Ok(StartOutcome::AlreadyRunning { pid });
            }
            StartupLine::Failed { error } => return Err(error),
        }
    }
}

/// Stops the daemon at `paths`. Returns `false` if none was running.
///
/// # Errors
///
/// Returns [`ErrorCode::Timeout`] if the daemon does not exit in time.
pub async fn stop_daemon(paths: &DaemonPaths) -> Result<bool, ServiceError> {
    match DaemonClient::connect(&paths.socket).await {
        Ok(Some(mut client)) => client.stop().await?,
        Ok(None) => return Ok(false),
        Err(error) if error.code == ErrorCode::DaemonVersionMismatch => {
            legacy_stop(&paths.socket).await?;
        }
        Err(error) => return Err(error),
    }
    wait_until_stopped(paths).await?;
    Ok(true)
}

/// Sends a protocol-1 `stop` so an older daemon can be replaced after an upgrade. Protocol-1
/// daemons only shut down after successfully writing their reply, so the reply is awaited.
async fn legacy_stop(socket: &Path) -> Result<(), ServiceError> {
    let stream = UnixStream::connect(socket)
        .await
        .map_err(|err| io_error(socket, &err))?;
    let (read_half, mut write_half) = stream.into_split();
    write_half
        .write_all(b"{\"id\":1,\"method\":\"stop\"}\n")
        .await
        .map_err(|err| io_error(socket, &err))?;
    let mut reader = BufReader::new(read_half);
    let mut buf = Vec::new();
    let reply: Option<Value> = timeout(Duration::from_secs(5), read_json(&mut reader, &mut buf))
        .await
        .map_err(|_| ServiceError::new(ErrorCode::Timeout, "older daemon did not answer stop"))?
        .map_err(|err| io_error(socket, &err))?;
    match reply {
        Some(_) => Ok(()),
        None => Err(ServiceError::new(
            ErrorCode::DaemonUnavailable,
            "older daemon closed the connection without stopping",
        )),
    }
}

async fn wait_until_stopped(paths: &DaemonPaths) -> Result<(), ServiceError> {
    let deadline = Instant::now() + STOP_TIMEOUT;
    while paths.socket.exists() {
        if Instant::now() >= deadline {
            return Err(
                ServiceError::new(ErrorCode::Timeout, "daemon did not stop within 20 s")
                    .with_hint(log_hint(&paths.log)),
            );
        }
        sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}

/// Returns the last `lines` lines of the daemon log.
pub async fn tail_log(path: &Path, lines: usize) -> Vec<String> {
    let Ok(content) = fs::read_to_string(path).await else {
        return Vec::new();
    };
    let all: Vec<&str> = content.lines().collect();
    let skip = all.len().saturating_sub(lines);
    all.into_iter().skip(skip).map(str::to_owned).collect()
}

/// Prints the log tail, then follows appended lines until interrupted when `follow` is set.
///
/// # Errors
///
/// Returns [`ErrorCode::Io`] if the log cannot be read or stdout fails.
pub async fn show_logs(path: &Path, lines: usize, follow: bool) -> Result<(), ServiceError> {
    if !fs::try_exists(path).await.unwrap_or(false) {
        return Err(ServiceError::new(
            ErrorCode::Io,
            format!("no daemon log at `{}`", path.display()),
        )
        .with_hint("start the daemon with `itts daemon start`"));
    }
    let mut out = io::stdout().lock();
    for line in tail_log(path, lines).await {
        writeln!(out, "{line}").map_err(|err| io_error(path, &err))?;
    }
    drop(out);
    if !follow {
        return Ok(());
    }
    let mut offset = fs::metadata(path)
        .await
        .map_err(|err| io_error(path, &err))?
        .len();
    let mut fresh = Vec::new();
    loop {
        sleep(Duration::from_millis(500)).await;
        let len = fs::metadata(path).await.map_or(0, |m| m.len());
        if len < offset {
            // Rotated or truncated: start over from the beginning of the new file.
            offset = 0;
        }
        if len > offset {
            let mut file = fs::File::open(path)
                .await
                .map_err(|err| io_error(path, &err))?;
            file.seek(SeekFrom::Start(offset))
                .await
                .map_err(|err| io_error(path, &err))?;
            fresh.clear();
            file.read_to_end(&mut fresh)
                .await
                .map_err(|err| io_error(path, &err))?;
            let mut out = io::stdout().lock();
            out.write_all(&fresh).map_err(|err| io_error(path, &err))?;
            out.flush().map_err(|err| io_error(path, &err))?;
            offset = offset.saturating_add(u64::try_from(fresh.len()).unwrap_or(u64::MAX));
        }
    }
}

async fn rotate_log(path: &Path) {
    if fs::metadata(path)
        .await
        .is_ok_and(|m| m.len() > MAX_LOG_BYTES)
    {
        let _ = fs::rename(path, path.with_extension("log.1")).await;
    }
}

fn log_hint(log: &Path) -> String {
    format!("inspect `{}` or run `itts daemon logs`", log.display())
}

fn io_error(path: &Path, err: &io::Error) -> ServiceError {
    ServiceError::new(ErrorCode::Io, format!("`{}`: {err}", path.display()))
}

/// Abbreviates `$HOME` as `~` for display.
pub fn display_path(path: &Path) -> String {
    if let Ok(home) = env::var("HOME")
        && let Ok(rest) = path.strip_prefix(&home)
    {
        return format!("~/{}", rest.display());
    }
    path.display().to_string()
}

/// Renders one startup progress line for humans.
///
/// # Errors
///
/// Returns the writer's I/O error.
pub fn render_progress(out: &mut dyn Write, line: &StartupLine) -> io::Result<()> {
    match line {
        StartupLine::Progress {
            step,
            detail,
            elapsed_ms,
        } => {
            let detail = detail
                .as_deref()
                .map_or_else(String::new, |d| display_path(Path::new(d)));
            let line = match elapsed_ms {
                Some(ms) => format!("  ▸ {step:<20} {ms:>7.0} ms {detail}"),
                None => format!("  ▸ {step:<20} {detail}"),
            };
            writeln!(out, "{}", line.trim_end())
        }
        StartupLine::AlreadyRunning { pid, socket_path } => writeln!(
            out,
            "✔ itts daemon already running{} · socket {}",
            pid.map_or_else(String::new, |p| format!(" (pid {p})")),
            display_path(socket_path)
        ),
        StartupLine::Ready(_) | StartupLine::Failed { .. } => Ok(()),
    }
}

/// Renders the readiness summary for humans.
///
/// # Errors
///
/// Returns the writer's I/O error.
pub fn render_ready(out: &mut dyn Write, info: &ReadyInfo) -> io::Result<()> {
    writeln!(
        out,
        "✔ itts daemon {} ready in {:.0} ms",
        info.version, info.load_ms
    )?;
    writeln!(out, "  pid      {}", info.pid)?;
    writeln!(out, "  socket   {}", display_path(&info.socket_path))?;
    writeln!(out, "  model    {}", display_path(&info.model_dir))?;
    if info.voices.is_empty() {
        writeln!(
            out,
            "  voices   none in {} — add one with `itts voice add <ID> --ref <AUDIO>`",
            display_path(&info.voices_dir)
        )?;
    } else {
        writeln!(
            out,
            "  voices   {} — {}",
            info.voices.len(),
            info.voices.join(", ")
        )?;
    }
    if info.idle_timeout_secs == 0 {
        writeln!(out, "  idle     never auto-stops")?;
    } else {
        writeln!(
            out,
            "  idle     auto-stops after {} s without jobs",
            info.idle_timeout_secs
        )?;
    }
    writeln!(out, "  log      {}", display_path(&info.log_path))
}

/// Human-readable startup reporter writing to stderr (foreground daemons).
#[derive(Debug, Default)]
pub struct HumanReporter;

impl StartupReporter for HumanReporter {
    fn report(&mut self, line: &StartupLine) {
        let mut err = io::stderr().lock();
        let _ = match line {
            StartupLine::Ready(info) => render_ready(&mut err, info),
            other => render_progress(&mut err, other),
        };
    }
}

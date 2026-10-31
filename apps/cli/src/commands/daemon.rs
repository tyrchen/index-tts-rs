//! `itts daemon {start,run,status,stop,restart,ensure,logs}`.

use std::{
    io::{self, Write},
    path::PathBuf,
};

use anyhow::{Result, anyhow};
use clap::{Args, Subcommand};
use index_tts_rs_core::config::EngineConfig;
use index_tts_rs_service::{ErrorCode, ServiceError};
use serde_json::json;

use crate::{
    GlobalPaths,
    daemon::{
        DEFAULT_IDLE_TIMEOUT_SECS, DEFAULT_MAX_IN_FLIGHT, DaemonPaths,
        client::DaemonClient,
        default_socket_path,
        lifecycle::{
            HumanReporter, LaunchSpec, StartOutcome, display_path, render_progress, render_ready,
            show_logs, spawn_detached, stop_daemon,
        },
        protocol::{DaemonState, DaemonStatus, StartupLine},
        server::{DaemonOptions, JsonLinesReporter, StartupReporter, run_daemon},
    },
    output::OutputMode,
};

/// Options shared by commands that may start a daemon.
#[derive(Debug, Clone, Args)]
pub struct StartOptions {
    /// Idle auto-shutdown timeout in seconds (`0` keeps the daemon running).
    #[arg(long, default_value_t = DEFAULT_IDLE_TIMEOUT_SECS)]
    idle_timeout_secs: u64,
    /// Maximum concurrent requests before the daemon answers `BUSY`.
    #[arg(long, default_value_t = DEFAULT_MAX_IN_FLIGHT, value_parser = parse_max_in_flight)]
    max_in_flight: usize,
    /// Daemon socket path.
    #[arg(long)]
    socket: Option<PathBuf>,
}

impl StartOptions {
    fn paths(&self) -> DaemonPaths {
        DaemonPaths::new(&self.socket.clone().unwrap_or_else(default_socket_path))
    }

    fn launch_spec(&self, globals: &GlobalPaths) -> LaunchSpec {
        LaunchSpec {
            config: globals.config.clone(),
            model_dir: globals.model_dir.clone(),
            voices_dir: globals.voices_dir.clone(),
            paths: self.paths(),
            idle_timeout_secs: self.idle_timeout_secs,
            max_in_flight: self.max_in_flight,
        }
    }

    fn daemon_options(&self) -> DaemonOptions {
        DaemonOptions {
            paths: self.paths(),
            idle_timeout_secs: self.idle_timeout_secs,
            max_in_flight: self.max_in_flight,
        }
    }
}

fn parse_max_in_flight(raw: &str) -> Result<usize, String> {
    match raw.parse::<usize>() {
        Ok(value @ 1..=256) => Ok(value),
        _ => Err("expected an integer in 1..=256".to_owned()),
    }
}

/// Socket selection for control commands.
#[derive(Debug, Clone, Args)]
pub struct SocketOption {
    /// Daemon socket path.
    #[arg(long)]
    socket: Option<PathBuf>,
}

impl SocketOption {
    fn paths(&self) -> DaemonPaths {
        DaemonPaths::new(&self.socket.clone().unwrap_or_else(default_socket_path))
    }
}

/// Daemon subcommands.
#[derive(Debug, Subcommand)]
pub enum DaemonCommand {
    /// Start the warm-model daemon in the background and wait until it is ready.
    Start {
        /// Run in the foreground instead (for launchd or debugging).
        #[arg(long)]
        foreground: bool,
        #[command(flatten)]
        options: StartOptions,
    },
    /// Run the daemon in this process, reporting startup as JSON lines (used by `start`).
    #[command(hide = true)]
    Run {
        #[command(flatten)]
        options: StartOptions,
    },
    /// Show daemon state, loaded models, voices, and job counters.
    Status {
        #[command(flatten)]
        socket: SocketOption,
    },
    /// Stop the daemon gracefully.
    Stop {
        #[command(flatten)]
        socket: SocketOption,
    },
    /// Stop and start the daemon (e.g. after upgrading itts).
    Restart {
        #[command(flatten)]
        options: StartOptions,
    },
    /// Start the daemon only if it is not already running; idempotent.
    Ensure {
        #[command(flatten)]
        options: StartOptions,
    },
    /// Print the daemon log.
    Logs {
        /// Number of trailing lines.
        #[arg(long, short = 'n', default_value_t = 50)]
        lines: usize,
        /// Keep printing appended lines.
        #[arg(long, short = 'f')]
        follow: bool,
        #[command(flatten)]
        socket: SocketOption,
    },
}

/// Runs a daemon subcommand.
///
/// # Errors
///
/// Returns the classified failure.
pub async fn run(
    config: EngineConfig,
    globals: &GlobalPaths,
    command: DaemonCommand,
    output: OutputMode,
) -> Result<()> {
    match command {
        DaemonCommand::Start {
            foreground: true,
            options,
        } => {
            let mut human = HumanReporter;
            let mut json = JsonLinesReporter;
            let reporter: &mut dyn StartupReporter = if output.is_json() {
                &mut json
            } else {
                &mut human
            };
            Ok(run_daemon(config, options.daemon_options(), reporter).await?)
        }
        DaemonCommand::Run { options } => {
            Ok(run_daemon(config, options.daemon_options(), &mut JsonLinesReporter).await?)
        }
        DaemonCommand::Start {
            foreground: false,
            options,
        } => start(&options.launch_spec(globals), output).await,
        DaemonCommand::Status { socket } => status(&socket.paths(), output).await,
        DaemonCommand::Stop { socket } => {
            let paths = socket.paths();
            let stopped = stop_daemon(&paths).await?;
            output.emit(
                &json!({ "status": "ok", "stopped": stopped, "socketPath": paths.socket }),
                |out| {
                    if stopped {
                        writeln!(out, "✔ daemon stopped")
                    } else {
                        writeln!(
                            out,
                            "daemon was not running ({})",
                            display_path(&paths.socket)
                        )
                    }
                },
            )
        }
        DaemonCommand::Restart { options } => {
            let spec = options.launch_spec(globals);
            if stop_daemon(&spec.paths).await? && output == OutputMode::Text {
                let _ = writeln!(io::stderr(), "  ▸ stopped previous daemon");
            }
            start(&spec, output).await
        }
        DaemonCommand::Ensure { options } => {
            let spec = options.launch_spec(globals);
            match DaemonClient::connect(&spec.paths.socket).await? {
                Some(mut client) => render_status(&client.status().await?, output),
                None => start(&spec, output).await,
            }
        }
        DaemonCommand::Logs {
            lines,
            follow,
            socket,
        } => Ok(show_logs(&socket.paths().log, lines, follow).await?),
    }
}

async fn start(spec: &LaunchSpec, output: OutputMode) -> Result<()> {
    let outcome = spawn_detached(spec, |line| match (output, line) {
        (OutputMode::Jsonl, line) => {
            let _ = OutputMode::emit_line(line);
        }
        (OutputMode::Text, line @ StartupLine::Progress { .. }) => {
            let _ = render_progress(&mut io::stderr().lock(), line);
        }
        _ => {}
    })
    .await?;
    match outcome {
        StartOutcome::Started(info) => match output {
            OutputMode::Jsonl => Ok(()),
            OutputMode::Json => output.emit(&info, |_| Ok(())),
            OutputMode::Text => Ok(render_ready(&mut io::stdout().lock(), &info)?),
        },
        StartOutcome::AlreadyRunning { pid } => {
            if output == OutputMode::Jsonl {
                return Ok(());
            }
            let mut client = DaemonClient::connect(&spec.paths.socket)
                .await?
                .ok_or_else(|| {
                    anyhow!(
                        ServiceError::new(
                            ErrorCode::DaemonUnavailable,
                            format!("daemon (pid {pid:?}) holds the lock but is not answering"),
                        )
                        .with_hint("run `itts daemon logs`, or `itts daemon stop` and start again")
                    )
                })?;
            render_status(&client.status().await?, output)
        }
    }
}

async fn status(paths: &DaemonPaths, output: OutputMode) -> Result<()> {
    match DaemonClient::connect(&paths.socket).await? {
        Some(mut client) => render_status(&client.status().await?, output),
        None => output.emit(
            &json!({ "running": false, "socketPath": paths.socket }),
            |out| {
                writeln!(
                    out,
                    "daemon is not running ({})",
                    display_path(&paths.socket)
                )?;
                writeln!(out, "  hint: start it with `itts daemon start`")
            },
        ),
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "memory sizes are displayed approximately"
)]
fn render_status(status: &DaemonStatus, output: OutputMode) -> Result<()> {
    let value = json!({ "running": true, "status": status });
    output.emit(&value, |out| {
        let state = match status.state {
            DaemonState::Ready => "ready",
            DaemonState::Loading => "loading models",
        };
        writeln!(
            out,
            "✔ itts daemon {} {state} · pid {} · up {} s · idle {} s{}",
            status.version,
            status.pid,
            status.uptime_secs,
            status.idle_secs,
            if status.idle_timeout_secs == 0 {
                " (never auto-stops)".to_owned()
            } else {
                format!(" / {} s", status.idle_timeout_secs)
            }
        )?;
        writeln!(out, "  socket   {}", display_path(&status.socket_path))?;
        if let Some(dir) = &status.model_dir {
            writeln!(out, "  model    {}", display_path(dir))?;
        }
        if let Some(engine) = &status.engine {
            writeln!(
                out,
                "  memory   {:.2} GiB active · {:.2} GiB peak · voice extractor {}",
                engine.active_memory_bytes as f64 / f64::from(1_u32 << 30),
                engine.peak_memory_bytes as f64 / f64::from(1_u32 << 30),
                if engine.voice_extractor_loaded {
                    "loaded"
                } else {
                    "evicted"
                }
            )?;
        }
        let names: Vec<&str> = status.voices.iter().map(|v| v.id.as_str()).collect();
        writeln!(
            out,
            "  voices   {} in {}{}",
            names.len(),
            display_path(&status.voices_dir),
            if names.is_empty() {
                String::new()
            } else {
                format!(" — {}", names.join(", "))
            }
        )?;
        let jobs = &status.jobs;
        writeln!(
            out,
            "  jobs     {} in flight (max {}) · {} done · {} failed · {} cancelled",
            jobs.in_flight, jobs.max_in_flight, jobs.completed, jobs.failed, jobs.cancelled
        )?;
        if let Some(last) = &status.last_error {
            writeln!(
                out,
                "  last err {} {}",
                last.error.code.as_str(),
                last.error.message
            )?;
        }
        writeln!(out, "  log      {}", display_path(&status.log_path))
    })
}

//! `itts listen`: verbatim local speech recognition of a recording (warm daemon or
//! in-process; in-process loads only the recognizer, not the MLX models).

use std::{
    env,
    io::{self, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::Args;
use index_tts_rs_asr::AsrEngine;
use index_tts_rs_core::config::EngineConfig;
use index_tts_rs_service::{
    Backend, ListenJob, ListenResult, PathPolicy, ServiceError, run_listen,
};
use serde::Serialize;

use crate::{
    commands::warn_daemon_unusable,
    daemon::{client::DaemonClient, default_socket_path},
    output::OutputMode,
};

/// `itts listen` arguments.
#[derive(Debug, Args)]
pub struct ListenArgs {
    /// Recording to recognize (`.wav`, `.mp3`, `.flac`, `.m4a`; any sample rate; at most 120 s).
    audio: PathBuf,
    /// Spoken language (`zh`, `zh/en`, `yue`, `ja`, `en`; default from config).
    #[arg(long, short = 'l')]
    lang: Option<String>,
    /// Recognizer: `paraformer` (Mandarin default), `sense-voice` (zh / yue / ja / en; default for
    /// yue and ja), `parakeet` (English default).
    #[arg(long, short = 'm')]
    model: Option<String>,
    /// Run in-process instead of using the warm daemon.
    #[arg(long)]
    no_daemon: bool,
    /// Daemon socket path.
    #[arg(long)]
    socket: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ListenOutcome<'a> {
    status: &'static str,
    backend: Backend,
    #[serde(flatten)]
    result: &'a ListenResult,
}

/// Runs `itts listen`.
///
/// # Errors
///
/// Returns the classified recognition failure.
pub async fn run(config: EngineConfig, args: ListenArgs, output: OutputMode) -> Result<()> {
    let mut job = ListenJob::builder().audio(args.audio.clone()).build();
    job.lang.clone_from(&args.lang);
    job.model.clone_from(&args.model);
    job.resolve_paths(&env::current_dir().context("cannot read current directory")?)?;
    let socket = args.socket.clone().unwrap_or_else(default_socket_path);
    let (result, backend) = execute(&config, &job, &socket, args.no_daemon).await?;
    let outcome = ListenOutcome {
        status: "ok",
        backend,
        result: &result,
    };
    output.emit(&outcome, |out| render(out, &outcome))
}

async fn execute(
    config: &EngineConfig,
    job: &ListenJob,
    socket: &Path,
    no_daemon: bool,
) -> Result<(ListenResult, Backend)> {
    if !no_daemon {
        match DaemonClient::connect(socket).await {
            Ok(Some(mut client)) => return Ok((client.listen(job).await?, Backend::Daemon)),
            Ok(None) => {}
            Err(error) => warn_daemon_unusable(&error, "recognizing in-process instead"),
        }
    }
    let prepared = job.prepare(config, PathPolicy::LocalAbsolute)?;
    let asr = AsrEngine::start().map_err(ServiceError::from)?;
    let result = run_listen(&asr, prepared).await;
    asr.stop().map_err(ServiceError::from)?;
    Ok((result?, Backend::InProcess))
}

fn render(out: &mut dyn Write, outcome: &ListenOutcome<'_>) -> io::Result<()> {
    let result = outcome.result;
    let via = match outcome.backend {
        Backend::Daemon => "daemon",
        _ => "in-process",
    };
    writeln!(
        out,
        "✔ heard {:.2} s of audio in {:.2} s ({}, {via}{})",
        result.duration_secs,
        result.listen_secs,
        result.recognizer.model,
        if result.model_downloaded {
            ", model downloaded"
        } else {
            ""
        }
    )?;
    writeln!(out, "  {}", result.text)?;
    let line: Vec<String> = result
        .tokens
        .iter()
        .map(|token| format!("{}@{:.2}", token.text, token.start))
        .collect();
    writeln!(out, "  tokens: {}", line.join(" "))
}

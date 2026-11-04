//! `itts verify`: check that a recording says its script — forced alignment and verbatim
//! recognition at once, compared into findings (warm daemon or in-process).

use std::{
    env,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use clap::Args;
use index_tts_rs_asr::AsrEngine;
use index_tts_rs_core::{config::EngineConfig, engine::IndexTtsEngine};
use index_tts_rs_service::{
    Backend, PathPolicy, ServiceError, VerifyJob, VerifyResult, run_verify,
};
use serde::Serialize;
use tokio::task::spawn_blocking;

use crate::{
    commands::{
        synth::{MAX_STDIN_TEXT_BYTES, read_text},
        warn_daemon_unusable,
    },
    daemon::{client::DaemonClient, default_socket_path},
    output::OutputMode,
};

/// `itts verify` arguments.
#[derive(Debug, Args)]
pub struct VerifyArgs {
    /// Recording to check (`.wav`, `.mp3`, `.flac`, `.m4a`; any sample rate; at most ~72 s).
    audio: PathBuf,
    /// Script the recording should say (or use --stdin); `<字|pinyin>` overrides are honoured
    /// and `[[...]]` controls are removed.
    #[arg(long, short = 't', conflicts_with = "stdin")]
    text: Option<String>,
    /// Read the script from standard input.
    #[arg(long)]
    stdin: bool,
    /// Language (`zh`, `zh/en`, `yue`, `ja`, `en`; default from config).
    #[arg(long, short = 'l')]
    lang: Option<String>,
    /// Recognizer: `paraformer` (Mandarin default), `sense-voice` (zh / yue / ja / en; default for
    /// yue and ja), `parakeet` (English default).
    #[arg(long, short = 'm')]
    model: Option<String>,
    /// Condition the aligner on a registered voice instead of the recording itself.
    #[arg(long, short = 'v', conflicts_with = "voice_file")]
    voice: Option<String>,
    /// Condition the aligner on an `.itts_voice` snapshot instead of the recording itself.
    #[arg(long)]
    voice_file: Option<PathBuf>,
    /// Run in-process instead of using the warm daemon.
    #[arg(long)]
    no_daemon: bool,
    /// Daemon socket path.
    #[arg(long)]
    socket: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VerifyOutcome<'a> {
    status: &'static str,
    backend: Backend,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_load_ms: Option<f64>,
    #[serde(flatten)]
    result: &'a VerifyResult,
}

/// Runs `itts verify`.
///
/// # Errors
///
/// Returns the classified failure. Findings do not fail the command.
pub async fn run(config: EngineConfig, args: VerifyArgs, output: OutputMode) -> Result<()> {
    let mut job = VerifyJob::builder()
        .audio(args.audio.clone())
        .input(read_text(
            args.text.as_deref(),
            args.stdin,
            MAX_STDIN_TEXT_BYTES,
        )?)
        .build();
    job.lang.clone_from(&args.lang);
    job.model.clone_from(&args.model);
    job.voice.clone_from(&args.voice);
    job.voice_file.clone_from(&args.voice_file);
    job.resolve_paths(&env::current_dir().context("cannot read current directory")?)?;
    let socket = args.socket.clone().unwrap_or_else(default_socket_path);
    let (result, backend, model_load_ms) = execute(config, &job, &socket, args.no_daemon).await?;
    let outcome = VerifyOutcome {
        status: "ok",
        backend,
        model_load_ms,
        result: &result,
    };
    output.emit(&outcome, |out| render(out, &outcome))
}

async fn execute(
    config: EngineConfig,
    job: &VerifyJob,
    socket: &Path,
    no_daemon: bool,
) -> Result<(VerifyResult, Backend, Option<f64>)> {
    if !no_daemon {
        match DaemonClient::connect(socket).await {
            Ok(Some(mut client)) => return Ok((client.verify(job).await?, Backend::Daemon, None)),
            Ok(None) => {}
            Err(error) => warn_daemon_unusable(&error, "verifying in-process instead"),
        }
    }
    let started = Instant::now();
    let engine = spawn_blocking(move || IndexTtsEngine::start(config).map_err(ServiceError::from))
        .await
        .context("engine startup task failed")??;
    let model_load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let asr = AsrEngine::start().map_err(ServiceError::from)?;
    let result = match job.prepare(engine.config(), PathPolicy::LocalAbsolute) {
        Ok(prepared) => run_verify(&engine, &asr, prepared).await,
        Err(err) => Err(err),
    };
    let asr_stopped = asr.stop().map_err(ServiceError::from);
    let stopped = spawn_blocking(move || engine.stop().map_err(ServiceError::from))
        .await
        .context("engine shutdown task failed")?;
    let result = result?;
    stopped?;
    asr_stopped?;
    Ok((result, Backend::InProcess, Some(model_load_ms)))
}

fn render(out: &mut dyn Write, outcome: &VerifyOutcome<'_>) -> io::Result<()> {
    let result = outcome.result;
    let via = match outcome.backend {
        Backend::Daemon => "daemon".to_owned(),
        _ => format!(
            "in-process, model load {:.0} ms",
            outcome.model_load_ms.unwrap_or_default()
        ),
    };
    let mark = if result.findings.is_empty() {
        "✔"
    } else {
        "⚠"
    };
    writeln!(
        out,
        "{mark} {}: {} finding(s) in {:.2} s of audio, checked in {:.2} s ({}, {via})",
        result.verdict.as_str(),
        result.findings.len(),
        result.duration_secs,
        result.verify_secs,
        result.heard.recognizer.model,
    )?;
    for finding in &result.findings {
        writeln!(
            out,
            "  [{}] {} {:?} → {:?} @{:.2}–{:.2}",
            finding.severity.as_str(),
            finding.kind.as_str(),
            finding.expected,
            finding.heard,
            finding.start,
            finding.end
        )?;
    }
    writeln!(out, "  heard: {}", result.heard.text)?;
    let quality = &result.align.quality;
    writeln!(
        out,
        "  align: confidence {:.2} · coverage {:.2} · weak {:?}",
        quality.confidence, quality.coverage, quality.weak
    )
}

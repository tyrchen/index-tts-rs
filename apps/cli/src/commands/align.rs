//! `itts align`: force-align a recording to the text it speaks (warm daemon or in-process).

use std::{
    env,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use clap::Args;
use index_tts_rs_core::{config::EngineConfig, engine::IndexTtsEngine};
use index_tts_rs_service::{AlignJob, AlignResult, Backend, PathPolicy, ServiceError, run_align};
use serde::Serialize;
use tokio::task::spawn_blocking;

use crate::{
    commands::synth::{MAX_STDIN_TEXT_BYTES, read_text},
    daemon::{client::DaemonClient, default_socket_path},
    output::OutputMode,
};

/// `itts align` arguments.
#[derive(Debug, Args)]
pub struct AlignArgs {
    /// Recording to align (`.wav`, `.mp3`, `.flac`, `.m4a`; any sample rate; at most ~72 s).
    #[arg(long, short = 'a')]
    audio: PathBuf,
    /// Text the recording speaks (or use --stdin); `<字|pinyin>` overrides are honoured and
    /// `[[...]]` controls are removed.
    #[arg(long, short = 't', conflicts_with = "stdin")]
    text: Option<String>,
    /// Read the text from standard input.
    #[arg(long)]
    stdin: bool,
    /// Language (`zh`, `en`, `ja`, `yue`, `es`, `ar`, `zh/en`; default from config).
    #[arg(long, short = 'l')]
    lang: Option<String>,
    /// Condition the decoder on a registered voice instead of the recording itself.
    #[arg(long, short = 'v', conflicts_with = "voice_file")]
    voice: Option<String>,
    /// Condition the decoder on an `.itts_voice` snapshot instead of the recording itself.
    #[arg(long)]
    voice_file: Option<PathBuf>,
    /// Also report phone timings (pinyin initials/finals, English `ARPAbet`).
    #[arg(long)]
    phones: bool,
    /// Run in-process instead of using the warm daemon.
    #[arg(long)]
    no_daemon: bool,
    /// Daemon socket path.
    #[arg(long)]
    socket: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AlignOutcome<'a> {
    status: &'static str,
    backend: Backend,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_load_ms: Option<f64>,
    #[serde(flatten)]
    result: &'a AlignResult,
}

/// Runs `itts align`.
///
/// # Errors
///
/// Returns the classified alignment failure.
pub async fn run(config: EngineConfig, args: AlignArgs, output: OutputMode) -> Result<()> {
    let mut job = AlignJob::builder()
        .audio(args.audio.clone())
        .input(read_text(
            args.text.as_deref(),
            args.stdin,
            MAX_STDIN_TEXT_BYTES,
        )?)
        .phones(args.phones)
        .build();
    job.lang.clone_from(&args.lang);
    job.voice.clone_from(&args.voice);
    job.voice_file.clone_from(&args.voice_file);
    job.resolve_paths(&env::current_dir().context("cannot read current directory")?)?;
    let socket = args.socket.clone().unwrap_or_else(default_socket_path);
    let (result, backend, model_load_ms) = execute(config, &job, &socket, args.no_daemon).await?;
    let outcome = AlignOutcome {
        status: "ok",
        backend,
        model_load_ms,
        result: &result,
    };
    output.emit(&outcome, |out| render(out, &outcome))
}

async fn execute(
    config: EngineConfig,
    job: &AlignJob,
    socket: &Path,
    no_daemon: bool,
) -> Result<(AlignResult, Backend, Option<f64>)> {
    if !no_daemon {
        match DaemonClient::connect(socket).await {
            Ok(Some(mut client)) => return Ok((client.align(job).await?, Backend::Daemon, None)),
            Ok(None) => {}
            Err(error) => {
                let mut err = io::stderr().lock();
                let _ = writeln!(
                    err,
                    "⚠ daemon unusable [{}]: {}; aligning in-process instead",
                    error.code.as_str(),
                    error.message
                );
                if let Some(hint) = &error.hint {
                    let _ = writeln!(err, "  hint: {hint}");
                }
            }
        }
    }
    let started = Instant::now();
    let engine = spawn_blocking(move || IndexTtsEngine::start(config).map_err(ServiceError::from))
        .await
        .context("engine startup task failed")??;
    let model_load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let result = match job.prepare(engine.config(), PathPolicy::LocalAbsolute) {
        Ok(prepared) => run_align(&engine, prepared).await,
        Err(err) => Err(err),
    };
    let stopped = spawn_blocking(move || engine.stop().map_err(ServiceError::from))
        .await
        .context("engine shutdown task failed")?;
    let result = result?;
    stopped?;
    Ok((result, Backend::InProcess, Some(model_load_ms)))
}

fn render(out: &mut dyn Write, outcome: &AlignOutcome<'_>) -> io::Result<()> {
    let result = outcome.result;
    let via = match outcome.backend {
        Backend::Daemon => "daemon".to_owned(),
        _ => format!(
            "in-process, model load {:.0} ms",
            outcome.model_load_ms.unwrap_or_default()
        ),
    };
    writeln!(
        out,
        "✔ aligned {:.2} s of audio in {:.2} s ({via}) · confidence {:.2} · coverage {:.2}",
        result.duration_secs, result.align_secs, result.quality.confidence, result.quality.coverage
    )?;
    if !result.quality.weak.is_empty() {
        let chars: Vec<char> = result.alignment.text.chars().collect();
        let weak: Vec<String> = result
            .quality
            .weak
            .iter()
            .map(|&i| format!("{}@{i}", chars.get(i).copied().unwrap_or('?')))
            .collect();
        writeln!(out, "  weak: {}", weak.join(" "))?;
    }
    for sentence in &result.sentences {
        writeln!(
            out,
            "  {:>7.2}–{:<7.2} {}",
            sentence.start, sentence.end, sentence.text
        )?;
    }
    if let Some(phones) = &result.phones {
        let line: Vec<String> = phones
            .iter()
            .map(|phone| {
                let tone = phone.t.map(|t| t.to_string()).unwrap_or_default();
                format!("{}{tone}@{:.2}", phone.p, phone.start)
            })
            .collect();
        writeln!(out, "  phones: {}", line.join(" "))?;
    }
    Ok(())
}

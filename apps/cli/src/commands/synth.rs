//! `itts synth`: one request model, two backends (warm daemon or in-process engine).

use std::{
    env,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, anyhow};
use clap::Args;
use index_tts_rs_core::{config::EngineConfig, engine::IndexTtsEngine};
use index_tts_rs_service::{
    AdvancedOptions, Backend, ChunkInfo, JobEvent, JobStarted, JobSummary, PathPolicy,
    ServiceError, SpeechJob, run_job,
};
use serde::Serialize;
use tokio::task::spawn_blocking;

use crate::{
    daemon::{client::DaemonClient, default_socket_path, lifecycle::display_path},
    output::OutputMode,
    player::Playback,
};

/// Bytes read from stdin at most (16 KiB text limit plus one byte to detect overflow).
const MAX_STDIN_BYTES: u64 = 16_385;

/// `itts synth` arguments.
#[derive(Debug, Args)]
#[allow(clippy::struct_excessive_bools, reason = "independent CLI flags")]
pub struct SynthArgs {
    /// Text to synthesize; supports inline controls such as `[[pause=600]]` (or use --stdin).
    #[arg(long, short = 't', conflicts_with = "stdin")]
    text: Option<String>,
    /// Read the text from standard input.
    #[arg(long)]
    stdin: bool,
    /// Registered voice ID (defaults to `default_voice` from the config).
    #[arg(long, short = 'v', conflicts_with_all = ["voice_file", "ref_audio"])]
    voice: Option<String>,
    /// Path to an `.itts_voice` snapshot file.
    #[arg(long, conflicts_with = "ref_audio")]
    voice_file: Option<PathBuf>,
    /// Reference audio for zero-shot cloning (`.wav`, `.mp3`, `.flac`, `.m4a`).
    #[arg(long = "ref", short = 'r')]
    ref_audio: Option<PathBuf>,
    /// Separate emotion reference audio (requires --ref).
    #[arg(long, requires = "ref_audio")]
    emo_ref: Option<PathBuf>,
    /// Emotion blending weight (`0.0..=2.0`, default `1.0`).
    #[arg(long)]
    emo_alpha: Option<f32>,
    /// Emotion mix, e.g. `melancholic:0.45,calm:0.25` or `低落:0.45,平静:0.25`.
    #[arg(long)]
    emo: Option<String>,
    /// Language (`zh`, `en`, `ja`, `yue`, `es`, `ar`, `zh/en`; default from config).
    #[arg(long, short = 'l')]
    lang: Option<String>,
    /// Speaking rate (`0.25..=4.0`, default from config: `1.2`; larger is faster).
    #[arg(long, short = 's')]
    speed: Option<f32>,
    /// Silence between blank-line separated paragraphs (`0..=5000` ms, default `600`).
    #[arg(long)]
    paragraph_pause: Option<u32>,
    /// Silence between sentences (`0..=5000` ms, default from config).
    #[arg(long)]
    interval_silence: Option<u32>,
    /// Output `.wav` path, or `-` for raw 16-bit little-endian PCM on stdout.
    #[arg(long, short = 'o')]
    out: Option<PathBuf>,
    /// Play through the default audio output while synthesizing.
    #[arg(long, short = 'p')]
    play: bool,
    /// Report per-character and per-sentence timings (`alignment` / `sentences` in JSON).
    #[arg(long)]
    timestamps: bool,
    /// Continue each sentence after the previous one (intonation and pace carry across
    /// sentences; inter-sentence silence defaults to 0).
    #[arg(long)]
    continuity: bool,
    /// Text spoken just before this one (e.g. the previous script line); the first sentence
    /// continues after it. Not spoken.
    #[arg(long)]
    previous_text: Option<String>,
    /// Output sample rate: 22050 (native) or 48000 (bandwidth-extended above 11 kHz).
    #[arg(long)]
    sample_rate: Option<u32>,
    /// Run in-process instead of using the warm daemon.
    #[arg(long)]
    no_daemon: bool,
    /// Daemon socket path.
    #[arg(long)]
    socket: Option<PathBuf>,
    #[command(flatten)]
    advanced: AdvancedArgs,
}

/// Low-level sampling overrides (defaults come from the config).
#[derive(Debug, Args)]
#[command(next_help_heading = "Advanced sampling")]
#[allow(clippy::struct_field_names)]
struct AdvancedArgs {
    /// Deterministic greedy decoding.
    #[arg(long)]
    greedy: bool,
    /// RNG seed for reproducible output (a random one is reported when omitted).
    #[arg(long)]
    seed: Option<u64>,
    /// Sampling temperature (`0.05..=2.0`).
    #[arg(long)]
    temperature: Option<f32>,
    /// Top-K cutoff (`0..=8194`).
    #[arg(long)]
    top_k: Option<u32>,
    /// Nucleus threshold (`0.01..=1.0`).
    #[arg(long)]
    top_p: Option<f32>,
    /// Repetition penalty (`1.0..=50.0`).
    #[arg(long)]
    repetition_penalty: Option<f32>,
    /// Maximum Mel tokens per segment (`1..=1815`).
    #[arg(long)]
    max_mel_tokens: Option<u32>,
    /// Maximum text tokens per segment (`1..=600`).
    #[arg(long)]
    max_text_tokens: Option<u32>,
    /// CFM integration steps (`1..=100`).
    #[arg(long)]
    n_timesteps: Option<u32>,
    /// Classifier-free guidance rate (`0.0..=2.0`).
    #[arg(long)]
    cfg_rate: Option<f32>,
}

impl From<&AdvancedArgs> for AdvancedOptions {
    fn from(args: &AdvancedArgs) -> Self {
        let mut options = Self::default();
        options.greedy = args.greedy.then_some(true);
        options.seed = args.seed;
        options.temperature = args.temperature;
        options.top_k = args.top_k;
        options.top_p = args.top_p;
        options.repetition_penalty = args.repetition_penalty;
        options.max_mel_tokens = args.max_mel_tokens;
        options.max_text_tokens = args.max_text_tokens;
        options.n_timesteps = args.n_timesteps;
        options.cfg_rate = args.cfg_rate;
        options
    }
}

/// Where synthesized audio goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sink {
    /// Raw PCM to stdout.
    StdoutPcm,
    /// WAV file and/or playback.
    Default,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SynthOutcome<'a> {
    status: &'static str,
    backend: Backend,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_load_ms: Option<f64>,
    #[serde(flatten)]
    summary: &'a JobSummary,
}

/// Reads input text from `--text` or `--stdin`.
///
/// # Errors
///
/// Returns an error if neither source is given or stdin cannot be read.
pub fn read_text(text: Option<&str>, from_stdin: bool) -> Result<String> {
    if from_stdin {
        let mut raw = String::new();
        io::stdin()
            .take(MAX_STDIN_BYTES)
            .read_to_string(&mut raw)
            .context("failed to read text from stdin")?;
        return Ok(raw);
    }
    text.map(str::to_owned).ok_or_else(|| {
        anyhow!(ServiceError::invalid("no text given").with_hint("pass --text <TEXT> or --stdin"))
    })
}

/// Runs `itts synth`.
///
/// # Errors
///
/// Returns the classified job failure.
pub async fn run(config: EngineConfig, args: SynthArgs, output: OutputMode) -> Result<()> {
    let sink = if args.out.as_deref() == Some(Path::new("-")) {
        Sink::StdoutPcm
    } else {
        Sink::Default
    };
    if sink == Sink::Default && args.out.is_none() && !args.play {
        return Err(anyhow!(
            ServiceError::invalid("nothing to do with the audio")
                .with_hint("pass --out <file.wav>, --out - (raw PCM to stdout), or --play")
        ));
    }
    let mut job = build_job(&args)?;
    job.resolve_paths(&env::current_dir().context("cannot read current directory")?)?;

    let playback = if args.play {
        Some(Playback::start()?)
    } else {
        None
    };
    let mut events = EventSink {
        output,
        sink,
        playback: playback.as_ref(),
    };
    let socket = args.socket.clone().unwrap_or_else(default_socket_path);
    let (summary, backend, model_load_ms) =
        execute(config, &job, &socket, args.no_daemon, &mut events).await?;
    if let Some(playback) = playback {
        spawn_blocking(move || playback.finish())
            .await
            .context("playback task failed")??;
    }
    emit_outcome(
        output,
        sink,
        &SynthOutcome {
            status: "ok",
            backend,
            model_load_ms,
            summary: &summary,
        },
    )
}

fn build_job(args: &SynthArgs) -> Result<SpeechJob> {
    let mut job = SpeechJob::builder()
        .input(read_text(args.text.as_deref(), args.stdin)?)
        .advanced(AdvancedOptions::from(&args.advanced))
        .build();
    job.voice.clone_from(&args.voice);
    job.voice_file.clone_from(&args.voice_file);
    job.ref_audio.clone_from(&args.ref_audio);
    job.emo_ref.clone_from(&args.emo_ref);
    job.emo_alpha = args.emo_alpha;
    job.emo.clone_from(&args.emo);
    job.lang.clone_from(&args.lang);
    job.speed = args.speed;
    job.paragraph_pause_ms = args.paragraph_pause;
    job.interval_silence_ms = args.interval_silence;
    job.output_path = args.out.clone().filter(|out| out != Path::new("-"));
    job.timestamps = args.timestamps;
    job.sample_rate = args.sample_rate;
    job.continuity = args.continuity;
    job.previous_text.clone_from(&args.previous_text);
    Ok(job)
}

async fn execute(
    config: EngineConfig,
    job: &SpeechJob,
    socket: &Path,
    no_daemon: bool,
    events: &mut EventSink<'_>,
) -> Result<(JobSummary, Backend, Option<f64>)> {
    if !no_daemon {
        match DaemonClient::connect(socket).await {
            Ok(Some(mut client)) => {
                let summary = client
                    .synth(job, events.needs_pcm(), |event| {
                        events.on_event(&event, None)
                    })
                    .await?;
                return Ok((summary, Backend::Daemon, None));
            }
            Ok(None) => {}
            Err(error) => {
                let mut err = io::stderr().lock();
                let _ = writeln!(
                    err,
                    "⚠ daemon unusable [{}]: {}; synthesizing in-process instead",
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
    let prepared = job.prepare(engine.config(), PathPolicy::LocalAbsolute)?;
    events.on_event(
        &JobEvent::Started(JobStarted::new(&prepared, Backend::InProcess)),
        None,
    )?;
    let result = run_job(engine.clone(), prepared, |chunk| {
        events.on_event(
            &JobEvent::Chunk(chunk.info(false)),
            Some(&chunk.audio.pcm_i16),
        )
    })
    .await;
    let stopped = spawn_blocking(move || engine.stop().map_err(ServiceError::from))
        .await
        .context("engine shutdown task failed")?;
    let summary = result?;
    stopped?;
    Ok((summary, Backend::InProcess, Some(model_load_ms)))
}

/// Delivers job events to the terminal, speaker, and stdout.
struct EventSink<'a> {
    output: OutputMode,
    sink: Sink,
    playback: Option<&'a Playback>,
}

impl EventSink<'_> {
    const fn needs_pcm(&self) -> bool {
        self.playback.is_some() || matches!(self.sink, Sink::StdoutPcm)
    }

    fn on_event(&mut self, event: &JobEvent, pcm: Option<&[i16]>) -> Result<(), ServiceError> {
        let io_err = |err: io::Error| ServiceError::internal(format!("stdout: {err}"));
        match event {
            JobEvent::Started(started) => {
                if self.output == OutputMode::Jsonl && self.sink != Sink::StdoutPcm {
                    OutputMode::emit_line(event)
                        .map_err(|e| ServiceError::internal(e.to_string()))?;
                } else if self.output == OutputMode::Text {
                    let via = match started.backend {
                        Backend::Daemon => "daemon",
                        _ => "in-process engine",
                    };
                    let _ = writeln!(
                        io::stderr(),
                        "▸ Synthesizing {} utterance(s) with `{}` [{}] via {via}",
                        started.utterances,
                        started.speaker,
                        started.language
                    );
                }
            }
            JobEvent::Chunk(info) => {
                let decoded;
                let samples = match pcm {
                    Some(samples) => Some(samples),
                    None if info.pcm_base64.is_some() => {
                        decoded = ChunkInfo::clone(info).into_job_chunk()?;
                        Some(decoded.audio.pcm_i16.as_slice())
                    }
                    None => None,
                };
                if let (Some(playback), Some(samples)) = (self.playback, samples) {
                    playback.enqueue(samples.to_vec(), info.sample_rate)?;
                }
                if self.sink == Sink::StdoutPcm {
                    let bytes: Vec<u8> = samples
                        .unwrap_or_default()
                        .iter()
                        .flat_map(|s| s.to_le_bytes())
                        .collect();
                    let mut out = io::stdout().lock();
                    out.write_all(&bytes).map_err(io_err)?;
                    out.flush().map_err(io_err)?;
                } else if self.output == OutputMode::Jsonl {
                    let mut line = ChunkInfo::clone(info);
                    line.pcm_base64 = None;
                    OutputMode::emit_line(&JobEvent::Chunk(line))
                        .map_err(|e| ServiceError::internal(e.to_string()))?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn emit_outcome(output: OutputMode, sink: Sink, outcome: &SynthOutcome<'_>) -> Result<()> {
    let summary = outcome.summary;
    let text = |out: &mut dyn Write| -> io::Result<()> {
        let via = match outcome.backend {
            Backend::Daemon => "daemon".to_owned(),
            _ => format!(
                "in-process, model load {:.0} ms",
                outcome.model_load_ms.unwrap_or_default()
            ),
        };
        writeln!(
            out,
            "✔ {:.2} s audio · {} segment(s) · synthesized in {:.2} s (RTF {:.3}, {via}) · seed {}",
            summary.duration_secs,
            summary.segments,
            summary.synthesis_secs,
            summary.rtf,
            summary.seed
        )?;
        if let Some(path) = &summary.output_path {
            writeln!(out, "  → {}", display_path(path))?;
        }
        for sentence in &summary.sentences {
            writeln!(
                out,
                "  {:>7.2}–{:<7.2} {}",
                sentence.start, sentence.end, sentence.text
            )?;
        }
        Ok(())
    };
    match (sink, output) {
        // stdout carries PCM: the summary must go to stderr.
        (Sink::StdoutPcm, OutputMode::Text) => Ok(text(&mut io::stderr().lock())?),
        (Sink::StdoutPcm, _) => {
            let line = serde_json::to_string(outcome)?;
            Ok(writeln!(io::stderr(), "{line}")?)
        }
        (_, OutputMode::Jsonl) => OutputMode::emit_line(&JobEvent::Done(summary.clone())),
        _ => output.emit(outcome, text),
    }
}

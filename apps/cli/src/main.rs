//! `itts`: AI-Agent friendly CLI, Unix Domain Socket daemon, and MCP server for
//! `IndexTTS-2.5` on Apple Silicon.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

mod daemon;
mod mcp;
mod player;

use std::{
    fs,
    io::{self, Read, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use anyhow::{Context, Result, anyhow};
use clap::{Args, Parser, Subcommand};
use hf_hub::{HFClient, split_id};
use index_tts_rs_core::{
    config::{
        CfgRate, CfmParams, DEFAULT_INTERVAL_SILENCE_MS, DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT,
        DEFAULT_MODEL_REPO, DurationFactor, EmoAlpha, EngineConfig, Language, RepetitionPenalty,
        SamplingParams, SpeakerSource, SynthesisRequest, SynthesisText, Temperature, TopK, TopP,
        VoiceId,
    },
    dsp::write_wav_i16_file,
    engine::{AudioChunk, IndexTtsEngine, StageTimings, SynthesisResult},
    models::{LengthRegulator, ModelWeightKind, resolve_model_dir},
    voice::{VoiceExtractor, VoiceStore, default_voices_dir},
};
use index_tts_rs_director::{
    Director, DirectorConfig, PlannedUtterance, plan_utterances, prepend_silence,
};
use serde::Serialize;
use serde_json::{json, to_string, to_string_pretty};
use tokio::task::spawn_blocking;
use tracing_subscriber::EnvFilter;

use crate::{
    daemon::{
        DEFAULT_IDLE_TIMEOUT_SECS, default_daemon_socket_path, query_daemon_status,
        run_daemon_server, stop_daemon, try_synthesize_via_daemon,
    },
    mcp::run_mcp_stdio_server,
    player::AudioPlayer,
};

const REQUIRED_REPO_FILES: &[&str] = &[
    "config.yaml",
    "glossary.yaml",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "spk_proj.npz",
    "gpt_int8.safetensors",
    "cfm_estimator.safetensors",
    "raw/semantic_model.safetensors",
    "raw/emo.safetensors",
    "raw/campplus.safetensors",
    "raw/length_regulator.safetensors",
    "raw/semantic_codec.safetensors",
    "raw/bigvgan.safetensors",
];

#[derive(Debug, Parser)]
#[command(
    name = "itts",
    version,
    about = "Native Rust & MLX Metal speech synthesis engine for IndexTTS-2.5 on Apple Silicon"
)]
struct Cli {
    /// Optional YAML configuration file path.
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Explicit local model directory override.
    #[arg(long, global = true)]
    model_dir: Option<PathBuf>,

    /// Directory where `.itts_voice` snapshots are stored.
    #[arg(long, global = true)]
    voices_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Synthesize speech from text using a registered `.itts_voice` or raw reference audio.
    Synth(Box<SynthArgs>),
    /// Add optional `DeepSeek` pacing controls without rewriting the source text.
    Annotate(AnnotateArgs),
    /// Manage `.itts_voice` speaker conditioning snapshots.
    Voice {
        #[command(subcommand)]
        action: VoiceSubcommand,
    },
    /// Pull or inspect local `IndexTTS-2.5-mlx` model weights.
    Models {
        #[command(subcommand)]
        action: ModelsSubcommand,
    },
    /// Manage the warm-model Unix Domain Socket background daemon.
    Daemon {
        #[command(subcommand)]
        action: DaemonSubcommand,
    },
    /// Run the Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio.
    Mcp,
}

#[derive(Debug, Args)]
#[allow(clippy::struct_excessive_bools)]
struct SynthArgs {
    /// Additional silence between explicit paragraphs (`0..=5000` ms).
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u32).range(0..=5000))]
    paragraph_pause: u32,

    /// Registered voice ID (`<voice_id>`) or path to an `.itts_voice` file.
    #[arg(long, short = 'v')]
    voice: Option<String>,

    /// Path to a raw reference audio file (`.wav`, `.mp3`, `.flac`, `.m4a`).
    #[arg(long = "ref", short = 'r')]
    ref_audio: Option<PathBuf>,

    /// Optional separate emotion reference audio file when `--ref` is used.
    #[arg(long = "emo-ref")]
    emo_ref: Option<PathBuf>,

    /// Emotion blending multiplier (`0.0..=2.0`, default `1.0`).
    #[arg(long, default_value_t = 1.0)]
    emo_alpha: f32,

    /// Input text to synthesize (omit when `--stdin` is passed).
    #[arg(long, short = 't')]
    text: Option<String>,

    /// Read synthesis text from standard input (`stdin`).
    #[arg(long)]
    stdin: bool,

    /// Language code (`zh`, `en`, `ja`, `yue`, `es`, `ar`, `zh/en`).
    #[arg(long, short = 'l', default_value = "zh")]
    lang: String,

    /// Output `.wav` file path.
    #[arg(long, short = 'o')]
    out: Option<PathBuf>,

    /// Stream synthesized audio directly through macOS `CoreAudio`.
    #[arg(long, short = 'p')]
    play: bool,

    /// Emit structured JSON summary to `stdout`.
    #[arg(long)]
    json: bool,

    /// Emit streaming JSON Lines (`chunk` + `done` events) to `stdout`.
    #[arg(long)]
    jsonl: bool,

    /// Bypass the Unix Domain Socket daemon and run in-process.
    #[arg(long)]
    no_daemon: bool,

    /// Unix Domain Socket path for daemon communication.
    #[arg(long)]
    socket: Option<PathBuf>,

    /// Use deterministic greedy (`argmax`) GPT token decoding.
    #[arg(long)]
    greedy: bool,

    /// Optional RNG seed for reproducible synthesis.
    #[arg(long)]
    seed: Option<u64>,

    /// Top-K sampling cutoff (`0..=8194`, default `30`).
    #[arg(long, default_value_t = 30)]
    top_k: u32,

    /// Nucleus sampling `top_p` threshold (`0.01..=1.0`, default `0.8`).
    #[arg(long, default_value_t = 0.8)]
    top_p: f32,

    /// Sampling temperature (`0.05..=2.0`, default `0.8`).
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Repetition penalty (`1.0..=50.0`, default `10.0`).
    #[arg(long, default_value_t = 10.0)]
    repetition_penalty: f32,

    /// Maximum generated Mel tokens per segment (`1..=1815`, default `1500`).
    #[arg(long, default_value_t = 1500)]
    max_mel_tokens: u32,

    /// Maximum text BPE tokens per sentence segment (`1..=600`, default `120`).
    #[arg(long, default_value_t = DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT as u32)]
    max_text_tokens: u32,

    /// Inter-sentence silence in milliseconds (`0..=5000`, default `200`).
    #[arg(long, default_value_t = DEFAULT_INTERVAL_SILENCE_MS)]
    interval_silence: u32,

    /// Speaking rate duration multiplier (`0.25..=4.0`, default `1.0`).
    #[arg(long, default_value_t = 1.0)]
    duration_factor: f32,

    /// Number of CFM Euler ODE integration steps (`1..=100`, default `25`).
    #[arg(long, default_value_t = 25)]
    n_timesteps: u32,

    /// Classifier-free guidance rate (`0.0..=2.0`, default `0.7`).
    #[arg(long, default_value_t = 0.7)]
    cfg_rate: f32,
}

#[derive(Debug, Args)]
struct AnnotateArgs {
    /// Text to annotate (use --stdin for a file or pipeline).
    #[arg(long, short = 't', conflicts_with = "stdin")]
    text: Option<String>,
    /// Read untagged source text from standard input.
    #[arg(long)]
    stdin: bool,
    /// Request poetry pacing rather than conversational pacing.
    #[arg(long)]
    poetry: bool,
    /// Optional YAML configuration for the `DeepSeek` provider.
    #[arg(long)]
    director_config: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum VoiceSubcommand {
    /// Extract speaker conditioning tensors from reference audio and save `<voice_id>.itts_voice`.
    Add {
        /// Voice identifier (`^[a-zA-Z0-9_-]{1,64}$`).
        voice_id: String,
        /// Reference audio file path (`.wav`, `.mp3`, `.flac`, `.m4a`).
        #[arg(long = "ref", short = 'r')]
        ref_audio: PathBuf,
        /// Optional separate emotion reference audio file path.
        #[arg(long = "emo-ref")]
        emo_ref: Option<PathBuf>,
        /// Emotion blending multiplier (`0.0..=2.0`, default `1.0`).
        #[arg(long, default_value_t = 1.0)]
        emo_alpha: f32,
        /// Emit structured JSON output.
        #[arg(long)]
        json: bool,
    },
    /// List all registered `.itts_voice` snapshots.
    List {
        /// Emit structured JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Inspect metadata and tensor dimensions of a registered voice.
    Inspect {
        /// Voice identifier.
        voice_id: String,
        /// Emit structured JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Remove a registered `.itts_voice` snapshot.
    #[command(visible_alias = "remove")]
    Rm {
        /// Voice identifier.
        voice_id: String,
        /// Emit structured JSON output.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum ModelsSubcommand {
    /// Download or verify the `IndexTTS-2.5-mlx` model weights from Hugging Face Hub.
    Pull {
        /// Hugging Face repository ID.
        #[arg(long, default_value = DEFAULT_MODEL_REPO)]
        repo: String,
        /// Emit structured JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Inspect local model snapshot files and dead-weight stripping statistics.
    #[command(visible_alias = "inspect")]
    Info {
        /// Emit structured JSON output.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum DaemonSubcommand {
    /// Start the warm-model Unix Domain Socket daemon.
    Start {
        /// Idle auto-shutdown timeout in seconds (`0` disables auto-shutdown).
        #[arg(long, default_value_t = DEFAULT_IDLE_TIMEOUT_SECS)]
        idle_timeout_secs: u64,
        /// Custom Unix socket path override.
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// Query status of the running daemon.
    Status {
        /// Custom Unix socket path override.
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Emit structured JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Gracefully stop the running daemon.
    Stop {
        /// Custom Unix socket path override.
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Emit structured JSON output.
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let config = build_engine_config(
        cli.config.as_deref(),
        cli.model_dir.clone(),
        cli.voices_dir.clone(),
    )?;

    match cli.command {
        Commands::Synth(args) => handle_synth_command(config, *args).await,
        Commands::Annotate(args) => handle_annotate_command(args).await,
        Commands::Voice { action } => handle_voice_command(&config, action),
        Commands::Models { action } => handle_models_command(&config, action).await,
        Commands::Daemon { action } => handle_daemon_command(config, action).await,
        Commands::Mcp => run_mcp_stdio_server(config).await,
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(io::stderr)
        .try_init();
}

fn build_engine_config(
    config_path: Option<&Path>,
    model_dir: Option<PathBuf>,
    voices_dir: Option<PathBuf>,
) -> Result<EngineConfig> {
    let mut cfg = match config_path {
        Some(path) => EngineConfig::from_yaml_file(path)?,
        None => EngineConfig::default(),
    };
    if model_dir.is_some() {
        cfg.model_dir = model_dir;
    }
    if voices_dir.is_some() {
        cfg.voices_dir = voices_dir;
    }
    cfg.validate_config()?;
    Ok(cfg)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
async fn handle_synth_command(config: EngineConfig, args: SynthArgs) -> Result<()> {
    let req = build_synthesis_request(&args)?;
    let total_start = Instant::now();
    let socket_path = args.socket.unwrap_or_else(default_daemon_socket_path);
    let sample_rate = config.sample_rate;
    let plan = plan_utterances(&req, args.paragraph_pause)?;

    let player = if args.play {
        Some(AudioPlayer::try_new()?)
    } else {
        None
    };
    let stream_stdout_pcm = args.out.as_deref() == Some(Path::new("-"));

    let mut chunks: Vec<AudioChunk> = Vec::new();
    let total_utterances = plan.len();
    let mut on_chunk = |chunk: AudioChunk, utterance_index: usize| -> Result<()> {
        if let Some(p) = &player {
            p.enqueue_pcm_i16(&chunk.pcm_i16, chunk.sample_rate)?;
        }
        if stream_stdout_pcm {
            let mut raw_bytes = Vec::with_capacity(chunk.pcm_i16.len() * 2);
            for &sample in &chunk.pcm_i16 {
                raw_bytes.extend_from_slice(&sample.to_le_bytes());
            }
            let mut out = io::stdout().lock();
            out.write_all(&raw_bytes)?;
            out.flush()?;
        } else if args.jsonl {
            let line = to_string(&json!({
                "event": "chunk",
                "utteranceIndex": utterance_index,
                "totalUtterances": total_utterances,
                "segmentIndex": chunk.segment_index,
                "totalSegments": chunk.total_segments,
                "samples": chunk.pcm_i16.len(),
                "durationSecs": chunk.duration_secs(),
                "melTokens": chunk.mel_tokens,
                "timings": chunk.timings,
            }))?;
            writeln!(io::stdout(), "{line}")?;
        }
        chunks.push(chunk);
        Ok(())
    };

    let execution =
        execute_synthesis_plan(config, plan, &socket_path, args.no_daemon, &mut on_chunk).await?;

    let wall_secs = total_start.elapsed().as_secs_f32();
    let res = merge_chunks_into_result(chunks, sample_rate, wall_secs);

    if let Some(out_path) = &args.out
        && !stream_stdout_pcm
    {
        write_wav_i16_file(out_path, &res.pcm_i16, res.sample_rate)?;
    }
    if let Some(p) = player {
        p.drain();
    }

    emit_synth_summary(
        &res,
        args.out.as_deref(),
        execution.used_daemon,
        execution.model_load_ms,
        wall_secs,
        args.json,
        args.jsonl,
        stream_stdout_pcm,
    )
}

#[derive(Debug)]
struct PlanExecution {
    used_daemon: bool,
    model_load_ms: f64,
}

async fn execute_synthesis_plan(
    config: EngineConfig,
    plan: Vec<PlannedUtterance>,
    socket_path: &Path,
    no_daemon: bool,
    on_chunk: &mut impl FnMut(AudioChunk, usize) -> Result<()>,
) -> Result<PlanExecution> {
    let mut used_daemon = false;
    let mut model_load_ms = 0.0_f64;
    let mut local_engine = None;
    let synthesis_result: Result<()> = async {
        for (utterance_index, part) in plan.into_iter().enumerate() {
            let interval_ms = part.request.interval_silence_ms;
            let pause_ms = part.pause_before_ms;
            let mut on_part = |mut chunk: AudioChunk| -> Result<()> {
                let silence = if chunk.segment_index == 0 {
                    pause_ms
                } else {
                    interval_ms
                };
                prepend_silence(&mut chunk.pcm_i16, chunk.sample_rate, silence)?;
                on_chunk(chunk, utterance_index)
            };
            let forwarded = if no_daemon {
                false
            } else {
                try_synthesize_via_daemon(socket_path, &part.request, &mut on_part).await?
            };
            used_daemon |= forwarded;
            if !forwarded {
                if local_engine.is_none() {
                    let started = Instant::now();
                    let config = config.clone();
                    local_engine = Some(
                        spawn_blocking(move || {
                            IndexTtsEngine::start(config)
                                .context("failed to start local synthesis engine")
                        })
                        .await
                        .context("local engine startup task failed")??,
                    );
                    model_load_ms = started.elapsed().as_secs_f64() * 1000.0;
                }
                if let Some(engine) = &local_engine {
                    let stream = engine.synthesize_stream(part.request)?;
                    while let Some(chunk) = stream.next_chunk().await {
                        on_part(chunk?)?;
                    }
                }
            }
        }
        Ok(())
    }
    .await;
    if let Some(engine) = local_engine {
        let stop_result = spawn_blocking(move || {
            engine
                .stop()
                .context("failed to stop local synthesis engine")
        })
        .await
        .context("local engine shutdown task failed");
        synthesis_result?;
        stop_result??;
    } else {
        synthesis_result?;
    }

    Ok(PlanExecution {
        used_daemon,
        model_load_ms,
    })
}

fn read_input_text(text: Option<&str>, from_stdin: bool) -> Result<SynthesisText> {
    let raw = if from_stdin {
        let mut text = String::new();
        io::stdin()
            .take(16_385)
            .read_to_string(&mut text)
            .context("failed to read input text from stdin")?;
        text
    } else {
        text.ok_or_else(|| anyhow!("provide --text <TEXT> or --stdin"))?
            .to_owned()
    };
    Ok(SynthesisText::new(raw)?)
}

async fn handle_annotate_command(args: AnnotateArgs) -> Result<()> {
    let text = read_input_text(args.text.as_deref(), args.stdin)?;
    let config = match args.director_config {
        Some(path) => DirectorConfig::from_yaml_file(path).await?,
        None => DirectorConfig::default(),
    };
    let director = Director::from_env(config)?;
    let script = director.annotate(text.as_str(), args.poetry).await?;
    writeln!(io::stdout(), "{script}")?;
    Ok(())
}

fn build_synthesis_request(args: &SynthArgs) -> Result<SynthesisRequest> {
    let text = read_input_text(args.text.as_deref(), args.stdin)?;
    let speaker = match (&args.voice, &args.ref_audio) {
        (Some(v), _) => {
            let path = PathBuf::from(v);
            if path.extension().and_then(|e| e.to_str()) == Some("itts_voice") || path.is_file() {
                SpeakerSource::VoiceFile(path)
            } else {
                SpeakerSource::VoiceId(VoiceId::new(v)?)
            }
        }
        (None, Some(ref_path)) => SpeakerSource::ReferenceAudio(ref_path.clone()),
        (None, None) => {
            return Err(anyhow!(
                "either `--voice <VOICE_ID>` or `--ref <AUDIO_PATH>` must be provided"
            ));
        }
    };

    let language = Language::from_str(&args.lang)?;
    let sampling = SamplingParams::builder()
        .greedy(args.greedy)
        .seed(args.seed)
        .top_k(TopK::new(args.top_k)?)
        .top_p(TopP::new(args.top_p)?)
        .temperature(Temperature::new(args.temperature)?)
        .repetition_penalty(RepetitionPenalty::new(args.repetition_penalty)?)
        .max_mel_tokens(args.max_mel_tokens)
        .build();
    let cfm = CfmParams::builder()
        .duration_factor(DurationFactor::new(args.duration_factor)?)
        .n_timesteps(args.n_timesteps)
        .cfg_rate(CfgRate::new(args.cfg_rate)?)
        .build();
    let max_text_tokens = NonZeroU32::new(args.max_text_tokens)
        .ok_or_else(|| anyhow!("--max-text-tokens must be >= 1"))?;

    Ok(SynthesisRequest::builder()
        .text(text)
        .speaker(speaker)
        .language(language)
        .sampling(sampling)
        .cfm(cfm)
        .max_text_tokens_per_segment(max_text_tokens)
        .interval_silence_ms(args.interval_silence)
        .emo_audio_path(args.emo_ref.clone())
        .emo_alpha(EmoAlpha::new(args.emo_alpha)?)
        .build())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn merge_chunks_into_result(
    chunks: Vec<AudioChunk>,
    sample_rate: u32,
    wall_secs: f32,
) -> SynthesisResult {
    let segments_count = chunks.len();
    let mut pcm_i16 = Vec::new();
    let mut total_mel_tokens = 0_usize;
    let mut timings = StageTimings::default();

    for chunk in chunks {
        pcm_i16.extend_from_slice(&chunk.pcm_i16);
        total_mel_tokens += chunk.mel_tokens;
        timings.accumulate_segment(&chunk.timings);
    }
    let stage_synth_ms = timings.frontend_ms
        + timings.gpt_ms
        + timings.codec_ms
        + timings.regulator_ms
        + timings.cfm_ms
        + timings.bigvgan_ms;
    let synthesis_secs = if stage_synth_ms > 0.0 {
        (stage_synth_ms / 1000.0) as f32
    } else {
        wall_secs
    };
    timings.total_ms = timings.voice_extract_ms + stage_synth_ms;

    SynthesisResult::new(
        pcm_i16,
        sample_rate,
        synthesis_secs,
        segments_count,
        total_mel_tokens,
        timings,
    )
}

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn emit_synth_summary(
    res: &SynthesisResult,
    output_path: Option<&Path>,
    used_daemon: bool,
    model_load_ms: f64,
    wall_secs: f32,
    pretty_json: bool,
    stream_lines: bool,
    stream_stdout_pcm: bool,
) -> Result<()> {
    let summary = json!({
        "event": "done",
        "status": "ok",
        "usedDaemon": used_daemon,
        "modelLoadMs": model_load_ms,
        "wallSecs": wall_secs,
        "outputPath": output_path,
        "sampleRate": res.sample_rate,
        "samples": res.pcm_i16.len(),
        "durationSecs": res.duration_secs,
        "synthesisSecs": res.synthesis_secs,
        "rtf": res.rtf,
        "segmentsCount": res.segments_count,
        "totalMelTokens": res.total_mel_tokens,
        "timings": res.timings,
    });

    if stream_stdout_pcm {
        writeln!(
            io::stderr(),
            "Synthesized {:.2}s audio ({} segments, {} mel tokens) in {:.3}s (RTF: {:.3}, daemon: \
             {})",
            res.duration_secs,
            res.segments_count,
            res.total_mel_tokens,
            res.synthesis_secs,
            res.rtf,
            used_daemon,
        )?;
    } else if stream_lines {
        writeln!(io::stdout(), "{}", to_string(&summary)?)?;
    } else if pretty_json {
        writeln!(io::stdout(), "{}", to_string_pretty(&summary)?)?;
    } else {
        writeln!(
            io::stdout(),
            "Synthesized {:.2}s audio ({} segments, {} mel tokens) in {:.3}s (RTF: {:.3}, daemon: \
             {})",
            res.duration_secs,
            res.segments_count,
            res.total_mel_tokens,
            res.synthesis_secs,
            res.rtf,
            used_daemon,
        )?;
        if let Some(p) = output_path {
            writeln!(io::stdout(), "Saved WAV -> {}", p.display())?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn handle_voice_command(config: &EngineConfig, action: VoiceSubcommand) -> Result<()> {
    let voices_dir = config.voices_dir.clone().unwrap_or_else(default_voices_dir);
    let store = VoiceStore::new(voices_dir)?;

    match action {
        VoiceSubcommand::Add {
            voice_id,
            ref_audio,
            emo_ref,
            emo_alpha,
            json,
        } => {
            let t0 = Instant::now();
            let id = VoiceId::new(voice_id)?;
            let alpha = EmoAlpha::new(emo_alpha)?;
            let model_dir = resolve_model_dir(config.model_dir.as_deref(), &config.model_repo)?;

            // Sequentially load & drop each Stage 0 encoder so peak RSS stays minimal
            let regulator = LengthRegulator::load_from_path(
                model_dir.join("raw/length_regulator.safetensors"),
            )?;
            let ctx = VoiceExtractor::extract_from_model_dir(
                &model_dir,
                &ref_audio,
                emo_ref.as_deref(),
                alpha,
                &regulator,
            )?;
            let _ = store.register(id.clone(), ctx)?;
            let entry = store.inspect(&id)?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

            if json {
                let payload = json!({
                    "status": "ok",
                    "extractMs": elapsed_ms,
                    "voice": entry,
                });
                writeln!(io::stdout(), "{}", to_string_pretty(&payload)?)?;
            } else {
                writeln!(
                    io::stdout(),
                    "Registered voice `{}` -> {} ({} frames, {:.2}s, {} bytes) in {:.1} ms",
                    entry.voice_id,
                    entry.path.display(),
                    entry.prompt_frames,
                    entry.duration_secs,
                    entry.size_bytes,
                    elapsed_ms,
                )?;
            }
            Ok(())
        }
        VoiceSubcommand::List { json } => {
            let entries = store.list()?;
            if json {
                writeln!(io::stdout(), "{}", to_string_pretty(&entries)?)?;
            } else if entries.is_empty() {
                writeln!(
                    io::stdout(),
                    "No registered voices in {}",
                    store.voices_dir().display()
                )?;
            } else {
                for e in entries {
                    writeln!(
                        io::stdout(),
                        "{:<20} {:>6.2}s ({:>4} frames, {:>7} B)  {}",
                        e.voice_id,
                        e.duration_secs,
                        e.prompt_frames,
                        e.size_bytes,
                        e.path.display()
                    )?;
                }
            }
            Ok(())
        }
        VoiceSubcommand::Inspect { voice_id, json } => {
            let id = VoiceId::new(voice_id)?;
            let entry = store.inspect(&id)?;
            if json {
                writeln!(io::stdout(), "{}", to_string_pretty(&entry)?)?;
            } else {
                writeln!(io::stdout(), "Voice ID:       {}", entry.voice_id)?;
                writeln!(io::stdout(), "Path:           {}", entry.path.display())?;
                writeln!(io::stdout(), "Prompt Frames:  {}", entry.prompt_frames)?;
                writeln!(io::stdout(), "Duration:       {:.2}s", entry.duration_secs)?;
                writeln!(io::stdout(), "File Size:      {} bytes", entry.size_bytes)?;
                writeln!(
                    io::stdout(),
                    "BLAKE3:         {}",
                    entry.metadata.source_audio_blake3
                )?;
                writeln!(
                    io::stdout(),
                    "Emo Alpha:      {:.2}",
                    entry.metadata.emo_alpha.get()
                )?;
            }
            Ok(())
        }
        VoiceSubcommand::Rm { voice_id, json } => {
            let id = VoiceId::new(voice_id)?;
            let removed = store.remove(&id)?;
            if json {
                writeln!(
                    io::stdout(),
                    "{}",
                    to_string_pretty(&json!({
                        "voiceId": id.as_str(),
                        "removed": removed,
                    }))?
                )?;
            } else if removed {
                writeln!(io::stdout(), "Removed voice `{}`", id.as_str())?;
            } else {
                writeln!(io::stdout(), "Voice `{}` was not found", id.as_str())?;
            }
            Ok(())
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelArtifactInfo {
    relative_path: &'static str,
    size_bytes: u64,
}

#[allow(clippy::disallowed_methods)]
async fn handle_models_command(config: &EngineConfig, action: ModelsSubcommand) -> Result<()> {
    match action {
        ModelsSubcommand::Pull { repo, json } => {
            let model_dir = if let Ok(dir) = resolve_model_dir(config.model_dir.as_deref(), &repo) {
                dir
            } else {
                let client =
                    HFClient::new().context("failed to initialize Hugging Face Hub client")?;
                let (owner, name) = split_id(&repo);
                let hf_repo = client.model(owner, name);
                let mut last_parent = None;
                for &rel in REQUIRED_REPO_FILES {
                    let downloaded = hf_repo
                        .download_file()
                        .filename(rel)
                        .send()
                        .await
                        .with_context(|| format!("failed to download `{rel}` from `{repo}`"))?;
                    if rel == "config.yaml" {
                        last_parent = downloaded.parent().map(Path::to_path_buf);
                    }
                }
                last_parent.ok_or_else(|| anyhow!("failed to resolve downloaded snapshot root"))?
            };

            if json {
                writeln!(
                    io::stdout(),
                    "{}",
                    to_string_pretty(&json!({
                        "status": "ok",
                        "repo": repo,
                        "modelDir": model_dir,
                    }))?
                )?;
            } else {
                writeln!(
                    io::stdout(),
                    "Model snapshot ready at {}",
                    model_dir.display()
                )?;
            }
            Ok(())
        }
        ModelsSubcommand::Info { json } => {
            let model_dir = resolve_model_dir(config.model_dir.as_deref(), &config.model_repo)?;
            let mut artifacts = Vec::new();
            let mut total_bytes = 0_u64;
            for kind in ModelWeightKind::ALL {
                let rel = kind.relative_path();
                let path = model_dir.join(rel);
                let size_bytes = fs::metadata(&path).map_or(0, |m| m.len());
                total_bytes += size_bytes;
                artifacts.push(ModelArtifactInfo {
                    relative_path: rel,
                    size_bytes,
                });
            }

            if json {
                writeln!(
                    io::stdout(),
                    "{}",
                    to_string_pretty(&json!({
                        "modelRepo": config.model_repo,
                        "modelDir": model_dir,
                        "totalWeightBytes": total_bytes,
                        "artifacts": artifacts,
                    }))?
                )?;
            } else {
                writeln!(io::stdout(), "Model Repo: {}", config.model_repo)?;
                writeln!(io::stdout(), "Model Dir:  {}", model_dir.display())?;
                for a in &artifacts {
                    #[allow(clippy::cast_precision_loss)]
                    let mb = (a.size_bytes as f64) / (1024.0 * 1024.0);
                    writeln!(io::stdout(), "  {:<34} {:>8.2} MiB", a.relative_path, mb)?;
                }
            }
            Ok(())
        }
    }
}

async fn handle_daemon_command(config: EngineConfig, action: DaemonSubcommand) -> Result<()> {
    match action {
        DaemonSubcommand::Start {
            idle_timeout_secs,
            socket,
        } => {
            let sock = socket.unwrap_or_else(default_daemon_socket_path);
            run_daemon_server(config, &sock, idle_timeout_secs).await
        }
        DaemonSubcommand::Status { socket, json } => {
            let sock = socket.unwrap_or_else(default_daemon_socket_path);
            let status = query_daemon_status(&sock).await?;
            if json {
                writeln!(
                    io::stdout(),
                    "{}",
                    to_string_pretty(&json!({
                        "running": status.is_some(),
                        "status": status,
                    }))?
                )?;
            } else if let Some(s) = status {
                writeln!(
                    io::stdout(),
                    "Daemon running (PID: {}, uptime: {}s, idle: {}s/{}s, socket: {})",
                    s.pid,
                    s.uptime_secs,
                    s.idle_secs,
                    s.idle_timeout_secs,
                    s.socket_path.display()
                )?;
            } else {
                writeln!(
                    io::stdout(),
                    "Daemon is not running (socket: {})",
                    sock.display()
                )?;
            }
            Ok(())
        }
        DaemonSubcommand::Stop { socket, json } => {
            let sock = socket.unwrap_or_else(default_daemon_socket_path);
            let stopped = stop_daemon(&sock).await?;
            if json {
                writeln!(
                    io::stdout(),
                    "{}",
                    to_string_pretty(&json!({
                        "stopped": stopped,
                        "socketPath": sock,
                    }))?
                )?;
            } else if stopped {
                writeln!(io::stdout(), "Daemon stopped")?;
            } else {
                writeln!(io::stdout(), "Daemon was not running")?;
            }
            Ok(())
        }
    }
}

//! `itts voice {add,list,inspect,rm}`.

use std::{
    env,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::{Context, Result};
use clap::Subcommand;
use index_tts_rs_core::{
    config::{EmoAlpha, EmotionVector, EngineConfig, VoiceId},
    models::{LengthRegulator, resolve_model_dir},
    voice::{VoiceEntry, VoiceExtractor, VoiceStore, default_voices_dir},
};
use index_tts_rs_service::{Backend, ServiceError, resolve_input_path};
use serde_json::json;
use tokio::task::spawn_blocking;

use crate::{
    daemon::{
        client::{DaemonClient, VoiceAddRequest},
        default_socket_path,
        lifecycle::display_path,
    },
    output::OutputMode,
};

/// Voice subcommands.
#[derive(Debug, Subcommand)]
pub enum VoiceCommand {
    /// Extract a reference clip into a reusable `<ID>.itts_voice` snapshot.
    Add {
        /// Voice identifier (`^[a-zA-Z0-9_-]{1,64}$`).
        voice_id: String,
        /// Reference audio (3-15 s of clean speech; `.wav`, `.mp3`, `.flac`, `.m4a`).
        #[arg(long = "ref", short = 'r')]
        ref_audio: PathBuf,
        /// Separate emotion reference audio.
        #[arg(long)]
        emo_ref: Option<PathBuf>,
        /// Emotion blending weight (`0.0..=2.0`, default `1.0`).
        #[arg(long)]
        emo_alpha: Option<f32>,
        /// Emotion mix baked into the snapshot.
        #[arg(long)]
        emo: Option<String>,
        /// Extract in-process even if the daemon is running.
        #[arg(long)]
        no_daemon: bool,
        /// Daemon socket path.
        #[arg(long)]
        socket: Option<PathBuf>,
    },
    /// List registered voices.
    List,
    /// Show a voice's metadata.
    Inspect {
        /// Voice identifier.
        voice_id: String,
    },
    /// Remove a registered voice.
    #[command(visible_alias = "remove")]
    Rm {
        /// Voice identifier.
        voice_id: String,
    },
}

/// Runs a voice subcommand.
///
/// # Errors
///
/// Returns the classified failure.
pub async fn run(config: EngineConfig, command: VoiceCommand, output: OutputMode) -> Result<()> {
    let store = VoiceStore::new(config.voices_dir.clone().unwrap_or_else(default_voices_dir))?;
    match command {
        VoiceCommand::Add {
            voice_id,
            ref_audio,
            emo_ref,
            emo_alpha,
            emo,
            no_daemon,
            socket,
        } => {
            let cwd = env::current_dir().context("cannot read current directory")?;
            let request = VoiceAddRequest {
                voice_id,
                ref_audio: resolve_input_path(&cwd, &ref_audio, "--ref")?,
                emo_ref: emo_ref
                    .map(|path| resolve_input_path(&cwd, &path, "--emo-ref"))
                    .transpose()?,
                emo_alpha,
                emo,
            };
            let started = Instant::now();
            let socket = socket.unwrap_or_else(default_socket_path);
            let (entry, backend) = match (no_daemon, DaemonClient::connect(&socket).await) {
                (false, Ok(Some(mut client))) => {
                    (client.voice_add(request).await?, Backend::Daemon)
                }
                _ => (add_in_process(config, request).await?, Backend::InProcess),
            };
            let extract_ms = started.elapsed().as_secs_f64() * 1000.0;
            output.emit(
                &json!({ "status": "ok", "backend": backend, "extractMs": extract_ms, "voice": entry }),
                |out| {
                    writeln!(
                        out,
                        "✔ registered voice `{}` ({:.2} s prompt) in {extract_ms:.0} ms",
                        entry.voice_id, entry.duration_secs
                    )?;
                    writeln!(out, "  → {}", display_path(&entry.path))
                },
            )
        }
        VoiceCommand::List => {
            let entries = store.list()?;
            output.emit(&entries, |out| {
                if entries.is_empty() {
                    writeln!(out, "no voices in {}", display_path(store.voices_dir()))?;
                    return writeln!(out, "  hint: `itts voice add <ID> --ref <AUDIO>`");
                }
                for e in &entries {
                    writeln!(
                        out,
                        "{:<24} {:>6.2} s  {:>8} B  {}",
                        e.voice_id,
                        e.duration_secs,
                        e.size_bytes,
                        display_path(&e.path)
                    )?;
                }
                Ok(())
            })
        }
        VoiceCommand::Inspect { voice_id } => {
            let entry = store.inspect(&VoiceId::new(voice_id)?)?;
            output.emit(&entry, |out| render_entry(out, &entry))
        }
        VoiceCommand::Rm { voice_id } => {
            let id = VoiceId::new(voice_id)?;
            if !store.remove(&id)? {
                return Err(ServiceError::voice_not_found(id.as_str()).into());
            }
            output.emit(
                &json!({ "status": "ok", "voiceId": id.as_str(), "removed": true }),
                |out| writeln!(out, "✔ removed voice `{}`", id.as_str()),
            )
        }
    }
}

fn render_entry(out: &mut dyn Write, entry: &VoiceEntry) -> io::Result<()> {
    writeln!(out, "voice      {}", entry.voice_id)?;
    writeln!(out, "path       {}", display_path(&entry.path))?;
    writeln!(
        out,
        "prompt     {:.2} s ({} frames)",
        entry.duration_secs, entry.prompt_frames
    )?;
    writeln!(out, "size       {} B", entry.size_bytes)?;
    writeln!(
        out,
        "source     blake3:{}",
        entry.metadata.source_audio_blake3
    )?;
    writeln!(out, "emo alpha  {:.2}", entry.metadata.emo_alpha.get())
}

/// Loads and drops each Stage 0 encoder sequentially to keep peak memory low.
async fn add_in_process(
    config: EngineConfig,
    request: VoiceAddRequest,
) -> Result<VoiceEntry, ServiceError> {
    spawn_blocking(move || -> Result<VoiceEntry, ServiceError> {
        let id = VoiceId::new(request.voice_id)?;
        let alpha = EmoAlpha::new(request.emo_alpha.unwrap_or(1.0))?;
        let emo_vector = match request.emo.as_deref() {
            Some(spec) => EmotionVector::parse_spec(spec)?,
            None => None,
        };
        let model_dir = resolve_model_dir(config.model_dir.as_deref(), &config.model_repo)?;
        let regulator =
            LengthRegulator::load_from_path(model_dir.join("raw/length_regulator.safetensors"))?;
        let mut ctx = VoiceExtractor::extract_from_model_dir(
            &model_dir,
            &request.ref_audio,
            request.emo_ref.as_deref(),
            alpha,
            &regulator,
        )?;
        if let Some(emo_vec) = &emo_vector {
            ctx.apply_emotion_from_model_dir(&model_dir, emo_vec, alpha)?;
        }
        let store = VoiceStore::new(config.voices_dir.clone().unwrap_or_else(default_voices_dir))?;
        store.register(id.clone(), ctx)?;
        Ok(store.inspect(&id)?)
    })
    .await
    .map_err(|err| ServiceError::internal(format!("voice extraction task failed: {err}")))?
}

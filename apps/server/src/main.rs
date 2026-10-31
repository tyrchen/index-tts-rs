//! Local speech studio and HTTP speech synthesis API for `index-tts-rs`.
//!
//! Open `/` for the browser UI, `/api/voices` for registered voices, and `/health` for status.
//! `POST /v1/audio/speech` accepts text and a registered voice ID and returns mono WAV audio.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

mod web;

use std::{net::SocketAddr, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use axum::serve;
use clap::Parser;
use index_tts_rs_core::{config::EngineConfig, engine::IndexTtsEngine};
use index_tts_rs_director::DirectorConfig;
use tokio::{net::TcpListener, signal::ctrl_c, task::spawn_blocking};
use tracing::{info, warn};

use crate::web::router;

#[derive(Debug, Parser)]
#[command(
    name = "index-tts-rs-server",
    version,
    about = "Local speech studio and HTTP API for IndexTTS-2.5 on Apple Silicon"
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

    /// Directory containing registered `.itts_voice` snapshots.
    #[arg(long)]
    voices_dir: Option<PathBuf>,

    /// Optional YAML settings for `DeepSeek` pacing annotation.
    #[arg(long)]
    director_config: Option<PathBuf>,
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
    if args.voices_dir.is_some() {
        cfg.voices_dir = args.voices_dir;
    }
    cfg.validate_config()?;
    let director_config = match args.director_config {
        Some(path) => DirectorConfig::from_yaml_file(path).await?,
        None => DirectorConfig::default(),
    };

    let listener = TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("failed to bind TCP listener on {}", args.bind))?;
    let engine = Arc::new(
        spawn_blocking(move || {
            IndexTtsEngine::start(cfg)
                .context("failed to initialize IndexTtsEngine for HTTP server")
        })
        .await
        .context("engine startup task failed")??,
    );
    info!(bind = %listener.local_addr()?, "Open http://{} for the speech studio", listener.local_addr()?);

    let result = serve(listener, router(Arc::clone(&engine), director_config))
        .with_graceful_shutdown(async {
            if let Err(err) = ctrl_c().await {
                warn!(error = %err, "failed to listen for shutdown signal");
            }
        })
        .await;
    spawn_blocking(move || engine.stop().context("failed to stop speech engine"))
        .await
        .context("engine shutdown task failed")??;
    result.context("HTTP server failed")
}

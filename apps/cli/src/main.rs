//! `itts`: AI-agent friendly CLI, warm-model daemon, and MCP server for `IndexTTS-2.5` on
//! Apple Silicon.
//!
//! Every command accepts `--output text|json|jsonl` (or `ITTS_OUTPUT`). In JSON modes results
//! and failures are machine-readable, and failures exit with the stable code of their
//! [`ErrorCode`](index_tts_rs_service::ErrorCode).

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

mod commands;
mod daemon;
mod mcp;
mod output;
mod player;

use std::{
    env,
    io::{self, IsTerminal},
    path::PathBuf,
    process::ExitCode,
};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use index_tts_rs_core::config::EngineConfig;
use tracing_subscriber::EnvFilter;

use crate::{
    commands::{
        AnnotateArgs, SchemaTarget, annotate, daemon::DaemonCommand, doctor, models::ModelsCommand,
        schema, synth::SynthArgs, voice::VoiceCommand,
    },
    output::{OutputMode, report_failure},
};

#[derive(Debug, Parser)]
#[command(
    name = "itts",
    version,
    about = "Expressive speech synthesis and voice cloning (IndexTTS-2.5) on Apple Silicon",
    after_help = "Start here: `itts doctor`. Agents: add `--output json` for machine-readable \
                  results and errors; `itts schema controls` lists inline controls."
)]
struct Cli {
    /// YAML configuration file (see config/default.yaml).
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Local model directory override.
    #[arg(long, global = true)]
    model_dir: Option<PathBuf>,
    /// Directory holding `.itts_voice` snapshots.
    #[arg(long, global = true)]
    voices_dir: Option<PathBuf>,
    /// Output format for results and errors.
    #[arg(long, global = true, value_enum, default_value_t, env = "ITTS_OUTPUT")]
    output: OutputMode,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Synthesize speech from text with a registered voice or reference audio.
    Synth(Box<SynthArgs>),
    /// Add `DeepSeek` pacing and emotion controls without rewriting the text.
    Annotate(AnnotateArgs),
    /// Manage `.itts_voice` speaker snapshots.
    Voice {
        #[command(subcommand)]
        action: VoiceCommand,
    },
    /// Download or inspect model weights.
    Models {
        #[command(subcommand)]
        action: ModelsCommand,
    },
    /// Manage the warm-model background daemon.
    Daemon {
        #[command(subcommand)]
        action: DaemonCommand,
    },
    /// Diagnose the platform, model, voices, daemon, annotation key, and audio output.
    Doctor,
    /// Print JSON Schemas and the inline control catalog.
    Schema {
        /// Document to print.
        #[arg(value_enum, default_value = "speech-job")]
        target: SchemaTarget,
    },
    /// Run the Model Context Protocol server over stdio (uses the daemon).
    Mcp,
}

/// Global path flags, made absolute so they stay valid in a daemon started elsewhere.
#[derive(Debug, Clone, Default)]
pub struct GlobalPaths {
    /// `--config`.
    pub config: Option<PathBuf>,
    /// `--model-dir`.
    pub model_dir: Option<PathBuf>,
    /// `--voices-dir`.
    pub voices_dir: Option<PathBuf>,
}

impl GlobalPaths {
    fn new(cli: &Cli) -> Result<Self> {
        let cwd = env::current_dir().context("cannot read current directory")?;
        let absolute = |path: &Option<PathBuf>| path.as_deref().map(|p| cwd.join(p));
        Ok(Self {
            config: absolute(&cli.config),
            model_dir: absolute(&cli.model_dir),
            voices_dir: absolute(&cli.voices_dir),
        })
    }

    fn engine_config(&self) -> Result<EngineConfig> {
        let mut cfg = match &self.config {
            Some(path) => EngineConfig::from_yaml_file(path)?,
            None => EngineConfig::default(),
        };
        if self.model_dir.is_some() {
            cfg.model_dir.clone_from(&self.model_dir);
        }
        if self.voices_dir.is_some() {
            cfg.voices_dir.clone_from(&self.voices_dir);
        }
        cfg.validate_config()?;
        Ok(cfg)
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(&cli.command);
    let output = cli.output;
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => report_failure(output, &err),
    }
}

async fn run(cli: Cli) -> Result<()> {
    let globals = GlobalPaths::new(&cli)?;
    let output = cli.output;
    match cli.command {
        Commands::Schema { target } => schema(target),
        Commands::Annotate(args) => annotate(args, output).await,
        Commands::Synth(args) => {
            commands::synth::run(globals.engine_config()?, *args, output).await
        }
        Commands::Voice { action } => {
            commands::voice::run(globals.engine_config()?, action, output).await
        }
        Commands::Models { action } => {
            commands::models::run(&globals.engine_config()?, action, output).await
        }
        Commands::Daemon { action } => {
            commands::daemon::run(globals.engine_config()?, &globals, action, output).await
        }
        Commands::Doctor => doctor::run(&globals.engine_config()?, output).await,
        Commands::Mcp => mcp::run(globals.engine_config()?, globals).await,
    }
}

/// Logs go to stderr. Daemons log at `info` (JSON when stderr is the log file); other
/// commands log warnings only so stdout/stderr stay readable for agents.
fn init_tracing(command: &Commands) {
    let is_daemon = matches!(
        command,
        Commands::Daemon {
            action: DaemonCommand::Run { .. }
                | DaemonCommand::Start {
                    foreground: true,
                    ..
                }
        }
    );
    let default_level = if is_daemon { "info" } else { "warn" };
    let filter = EnvFilter::try_from_env("ITTS_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new(default_level));
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(io::stderr);
    let _ = if is_daemon && !io::stderr().is_terminal() {
        builder.json().try_init()
    } else {
        builder.try_init()
    };
}

//! Subcommand implementations.

pub mod daemon;
pub mod doctor;
pub mod models;
pub mod synth;
pub mod voice;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, ValueEnum};
use index_tts_rs_director::{Director, DirectorConfig};
use index_tts_rs_service::{SchemaKind, ServiceError, catalog, json_schema};
use serde_json::{json, to_value};

use crate::output::OutputMode;

/// `itts annotate` arguments.
#[derive(Debug, Args)]
pub struct AnnotateArgs {
    /// Text to annotate (or use --stdin).
    #[arg(long, short = 't', conflicts_with = "stdin")]
    text: Option<String>,
    /// Read the source text from standard input.
    #[arg(long)]
    stdin: bool,
    /// Use poetry pacing instead of conversational pacing.
    #[arg(long)]
    poetry: bool,
    /// YAML configuration for the `DeepSeek` provider.
    #[arg(long)]
    director_config: Option<PathBuf>,
}

/// Documents published by `itts schema`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SchemaTarget {
    /// JSON Schema of the speech request (`SpeechJob`).
    SpeechJob,
    /// JSON Schema of `synth --output jsonl` / daemon events.
    Events,
    /// JSON Schema of structured errors.
    Error,
    /// Inline controls, emotions, languages, ranges, and error codes.
    Controls,
}

/// Runs `itts annotate`.
///
/// # Errors
///
/// Returns the classified annotation failure.
pub async fn annotate(args: AnnotateArgs, output: OutputMode) -> Result<()> {
    let text = synth::read_text(args.text.as_deref(), args.stdin)?;
    let config = match args.director_config {
        Some(path) => DirectorConfig::from_yaml_file(path).await?,
        None => DirectorConfig::default(),
    };
    let director = Director::from_env(config).map_err(ServiceError::from)?;
    let script = director.annotate(&text, args.poetry).await?;
    output.emit(&json!({ "status": "ok", "script": script }), |out| {
        writeln!(out, "{script}")
    })
}

/// Runs `itts schema`.
///
/// # Errors
///
/// Returns an output failure.
pub fn schema(target: SchemaTarget) -> Result<()> {
    let value = match target {
        SchemaTarget::SpeechJob => json_schema(SchemaKind::SpeechJob),
        SchemaTarget::Events => json_schema(SchemaKind::Events),
        SchemaTarget::Error => json_schema(SchemaKind::Error),
        SchemaTarget::Controls => to_value(catalog())?,
    };
    OutputMode::Json.emit(&value, |_| Ok(()))
}

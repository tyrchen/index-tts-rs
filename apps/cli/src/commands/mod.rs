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
use index_tts_rs_service::{AnnotateJob, SchemaKind, ServiceError, catalog, json_schema};
use serde_json::{from_str, json, to_value};

use crate::output::OutputMode;

/// Bytes of an `AnnotateJob` document read from stdin at most (the daemon's 256 KiB line limit,
/// plus one byte to detect overflow).
const MAX_UNITS_JSON_BYTES: u64 = 256 * 1024 + 1;

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
    #[arg(long, conflicts_with = "units")]
    poetry: bool,
    /// Treat the input as an `AnnotateJob` JSON document (`{"units":[{"id","text"}],...}`, see
    /// `itts schema annotate-job`) and print one delivery per unit instead of a script.
    #[arg(long)]
    units: bool,
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
    /// JSON Schema of `itts annotate --units` input (`AnnotateJob`).
    AnnotateJob,
    /// JSON Schema of `itts annotate --units` output (`AnnotateResult`).
    AnnotateResult,
}

/// Runs `itts annotate`.
///
/// # Errors
///
/// Returns the classified annotation failure.
pub async fn annotate(args: AnnotateArgs, output: OutputMode) -> Result<()> {
    // A units document carries up to 16 KiB of text plus IDs, a brief, and JSON escaping; the
    // director enforces the decoded limits.
    let max_stdin = if args.units {
        MAX_UNITS_JSON_BYTES
    } else {
        synth::MAX_STDIN_TEXT_BYTES
    };
    let text = synth::read_text(args.text.as_deref(), args.stdin, max_stdin)?;
    if args.units && text.len() as u64 >= MAX_UNITS_JSON_BYTES {
        return Err(ServiceError::invalid(format!(
            "AnnotateJob documents are limited to {} bytes",
            MAX_UNITS_JSON_BYTES - 1
        ))
        .into());
    }
    let config = match args.director_config {
        Some(path) => DirectorConfig::from_yaml_file(path).await?,
        None => DirectorConfig::default(),
    };
    if args.units {
        let job: AnnotateJob = from_str(&text).map_err(|err| {
            ServiceError::invalid(format!("input is not an AnnotateJob document: {err}"))
                .with_hint("see `itts schema annotate-job`")
        })?;
        let result = job.run(config, None).await?;
        return output.emit(
            &json!({ "status": "ok", "deliveries": result.deliveries }),
            |out| {
                for delivery in &result.deliveries {
                    writeln!(
                        out,
                        "{:<16} {} pause {} ms",
                        delivery.id, delivery.controls, delivery.pause_after_ms
                    )?;
                }
                Ok(())
            },
        );
    }
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
        SchemaTarget::AnnotateJob => json_schema(SchemaKind::AnnotateJob),
        SchemaTarget::AnnotateResult => json_schema(SchemaKind::AnnotateResult),
        SchemaTarget::Controls => to_value(catalog())?,
    };
    OutputMode::Json.emit(&value, |_| Ok(()))
}

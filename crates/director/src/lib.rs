//! Speech control scripts and optional, text-preserving `DeepSeek` pacing annotation.
//!
//! Controls are executed by the application before inference; unsupported acoustic controls
//! are rejected. Automatic annotation chooses numbers for fixed source paragraphs and cannot
//! rewrite the supplied text.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

mod annotation;
mod controls;
mod reading;

use std::{io, result::Result as StdResult};

pub use annotation::{Direction, Director, DirectorConfig, ReadingStyle};
use config::ConfigError;
pub use controls::{PlannedUtterance, ReadingPlan, plan_utterances, prepend_silence};
use index_tts_rs_core::error::IndexTtsError;
use reqwest::Error as RequestError;
use serde_json::Error as JsonError;
use thiserror::Error;

/// Errors from script validation, annotation, and provider communication.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DirectorError {
    /// Invalid control, input, or model output.
    #[error("{0}")]
    Invalid(String),
    /// The configured provider key is missing or malformed.
    #[error("DEEPSEEK_API_KEY is not configured or is invalid")]
    MissingKey,
    /// Configuration could not be read.
    #[error("could not read director configuration: {0}")]
    Io(#[from] io::Error),
    /// YAML configuration could not be parsed.
    #[error("invalid director configuration: {source}")]
    Config {
        /// Underlying configuration parser error.
        #[source]
        source: Box<ConfigError>,
    },
    /// Validated model input was rejected by the speech engine.
    #[error(transparent)]
    Speech(#[from] Box<IndexTtsError>),
    /// Provider request failed. Authentication headers are never included.
    #[error("annotation request failed: {0}")]
    Transport(#[from] RequestError),
    /// The provider returned a non-success status.
    #[error("annotation provider returned HTTP {0}")]
    Provider(u16),
    /// A provider response was not valid JSON.
    #[error("invalid annotation response: {0}")]
    Json(#[from] JsonError),
}

impl From<IndexTtsError> for DirectorError {
    fn from(error: IndexTtsError) -> Self {
        Self::Speech(Box::new(error))
    }
}

/// Result type for speech control and annotation operations.
pub type Result<T> = StdResult<T, DirectorError>;

//! Error type of the recognizer and the script comparison.

use std::{io, path::PathBuf};

use thiserror::Error;

/// Result alias for this crate.
pub type Result<T, E = AsrError> = std::result::Result<T, E>;

/// Recognition and comparison failures.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AsrError {
    /// A model name, language, or input is not accepted.
    #[error("{0}")]
    InvalidArgument(String),
    /// Model files are missing, fail verification, or do not load.
    #[error("speech recognition model `{model}` unavailable: {message}")]
    ModelUnavailable {
        /// Model name.
        model: &'static str,
        /// What went wrong.
        message: String,
    },
    /// An input exceeds what one request can process.
    #[error("{0}")]
    TooLong(String),
    /// A filesystem operation failed.
    #[error("I/O error at `{path}`: {source}")]
    Io {
        /// Path involved.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: io::Error,
    },
    /// The recognizer worker thread is gone or failed.
    #[error("speech recognition worker: {0}")]
    Actor(String),
}

impl AsrError {
    /// Shorthand for [`AsrError::InvalidArgument`].
    #[must_use]
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    /// Shorthand for [`AsrError::Actor`].
    #[must_use]
    pub fn actor(message: impl Into<String>) -> Self {
        Self::Actor(message.into())
    }

    /// Shorthand for [`AsrError::ModelUnavailable`].
    #[must_use]
    pub fn model(model: &'static str, message: impl Into<String>) -> Self {
        Self::ModelUnavailable {
            model,
            message: message.into(),
        }
    }
}

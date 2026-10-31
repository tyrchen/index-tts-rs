//! Domain-specific error types for the `index-tts-rs-core` crate.
//!
//! All fallible operations in `index-tts-rs-core` return [`Result<T>`], backed
//! by [`IndexTtsError`]. No `unwrap()` or `expect()` is permitted in production
//! code paths.

use std::{io, path::PathBuf};

use thiserror::Error;

/// Convenience result alias using [`IndexTtsError`].
pub type Result<T, E = IndexTtsError> = std::result::Result<T, E>;

/// Primary error enum for all `index-tts-rs-core` subsystems.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IndexTtsError {
    /// Configuration file parsing or layering failure.
    #[error("configuration error: {message}")]
    Config {
        /// Human-readable error context.
        message: String,
        /// Optional underlying `config::ConfigError`.
        #[source]
        source: Option<config::ConfigError>,
    },

    /// Input validation failure at the trust boundary.
    #[error("validation error on `{field}`: {reason}")]
    Validation {
        /// Name of the invalid field or parameter.
        field: &'static str,
        /// Explanation of why validation failed.
        reason: String,
    },

    /// File system or I/O error with associated path context.
    #[error("I/O error at `{path}`: {source}")]
    Io {
        /// Path associated with the failed I/O operation.
        path: PathBuf,
        /// Underlying standard I/O error.
        #[source]
        source: io::Error,
    },

    /// Audio container or codec decoding error.
    #[error("audio decode error: {message}")]
    AudioDecode {
        /// Detailed reason for the decode failure.
        message: String,
    },

    /// Audio WAV encoding error.
    #[error("audio encode error: {message}")]
    AudioEncode {
        /// Detailed reason for the encode failure.
        message: String,
    },

    /// Digital signal processing (resampling, FFT, Mel/Fbank) error.
    #[error("DSP processing error: {message}")]
    Dsp {
        /// Detailed reason for the DSP failure.
        message: String,
    },

    /// Multilingual frontend or BPE tokenizer error.
    #[error("tokenization error: {message}")]
    Tokenize {
        /// Detailed reason for the tokenization failure.
        message: String,
    },

    /// `SafeTensors` or `.npz` model weight loading and shape/dtype validation error.
    #[error("weight load error in `{artifact}`: {message}")]
    WeightLoad {
        /// Model file or tensor name.
        artifact: String,
        /// Detailed reason for the weight loading failure.
        message: String,
    },

    /// MLX / GPU tensor execution error.
    #[error("MLX execution error: {message}")]
    Mlx {
        /// Detailed reason for the tensor or graph failure.
        message: String,
    },

    /// Requested voice snapshot was not found in memory or on disk.
    #[error("voice `{voice_id}` not found")]
    VoiceNotFound {
        /// Identifier of the missing voice.
        voice_id: String,
    },

    /// Actor communication channel closed unexpectedly.
    #[error("engine actor channel closed: {context}")]
    ChannelClosed {
        /// Operation context when the channel closed.
        context: &'static str,
    },

    /// Synthesis or extraction task was cancelled by caller or shutdown signal.
    #[error("operation cancelled: {reason}")]
    Cancelled {
        /// Reason for cancellation.
        reason: &'static str,
    },
}

impl IndexTtsError {
    /// Constructs a [`IndexTtsError::Validation`] variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::error::IndexTtsError;
    ///
    /// let err = IndexTtsError::validation("temperature", "must be in 0.05..=2.0");
    /// assert!(err.to_string().contains("temperature"));
    /// ```
    #[must_use]
    pub fn validation(field: &'static str, reason: impl Into<String>) -> Self {
        Self::Validation {
            field,
            reason: reason.into(),
        }
    }

    /// Constructs an [`IndexTtsError::Io`] variant with path context.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io;
    /// use index_tts_rs_core::error::IndexTtsError;
    ///
    /// let err = IndexTtsError::io("/tmp/missing.wav", io::Error::from(io::ErrorKind::NotFound));
    /// assert!(err.to_string().contains("/tmp/missing.wav"));
    /// ```
    #[must_use]
    pub fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Constructs an [`IndexTtsError::WeightLoad`] variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::error::IndexTtsError;
    ///
    /// let err = IndexTtsError::weight_load("spk_proj.npz", "missing weight array");
    /// assert!(err.to_string().contains("spk_proj.npz"));
    /// ```
    #[must_use]
    pub fn weight_load(artifact: impl Into<String>, message: impl Into<String>) -> Self {
        Self::WeightLoad {
            artifact: artifact.into(),
            message: message.into(),
        }
    }
}

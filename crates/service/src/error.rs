//! Stable, machine-readable error codes shared by every `itts` frontend.
//!
//! Agents branch on [`ErrorCode`]; humans read `message`; `hint` is always an actionable next
//! step (usually a concrete `itts` command). Each code maps to a fixed process exit code.

use index_tts_rs_core::error::IndexTtsError;
use index_tts_rs_director::DirectorError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Hint attached to every voice lookup failure.
const VOICE_HINT: &str = "run `itts voice list` to see registered voices, or `itts voice add <ID> \
                          --ref <AUDIO>` to register one";
/// Hint attached to every model resolution or weight loading failure.
const MODEL_HINT: &str =
    "run `itts models pull` to download the weights, or `itts doctor` to diagnose";

/// Stable error classification. Serialized as `SCREAMING_SNAKE_CASE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[non_exhaustive]
pub enum ErrorCode {
    /// A request field, CLI flag, control tag, or configuration value is invalid.
    InvalidArgument,
    /// The requested voice is not registered.
    VoiceNotFound,
    /// Model weights are missing, incomplete, or failed to load.
    ModelUnavailable,
    /// Reference audio could not be decoded.
    AudioDecodeFailed,
    /// No usable audio output device for playback.
    AudioOutputUnavailable,
    /// The daemon is not running or not reachable.
    DaemonUnavailable,
    /// The running daemon speaks an incompatible protocol version.
    DaemonVersionMismatch,
    /// Too many requests are already queued.
    Busy,
    /// The request was cancelled before completion.
    Cancelled,
    /// The request exceeded its time budget.
    Timeout,
    /// Annotation requires `DEEPSEEK_API_KEY`, which is not configured.
    AnnotationUnavailable,
    /// The annotation provider failed or returned invalid output.
    AnnotationFailed,
    /// A filesystem operation failed.
    Io,
    /// An unexpected internal failure.
    Internal,
}

impl ErrorCode {
    /// Every code, in documentation order.
    pub const ALL: [Self; 14] = [
        Self::InvalidArgument,
        Self::VoiceNotFound,
        Self::ModelUnavailable,
        Self::AudioDecodeFailed,
        Self::AudioOutputUnavailable,
        Self::DaemonUnavailable,
        Self::DaemonVersionMismatch,
        Self::Busy,
        Self::Cancelled,
        Self::Timeout,
        Self::AnnotationUnavailable,
        Self::AnnotationFailed,
        Self::Io,
        Self::Internal,
    ];

    /// Process exit code for CLI invocations failing with this code.
    ///
    /// ```
    /// use index_tts_rs_service::ErrorCode;
    /// assert_eq!(ErrorCode::InvalidArgument.exit_code(), 2);
    /// assert_eq!(ErrorCode::Internal.exit_code(), 1);
    /// ```
    #[must_use]
    pub const fn exit_code(self) -> u8 {
        match self {
            Self::Internal => 1,
            Self::InvalidArgument => 2,
            Self::VoiceNotFound => 3,
            Self::ModelUnavailable => 4,
            Self::AudioDecodeFailed | Self::AudioOutputUnavailable => 5,
            Self::DaemonUnavailable | Self::DaemonVersionMismatch => 6,
            Self::Busy => 7,
            Self::Cancelled => 8,
            Self::Timeout => 9,
            Self::AnnotationUnavailable | Self::AnnotationFailed => 10,
            Self::Io => 11,
        }
    }

    /// Wire name of this code (`VOICE_NOT_FOUND`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgument => "INVALID_ARGUMENT",
            Self::VoiceNotFound => "VOICE_NOT_FOUND",
            Self::ModelUnavailable => "MODEL_UNAVAILABLE",
            Self::AudioDecodeFailed => "AUDIO_DECODE_FAILED",
            Self::AudioOutputUnavailable => "AUDIO_OUTPUT_UNAVAILABLE",
            Self::DaemonUnavailable => "DAEMON_UNAVAILABLE",
            Self::DaemonVersionMismatch => "DAEMON_VERSION_MISMATCH",
            Self::Busy => "BUSY",
            Self::Cancelled => "CANCELLED",
            Self::Timeout => "TIMEOUT",
            Self::AnnotationUnavailable => "ANNOTATION_UNAVAILABLE",
            Self::AnnotationFailed => "ANNOTATION_FAILED",
            Self::Io => "IO",
            Self::Internal => "INTERNAL",
        }
    }

    /// One-line description used by `itts schema errors` and MCP `describe_controls`.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::InvalidArgument => "A request field, flag, control tag, or config is invalid.",
            Self::VoiceNotFound => "The requested voice is not registered.",
            Self::ModelUnavailable => "Model weights are missing, incomplete, or failed to load.",
            Self::AudioDecodeFailed => "Reference audio could not be decoded.",
            Self::AudioOutputUnavailable => "No usable audio output device for playback.",
            Self::DaemonUnavailable => "The daemon is not running or not reachable.",
            Self::DaemonVersionMismatch => "The running daemon uses an incompatible protocol.",
            Self::Busy => "Too many requests are already queued; retry later.",
            Self::Cancelled => "The request was cancelled before completion.",
            Self::Timeout => "The request exceeded its time budget.",
            Self::AnnotationUnavailable => "Annotation requires DEEPSEEK_API_KEY.",
            Self::AnnotationFailed => "The annotation provider failed or returned bad output.",
            Self::Io => "A filesystem operation failed.",
            Self::Internal => "An unexpected internal failure.",
        }
    }
}

/// Structured error returned by every frontend (CLI `--output json`, daemon, MCP, HTTP).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Error)]
#[serde(rename_all = "camelCase")]
#[error("{message}")]
#[non_exhaustive]
pub struct ServiceError {
    /// Stable classification.
    pub code: ErrorCode,
    /// Human-readable explanation.
    pub message: String,
    /// Actionable next step, when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl ServiceError {
    /// Creates an error without a hint.
    ///
    /// ```
    /// use index_tts_rs_service::{ErrorCode, ServiceError};
    /// let err = ServiceError::new(ErrorCode::Busy, "queue full").with_hint("retry later");
    /// assert_eq!(err.hint.as_deref(), Some("retry later"));
    /// ```
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            hint: None,
        }
    }

    /// Shorthand for [`ErrorCode::InvalidArgument`].
    #[must_use]
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, message)
    }

    /// [`ErrorCode::VoiceNotFound`] with the standard registration hint.
    #[must_use]
    pub fn voice_not_found(voice_id: &str) -> Self {
        Self::from(&IndexTtsError::VoiceNotFound {
            voice_id: voice_id.to_owned(),
        })
    }

    /// Shorthand for [`ErrorCode::Internal`].
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }

    /// Attaches an actionable hint.
    #[must_use]
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// Process exit code for this error.
    #[must_use]
    pub const fn exit_code(&self) -> u8 {
        self.code.exit_code()
    }
}

impl From<IndexTtsError> for ServiceError {
    fn from(err: IndexTtsError) -> Self {
        Self::from(&err)
    }
}

impl From<&IndexTtsError> for ServiceError {
    fn from(err: &IndexTtsError) -> Self {
        let message = err.to_string();
        match err {
            IndexTtsError::Validation { .. } | IndexTtsError::Tokenize { .. } => {
                Self::invalid(message)
            }
            IndexTtsError::Config { .. } => Self::invalid(message)
                .with_hint("check the YAML passed to `--config` against config/default.yaml"),
            IndexTtsError::VoiceNotFound { .. } => {
                Self::new(ErrorCode::VoiceNotFound, message).with_hint(VOICE_HINT)
            }
            IndexTtsError::WeightLoad { .. } => {
                Self::new(ErrorCode::ModelUnavailable, message).with_hint(MODEL_HINT)
            }
            IndexTtsError::AudioDecode { .. } => Self::new(ErrorCode::AudioDecodeFailed, message)
                .with_hint("use a 3-15 s mono speech clip in .wav, .mp3, .flac, or .m4a format"),
            IndexTtsError::Io { .. } => Self::new(ErrorCode::Io, message),
            IndexTtsError::Cancelled { .. } => Self::new(ErrorCode::Cancelled, message),
            _ => Self::internal(message),
        }
    }
}

impl From<DirectorError> for ServiceError {
    fn from(err: DirectorError) -> Self {
        Self::from(&err)
    }
}

impl From<&DirectorError> for ServiceError {
    fn from(err: &DirectorError) -> Self {
        match err {
            DirectorError::Speech(inner) => Self::from(inner.as_ref()),
            DirectorError::Invalid(message) => Self::invalid(message.as_str()).with_hint(
                "supported controls: [[pause=MS]] [[speed=X]] [[emo=NAME:W,...]] [[temp=X]] \
                 [[voice=ID]] [[lang=CODE]] [[reset]]",
            ),
            DirectorError::MissingKey => {
                Self::new(ErrorCode::AnnotationUnavailable, err.to_string())
                    .with_hint("export DEEPSEEK_API_KEY=<key> before starting itts")
            }
            DirectorError::Io(_) | DirectorError::Config { .. } => Self::invalid(err.to_string()),
            _ => Self::new(ErrorCode::AnnotationFailed, err.to_string())
                .with_hint("retry later; the source text was not modified"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use index_tts_rs_core::error::IndexTtsError;
    use index_tts_rs_director::DirectorError;
    use rstest::rstest;
    use serde_json::{json, to_value};

    use super::{ErrorCode, ServiceError};

    #[test]
    fn test_should_serialize_codes_in_screaming_snake_case() -> serde_json::Result<()> {
        for code in ErrorCode::ALL {
            assert_eq!(to_value(code)?, json!(code.as_str()));
        }
        Ok(())
    }

    #[test]
    fn test_should_keep_wire_names_unique() {
        let names: HashSet<_> = ErrorCode::ALL.iter().map(|c| c.as_str()).collect();
        assert_eq!(names.len(), ErrorCode::ALL.len());
    }

    #[rstest]
    #[case(IndexTtsError::VoiceNotFound { voice_id: "x".to_owned() }, ErrorCode::VoiceNotFound)]
    #[case(IndexTtsError::validation("lang", "bad"), ErrorCode::InvalidArgument)]
    #[case(
        IndexTtsError::weight_load("gpt", "missing"),
        ErrorCode::ModelUnavailable
    )]
    #[case(IndexTtsError::actor("boom"), ErrorCode::Internal)]
    fn test_should_map_engine_errors_to_stable_codes(
        #[case] err: IndexTtsError,
        #[case] code: ErrorCode,
    ) {
        assert_eq!(ServiceError::from(err).code, code);
    }

    #[test]
    fn test_should_give_actionable_hints_for_missing_resources() {
        let voice = ServiceError::from(IndexTtsError::VoiceNotFound {
            voice_id: "x".to_owned(),
        });
        assert!(voice.hint.is_some_and(|h| h.contains("itts voice list")));
        let model = ServiceError::from(IndexTtsError::weight_load("gpt", "missing"));
        assert!(model.hint.is_some_and(|h| h.contains("itts models pull")));
    }

    #[test]
    fn test_should_unwrap_speech_errors_from_director() {
        let err = DirectorError::from(IndexTtsError::VoiceNotFound {
            voice_id: "x".to_owned(),
        });
        assert_eq!(ServiceError::from(err).code, ErrorCode::VoiceNotFound);
        assert_eq!(
            ServiceError::from(DirectorError::MissingKey).code,
            ErrorCode::AnnotationUnavailable
        );
    }
}

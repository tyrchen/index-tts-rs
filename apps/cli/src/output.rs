//! Output modes and structured error rendering shared by every `itts` subcommand.
//!
//! stdout carries results only; progress and diagnostics go to stderr. In `json`/`jsonl` modes
//! failures are also JSON (`{"status":"error","error":{code,message,hint}}`) and the process exits
//! with the code's stable exit status.

use std::{
    io::{self, Write},
    process::ExitCode,
};

use anyhow::{Error, Result};
use clap::ValueEnum;
use index_tts_rs_core::error::IndexTtsError;
use index_tts_rs_director::DirectorError;
use index_tts_rs_service::ServiceError;
use serde::Serialize;
use serde_json::{json, to_string, to_string_pretty};

/// How command results are written to stdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum OutputMode {
    /// Human-readable text.
    #[default]
    Text,
    /// One pretty-printed JSON document.
    Json,
    /// Compact JSON Lines; streaming commands emit one event per line.
    Jsonl,
}

impl OutputMode {
    /// Whether output is machine-readable JSON.
    pub const fn is_json(self) -> bool {
        matches!(self, Self::Json | Self::Jsonl)
    }

    /// Writes `value` as JSON, or calls `text` to render it for humans.
    pub fn emit<T: Serialize>(
        self,
        value: &T,
        text: impl FnOnce(&mut dyn Write) -> io::Result<()>,
    ) -> Result<()> {
        let mut out = io::stdout().lock();
        match self {
            Self::Text => text(&mut out)?,
            Self::Json => writeln!(out, "{}", to_string_pretty(value)?)?,
            Self::Jsonl => writeln!(out, "{}", to_string(value)?)?,
        }
        out.flush()?;
        Ok(())
    }

    /// Writes one JSON line to stdout (streaming events).
    pub fn emit_line<T: Serialize>(value: &T) -> Result<()> {
        let mut out = io::stdout().lock();
        writeln!(out, "{}", to_string(value)?)?;
        out.flush()?;
        Ok(())
    }
}

/// Classifies any error into a [`ServiceError`], keeping the full context chain as message.
pub fn classify(err: &Error) -> ServiceError {
    let message = format!("{err:#}");
    for cause in err.chain() {
        let classified = if let Some(found) = cause.downcast_ref::<ServiceError>() {
            Some(found.clone())
        } else if let Some(found) = cause.downcast_ref::<IndexTtsError>() {
            Some(ServiceError::from(found))
        } else {
            cause
                .downcast_ref::<DirectorError>()
                .map(ServiceError::from)
        };
        if let Some(mut classified) = classified {
            classified.message = message;
            return classified;
        }
    }
    ServiceError::internal(message)
}

/// Reports a failed command and returns its exit code.
pub fn report_failure(mode: OutputMode, err: &Error) -> ExitCode {
    let error = classify(err);
    let code = ExitCode::from(error.exit_code());
    if mode.is_json() {
        let body = json!({ "status": "error", "error": error });
        let rendered = if mode == OutputMode::Json {
            to_string_pretty(&body)
        } else {
            to_string(&body)
        };
        if let Ok(line) = rendered {
            let _ = writeln!(io::stdout(), "{line}");
        }
    } else {
        let mut stderr = io::stderr().lock();
        let _ = writeln!(stderr, "error[{}]: {}", error.code.as_str(), error.message);
        if let Some(hint) = &error.hint {
            let _ = writeln!(stderr, "  hint: {hint}");
        }
    }
    code
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use index_tts_rs_core::error::IndexTtsError;
    use index_tts_rs_service::ErrorCode;

    use super::classify;

    #[test]
    fn test_should_classify_through_context_layers() {
        let err = Err::<(), _>(IndexTtsError::VoiceNotFound {
            voice_id: "ghost".to_owned(),
        })
        .context("synthesis failed")
        .err();
        let classified = err.as_ref().map(classify);
        assert_eq!(
            classified.as_ref().map(|e| e.code),
            Some(ErrorCode::VoiceNotFound)
        );
        assert!(
            classified
                .is_some_and(|e| e.message.starts_with("synthesis failed") && e.hint.is_some())
        );
    }

    #[test]
    fn test_should_default_unknown_errors_to_internal() {
        let classified = classify(&anyhow::anyhow!("boom"));
        assert_eq!(classified.code, ErrorCode::Internal);
    }
}

//! Local verbatim speech recognition and script verification for `itts`.
//!
//! - [`AsrModel`] / [`models`]: the pinned sherpa-onnx exports itts runs (SenseVoice-Small,
//!   Paraformer-large, Parakeet-TDT), where they are cached, and first-use downloads.
//! - [`AsrEngine`] (feature `recognizer`): the ONNX Runtime actor. It owns the loaded recognizers
//!   on its own thread, independent of the MLX engine, and loads each model the first time it is
//!   used.
//! - [`heard_tokens`]: recognizer tokens → one unit per CJK character or Latin word, timed.
//! - [`compare`](compare::compare): script vs heard units → findings (repeat, insertion, omission,
//!   substitution, number), graded with forced-alignment evidence.
//!
//! The recognizers are non-autoregressive (CTC / CIF / transducer), so they keep the stutters
//! and repeats that an autoregressive recognizer such as Whisper smooths away.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

pub mod compare;
#[cfg(feature = "recognizer")]
mod engine;
mod error;
pub mod models;
mod tokens;

#[cfg(feature = "recognizer")]
pub use engine::{ASR_THREADS, AsrEngine, Transcript};
pub use error::{AsrError, Result};
pub use models::{AsrModel, ModelFile, ModelFiles, SpokenLanguage};
pub use tokens::{HeardToken, MAX_TOKEN_SECS, RawToken, heard_text, heard_tokens, is_cjk};

/// Sample rate the recognizers take.
pub const ASR_SAMPLE_RATE: u32 = 16_000;

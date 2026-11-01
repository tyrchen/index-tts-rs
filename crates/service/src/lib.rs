//! Transport-neutral service layer shared by every `itts` frontend (CLI, daemon, MCP, HTTP).
//!
//! - [`SpeechJob`]: the one request model. Each transport maps its input onto it, then
//!   [`SpeechJob::prepare`] validates and compiles inline controls exactly once.
//! - [`JobRun`] / [`run_job`]: execute a [`PreparedJob`] on an engine with pauses applied, placing
//!   every chunk on the job timeline with its text and (on request) character [`Alignment`].
//! - [`AnnotateJob`]: per-unit delivery direction (emotion, temperature, speed, pause) for
//!   caller-supplied units, with an optional per-request provider [`ApiKey`].
//! - [`JobEvent`]: the `started` → `chunk`* → `done` | `error` stream every frontend emits.
//! - [`ServiceError`] / [`ErrorCode`]: stable codes, actionable hints, and CLI exit codes.
//! - [`catalog`] / [`json_schema`]: machine-readable self-description for agents.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

mod annotation;
mod catalog;
mod error;
mod events;
mod request;
mod run;
mod transcript;

pub use annotation::{AnnotateJob, AnnotateResult, AnnotationUnit, ApiKey, Delivery};
pub use catalog::{Catalog, ControlSpec, EmotionSpec, ErrorSpec, SchemaKind, catalog, json_schema};
pub use error::{ErrorCode, ServiceError};
pub use events::{Backend, ChunkInfo, JobEvent, JobStarted, JobSummary};
pub use request::{
    AdvancedOptions, DEFAULT_PARAGRAPH_PAUSE_MS, MAX_SILENCE_MS, MAX_SPEED, MIN_SPEED, PathPolicy,
    PreparedJob, SpeechJob, resolve_input_path,
};
pub use run::{JobAccumulator, JobChunk, JobRun, run_job};
pub use transcript::{Alignment, SentenceTiming, Transcript, sentences};

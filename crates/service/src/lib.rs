//! Transport-neutral service layer shared by every `itts` frontend (CLI, daemon, MCP, HTTP).
//!
//! - [`SpeechJob`]: the one request model. Each transport maps its input onto it, then
//!   [`SpeechJob::prepare`] validates and compiles inline controls exactly once.
//! - [`JobRun`] / [`run_job`]: execute a [`PreparedJob`] on an engine with pauses applied, placing
//!   every chunk on the job timeline with its text and (on request) character [`Alignment`].
//! - [`AlignJob`] / [`run_align`]: force-align a recording to its text ([`AlignResult`]: character
//!   timings in the `synth` shape, attention quality, optional phones).
//! - [`ListenJob`] / `run_listen`: verbatim local speech recognition of a recording
//!   ([`ListenResult`]: text and one timed unit per CJK character or Latin word).
//! - [`VerifyJob`] / `run_verify`: align and recognize a recording at once and compare it with its
//!   script ([`VerifyResult`]: findings — repeats, insertions, omissions, substitutions, numbers
//!   read as other values — with times and severities). Running listen / verify jobs needs the
//!   `asr` feature (on by default); their types are always available.
//! - [`AnnotateJob`]: per-unit delivery direction (emotion, temperature, speed, pause) for
//!   caller-supplied units, with an optional per-request provider [`ApiKey`].
//! - [`JobEvent`]: the `started` → `chunk`* → `done` | `error` stream every frontend emits.
//! - [`ServiceError`] / [`ErrorCode`]: stable codes, actionable hints, and CLI exit codes.
//! - [`catalog`] / [`json_schema`]: machine-readable self-description for agents.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

mod align;
mod annotation;
mod catalog;
mod error;
mod events;
mod listen;
mod request;
mod run;
mod transcript;
mod verify;

pub use align::{
    AlignJob, AlignQuality, AlignResult, AlignTimings, AlignerInfo, Phone, PhoneKindWire,
    PhonesMethod, PreparedAlign, run_align,
};
pub use annotation::{AnnotateJob, AnnotateResult, AnnotationUnit, ApiKey, Delivery};
pub use catalog::{Catalog, ControlSpec, EmotionSpec, ErrorSpec, SchemaKind, catalog, json_schema};
pub use error::{ErrorCode, ServiceError};
pub use events::{Backend, ChunkInfo, JobEvent, JobStarted, JobSummary};
#[cfg(feature = "asr")]
pub use listen::run_listen;
pub use listen::{
    HeardUnit, ListenJob, ListenResult, ListenTimings, MAX_LISTEN_SECS, MIN_LISTEN_SECS,
    PreparedListen, RECOGNIZER_NAME, RECOGNIZER_VERSION, RecognizerInfo, asr_unavailable,
};
pub use request::{
    AdvancedOptions, DEFAULT_PARAGRAPH_PAUSE_MS, MAX_SILENCE_MS, MAX_SPEED, MIN_SPEED, PathPolicy,
    PreparedJob, SpeechJob, resolve_input_path,
};
pub use run::{JobAccumulator, JobChunk, JobRun, run_job};
pub use transcript::{Alignment, SentenceTiming, Transcript, sentences};
#[cfg(feature = "asr")]
pub use verify::run_verify;
pub use verify::{
    Finding, FindingKind, PreparedVerify, Severity, Verdict, VerifyJob, VerifyResult, VerifyTimings,
};

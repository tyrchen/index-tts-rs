//! Actor-based GPU inference engine (`IndexTtsEngine`) and streaming output pipeline.

pub mod actor;
pub mod stream;
pub mod telemetry;

pub use actor::{
    ENGINE_CHANNEL_CAPACITY, IndexTtsEngine, MEL_TO_CODEC_RATIO, STREAM_CHANNEL_CAPACITY,
};
pub use stream::{AudioChunk, StageTimings, SynthesisOutput, SynthesisResult, SynthesisStream};
pub use telemetry::{StartupEvent, StartupObserver, TelemetrySnapshot};

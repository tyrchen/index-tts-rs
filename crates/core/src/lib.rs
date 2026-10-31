//! High-performance Rust inference core, pure-Rust DSP pipeline, and multilingual
//! text frontend for `IndexTTS-2.5` on Apple Silicon.
//!
//! # Architecture Overview
//!
//! - [`config`]: Validated domain primitives (`SynthesisText`, `VoiceId`, `Temperature`, `TopP`,
//!   `TopK`, `Language`) and YAML configuration management (`EngineConfig`).
//! - [`error`]: Unified domain error hierarchy (`IndexTtsError`).
//! - [`dsp`]: Pure-Rust audio decoding/encoding (`symphonia`/`hound`), Hann-sinc resampling
//!   (`resample_sinc_hann`), 22.05 kHz Slaney Mel spectrograms, and 16 kHz Kaldi Povey filterbanks
//!   (`campplus_fbank` & `w2v_features`).
//! - [`frontend`]: `winnow`-based `<word|PRON>` pronunciation parser, YAML glossary and number
//!   normalizer, Tiktoken BPE encoder, and token-budget sentence splitter.
//! - [`models`]: Local Hugging Face snapshot resolution, `spk_proj.npz` parser, and dead-weight
//!   filtering `SafeTensors` loader with automatic PyTorch-to-MLX layout transposition.
//! - [`voice`]: `.itts_voice` speaker conditioning snapshot (`SpeakerContext`) serialization and
//!   validation.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

pub mod config;
pub mod dsp;
pub mod engine;
pub mod error;
pub mod frontend;
pub mod models;
pub mod voice;

pub use config::{
    CfgRate, CfmParams, DurationFactor, EmoAlpha, EngineConfig, Language, RepetitionPenalty,
    SamplingParams, SpeakerSource, SynthesisRequest, SynthesisText, Temperature, TopK, TopP,
    VoiceExtractRequest, VoiceId,
};
pub use engine::{
    AudioChunk, IndexTtsEngine, StageTimings, SynthesisOutput, SynthesisResult, SynthesisStream,
};
pub use error::{IndexTtsError, Result};
pub use frontend::Frontend;
pub use models::{
    BigVgan, CampPlus, CfmDit, EmoVec, Gpt, KvCache, LengthRegulator, SamplerRng, SeenTokenBitSet,
    SemanticCodec, SemanticModel, generate_codes, nearest_expand_indices, sample_next_token,
};
pub use voice::{SpeakerContext, VoiceEntry, VoiceExtractor, VoiceMetadata, VoiceStore};

//! Request and result types for aligning an external recording ([`AlignmentRequest`]).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use crate::{
    align::{AlignmentQuality, SpanTiming, phones::PhoneTiming},
    config::{Language, SpeakerSource, SynthesisText},
};

/// Aligner name reported to callers (`itts-attn`).
pub const ALIGNER_NAME: &str = "itts-attn";
/// Aligner major version; bumped whenever the encoder, head selection, decoding rules, or the
/// quality definition change the output for the same input.
pub const ALIGNER_VERSION: u32 = 1;
/// Most semantic codes one alignment may replay (about 72.6 s of audio): the decoder's Mel
/// positions minus `<START_MEL>`, `<STOP_MEL>`, and one spare.
pub const MAX_ALIGN_CODES: usize = 1_815;
/// Shortest recording accepted, in seconds.
pub const MIN_ALIGN_SECS: f32 = 0.1;

/// Where the GPT speaker conditioning (`conds`) for an alignment comes from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Conditioning {
    /// The recording itself (its first 15 s), computed like a voice snapshot.
    #[default]
    Recording,
    /// A registered voice, a snapshot file, or reference audio.
    Speaker(SpeakerSource),
}

impl Conditioning {
    /// Wire label: `recording` or `voice`.
    ///
    /// ```
    /// use index_tts_rs_core::align::Conditioning;
    /// assert_eq!(Conditioning::Recording.label(), "recording");
    /// ```
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Recording => "recording",
            Self::Speaker(_) => "voice",
        }
    }
}

/// The audio of an alignment request.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Recording {
    /// An audio file (`.wav`/`.mp3`/`.flac`/`.m4a`), decoded on the engine's worker thread.
    File(PathBuf),
    /// Mono samples in `[-1, 1]` at `sample_rate` Hz.
    Pcm {
        /// Samples.
        samples: Vec<f32>,
        /// Sample rate in Hz.
        sample_rate: u32,
    },
}

/// A recording and the text it speaks, to be force-aligned.
#[derive(Debug, Clone, TypedBuilder)]
#[non_exhaustive]
pub struct AlignmentRequest {
    /// The recording.
    pub recording: Recording,
    /// Spoken text (controls already removed); may contain pronunciation overrides.
    pub text: SynthesisText,
    /// Language of the text.
    pub language: Language,
    /// Source of the GPT speaker conditioning.
    #[builder(default)]
    pub conditioning: Conditioning,
    /// Also split every syllable or word into phones.
    #[builder(default)]
    pub phones: bool,
}

/// Wall-clock milliseconds spent in each alignment stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct AlignmentTimings {
    /// Reading and decoding the audio file.
    pub audio_ms: f64,
    /// Resampling, fbank, and `w2v-BERT` features.
    pub features_ms: f64,
    /// Codec encoder quantization.
    pub encode_ms: f64,
    /// Speaker conditioning (snapshot lookup or recording analysis).
    pub condition_ms: f64,
    /// Text preparation, teacher-forced attention, decoding, and phones.
    pub attention_ms: f64,
    /// Whole request inside the engine, including lazy model loads.
    pub total_ms: f64,
}

/// Result of aligning one recording.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct AlignmentOutput {
    /// Timing of every spoken token on request-text characters, seconds from the recording start.
    pub spans: Vec<SpanTiming>,
    /// Attention support of the alignment.
    pub quality: AlignmentQuality,
    /// Phone timings when requested.
    pub phones: Vec<PhoneTiming>,
    /// Recording duration in seconds.
    pub duration_secs: f32,
    /// Semantic codes replayed through the decoder.
    pub codes: usize,
    /// Per-stage timings.
    pub timings: AlignmentTimings,
}

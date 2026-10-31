//! Voice snapshot (`.itts_voice`) representation, extraction, and store management.

pub mod context;
pub mod store;

pub use context::{
    CONDS_DIM, CONDS_FRAMES, EMO_PROTOTYPE_COUNTS, EMO_PROTOTYPE_ROWS, EmoMatrix,
    ITTS_VOICE_FORMAT_VERSION, PROMPT_CONDITION_DIM, REF_MEL_BINS, STYLE_DIM, SpeakerContext,
    VoiceMetadata,
};
pub use store::{
    DEFAULT_VOICES_SUBDIR, MAX_REF_SAMPLES_16K, MAX_REF_SAMPLES_22K, VoiceEntry, VoiceExtractor,
    VoiceStore, default_voices_dir,
};

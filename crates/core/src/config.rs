//! Configuration management and validated domain primitives (Newtypes).
//!
//! All external inputs (text, voice identifiers, sampling hyperparameters,
//! CFM ODE parameters, and YAML configuration files) are validated immediately
//! at the boundary before entering the synthesis domain.

use std::{
    fmt,
    num::NonZeroU32,
    path::{Component, Path, PathBuf},
    str::FromStr,
    sync::LazyLock,
};

use config::{Config, File, FileFormat};
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize};
use typed_builder::TypedBuilder;
use validator::Validate;

use crate::error::{IndexTtsError, Result};

/// Maximum allowed byte length for synthesis input text (16 KiB).
pub const MAX_SYNTHESIS_TEXT_BYTES: usize = 16_384;

/// Maximum allowed byte length for a [`VoiceId`].
pub const MAX_VOICE_ID_BYTES: usize = 64;

/// Maximum Mel vocabulary size (`8194`).
pub const MEL_VOCAB_SIZE: u32 = 8_194;

/// Output audio sample rate in Hz (`22050`).
pub const DEFAULT_SAMPLE_RATE: u32 = 22_050;

/// Maximum reference audio duration in seconds (`15`).
pub const DEFAULT_MAX_REF_SECONDS: u32 = 15;

/// Default GPT text positional embedding capacity (`602`).
pub const DEFAULT_TEXT_POS_CAPACITY: usize = 602;

/// Default maximum text tokens per segment (`120`).
pub const DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT: usize = 120;

/// Default silence inserted between synthesized segments in milliseconds (`200`).
pub const DEFAULT_INTERVAL_SILENCE_MS: u32 = 200;

/// Default Hugging Face repository ID for IndexTTS-2.5-mlx weights.
pub const DEFAULT_MODEL_REPO: &str = "yunfengwang/IndexTTS-2.5-mlx";

static VOICE_ID_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z0-9_-]{1,64}$").unwrap_or_else(|_| unreachable!()));

/// Bounded, non-empty synthesis text without NUL bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SynthesisText(String);

impl SynthesisText {
    /// Validates and constructs a [`SynthesisText`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if the string is empty, whitespace-only,
    /// contains NUL bytes, or exceeds [`MAX_SYNTHESIS_TEXT_BYTES`].
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::SynthesisText;
    ///
    /// let text = SynthesisText::new("大家好，欢迎使用 IndexTTS-2.5！")?;
    /// assert_eq!(text.as_str(), "大家好，欢迎使用 IndexTTS-2.5！");
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(raw: impl Into<String>) -> Result<Self> {
        let s = raw.into();
        if s.len() > MAX_SYNTHESIS_TEXT_BYTES {
            return Err(IndexTtsError::validation(
                "text",
                format!(
                    "byte length {} exceeds maximum allowed {} bytes",
                    s.len(),
                    MAX_SYNTHESIS_TEXT_BYTES
                ),
            ));
        }
        if s.as_bytes().contains(&0) {
            return Err(IndexTtsError::validation(
                "text",
                "text must not contain NUL bytes",
            ));
        }
        if s.trim().is_empty() {
            return Err(IndexTtsError::validation(
                "text",
                "text must not be empty or whitespace-only",
            ));
        }
        Ok(Self(s))
    }

    /// Returns the validated text slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::SynthesisText;
    ///
    /// let text = SynthesisText::new("Hello world")?;
    /// assert_eq!(text.as_str(), "Hello world");
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for SynthesisText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SynthesisText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for SynthesisText {
    type Error = IndexTtsError;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<&str> for SynthesisText {
    type Error = IndexTtsError;

    fn try_from(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl FromStr for SynthesisText {
    type Err = IndexTtsError;

    fn from_str(s: &str) -> Result<Self> {
        Self::new(s)
    }
}

impl<'de> Deserialize<'de> for SynthesisText {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

/// Validated voice identifier matching `^[a-zA-Z0-9_-]{1,64}$`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct VoiceId(String);

impl VoiceId {
    /// Validates and constructs a [`VoiceId`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if the identifier does not match
    /// `^[a-zA-Z0-9_-]{1,64}$`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::VoiceId;
    ///
    /// let id = VoiceId::new("narrator_zh_01")?;
    /// assert_eq!(id.as_str(), "narrator_zh_01");
    /// assert!(VoiceId::new("../etc/passwd").is_err());
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(raw: impl Into<String>) -> Result<Self> {
        let s = raw.into();
        if s.is_empty() || s.len() > MAX_VOICE_ID_BYTES || !VOICE_ID_REGEX.is_match(&s) {
            return Err(IndexTtsError::validation(
                "voice_id",
                "must match ^[a-zA-Z0-9_-]{1,64}$ without path separators or control chars",
            ));
        }
        Ok(Self(s))
    }

    /// Returns the voice identifier as a string slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::VoiceId;
    ///
    /// let id = VoiceId::new("default_voice")?;
    /// assert_eq!(id.as_str(), "default_voice");
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for VoiceId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for VoiceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for VoiceId {
    type Error = IndexTtsError;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<&str> for VoiceId {
    type Error = IndexTtsError;

    fn try_from(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl FromStr for VoiceId {
    type Err = IndexTtsError;

    fn from_str(s: &str) -> Result<Self> {
        Self::new(s)
    }
}

impl<'de> Deserialize<'de> for VoiceId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

/// Supported synthesis languages and their `LANGUAGE_DICT` token IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Language {
    /// Chinese (`zh`, token ID `1`).
    #[default]
    Zh,
    /// English (`en`, token ID `0`).
    En,
    /// Spanish (`es`, token ID `3`).
    Es,
    /// Japanese (`ja`, token ID `7`).
    Ja,
    /// Arabic (`ar`, token ID `13`).
    Ar,
    /// Cantonese (`yue`, token ID `87`).
    Yue,
    /// Mixed Chinese/English (`zh/en` or `zhen`, token ID `91`).
    #[serde(rename = "zh/en", alias = "zhen")]
    ZhEn,
    /// Fallback multilingual (`common`, token ID `93`).
    Common,
}

impl Language {
    /// Returns the `LANGUAGE_DICT` integer index used by `Gpt::lang_embedding`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::Language;
    ///
    /// assert_eq!(Language::En.token_id(), 0);
    /// assert_eq!(Language::Zh.token_id(), 1);
    /// assert_eq!(Language::Ja.token_id(), 7);
    /// assert_eq!(Language::Yue.token_id(), 87);
    /// assert_eq!(Language::ZhEn.token_id(), 91);
    /// ```
    #[must_use]
    pub const fn token_id(self) -> i32 {
        match self {
            Self::En => 0,
            Self::Zh => 1,
            Self::Es => 3,
            Self::Ja => 7,
            Self::Ar => 13,
            Self::Yue => 87,
            Self::ZhEn => 91,
            Self::Common => 93,
        }
    }

    /// Returns the canonical lowercase code used to form `<|lang|> ` prefixes.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::Language;
    ///
    /// assert_eq!(Language::ZhEn.as_code(), "zh/en");
    /// assert_eq!(Language::Zh.as_code(), "zh");
    /// ```
    #[must_use]
    pub const fn as_code(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Zh => "zh",
            Self::Es => "es",
            Self::Ja => "ja",
            Self::Ar => "ar",
            Self::Yue => "yue",
            Self::ZhEn => "zh/en",
            Self::Common => "common",
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_code())
    }
}

impl FromStr for Language {
    type Err = IndexTtsError;

    fn from_str(s: &str) -> Result<Self> {
        if s.len() > 32 {
            return Err(IndexTtsError::validation(
                "language",
                "language code exceeds 32 bytes",
            ));
        }
        match s.trim().to_ascii_lowercase().as_str() {
            "en" | "english" => Ok(Self::En),
            "zh" | "chinese" => Ok(Self::Zh),
            "es" | "spanish" => Ok(Self::Es),
            "ja" | "japanese" => Ok(Self::Ja),
            "ar" | "arabic" => Ok(Self::Ar),
            "yue" | "cantonese" => Ok(Self::Yue),
            "zh/en" | "zhen" | "en/zh" => Ok(Self::ZhEn),
            "common" => Ok(Self::Common),
            other => Err(IndexTtsError::validation(
                "language",
                format!("unsupported language `{other}`"),
            )),
        }
    }
}

impl TryFrom<&str> for Language {
    type Error = IndexTtsError;

    fn try_from(value: &str) -> Result<Self> {
        Self::from_str(value)
    }
}

/// Validated sampling temperature in `0.05..=2.0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Temperature(f32);

impl Temperature {
    /// Validates and constructs a [`Temperature`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `value` is not finite or outside `0.05..=2.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::Temperature;
    ///
    /// let temp = Temperature::new(0.8)?;
    /// assert!((temp.get() - 0.8).abs() < f32::EPSILON);
    /// assert!(Temperature::new(0.0).is_err());
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(value: f32) -> Result<Self> {
        if !value.is_finite() || !(0.05..=2.0).contains(&value) {
            return Err(IndexTtsError::validation(
                "temperature",
                format!("must be finite and in 0.05..=2.0, got {value}"),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the inner `f32` temperature value.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for Temperature {
    fn default() -> Self {
        Self(0.8)
    }
}

impl TryFrom<f32> for Temperature {
    type Error = IndexTtsError;

    fn try_from(value: f32) -> Result<Self> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for Temperature {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = f32::deserialize(deserializer)?;
        Self::new(v).map_err(serde::de::Error::custom)
    }
}

/// Validated nucleus sampling Top-P probability in `0.01..=1.0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TopP(f32);

impl TopP {
    /// Validates and constructs a [`TopP`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `value` is not finite or outside `0.01..=1.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::TopP;
    ///
    /// let p = TopP::new(0.8)?;
    /// assert!((p.get() - 0.8).abs() < f32::EPSILON);
    /// assert!(TopP::new(1.5).is_err());
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(value: f32) -> Result<Self> {
        if !value.is_finite() || !(0.01..=1.0).contains(&value) {
            return Err(IndexTtsError::validation(
                "top_p",
                format!("must be finite and in 0.01..=1.0, got {value}"),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the inner `f32` value.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for TopP {
    fn default() -> Self {
        Self(0.8)
    }
}

impl TryFrom<f32> for TopP {
    type Error = IndexTtsError;

    fn try_from(value: f32) -> Result<Self> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for TopP {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = f32::deserialize(deserializer)?;
        Self::new(v).map_err(serde::de::Error::custom)
    }
}

/// Validated Top-K candidate count in `0..=8194` (`0` disables Top-K filtering).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TopK(u32);

impl TopK {
    /// Validates and constructs a [`TopK`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `value > MEL_VOCAB_SIZE`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::TopK;
    ///
    /// let k = TopK::new(30)?;
    /// assert_eq!(k.get(), 30);
    /// assert!(TopK::new(9000).is_err());
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(value: u32) -> Result<Self> {
        if value > MEL_VOCAB_SIZE {
            return Err(IndexTtsError::validation(
                "top_k",
                format!("must be <= {MEL_VOCAB_SIZE}, got {value}"),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the inner `u32` candidate count.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Default for TopK {
    fn default() -> Self {
        Self(30)
    }
}

impl TryFrom<u32> for TopK {
    type Error = IndexTtsError;

    fn try_from(value: u32) -> Result<Self> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for TopK {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = u32::deserialize(deserializer)?;
        Self::new(v).map_err(serde::de::Error::custom)
    }
}

/// Validated repetition penalty in `1.0..=50.0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RepetitionPenalty(f32);

impl RepetitionPenalty {
    /// Validates and constructs a [`RepetitionPenalty`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `value` is not finite or outside `1.0..=50.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::RepetitionPenalty;
    ///
    /// let p = RepetitionPenalty::new(10.0)?;
    /// assert!((p.get() - 10.0).abs() < f32::EPSILON);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(value: f32) -> Result<Self> {
        if !value.is_finite() || !(1.0..=50.0).contains(&value) {
            return Err(IndexTtsError::validation(
                "repetition_penalty",
                format!("must be finite and in 1.0..=50.0, got {value}"),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the inner `f32` value.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for RepetitionPenalty {
    fn default() -> Self {
        Self(10.0)
    }
}

impl TryFrom<f32> for RepetitionPenalty {
    type Error = IndexTtsError;

    fn try_from(value: f32) -> Result<Self> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for RepetitionPenalty {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = f32::deserialize(deserializer)?;
        Self::new(v).map_err(serde::de::Error::custom)
    }
}

/// Validated duration factor (speaking rate multiplier) in `0.25..=4.0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DurationFactor(f32);

impl DurationFactor {
    /// Validates and constructs a [`DurationFactor`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `value` is not finite or outside `0.25..=4.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::DurationFactor;
    ///
    /// let df = DurationFactor::new(1.0)?;
    /// assert!((df.get() - 1.0).abs() < f32::EPSILON);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(value: f32) -> Result<Self> {
        if !value.is_finite() || !(0.25..=4.0).contains(&value) {
            return Err(IndexTtsError::validation(
                "duration_factor",
                format!("must be finite and in 0.25..=4.0, got {value}"),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the inner `f32` value.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for DurationFactor {
    fn default() -> Self {
        Self(1.0)
    }
}

impl TryFrom<f32> for DurationFactor {
    type Error = IndexTtsError;

    fn try_from(value: f32) -> Result<Self> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for DurationFactor {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = f32::deserialize(deserializer)?;
        Self::new(v).map_err(serde::de::Error::custom)
    }
}

/// Validated classifier-free guidance rate in `0.0..=2.0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CfgRate(f32);

impl CfgRate {
    /// Validates and constructs a [`CfgRate`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `value` is not finite or outside `0.0..=2.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::CfgRate;
    ///
    /// let cfg = CfgRate::new(0.7)?;
    /// assert!((cfg.get() - 0.7).abs() < f32::EPSILON);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(value: f32) -> Result<Self> {
        if !value.is_finite() || !(0.0..=2.0).contains(&value) {
            return Err(IndexTtsError::validation(
                "cfg_rate",
                format!("must be finite and in 0.0..=2.0, got {value}"),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the inner `f32` value.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for CfgRate {
    fn default() -> Self {
        Self(0.7)
    }
}

impl TryFrom<f32> for CfgRate {
    type Error = IndexTtsError;

    fn try_from(value: f32) -> Result<Self> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for CfgRate {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = f32::deserialize(deserializer)?;
        Self::new(v).map_err(serde::de::Error::custom)
    }
}

/// Validated emotion blending alpha weight in `0.0..=2.0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EmoAlpha(f32);

impl EmoAlpha {
    /// Validates and constructs an [`EmoAlpha`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `value` is not finite or outside `0.0..=2.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EmoAlpha;
    ///
    /// let alpha = EmoAlpha::new(1.0)?;
    /// assert!((alpha.get() - 1.0).abs() < f32::EPSILON);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(value: f32) -> Result<Self> {
        if !value.is_finite() || !(0.0..=2.0).contains(&value) {
            return Err(IndexTtsError::validation(
                "emo_alpha",
                format!("must be finite and in 0.0..=2.0, got {value}"),
            ));
        }
        Ok(Self(value))
    }

    /// Returns the inner `f32` value.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for EmoAlpha {
    fn default() -> Self {
        Self(1.0)
    }
}

impl TryFrom<f32> for EmoAlpha {
    type Error = IndexTtsError;

    fn try_from(value: f32) -> Result<Self> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for EmoAlpha {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = f32::deserialize(deserializer)?;
        Self::new(v).map_err(serde::de::Error::custom)
    }
}

/// Canonical 8-class emotion names in upstream `IndexTTS-2.5` order.
pub const EMOTION_NAMES: [&str; 8] = [
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
];

/// Upstream empirical bias weights suppressing dominant emotion prototypes
/// (`indextts/infer_v2_5.py`).
pub const EMOTION_BIAS: [f32; 8] = [
    0.9375, // happy (15/16)
    0.8750, // angry (7/8)
    1.0000, // sad (1)
    1.0000, // afraid (1)
    0.9375, // disgusted (15/16)
    0.9375, // melancholic (15/16)
    0.6875, // surprised (11/16)
    0.5625, // calm (9/16)
];

/// Maximum allowed byte length for an emotion specification string.
const MAX_EMOTION_SPEC_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "camelCase", default, deny_unknown_fields)]
struct RawEmotionVector {
    happy: f32,
    angry: f32,
    sad: f32,
    afraid: f32,
    disgusted: f32,
    melancholic: f32,
    surprised: f32,
    calm: f32,
}

/// Validated 8-class emotion intensity vector (`happy`, `angry`, `sad`, `afraid`,
/// `disgusted`, `melancholic`, `surprised`, `calm`), with each weight in `0.0..=1.0`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", try_from = "RawEmotionVector")]
pub struct EmotionVector {
    /// Happy / joyful (`高兴`, index `0`).
    #[serde(default)]
    pub happy: f32,
    /// Angry (`愤怒`, index `1`).
    #[serde(default)]
    pub angry: f32,
    /// Sad (`悲伤`, index `2`).
    #[serde(default)]
    pub sad: f32,
    /// Afraid / fearful (`恐惧`, index `3`).
    #[serde(default)]
    pub afraid: f32,
    /// Disgusted (`反感`, index `4`).
    #[serde(default)]
    pub disgusted: f32,
    /// Melancholic / low / wistful (`低落`, index `5`).
    #[serde(default)]
    pub melancholic: f32,
    /// Surprised (`惊讶`, index `6`).
    #[serde(default)]
    pub surprised: f32,
    /// Calm / neutral / serene (`平静`, index `7`).
    #[serde(default)]
    pub calm: f32,
}

impl EmotionVector {
    /// Validates and constructs an [`EmotionVector`] from an 8-element array in canonical order:
    /// `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if any weight is non-finite or outside `0.0..=1.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EmotionVector;
    ///
    /// let emo = EmotionVector::new([0.0, 0.0, 0.2, 0.0, 0.0, 0.45, 0.0, 0.25])?;
    /// assert!((emo.melancholic - 0.45).abs() < f32::EPSILON);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(weights: [f32; 8]) -> Result<Self> {
        for (idx, &w) in weights.iter().enumerate() {
            if !w.is_finite() || !(0.0..=1.0).contains(&w) {
                return Err(IndexTtsError::validation(
                    "emo_vector",
                    format!(
                        "emotion `{}` weight must be finite and in 0.0..=1.0, got {w}",
                        EMOTION_NAMES[idx]
                    ),
                ));
            }
        }
        Ok(Self {
            happy: weights[0],
            angry: weights[1],
            sad: weights[2],
            afraid: weights[3],
            disgusted: weights[4],
            melancholic: weights[5],
            surprised: weights[6],
            calm: weights[7],
        })
    }

    /// Validates all 8 weights in `self`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if any component is non-finite or outside `0.0..=1.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EmotionVector;
    ///
    /// let emo = EmotionVector::default();
    /// assert!(emo.validate_weights().is_ok());
    /// ```
    pub fn validate_weights(&self) -> Result<()> {
        Self::new(self.as_array()).map(|_| ())
    }

    /// Returns the 8 emotion weights in canonical order.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EmotionVector;
    ///
    /// let emo = EmotionVector::default();
    /// assert_eq!(emo.as_array(), [0.0; 8]);
    /// ```
    #[must_use]
    pub const fn as_array(&self) -> [f32; 8] {
        [
            self.happy,
            self.angry,
            self.sad,
            self.afraid,
            self.disgusted,
            self.melancholic,
            self.surprised,
            self.calm,
        ]
    }

    /// Returns `true` if all 8 emotion weights are effectively zero (`<= 1e-4`).
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EmotionVector;
    ///
    /// assert!(EmotionVector::default().is_zero());
    /// ```
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.as_array().iter().all(|&w| w <= 1e-4)
    }

    /// Normalizes the emotion vector following upstream `IndexTTS-2.5` `normalize_emo_vec`:
    /// 1. Multiplies each component by [`EMOTION_BIAS`] when `apply_bias` is `true`.
    /// 2. Caps the total sum to `0.8` so the speaker identity is preserved.
    /// 3. Scales by `emo_alpha` clamped to `0.0..=1.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::{EmoAlpha, EmotionVector};
    ///
    /// let emo: EmotionVector = "melancholic:0.6,calm:0.6".parse()?;
    /// let norm = emo.normalize(true, EmoAlpha::default());
    /// let sum: f32 = norm.iter().sum();
    /// assert!((sum - 0.8).abs() < 1e-5);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[must_use]
    pub fn normalize(&self, apply_bias: bool, emo_alpha: EmoAlpha) -> [f32; 8] {
        let raw = self.as_array();
        let mut biased = [0.0_f32; 8];
        for i in 0..8 {
            let w = raw[i].max(0.0);
            biased[i] = if apply_bias { w * EMOTION_BIAS[i] } else { w };
        }
        let total: f32 = biased.iter().sum();
        if total > 0.8 {
            let scale = 0.8 / total;
            for item in &mut biased {
                *item *= scale;
            }
        }
        let alpha = emo_alpha.get().clamp(0.0, 1.0);
        if (alpha - 1.0).abs() > f32::EPSILON {
            for item in &mut biased {
                *item *= alpha;
            }
        }
        biased
    }

    /// Parses an emotion specification string such as `"melancholic:0.45,calm:0.25"`,
    /// `"低落:0.45,平静:0.25"`, `"calm"` (default weight `0.6`), or `"none"` (returns `Ok(None)`).
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if the string is malformed, references an unknown
    /// emotion category, or specifies a weight outside `0.0..=1.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EmotionVector;
    ///
    /// let parsed = EmotionVector::parse_spec("低落:0.45,平静:0.25")?.unwrap();
    /// assert!((parsed.melancholic - 0.45).abs() < 1e-6);
    /// assert!((parsed.calm - 0.25).abs() < 1e-6);
    /// assert!(EmotionVector::parse_spec("none")?.is_none());
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn parse_spec(spec: &str) -> Result<Option<Self>> {
        if spec.len() > MAX_EMOTION_SPEC_BYTES {
            return Err(IndexTtsError::validation(
                "emo_vector",
                "emotion specification exceeds 256 bytes",
            ));
        }
        let trimmed = spec.trim();
        if trimmed.is_empty() {
            return Err(IndexTtsError::validation(
                "emo_vector",
                "emotion specification must not be empty",
            ));
        }
        if matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "none" | "default" | "reset" | "无" | "默认"
        ) {
            return Ok(None);
        }

        let tokens: Vec<&str> = trimmed
            .split([',', '，', '、', ';', '；'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if tokens.is_empty() {
            return Err(IndexTtsError::validation(
                "emo_vector",
                "emotion specification must contain at least one emotion",
            ));
        }

        if tokens.len() == 8 && tokens.iter().all(|t| !t.contains([':', '：', '='])) {
            let mut weights = [0.0_f32; 8];
            for (idx, token) in tokens.iter().enumerate() {
                let value = token.parse::<f32>().map_err(|_| {
                    IndexTtsError::validation(
                        "emo_vector",
                        format!("invalid numeric weight `{token}` at index {idx}"),
                    )
                })?;
                weights[idx] = value;
            }
            let vec = Self::new(weights)?;
            return Ok((!vec.is_zero()).then_some(vec));
        }

        let mut weights = [0.0_f32; 8];
        for token in tokens {
            let (name, value) = if let Some((k, v)) = token.split_once([':', '：', '=']) {
                let parsed = v.trim().parse::<f32>().map_err(|_| {
                    IndexTtsError::validation(
                        "emo_vector",
                        format!("invalid weight `{}` for emotion `{}`", v.trim(), k.trim()),
                    )
                })?;
                (k.trim(), parsed)
            } else {
                (token, 0.6_f32)
            };
            let idx = resolve_emotion_index(name)?;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(IndexTtsError::validation(
                    "emo_vector",
                    format!(
                        "emotion `{}` weight must be finite and in 0.0..=1.0, got {value}",
                        EMOTION_NAMES[idx]
                    ),
                ));
            }
            weights[idx] = value;
        }

        let vec = Self::new(weights)?;
        Ok((!vec.is_zero()).then_some(vec))
    }

    /// Formats the emotion vector into a compact `name:weight` string (e.g.
    /// `"melancholic:0.45,calm:0.25"`), or `"none"` when all weights are zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EmotionVector;
    ///
    /// let emo: EmotionVector = "melancholic:0.45,calm:0.25".parse()?;
    /// assert_eq!(emo.format_compact(), "melancholic:0.45,calm:0.25");
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[must_use]
    pub fn format_compact(&self) -> String {
        let mut parts = Vec::new();
        for (idx, &w) in self.as_array().iter().enumerate() {
            if w > 0.005 {
                let text = format!("{w:.2}");
                let clean = text.trim_end_matches('0').trim_end_matches('.');
                parts.push(format!("{}:{clean}", EMOTION_NAMES[idx]));
            }
        }
        if parts.is_empty() {
            "none".to_owned()
        } else {
            parts.join(",")
        }
    }
}

fn resolve_emotion_index(name: &str) -> Result<usize> {
    match name.trim().to_ascii_lowercase().as_str() {
        "happy" | "joy" | "高兴" | "快乐" | "喜悦" | "欢快" => Ok(0),
        "angry" | "anger" | "愤怒" | "生气" | "恼怒" | "激愤" => Ok(1),
        "sad" | "sadness" | "悲伤" | "伤心" | "哀伤" | "悲痛" => Ok(2),
        "afraid" | "fear" | "scared" | "恐惧" | "害怕" | "惊恐" | "畏惧" => Ok(3),
        "disgusted" | "disgust" | "反感" | "厌恶" | "嫌弃" => Ok(4),
        "melancholic" | "melancholy" | "depressed" | "low" | "wistful" | "低落" | "忧郁"
        | "惆怅" | "感伤" | "眷恋" | "伤感" => Ok(5),
        "surprised" | "surprise" | "惊讶" | "惊奇" | "诧异" | "惊叹" => Ok(6),
        "calm" | "neutral" | "peaceful" | "serene" | "gentle" | "平静" | "自然" | "温和"
        | "宁静" | "轻柔" | "舒缓" => Ok(7),
        other => Err(IndexTtsError::validation(
            "emo_vector",
            format!(
                "unknown emotion `{other}` (expected happy/高兴, angry/愤怒, sad/悲伤, \
                 afraid/恐惧, disgusted/反感, melancholic/低落, surprised/惊讶, calm/平静)"
            ),
        )),
    }
}

impl fmt::Display for EmotionVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_compact())
    }
}

impl FromStr for EmotionVector {
    type Err = IndexTtsError;

    fn from_str(s: &str) -> Result<Self> {
        Ok(Self::parse_spec(s)?.unwrap_or_default())
    }
}

impl TryFrom<&str> for EmotionVector {
    type Error = IndexTtsError;

    fn try_from(value: &str) -> Result<Self> {
        Self::from_str(value)
    }
}

impl TryFrom<RawEmotionVector> for EmotionVector {
    type Error = IndexTtsError;

    fn try_from(raw: RawEmotionVector) -> Result<Self> {
        Self::new([
            raw.happy,
            raw.angry,
            raw.sad,
            raw.afraid,
            raw.disgusted,
            raw.melancholic,
            raw.surprised,
            raw.calm,
        ])
    }
}

/// Autoregressive GPT sampling parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder, Validate)]
#[serde(rename_all = "snake_case", default)]
#[non_exhaustive]
pub struct SamplingParams {
    /// Whether to use deterministic greedy (`argmax`) decoding.
    #[builder(default = false)]
    pub greedy: bool,
    /// Optional RNG seed for reproducible stochastic sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub seed: Option<u64>,
    /// Top-K candidate truncation (`0..=8194`, default `30`).
    #[builder(default)]
    pub top_k: TopK,
    /// Nucleus sampling threshold (`0.01..=1.0`, default `0.8`).
    #[builder(default)]
    pub top_p: TopP,
    /// Logits temperature (`0.05..=2.0`, default `0.8`).
    #[builder(default)]
    pub temperature: Temperature,
    /// Repetition penalty (`1.0..=50.0`, default `10.0`).
    #[builder(default)]
    pub repetition_penalty: RepetitionPenalty,
    /// Maximum generated Mel tokens per segment (`1..=1815`, default `1500`).
    #[validate(range(min = 1, max = 1815))]
    #[builder(default = 1500)]
    pub max_mel_tokens: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            greedy: false,
            seed: None,
            top_k: TopK::default(),
            top_p: TopP::default(),
            temperature: Temperature::default(),
            repetition_penalty: RepetitionPenalty::default(),
            max_mel_tokens: 1500,
        }
    }
}

/// Conditional Flow Matching (CFM) Euler ODE solver parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder, Validate)]
#[serde(rename_all = "snake_case", default)]
#[non_exhaustive]
pub struct CfmParams {
    /// Speaking rate duration multiplier (`0.25..=4.0`, default `1.0`).
    #[builder(default)]
    pub duration_factor: DurationFactor,
    /// Number of Euler ODE integration steps (`1..=100`, default `25`).
    #[validate(range(min = 1, max = 100))]
    #[builder(default = 25)]
    pub n_timesteps: u32,
    /// Classifier-free guidance rate (`0.0..=2.0`, default `0.7`).
    #[builder(default)]
    pub cfg_rate: CfgRate,
}

impl Default for CfmParams {
    fn default() -> Self {
        Self {
            duration_factor: DurationFactor::default(),
            n_timesteps: 25,
            cfg_rate: CfgRate::default(),
        }
    }
}

/// Top-level runtime configuration for `IndexTtsEngine`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder, Validate)]
#[serde(rename_all = "snake_case", default)]
#[non_exhaustive]
pub struct EngineConfig {
    /// Explicit local model directory override. When `None`, resolves from local HF cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub model_dir: Option<PathBuf>,
    /// Hugging Face repository identifier (`yunfengwang/IndexTTS-2.5-mlx`).
    #[validate(length(min = 1, max = 256))]
    #[builder(default = DEFAULT_MODEL_REPO.to_owned())]
    pub model_repo: String,
    /// Optional glossary YAML path override.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub glossary_path: Option<PathBuf>,
    /// Optional directory for `.itts_voice` snapshots.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub voices_dir: Option<PathBuf>,
    /// Default synthesis language.
    #[builder(default)]
    pub default_language: Language,
    /// Maximum text positional capacity (`10..=602`, default `602`).
    #[validate(range(min = 10, max = 602))]
    #[builder(default = DEFAULT_TEXT_POS_CAPACITY)]
    pub text_pos_capacity: usize,
    /// Default token budget per sentence segment (`1..=600`, default `120`).
    #[validate(range(min = 1, max = 600))]
    #[builder(default = DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT)]
    pub max_text_tokens_per_segment: usize,
    /// Whether to run text normalization (numbers, tech terms, contractions).
    #[builder(default = true)]
    pub use_normalization: bool,
    /// Maximum reference audio duration in seconds (`1..=30`, default `15`).
    #[validate(range(min = 1, max = 30))]
    #[builder(default = DEFAULT_MAX_REF_SECONDS)]
    pub max_ref_seconds: u32,
    /// Output waveform sample rate (`22050` Hz).
    #[validate(range(min = 8000, max = 48000))]
    #[builder(default = DEFAULT_SAMPLE_RATE)]
    pub sample_rate: u32,
    /// Silence inserted between segments in milliseconds (`0..=5000`, default `200`).
    #[validate(range(min = 0, max = 5000))]
    #[builder(default = DEFAULT_INTERVAL_SILENCE_MS)]
    pub interval_silence_ms: u32,
    /// Default GPT sampling parameters.
    #[validate(nested)]
    #[builder(default)]
    pub sampling: SamplingParams,
    /// Default CFM solver parameters.
    #[validate(nested)]
    #[builder(default)]
    pub cfm: CfmParams,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            model_repo: DEFAULT_MODEL_REPO.to_owned(),
            glossary_path: None,
            voices_dir: None,
            default_language: Language::Zh,
            text_pos_capacity: DEFAULT_TEXT_POS_CAPACITY,
            max_text_tokens_per_segment: DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT,
            use_normalization: true,
            max_ref_seconds: DEFAULT_MAX_REF_SECONDS,
            sample_rate: DEFAULT_SAMPLE_RATE,
            interval_silence_ms: DEFAULT_INTERVAL_SILENCE_MS,
            sampling: SamplingParams::default(),
            cfm: CfmParams::default(),
        }
    }
}

impl EngineConfig {
    /// Validates all fields and nested structures of [`EngineConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if any range, path, or invariant check fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EngineConfig;
    ///
    /// let cfg = EngineConfig::default();
    /// cfg.validate_config()?;
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn validate_config(&self) -> Result<()> {
        self.validate().map_err(|err| {
            IndexTtsError::validation("engine_config", format!("invalid configuration: {err}"))
        })?;
        if self.max_text_tokens_per_segment.saturating_add(2) > self.text_pos_capacity {
            return Err(IndexTtsError::validation(
                "max_text_tokens_per_segment",
                "must be <= text_pos_capacity - 2",
            ));
        }
        if let Some(dir) = &self.model_dir {
            validate_safe_file_path(dir, "model_dir")?;
        }
        if let Some(glossary) = &self.glossary_path {
            validate_safe_file_path(glossary, "glossary_path")?;
        }
        if let Some(voices) = &self.voices_dir {
            validate_safe_file_path(voices, "voices_dir")?;
        }
        Ok(())
    }

    /// Parses and validates an [`EngineConfig`] from a YAML string.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Config`] on YAML syntax/type errors or
    /// [`IndexTtsError::Validation`] if any parameter is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::{EngineConfig, Language};
    ///
    /// let yaml = r#"
    /// default_language: "en"
    /// max_text_tokens_per_segment: 100
    /// sampling:
    ///   greedy: true
    ///   temperature: 0.7
    /// "#;
    /// let cfg = EngineConfig::from_yaml_str(yaml)?;
    /// assert_eq!(cfg.default_language, Language::En);
    /// assert!(cfg.sampling.greedy);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_yaml_str(yaml_content: &str) -> Result<Self> {
        if yaml_content.len() > 65_536 {
            return Err(IndexTtsError::validation(
                "config_yaml",
                "YAML configuration exceeds 64 KiB limit",
            ));
        }
        let raw = Config::builder()
            .add_source(File::from_str(yaml_content, FileFormat::Yaml))
            .build()
            .map_err(|err| IndexTtsError::Config {
                message: format!("failed to parse YAML configuration: {err}"),
                source: Some(err),
            })?;
        let cfg: Self = raw.try_deserialize().map_err(|err| IndexTtsError::Config {
            message: format!("failed to deserialize configuration: {err}"),
            source: Some(err),
        })?;
        cfg.validate_config()?;
        Ok(cfg)
    }

    /// Loads and validates an [`EngineConfig`] from a YAML file path on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if the path is unsafe, [`IndexTtsError::Config`]
    /// if loading fails, or [`IndexTtsError::Validation`] if values are out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    /// use index_tts_rs_core::config::EngineConfig;
    ///
    /// let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/default.yaml");
    /// let cfg = EngineConfig::from_yaml_file(&path)?;
    /// assert_eq!(cfg.sample_rate, 22050);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        validate_safe_file_path(path_ref, "config_path")?;
        let raw = Config::builder()
            .add_source(File::from(path_ref).format(FileFormat::Yaml))
            .build()
            .map_err(|err| IndexTtsError::Config {
                message: format!(
                    "failed to load YAML configuration from `{}`: {err}",
                    path_ref.display()
                ),
                source: Some(err),
            })?;
        let cfg: Self = raw.try_deserialize().map_err(|err| IndexTtsError::Config {
            message: format!("failed to deserialize configuration: {err}"),
            source: Some(err),
        })?;
        cfg.validate_config()?;
        Ok(cfg)
    }
}

/// Speaker voice source for a [`SynthesisRequest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SpeakerSource {
    /// Pre-registered voice snapshot identifier (`<voice_id>.itts_voice`).
    VoiceId(VoiceId),
    /// Path to an `.itts_voice` snapshot file on disk.
    VoiceFile(PathBuf),
    /// Path to a raw reference audio file (`.wav`, `.mp3`, `.flac`, `.m4a`).
    ReferenceAudio(PathBuf),
}

/// Validated synthesis request parameters built via [`TypedBuilder`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct SynthesisRequest {
    /// Validated synthesis text.
    pub text: SynthesisText,
    /// Speaker conditioning source.
    pub speaker: SpeakerSource,
    /// Target synthesis language.
    #[builder(default)]
    pub language: Language,
    /// Autoregressive GPT sampling parameters.
    #[builder(default)]
    pub sampling: SamplingParams,
    /// CFM ODE solver parameters.
    #[builder(default)]
    pub cfm: CfmParams,
    /// Maximum text tokens per segment (`1..=600`).
    #[builder(default = NonZeroU32::new(120).unwrap_or(NonZeroU32::MIN))]
    pub max_text_tokens_per_segment: NonZeroU32,
    /// Inter-sentence silence in milliseconds.
    #[builder(default = DEFAULT_INTERVAL_SILENCE_MS)]
    pub interval_silence_ms: u32,
    /// Optional separate emotion reference audio file when `speaker` is `ReferenceAudio`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub emo_audio_path: Option<PathBuf>,
    /// Emotion blending multiplier (`0.0..=2.0`, default `1.0`) when `speaker` is
    /// `ReferenceAudio` or when `emo_vector` is provided.
    #[builder(default)]
    pub emo_alpha: EmoAlpha,
    /// Optional 8-class emotion intensity vector override (`happy`, `angry`, `sad`, `afraid`,
    /// `disgusted`, `melancholic`, `surprised`, `calm`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub emo_vector: Option<EmotionVector>,
}

/// Request to extract a [`crate::voice::SpeakerContext`] from reference audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct VoiceExtractRequest {
    /// Target voice identifier.
    pub voice_id: VoiceId,
    /// Path to the speaker reference audio file (3s..15s).
    pub ref_audio_path: PathBuf,
    /// Optional separate emotion reference audio file.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub emo_audio_path: Option<PathBuf>,
    /// Emotion blending multiplier (`0.0..=2.0`, default `1.0`).
    #[builder(default)]
    pub emo_alpha: EmoAlpha,
    /// Optional 8-class emotion intensity vector to bake into the voice snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub emo_vector: Option<EmotionVector>,
}

/// Validates that a file path does not contain NUL bytes, excessive length, or
/// parent-directory (`..`) traversal components unless safely resolved.
///
/// # Errors
///
/// Returns [`IndexTtsError::Validation`] if the path is empty, exceeds 4096 bytes,
/// contains NUL bytes, or contains a `..` component in a relative path without
/// existing parent resolution.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use index_tts_rs_core::config::validate_safe_file_path;
///
/// assert!(validate_safe_file_path(Path::new("voices/speaker.wav"), "ref").is_ok());
/// assert!(validate_safe_file_path(Path::new("../../../etc/passwd"), "ref").is_err());
/// ```
pub fn validate_safe_file_path(path: &Path, field: &'static str) -> Result<()> {
    let raw = path.as_os_str().to_string_lossy();
    if raw.is_empty() || raw.len() > 4096 {
        return Err(IndexTtsError::validation(
            field,
            "path must be between 1 and 4096 bytes",
        ));
    }
    if raw.as_bytes().contains(&0) {
        return Err(IndexTtsError::validation(
            field,
            "path must not contain NUL bytes",
        ));
    }
    if !path.is_absolute()
        && path
            .components()
            .any(|comp| matches!(comp, Component::ParentDir))
    {
        return Err(IndexTtsError::validation(
            field,
            "relative path must not contain `..` traversal segments",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("")]
    #[case("   \t\n  ")]
    #[case("hello\0world")]
    fn test_should_reject_empty_or_invalid_text(#[case] input: &str) {
        let err = SynthesisText::new(input);
        assert!(matches!(
            err,
            Err(IndexTtsError::Validation { field: "text", .. })
        ));
    }

    #[test]
    fn test_should_reject_oversized_text() {
        let oversized = "a".repeat(MAX_SYNTHESIS_TEXT_BYTES + 1);
        let err = SynthesisText::new(oversized);
        assert!(matches!(
            err,
            Err(IndexTtsError::Validation { field: "text", .. })
        ));
    }

    #[rstest]
    #[case("../etc/passwd")]
    #[case("foo/bar")]
    #[case("foo\\bar")]
    #[case("a\0b")]
    #[case("")]
    #[case("voice with spaces")]
    #[case("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")]
    fn test_should_reject_voice_id_with_path_traversal(#[case] bad_id: &str) {
        let err = VoiceId::new(bad_id);
        assert!(matches!(
            err,
            Err(IndexTtsError::Validation {
                field: "voice_id",
                ..
            })
        ));
    }

    #[rstest]
    #[case("default")]
    #[case("speaker_01-zh")]
    #[case("AgentVoice_2026")]
    fn test_should_accept_valid_voice_id(#[case] valid_id: &str) {
        let voice_id = VoiceId::new(valid_id);
        assert!(voice_id.is_ok());
    }

    #[rstest]
    #[case(0.0)]
    #[case(-1.0)]
    #[case(2.5)]
    #[case(f32::NAN)]
    #[case(f32::INFINITY)]
    fn test_should_reject_out_of_range_temperature(#[case] temp: f32) {
        assert!(matches!(
            Temperature::new(temp),
            Err(IndexTtsError::Validation {
                field: "temperature",
                ..
            })
        ));
    }

    #[rstest]
    #[case(0.0)]
    #[case(1.5)]
    #[case(-0.2)]
    #[case(f32::NAN)]
    fn test_should_reject_out_of_range_top_p(#[case] top_p: f32) {
        assert!(matches!(
            TopP::new(top_p),
            Err(IndexTtsError::Validation { field: "top_p", .. })
        ));
    }

    #[test]
    fn test_should_reject_out_of_range_sampling_params_in_yaml() {
        let bad_yaml = r"
sampling:
  temperature: 0.0
  top_p: 1.5
cfm:
  n_timesteps: 0
";
        assert!(EngineConfig::from_yaml_str(bad_yaml).is_err());
    }

    #[test]
    fn test_should_load_default_yaml_config() {
        let yaml = include_str!("../../../config/default.yaml");
        let cfg = EngineConfig::from_yaml_str(yaml);
        assert!(cfg.is_ok());
    }

    #[test]
    fn test_should_parse_and_normalize_emotion_vector() {
        let parsed = EmotionVector::parse_spec("低落:0.45, 平静:0.25")
            .ok()
            .flatten();
        assert!(parsed.is_some());
        let emo = parsed.unwrap_or_default();
        assert!((emo.melancholic - 0.45).abs() < 1e-6);
        assert!((emo.calm - 0.25).abs() < 1e-6);
        assert_eq!(emo.format_compact(), "melancholic:0.45,calm:0.25");

        let single = EmotionVector::parse_spec("calm")
            .ok()
            .flatten()
            .unwrap_or_default();
        assert!((single.calm - 0.6).abs() < 1e-6);

        let cleared = EmotionVector::parse_spec("none").ok().flatten();
        assert!(cleared.is_none());

        let high = EmotionVector::parse_spec("happy:0.9,sad:0.9")
            .ok()
            .flatten()
            .unwrap_or_default();
        let norm = high.normalize(true, EmoAlpha::default());
        let sum: f32 = norm.iter().sum();
        assert!((sum - 0.8).abs() < 1e-5);
    }

    #[rstest]
    #[case("unknown:0.5")]
    #[case("happy:1.5")]
    #[case("sad:-0.2")]
    #[case("")]
    fn test_should_reject_invalid_emotion_spec(#[case] spec: &str) {
        assert!(EmotionVector::parse_spec(spec).is_err());
    }

    proptest! {
        #[test]
        fn proptest_voice_id_never_panics(s in "\\PC*") {
            let _ = VoiceId::new(s);
        }

        #[test]
        fn proptest_synthesis_text_never_panics(s in "\\PC*") {
            let _ = SynthesisText::new(s);
        }

        #[test]
        fn proptest_emotion_spec_never_panics(s in "\\PC*") {
            let _ = EmotionVector::parse_spec(&s);
        }
    }
}

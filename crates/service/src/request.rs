//! [`SpeechJob`]: the single wire-level speech request accepted by every `itts` frontend.
//!
//! Fields are plain JSON primitives so the generated JSON Schema stays readable for agents;
//! [`SpeechJob::prepare`] validates everything at the boundary and compiles inline controls into
//! an executable [`PreparedJob`]. CLI flags, daemon requests, MCP tool arguments, and HTTP bodies
//! all deserialize into this one type, so they share identical semantics.

use std::{
    hash::{BuildHasher, Hasher, RandomState},
    num::NonZeroU32,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use index_tts_rs_core::{
    config::{
        CfgRate, CfmParams, DurationFactor, EmoAlpha, EmotionVector, EngineConfig, Language,
        RepetitionPenalty, SampleRate, SamplingParams, SpeakerSource, SpeechContext,
        SynthesisRequest, SynthesisText, Temperature, TopK, TopP, VoiceId, validate_safe_file_path,
    },
    voice::{VoiceStore, default_voices_dir},
};
use index_tts_rs_director::{PlannedUtterance, plan_utterances};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use crate::{
    error::{ErrorCode, ServiceError},
    transcript::Transcript,
};

/// Default silence between explicit paragraphs, in milliseconds.
pub const DEFAULT_PARAGRAPH_PAUSE_MS: u32 = 600;
/// Upper bound for paragraph pauses and inter-sentence silence, in milliseconds.
pub const MAX_SILENCE_MS: u32 = 5000;
/// Slowest accepted speaking rate multiplier.
pub const MIN_SPEED: f32 = 0.25;
/// Fastest accepted speaking rate multiplier.
pub const MAX_SPEED: f32 = 4.0;
/// Upper bound on generated Mel tokens per segment.
pub const MAX_MEL_TOKENS: u32 = 1815;
/// Upper bound on CFM Euler integration steps.
pub const MAX_TIMESTEPS: u32 = 100;
/// Maximum number of registered voices listed in a hint.
const MAX_HINT_VOICES: usize = 12;

/// A speech synthesis request. Exactly one of `voice`, `voiceFile`, or `refAudio` selects the
/// speaker; when all three are omitted the configured `default_voice` is used.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema, TypedBuilder)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[builder(field_defaults(default, setter(strip_option)))]
#[non_exhaustive]
pub struct SpeechJob {
    /// Text to speak (at most 16 KiB UTF-8). Blank lines separate paragraphs. Inline controls
    /// such as `[[pause=600]]`, `[[speed=0.9]]`, `[[emo=calm:0.6]]`, `[[temp=0.7]]`,
    /// `[[voice=ID]]`, `[[lang=en]]`, and `[[reset]]` are executed, never spoken.
    #[builder(!default, setter(!strip_option, into))]
    pub input: String,
    /// Registered voice ID (`^[a-zA-Z0-9_-]{1,64}$`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub voice: Option<String>,
    /// Path to an `.itts_voice` snapshot file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_file: Option<PathBuf>,
    /// Path to raw reference audio (3-15 s `.wav`/`.mp3`/`.flac`/`.m4a`) for zero-shot cloning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_audio: Option<PathBuf>,
    /// Optional separate emotion reference audio; requires `refAudio`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emo_ref: Option<PathBuf>,
    /// Emotion blending weight (`0.0..=2.0`, default `1.0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emo_alpha: Option<f32>,
    /// 8-class emotion mix, e.g. `melancholic:0.45,calm:0.25` or `低落:0.45,平静:0.25`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub emo: Option<String>,
    /// Language code: `zh`, `en`, `ja`, `yue`, `es`, `ar`, `zh/en` (default from config).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub lang: Option<String>,
    /// Speaking rate multiplier (`0.25..=4.0`, default from config: `1.2`; larger is faster).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Silence between blank-line separated paragraphs (`0..=5000` ms, default `600`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paragraph_pause_ms: Option<u32>,
    /// Silence between sentences within a paragraph (`0..=5000` ms, default from config).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval_silence_ms: Option<u32>,
    /// Write the result to this `.wav` path (the parent directory must exist).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_path: Option<PathBuf>,
    /// Output sample rate: `22050` (native) or `48000` (bandwidth-extended; needs the extender
    /// weights from `itts models pull`). Default from config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    /// Continue each sentence after the previous one instead of starting every sentence as a
    /// fresh utterance: intonation, pace, and pauses carry across sentence and paragraph
    /// boundaries (same voice and language). Inter-sentence silence defaults to `0` because
    /// the model speaks its own pauses.
    #[serde(default, skip_serializing_if = "is_false")]
    #[builder(setter(!strip_option))]
    pub continuity: bool,
    /// Text spoken just before `input` (for example the previous script line synthesized in
    /// another job), at most 16 KiB without controls. The first sentence continues after it;
    /// only its last sentence is used and it is not spoken.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub previous_text: Option<String>,
    /// Return per-character timings (`alignment`) and sentence timings (`sentences`).
    #[serde(default, skip_serializing_if = "is_false")]
    #[builder(setter(!strip_option))]
    pub timestamps: bool,
    /// Low-level sampling and solver overrides; omitted fields use the configured defaults.
    #[serde(default, skip_serializing_if = "AdvancedOptions::is_empty")]
    #[builder(setter(!strip_option))]
    pub advanced: AdvancedOptions,
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if signature"
)]
const fn is_false(value: &bool) -> bool {
    !*value
}

/// Low-level GPT sampling and CFM solver overrides.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema, TypedBuilder)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[builder(field_defaults(default, setter(strip_option)))]
#[non_exhaustive]
pub struct AdvancedOptions {
    /// Deterministic greedy (`argmax`) decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub greedy: Option<bool>,
    /// RNG seed for reproducible output. When omitted a random seed is drawn and reported in
    /// the job summary, so any result can be reproduced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Sampling temperature (`0.05..=2.0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-K cutoff (`0..=8194`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Nucleus threshold (`0.01..=1.0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Repetition penalty (`1.0..=50.0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// Maximum Mel tokens per segment (`1..=1815`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_mel_tokens: Option<u32>,
    /// Maximum text tokens per segment (`1..=600`, bounded by the configured capacity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_text_tokens: Option<u32>,
    /// CFM Euler integration steps (`1..=100`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_timesteps: Option<u32>,
    /// Classifier-free guidance rate (`0.0..=2.0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_rate: Option<f32>,
}

impl AdvancedOptions {
    /// Whether no override is set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Which filesystem references a transport may carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PathPolicy {
    /// Same-host transports (CLI, daemon, MCP): paths are allowed but must already be absolute,
    /// so they never depend on the receiving process's working directory.
    LocalAbsolute,
    /// Network transports (HTTP): file references are rejected; only registered voices work.
    Denied,
}

/// A validated job compiled into executable utterances.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreparedJob {
    /// Utterances to synthesize in order, with their leading pauses.
    pub plan: Vec<PlannedUtterance>,
    /// Where each utterance came from in the script, for projecting timings.
    pub transcript: Arc<Transcript>,
    /// Seed driving GPT sampling and CFM noise for every utterance.
    pub seed: u64,
    /// Whether character timings were requested.
    pub timestamps: bool,
    /// Output sample rate of every chunk.
    pub sample_rate: SampleRate,
    /// Whether consecutive utterances continue after one another.
    pub continuity: bool,
    /// Where to write the assembled WAV, if requested.
    pub output_path: Option<PathBuf>,
    /// Human-readable speaker of the first utterance (voice ID or file name).
    pub speaker: String,
    /// Language of the first utterance.
    pub language: Language,
}

impl SpeechJob {
    /// Resolves relative paths against `base` (normally the caller's working directory).
    ///
    /// Input files are canonicalized so a missing file fails here, in the caller's terms,
    /// rather than inside a daemon that runs from a different directory.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::InvalidArgument`] if an input file does not exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_service::SpeechJob;
    ///
    /// let dir = tempfile::tempdir()?;
    /// let mut job = SpeechJob::builder().input("你好").voice("narrator").output_path("out.wav".into()).build();
    /// job.resolve_paths(dir.path())?;
    /// assert_eq!(job.output_path, Some(dir.path().join("out.wav")));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resolve_paths(&mut self, base: &Path) -> Result<(), ServiceError> {
        for (field, slot) in [
            ("voiceFile", &mut self.voice_file),
            ("refAudio", &mut self.ref_audio),
            ("emoRef", &mut self.emo_ref),
        ] {
            if let Some(path) = slot.as_mut() {
                *path = resolve_input_path(base, path, field)?;
            }
        }
        if let Some(path) = self.output_path.as_mut() {
            *path = base.join(&*path);
        }
        Ok(())
    }

    /// Validates every field and compiles inline controls into a [`PreparedJob`].
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::InvalidArgument`] for out-of-range values, conflicting speaker
    /// selectors, malformed controls, or paths forbidden by `policy`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EngineConfig;
    /// use index_tts_rs_service::{PathPolicy, SpeechJob};
    ///
    /// let voices = tempfile::tempdir()?;
    /// std::fs::write(voices.path().join("narrator.itts_voice"), b"")?;
    /// let mut config = EngineConfig::default();
    /// config.voices_dir = Some(voices.path().to_path_buf());
    ///
    /// let job = SpeechJob::builder().input("你好。\n\n[[speed=1.2]]再见。").voice("narrator").build();
    /// let prepared = job.prepare(&config, PathPolicy::Denied)?;
    /// assert_eq!(prepared.plan.len(), 2);
    /// assert_eq!(prepared.speaker, "narrator");
    ///
    /// let ghost = SpeechJob::builder().input("你好[[voice=ghost]]再见").voice("narrator").build();
    /// assert!(ghost.prepare(&config, PathPolicy::Denied).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn prepare(
        &self,
        config: &EngineConfig,
        policy: PathPolicy,
    ) -> Result<PreparedJob, ServiceError> {
        self.check_paths(policy)?;
        let text = SynthesisText::new(self.input.as_str())?;
        let speaker = self.speaker(config)?;
        let language = match self.lang.as_deref() {
            Some(code) => Language::from_str(code)?,
            None => config.default_language,
        };
        let emo_vector = match self.emo.as_deref() {
            Some(spec) => EmotionVector::parse_spec(spec)?,
            None => None,
        };
        let default_interval = if self.continuity {
            0
        } else {
            config.interval_silence_ms
        };
        let interval_silence_ms = bounded_silence(
            "intervalSilenceMs",
            self.interval_silence_ms.unwrap_or(default_interval),
        )?;
        let context = self.previous_context()?;
        let paragraph_pause_ms = bounded_silence(
            "paragraphPauseMs",
            self.paragraph_pause_ms
                .unwrap_or(DEFAULT_PARAGRAPH_PAUSE_MS),
        )?;

        let sample_rate = self
            .sample_rate
            .map_or(Ok(config.sample_rate), SampleRate::try_from)?;
        let mut sampling = self.advanced.sampling(config)?;
        let seed = sampling.seed.unwrap_or_else(random_seed);
        sampling.seed = Some(seed);
        let base = SynthesisRequest::builder()
            .text(text)
            .speaker(speaker)
            .language(language)
            .sampling(sampling)
            .cfm(self.cfm(config)?)
            .max_text_tokens_per_segment(self.advanced.max_text_tokens(config)?)
            .interval_silence_ms(interval_silence_ms)
            .emo_audio_path(self.emo_ref.clone())
            .emo_alpha(EmoAlpha::new(self.emo_alpha.unwrap_or(1.0))?)
            .emo_vector(emo_vector)
            .align(self.timestamps)
            .sample_rate(sample_rate)
            .continuity(self.continuity)
            .build();
        let label = speaker_label(&base.speaker);
        let mut reading = plan_utterances(&base, paragraph_pause_ms)?;
        ensure_voices_registered(config, &reading.utterances)?;
        if let Some(first) = reading.utterances.first_mut() {
            first.request.context = context;
        }
        let transcript = Arc::new(Transcript::new(self.input.as_str(), &reading));
        Ok(PreparedJob {
            plan: reading.utterances,
            transcript,
            seed,
            timestamps: self.timestamps,
            sample_rate,
            continuity: self.continuity,
            output_path: self.output_path.clone(),
            speaker: label,
            language,
        })
    }

    fn previous_context(&self) -> Result<SpeechContext, ServiceError> {
        match self.previous_text.as_deref() {
            None => Ok(SpeechContext::None),
            Some(text) if text.contains("[[") => Err(ServiceError::invalid(
                "previousText must be plain text without [[controls]]",
            )),
            Some(text) => Ok(SpeechContext::Text(SynthesisText::new(text)?)),
        }
    }

    fn check_paths(&self, policy: PathPolicy) -> Result<(), ServiceError> {
        let paths = [
            ("voiceFile", self.voice_file.as_deref()),
            ("refAudio", self.ref_audio.as_deref()),
            ("emoRef", self.emo_ref.as_deref()),
            ("outputPath", self.output_path.as_deref()),
        ];
        for (field, path) in paths {
            let Some(path) = path else { continue };
            match policy {
                PathPolicy::Denied => {
                    return Err(ServiceError::invalid(format!(
                        "{field} is not accepted over this transport; use a registered `voice`"
                    )));
                }
                PathPolicy::LocalAbsolute => {
                    validate_safe_file_path(path, "path")?;
                    if !path.is_absolute() {
                        return Err(ServiceError::invalid(format!(
                            "{field} must be an absolute path, got `{}`",
                            path.display()
                        )));
                    }
                }
            }
        }
        if self.emo_ref.is_some() && self.ref_audio.is_none() {
            return Err(ServiceError::invalid("emoRef requires refAudio"));
        }
        Ok(())
    }

    fn speaker(&self, config: &EngineConfig) -> Result<SpeakerSource, ServiceError> {
        match (&self.voice, &self.voice_file, &self.ref_audio) {
            (Some(id), None, None) => Ok(SpeakerSource::VoiceId(VoiceId::new(id.as_str())?)),
            (None, Some(path), None) => Ok(SpeakerSource::VoiceFile(path.clone())),
            (None, None, Some(path)) => Ok(SpeakerSource::ReferenceAudio(path.clone())),
            (None, None, None) => config
                .default_voice
                .clone()
                .map(SpeakerSource::VoiceId)
                .ok_or_else(|| missing_voice_error(config)),
            _ => Err(ServiceError::invalid(
                "provide only one of voice, voiceFile, or refAudio",
            )),
        }
    }

    fn cfm(&self, config: &EngineConfig) -> Result<CfmParams, ServiceError> {
        let mut cfm = config.cfm.clone();
        if let Some(speed) = self.speed {
            if !speed.is_finite() || !(MIN_SPEED..=MAX_SPEED).contains(&speed) {
                return Err(ServiceError::invalid(format!(
                    "speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}"
                )));
            }
            cfm.duration_factor = DurationFactor::new(speed.recip())?;
        }
        if let Some(steps) = self.advanced.n_timesteps {
            cfm.n_timesteps = bounded("nTimesteps", steps, 1, MAX_TIMESTEPS)?;
        }
        if let Some(rate) = self.advanced.cfg_rate {
            cfm.cfg_rate = CfgRate::new(rate)?;
        }
        Ok(cfm)
    }
}

impl AdvancedOptions {
    fn sampling(&self, config: &EngineConfig) -> Result<SamplingParams, ServiceError> {
        let mut sampling = config.sampling.clone();
        if let Some(greedy) = self.greedy {
            sampling.greedy = greedy;
        }
        if self.seed.is_some() {
            sampling.seed = self.seed;
        }
        if let Some(value) = self.temperature {
            sampling.temperature = Temperature::new(value)?;
        }
        if let Some(value) = self.top_k {
            sampling.top_k = TopK::new(value)?;
        }
        if let Some(value) = self.top_p {
            sampling.top_p = TopP::new(value)?;
        }
        if let Some(value) = self.repetition_penalty {
            sampling.repetition_penalty = RepetitionPenalty::new(value)?;
        }
        if let Some(value) = self.max_mel_tokens {
            sampling.max_mel_tokens = bounded("maxMelTokens", value, 1, MAX_MEL_TOKENS)?;
        }
        Ok(sampling)
    }

    fn max_text_tokens(&self, config: &EngineConfig) -> Result<NonZeroU32, ServiceError> {
        let capacity = u32::try_from(config.text_pos_capacity.saturating_sub(2))
            .map_err(|_| ServiceError::internal("text capacity exceeds u32"))?;
        let configured = u32::try_from(config.max_text_tokens_per_segment)
            .map_err(|_| ServiceError::internal("text token limit exceeds u32"))?;
        let value = bounded(
            "maxTextTokens",
            self.max_text_tokens.unwrap_or(configured),
            1,
            capacity,
        )?;
        NonZeroU32::new(value).ok_or_else(|| ServiceError::invalid("maxTextTokens must be >= 1"))
    }
}

/// Resolves an input file relative to `base` and canonicalizes it, so a missing file is reported
/// in the caller's terms (`field` names the flag or JSON field).
///
/// # Errors
///
/// Returns [`ErrorCode::InvalidArgument`] if the file does not exist or is unreadable.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use index_tts_rs_service::resolve_input_path;
///
/// let dir = tempfile::tempdir()?;
/// assert!(resolve_input_path(dir.path(), Path::new("missing.wav"), "--ref").is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(
    clippy::disallowed_methods,
    reason = "a single canonicalize at the request boundary; callers are not latency sensitive"
)]
pub fn resolve_input_path(base: &Path, path: &Path, field: &str) -> Result<PathBuf, ServiceError> {
    let joined = base.join(path);
    joined.canonicalize().map_err(|err| {
        ServiceError::invalid(format!(
            "{field} `{}` is not readable: {err}",
            joined.display()
        ))
    })
}

/// Draws a fresh seed from the process's randomly keyed hasher; reproducibility comes from
/// reporting it, not from its quality, so OS-seeded `SipHash` keys suffice.
fn random_seed() -> u64 {
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u128(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos()),
    );
    hasher.finish()
}

fn bounded(field: &str, value: u32, min: u32, max: u32) -> Result<u32, ServiceError> {
    if (min..=max).contains(&value) {
        Ok(value)
    } else {
        Err(ServiceError::invalid(format!(
            "{field} must be between {min} and {max}, got {value}"
        )))
    }
}

fn bounded_silence(field: &str, value: u32) -> Result<u32, ServiceError> {
    bounded(field, value, 0, MAX_SILENCE_MS)
}

fn speaker_label(speaker: &SpeakerSource) -> String {
    let file_label = |path: &Path| {
        path.file_name().map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        )
    };
    match speaker {
        SpeakerSource::VoiceId(id) => id.as_str().to_owned(),
        SpeakerSource::VoiceFile(path) | SpeakerSource::ReferenceAudio(path) => file_label(path),
        _ => "unknown".to_owned(),
    }
}

/// Fails fast, before any GPU work, when the request or a `[[voice=...]]` switch names a voice
/// that is not registered; otherwise the job would fail midway after producing partial audio.
fn ensure_voices_registered(
    config: &EngineConfig,
    plan: &[PlannedUtterance],
) -> Result<(), ServiceError> {
    let store = VoiceStore::new(config.voices_dir.clone().unwrap_or_else(default_voices_dir))?;
    for part in plan {
        if let SpeakerSource::VoiceId(id) = &part.request.speaker
            && !store.voice_path(id).is_file()
        {
            return Err(ServiceError::voice_not_found(id.as_str()));
        }
    }
    Ok(())
}

fn missing_voice_error(config: &EngineConfig) -> ServiceError {
    let dir = config.voices_dir.clone().unwrap_or_else(default_voices_dir);
    let voices: Vec<String> = VoiceStore::new(dir)
        .ok()
        .and_then(|store| store.list().ok())
        .map(|entries| {
            entries
                .into_iter()
                .take(MAX_HINT_VOICES)
                .map(|entry| entry.voice_id)
                .collect()
        })
        .unwrap_or_default();
    let hint = if voices.is_empty() {
        "register one with `itts voice add <ID> --ref <AUDIO>`, or pass refAudio".to_owned()
    } else {
        format!(
            "registered voices: {}; set `default_voice` in the config to make one implicit",
            voices.join(", ")
        )
    };
    ServiceError::new(
        ErrorCode::InvalidArgument,
        "no speaker selected: provide voice, voiceFile, or refAudio",
    )
    .with_hint(hint)
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        fs::write,
        path::{Path, PathBuf},
    };

    use index_tts_rs_core::config::{
        EngineConfig, Language, SpeakerSource, SpeechContext, VoiceId,
    };
    use rstest::rstest;
    use serde_json::{Value, from_str, from_value, json};
    use tempfile::TempDir;

    use super::{PathPolicy, PreparedJob, SpeechJob};
    use crate::{ErrorCode, ServiceError};

    /// A config whose voices directory registers `narrator`, `tyr`, and `other`.
    struct Fixture {
        _voices: TempDir,
        config: EngineConfig,
    }

    #[allow(clippy::disallowed_methods, reason = "synchronous test fixture")]
    fn fixture() -> Result<Fixture, Box<dyn Error>> {
        let voices = tempfile::tempdir()?;
        for id in ["narrator", "tyr", "other"] {
            write(voices.path().join(format!("{id}.itts_voice")), b"")?;
        }
        let mut config = EngineConfig::default();
        config.voices_dir = Some(voices.path().to_path_buf());
        Ok(Fixture {
            _voices: voices,
            config,
        })
    }

    fn prepare(value: Value) -> Result<PreparedJob, Box<dyn Error>> {
        let fixture = fixture()?;
        let job: SpeechJob = from_value(value)?;
        Ok(job.prepare(&fixture.config, PathPolicy::LocalAbsolute)?)
    }

    #[test]
    fn test_should_compile_inline_controls_into_utterances() -> Result<(), Box<dyn Error>> {
        let prepared = prepare(json!({
            "input": "第一段。\n\n[[pause=900]][[emo=calm:0.6]]第二段。",
            "voice": "narrator",
            "speed": 2.0
        }))?;
        assert_eq!(prepared.plan.len(), 2);
        let second = prepared
            .plan
            .get(1)
            .ok_or_else(|| ServiceError::internal("plan"))?;
        assert_eq!(second.pause_before_ms, 900);
        assert!(second.request.emo_vector.is_some());
        assert!((second.request.cfm.duration_factor.get() - 0.5).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn test_should_continue_only_the_first_utterance_after_previous_text()
    -> Result<(), Box<dyn Error>> {
        let prepared = prepare(json!({
            "input": "第一段。\n\n第二段。",
            "voice": "narrator",
            "continuity": true,
            "previousText": "上一句。"
        }))?;
        assert!(prepared.continuity);
        let contexts: Vec<bool> = prepared
            .plan
            .iter()
            .map(|part| matches!(part.request.context, SpeechContext::Text(_)))
            .collect();
        assert_eq!(contexts, [true, false]);
        assert!(prepared.plan.iter().all(|part| part.request.continuity));
        assert!(
            prepared
                .plan
                .iter()
                .all(|part| part.request.interval_silence_ms == 0)
        );
        Ok(())
    }

    #[test]
    fn test_should_keep_explicit_interval_and_reject_controls_in_previous_text()
    -> Result<(), Box<dyn Error>> {
        let prepared = prepare(json!({
            "input": "你好。", "voice": "narrator", "continuity": true, "intervalSilenceMs": 150
        }))?;
        assert!(
            prepared
                .plan
                .iter()
                .all(|part| part.request.interval_silence_ms == 150)
        );
        let err = prepare(json!({
            "input": "你好。", "voice": "narrator", "previousText": "[[pause=100]]上一句。"
        }))
        .err()
        .map(|err| err.to_string());
        assert!(err.is_some_and(|message| message.contains("previousText")));
        Ok(())
    }

    #[test]
    fn test_should_apply_config_defaults_and_overrides() -> Result<(), Box<dyn Error>> {
        let fixture = fixture()?;
        let mut config = fixture.config.clone();
        config.default_language = Language::En;
        config.interval_silence_ms = 450;
        config.cfm.n_timesteps = 12;
        let job: SpeechJob =
            from_str(r#"{"input":"Hello","voice":"tyr","advanced":{"greedy":true}}"#)
                .map_err(|err| ServiceError::invalid(err.to_string()))?;
        let prepared = job.prepare(&config, PathPolicy::Denied)?;
        let request = &prepared
            .plan
            .first()
            .ok_or_else(|| ServiceError::internal("plan"))?
            .request;
        assert_eq!(request.language, Language::En);
        assert_eq!(request.interval_silence_ms, 450);
        assert_eq!(request.cfm.n_timesteps, 12);
        assert!(request.sampling.greedy);
        Ok(())
    }

    #[test]
    fn test_should_fall_back_to_configured_default_voice() -> Result<(), Box<dyn Error>> {
        let fixture = fixture()?;
        let mut config = fixture.config.clone();
        config.default_voice = Some(VoiceId::new("narrator")?);
        let job = SpeechJob::builder().input("你好").build();
        let prepared = job.prepare(&config, PathPolicy::Denied)?;
        let request = &prepared
            .plan
            .first()
            .ok_or_else(|| ServiceError::internal("plan"))?
            .request;
        assert_eq!(
            request.speaker,
            SpeakerSource::VoiceId(VoiceId::new("narrator")?)
        );
        Ok(())
    }

    #[rstest]
    #[case("你好", "ghost")]
    #[case("你好。[[voice=ghost]]再见。", "tyr")]
    fn test_should_fail_fast_on_unregistered_voices(
        #[case] input: &str,
        #[case] voice: &str,
    ) -> Result<(), Box<dyn Error>> {
        let fixture = fixture()?;
        let job = SpeechJob::builder().input(input).voice(voice).build();
        let err = job.prepare(&fixture.config, PathPolicy::Denied);
        assert!(matches!(
            err,
            Err(ServiceError {
                code: ErrorCode::VoiceNotFound,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn test_should_explain_missing_speaker_with_hint() {
        let temp = tempfile::tempdir().ok();
        let mut config = EngineConfig::default();
        config.voices_dir = temp.as_ref().map(|dir| dir.path().to_path_buf());
        let err = SpeechJob::builder()
            .input("你好")
            .build()
            .prepare(&config, PathPolicy::Denied);
        assert!(matches!(
            err,
            Err(ServiceError {
                code: ErrorCode::InvalidArgument,
                hint: Some(_),
                ..
            })
        ));
    }

    #[rstest]
    #[case(json!({"input":" ","voice":"tyr"}))]
    #[case(json!({"input":"你好","voice":"../secret"}))]
    #[case(json!({"input":"你好","voice":"tyr","speed":0}))]
    #[case(json!({"input":"你好","voice":"tyr","speed":5}))]
    #[case(json!({"input":"你好","voice":"tyr","lang":"invalid"}))]
    #[case(json!({"input":"你好","voice":"tyr","paragraphPauseMs":6000}))]
    #[case(json!({"input":"你好","voice":"tyr","advanced":{"nTimesteps":0}}))]
    #[case(json!({"input":"你好","voice":"tyr","voiceFile":"/tmp/a.itts_voice"}))]
    #[case(json!({"input":"你好","voice":"tyr","emoRef":"/tmp/a.wav"}))]
    #[case(json!({"input":"你好","refAudio":"relative.wav"}))]
    #[case(json!({"input":"你好","voice":"tyr","extra":"rejected"}))]
    #[case(json!({"input":"[[bogus=1]]你好","voice":"tyr"}))]
    #[case(json!({"input": "你".repeat(5462), "voice":"tyr"}))]
    fn test_should_reject_invalid_jobs(#[case] value: Value) {
        assert!(prepare(value).is_err());
    }

    #[test]
    fn test_should_reject_file_references_over_network_transports() {
        let job = SpeechJob::builder()
            .input("你好")
            .ref_audio(PathBuf::from("/tmp/ref.wav"))
            .build();
        let err = job.prepare(&EngineConfig::default(), PathPolicy::Denied);
        assert!(matches!(
            err,
            Err(ServiceError {
                code: ErrorCode::InvalidArgument,
                ..
            })
        ));
    }

    #[test]
    #[allow(clippy::disallowed_methods, reason = "synchronous test fixture")]
    fn test_should_canonicalize_inputs_and_reject_missing_files() -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        write(dir.path().join("ref.wav"), b"RIFF")?;
        let mut job = SpeechJob::builder()
            .input("你好")
            .ref_audio(PathBuf::from("ref.wav"))
            .build();
        job.resolve_paths(dir.path())?;
        assert!(job.ref_audio.as_deref().is_some_and(Path::is_absolute));

        let mut missing = SpeechJob::builder()
            .input("你好")
            .voice_file(PathBuf::from("nope.itts_voice"))
            .build();
        assert!(missing.resolve_paths(dir.path()).is_err());
        Ok(())
    }
}

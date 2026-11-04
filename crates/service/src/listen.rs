//! [`ListenJob`]: verbatim local speech recognition of a recording (`itts listen`, daemon
//! `listen`, MCP `listen`), returning [`ListenResult`]: the recognized text and one timed token
//! per CJK character or Latin word.
//!
//! The recognizers (sherpa-onnx `SenseVoice`, `Paraformer`, `Parakeet`) are non-autoregressive
//! and run without inverse text normalization, so stutters, repeats and numbers come back as
//! spoken. Running a job needs the `asr` feature; the job and result types are always available.

use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

use index_tts_rs_asr::{AsrModel, SpokenLanguage};
use index_tts_rs_core::config::{EngineConfig, Language, validate_safe_file_path};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use crate::{
    error::{ErrorCode, ServiceError},
    request::{PathPolicy, resolve_input_path},
};

/// Recognizer name reported in [`RecognizerInfo::name`].
pub const RECOGNIZER_NAME: &str = "sherpa-onnx";
/// Major version of token post-processing and script comparison; changes whenever results for
/// the same input change.
pub const RECOGNIZER_VERSION: u32 = 1;
/// Longest recording one `listen` accepts, in seconds.
pub const MAX_LISTEN_SECS: f32 = 120.0;
/// Shortest recording accepted, in seconds.
pub const MIN_LISTEN_SECS: f32 = 0.1;

/// A recognition request.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TypedBuilder,
)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[builder(field_defaults(default, setter(strip_option)))]
#[non_exhaustive]
pub struct ListenJob {
    /// Recording (`.wav`/`.mp3`/`.flac`/`.m4a`, any sample rate, at most 120 s).
    #[builder(!default, setter(!strip_option, into))]
    pub audio: PathBuf,
    /// Spoken language: `zh`, `zh/en`, `yue`, `ja`, `en` (default from config).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub lang: Option<String>,
    /// Recognizer: `paraformer` (Mandarin, default for `zh` and `zh/en`), `sense-voice` (zh, yue,
    /// ja, en; default for `yue` and `ja`), `parakeet` (English, default for `en`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub model: Option<String>,
}

/// A validated [`ListenJob`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreparedListen {
    /// Absolute recording path.
    pub audio: PathBuf,
    /// Spoken language.
    pub language: SpokenLanguage,
    /// Recognizer model.
    pub model: AsrModel,
}

impl ListenJob {
    /// Makes `audio` absolute against `base` (it must exist).
    ///
    /// # Errors
    ///
    /// Returns [`ServiceError`] with `INVALID_ARGUMENT` when the path is not readable.
    pub fn resolve_paths(&mut self, base: &Path) -> Result<(), ServiceError> {
        self.audio = resolve_input_path(base, &self.audio, "audio")?;
        Ok(())
    }

    /// Validates the job.
    ///
    /// # Errors
    ///
    /// Returns `INVALID_ARGUMENT` for an unknown language or model, a language without a local
    /// recognizer, a model that does not recognize the language, or a path that is relative or
    /// forbidden by `policy`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EngineConfig;
    /// use index_tts_rs_service::{ListenJob, PathPolicy};
    ///
    /// let job = ListenJob::builder().audio("/tmp/line.wav").lang("en").build();
    /// let prepared = job.prepare(&EngineConfig::default(), PathPolicy::LocalAbsolute)?;
    /// assert_eq!(prepared.model.as_str(), "parakeet");
    /// # Ok::<(), index_tts_rs_service::ServiceError>(())
    /// ```
    pub fn prepare(
        &self,
        config: &EngineConfig,
        policy: PathPolicy,
    ) -> Result<PreparedListen, ServiceError> {
        check_audio_path(&self.audio, policy)?;
        let language = match self.lang.as_deref() {
            Some(code) => Language::from_str(code)?,
            None => config.default_language,
        };
        let language = spoken_language(language)?;
        let model = self.model.as_deref().map(AsrModel::from_str).transpose()?;
        Ok(PreparedListen {
            audio: self.audio.clone(),
            language,
            model: AsrModel::choose(model, language)?,
        })
    }
}

/// Checks a recording path against the transport's path policy.
pub(crate) fn check_audio_path(path: &Path, policy: PathPolicy) -> Result<(), ServiceError> {
    match policy {
        PathPolicy::Denied => Err(ServiceError::invalid(
            "audio is not accepted over this transport",
        )),
        PathPolicy::LocalAbsolute => {
            validate_safe_file_path(path, "path")?;
            if path.is_absolute() {
                Ok(())
            } else {
                Err(ServiceError::invalid(format!(
                    "audio must be an absolute path, got `{}`",
                    path.display()
                )))
            }
        }
    }
}

/// The recognizer language for an itts language.
fn spoken_language(language: Language) -> Result<SpokenLanguage, ServiceError> {
    match language {
        Language::Zh => Ok(SpokenLanguage::Zh),
        Language::ZhEn => Ok(SpokenLanguage::ZhEn),
        Language::Yue => Ok(SpokenLanguage::Yue),
        Language::Ja => Ok(SpokenLanguage::Ja),
        Language::En => Ok(SpokenLanguage::En),
        other => Err(
            ServiceError::invalid(format!("no local speech recognizer for `{other}`"))
                .with_hint("speech recognition supports zh, zh/en, yue, ja, and en"),
        ),
    }
}

/// The error for a recognition request on a build without the `asr` feature.
#[must_use]
pub fn asr_unavailable() -> ServiceError {
    ServiceError::new(
        ErrorCode::ModelUnavailable,
        "this itts was built without speech recognition",
    )
    .with_hint("rebuild with `cargo install index-tts-rs-cli --features asr`")
}

/// The recognizer that produced a result; `name`, `version`, and `model` identify its output
/// for caching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct RecognizerInfo {
    /// Runtime (`sherpa-onnx`).
    pub name: String,
    /// Major version of token post-processing and comparison.
    pub version: u32,
    /// Model: `sense-voice`, `paraformer`, or `parakeet`.
    pub model: String,
    /// Hugging Face repository of the model.
    pub repo: String,
    /// Pinned repository revision.
    pub revision: String,
}

impl RecognizerInfo {
    /// Describes `model`.
    #[must_use]
    pub fn new(model: AsrModel) -> Self {
        Self {
            name: RECOGNIZER_NAME.to_owned(),
            version: RECOGNIZER_VERSION,
            model: model.as_str().to_owned(),
            repo: model.repo().to_owned(),
            revision: model.revision().to_owned(),
        }
    }
}

/// One recognized unit: a CJK character, or a Latin word / number. Times are approximate (the
/// recognizer reports token starts; an end is the next start, at most 0.5 s later).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct HeardUnit {
    /// Text as recognized.
    pub text: String,
    /// Start in seconds.
    pub start: f32,
    /// End in seconds.
    pub end: f32,
}

/// Milliseconds spent in each stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ListenTimings {
    /// Reading, decoding, and resampling the audio.
    pub audio_ms: f64,
    /// Loading the model (first use; downloads not included).
    pub load_ms: f64,
    /// Recognition.
    pub recognize_ms: f64,
    /// Total, including a first-use download.
    pub total_ms: f64,
}

/// Result of a [`ListenJob`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ListenResult {
    /// The recognizer.
    pub recognizer: RecognizerInfo,
    /// Recognized text: CJK characters run together, Latin words separated by spaces.
    pub text: String,
    /// Recognized units in order.
    pub tokens: Vec<HeardUnit>,
    /// Whether the model was downloaded for this request.
    #[serde(default, skip_serializing_if = "is_false")]
    pub model_downloaded: bool,
    /// Recording duration in seconds.
    pub duration_secs: f32,
    /// Wall-clock seconds spent (audio decoding included).
    pub listen_secs: f64,
    /// Per-stage milliseconds.
    pub timings: ListenTimings,
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if signature"
)]
const fn is_false(value: &bool) -> bool {
    !*value
}

#[cfg(feature = "asr")]
pub(crate) use run::listen_tokens;
#[cfg(feature = "asr")]
pub use run::run_listen;

#[cfg(feature = "asr")]
mod run {
    use std::{path::PathBuf, time::Instant};

    use index_tts_rs_asr::{ASR_SAMPLE_RATE, AsrEngine, HeardToken, heard_text};
    use index_tts_rs_core::dsp::{audio::decode_audio_file, resample_sinc_hann};
    use tokio::task::spawn_blocking;

    use super::{
        HeardUnit, ListenResult, ListenTimings, MAX_LISTEN_SECS, MIN_LISTEN_SECS, PreparedListen,
        RecognizerInfo,
    };
    use crate::error::{ErrorCode, ServiceError};

    /// Recognizes a prepared job on `asr`.
    ///
    /// # Errors
    ///
    /// Returns `AUDIO_DECODE_FAILED` for unreadable audio, `INVALID_ARGUMENT` for a recording
    /// shorter than 0.1 s, `TOO_LONG` beyond 120 s, or `MODEL_UNAVAILABLE` when the model
    /// cannot be downloaded or loaded.
    pub async fn run_listen(
        asr: &AsrEngine,
        prepared: PreparedListen,
    ) -> Result<ListenResult, ServiceError> {
        Ok(listen_tokens(asr, prepared).await?.0)
    }

    /// [`run_listen`], also returning the recognized units for comparison.
    pub(crate) async fn listen_tokens(
        asr: &AsrEngine,
        prepared: PreparedListen,
    ) -> Result<(ListenResult, Vec<HeardToken>), ServiceError> {
        let started = Instant::now();
        let samples = load_16k(prepared.audio.clone()).await?;
        #[allow(
            clippy::cast_precision_loss,
            reason = "recordings are at most minutes long"
        )]
        let duration_secs = samples.len() as f32 / ASR_SAMPLE_RATE as f32;
        if duration_secs < MIN_LISTEN_SECS {
            return Err(ServiceError::invalid(format!(
                "the recording is {duration_secs:.2} s long; at least {MIN_LISTEN_SECS} s is \
                 needed"
            )));
        }
        if duration_secs > MAX_LISTEN_SECS {
            return Err(ServiceError::new(
                ErrorCode::TooLong,
                format!(
                    "the recording is {duration_secs:.1} s long; one listen takes at most \
                     {MAX_LISTEN_SECS} s"
                ),
            )
            .with_hint("split the recording into pieces of at most 120 s"));
        }
        let audio_ms = started.elapsed().as_secs_f64() * 1000.0;
        let transcript = asr.transcribe(prepared.model, samples).await?;
        let tokens = transcript
            .tokens
            .iter()
            .map(|token| HeardUnit {
                text: token.text.clone(),
                start: token.start,
                end: token.end,
            })
            .collect();
        let elapsed = started.elapsed().as_secs_f64();
        let result = ListenResult {
            recognizer: RecognizerInfo::new(prepared.model),
            text: heard_text(&transcript.tokens),
            tokens,
            model_downloaded: transcript.downloaded,
            duration_secs,
            listen_secs: elapsed,
            timings: ListenTimings {
                audio_ms,
                load_ms: transcript.load_ms,
                recognize_ms: transcript.recognize_ms,
                total_ms: elapsed * 1000.0,
            },
        };
        Ok((result, transcript.tokens))
    }

    /// Decodes `path` to mono 16 kHz on a blocking thread.
    async fn load_16k(path: PathBuf) -> Result<Vec<f32>, ServiceError> {
        spawn_blocking(move || {
            let decoded = decode_audio_file(&path).map_err(ServiceError::from)?;
            resample_sinc_hann(&decoded.samples, decoded.sample_rate, ASR_SAMPLE_RATE)
                .map_err(ServiceError::from)
        })
        .await
        .map_err(|err| ServiceError::internal(format!("audio decoding task failed: {err}")))?
    }
}

#[cfg(test)]
mod tests {
    use index_tts_rs_core::config::EngineConfig;
    use serde_json::{from_value, json};

    use super::ListenJob;
    use crate::{ErrorCode, PathPolicy};

    fn prepare(lang: Option<&str>, model: Option<&str>) -> Result<String, ErrorCode> {
        let mut job = ListenJob::builder().audio("/tmp/line.wav").build();
        job.lang = lang.map(str::to_owned);
        job.model = model.map(str::to_owned);
        job.prepare(&EngineConfig::default(), PathPolicy::LocalAbsolute)
            .map(|p| p.model.as_str().to_owned())
            .map_err(|e| e.code)
    }

    #[test]
    fn test_should_choose_a_recognizer_per_language() {
        assert_eq!(prepare(None, None).as_deref(), Ok("paraformer"));
        assert_eq!(prepare(Some("yue"), None).as_deref(), Ok("sense-voice"));
        assert_eq!(prepare(Some("en"), None).as_deref(), Ok("parakeet"));
        assert_eq!(
            prepare(Some("zh"), Some("sense-voice")).as_deref(),
            Ok("sense-voice")
        );
        assert_eq!(
            prepare(Some("ja"), Some("sense-voice")).as_deref(),
            Ok("sense-voice")
        );
    }

    #[test]
    fn test_should_reject_unsupported_languages_and_models() {
        for (lang, model) in [
            (Some("es"), None),
            (Some("ar"), None),
            (Some("zh"), Some("parakeet")),
            (Some("en"), Some("paraformer")),
            (None, Some("whisper")),
            (Some("xx"), None),
        ] {
            assert_eq!(
                prepare(lang, model),
                Err(ErrorCode::InvalidArgument),
                "{lang:?} {model:?}"
            );
        }
    }

    #[test]
    fn test_should_reject_relative_paths_and_unknown_fields() {
        let relative = ListenJob::builder().audio("line.wav").build();
        assert!(
            relative
                .prepare(&EngineConfig::default(), PathPolicy::LocalAbsolute)
                .is_err()
        );
        let absolute = ListenJob::builder().audio("/tmp/line.wav").build();
        assert!(
            absolute
                .prepare(&EngineConfig::default(), PathPolicy::Denied)
                .is_err()
        );
        assert!(from_value::<ListenJob>(json!({ "audio": "/a.wav", "seed": 1 })).is_err());
    }
}

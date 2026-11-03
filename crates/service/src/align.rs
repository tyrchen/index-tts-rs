//! [`AlignJob`]: force-align a recording to the text it speaks (`itts align`, daemon `align`,
//! MCP `align`), returning [`AlignResult`] in the same alignment shape as `synth`.
//!
//! The text's `[[...]]` controls are removed (not executed); pronunciation overrides are kept and
//! read by the same frontend as synthesis. Timings are seconds from the start of the recording.

use std::{
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use index_tts_rs_core::{
    align::{
        ALIGNER_NAME, ALIGNER_VERSION, ALIGNMENT_HEADS, AlignmentOutput, AlignmentRequest,
        Conditioning, Recording,
        phones::{PhoneKind, PhoneTiming},
    },
    config::{
        EngineConfig, Language, SpeakerSource, SynthesisText, VoiceId, validate_safe_file_path,
    },
    engine::IndexTtsEngine,
};
use index_tts_rs_director::control_ranges;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use crate::{
    error::ServiceError,
    request::{PathPolicy, resolve_input_path},
    transcript::{Alignment, SentenceTiming, is_spoken_char, sentences},
};

/// A forced-alignment request. `voice` and `voiceFile` are mutually exclusive; without either,
/// the recording itself conditions the decoder.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TypedBuilder,
)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[builder(field_defaults(default, setter(strip_option)))]
#[non_exhaustive]
pub struct AlignJob {
    /// Recording to align (`.wav`/`.mp3`/`.flac`/`.m4a`, any sample rate, at most about 72 s).
    #[builder(!default, setter(!strip_option, into))]
    pub audio: PathBuf,
    /// Text the recording speaks (at most 16 KiB and 600 tokens). `<字|pinyin>` /
    /// `<word|ARPAbet>` pronunciation overrides are honoured; `[[...]]` controls are removed.
    #[builder(!default, setter(!strip_option, into))]
    pub input: String,
    /// Language code: `zh`, `en`, `ja`, `yue`, `es`, `ar`, `zh/en` (default from config).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub lang: Option<String>,
    /// Registered voice whose snapshot conditions the decoder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub voice: Option<String>,
    /// `.itts_voice` snapshot file whose conditioning is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_file: Option<PathBuf>,
    /// Also return phone timings (`phones`, `phonesMethod`).
    #[serde(default, skip_serializing_if = "is_false")]
    #[builder(setter(!strip_option))]
    pub phones: bool,
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if signature"
)]
const fn is_false(value: &bool) -> bool {
    !*value
}

/// A validated [`AlignJob`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreparedAlign {
    /// Absolute recording path.
    pub audio: PathBuf,
    /// The input with controls removed; the alignment text.
    pub text: String,
    /// Language of the text.
    pub language: Language,
    /// Source of the decoder conditioning.
    pub conditioning: Conditioning,
    /// Whether phones were requested.
    pub phones: bool,
    /// Model repository reported in [`AlignerInfo::model`].
    pub model: String,
}

impl AlignJob {
    /// Makes `audio` and `voiceFile` absolute against `base` (they must exist).
    ///
    /// # Errors
    ///
    /// Returns [`ServiceError`] with `INVALID_ARGUMENT` when a path is not readable.
    pub fn resolve_paths(&mut self, base: &Path) -> Result<(), ServiceError> {
        self.audio = resolve_input_path(base, &self.audio, "audio")?;
        if let Some(path) = self.voice_file.as_mut() {
            *path = resolve_input_path(base, path, "voiceFile")?;
        }
        Ok(())
    }

    /// Validates the job.
    ///
    /// # Errors
    ///
    /// Returns `INVALID_ARGUMENT` for an empty text, malformed controls, an unknown language,
    /// conflicting speakers, or paths that are relative or forbidden by `policy`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EngineConfig;
    /// use index_tts_rs_service::{AlignJob, PathPolicy};
    ///
    /// let job = AlignJob::builder()
    ///     .audio("/tmp/line.wav")
    ///     .input("你好[[pause=300]]世界。")
    ///     .lang("zh")
    ///     .build();
    /// let prepared = job.prepare(&EngineConfig::default(), PathPolicy::LocalAbsolute)?;
    /// assert_eq!(prepared.text, "你好世界。");
    /// # Ok::<(), index_tts_rs_service::ServiceError>(())
    /// ```
    pub fn prepare(
        &self,
        config: &EngineConfig,
        policy: PathPolicy,
    ) -> Result<PreparedAlign, ServiceError> {
        self.check_paths(policy)?;
        let text = spoken_text(&self.input)?;
        let language = match self.lang.as_deref() {
            Some(code) => Language::from_str(code)?,
            None => config.default_language,
        };
        let conditioning = match (&self.voice, &self.voice_file) {
            (None, None) => Conditioning::Recording,
            (Some(id), None) => {
                Conditioning::Speaker(SpeakerSource::VoiceId(VoiceId::new(id.as_str())?))
            }
            (None, Some(path)) => Conditioning::Speaker(SpeakerSource::VoiceFile(path.clone())),
            (Some(_), Some(_)) => {
                return Err(ServiceError::invalid(
                    "provide only one of voice or voiceFile",
                ));
            }
        };
        Ok(PreparedAlign {
            audio: self.audio.clone(),
            text,
            language,
            conditioning,
            phones: self.phones,
            model: config.model_repo.clone(),
        })
    }

    fn check_paths(&self, policy: PathPolicy) -> Result<(), ServiceError> {
        for (field, path) in [
            ("audio", Some(self.audio.as_path())),
            ("voiceFile", self.voice_file.as_deref()),
        ] {
            let Some(path) = path else { continue };
            match policy {
                PathPolicy::Denied => {
                    return Err(ServiceError::invalid(format!(
                        "{field} is not accepted over this transport"
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
        Ok(())
    }
}

/// `input` with every `[[...]]` control removed; it must still contain pronounced text.
fn spoken_text(input: &str) -> Result<String, ServiceError> {
    let mut text = String::with_capacity(input.len());
    let mut cursor = 0;
    for control in control_ranges(input)? {
        text.push_str(input.get(cursor..control.start).unwrap_or_default());
        cursor = control.end;
    }
    text.push_str(input.get(cursor..).unwrap_or_default());
    if !text.chars().any(is_spoken_char) {
        return Err(ServiceError::invalid("the text to align is empty")
            .with_hint("pass the words the recording speaks with --text or --stdin"));
    }
    SynthesisText::new(text.as_str())?;
    Ok(text)
}

/// Which aligner produced a result; `name` and `version` identify its output for caching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct AlignerInfo {
    /// Aligner name (`itts-attn`).
    pub name: String,
    /// Major version; changes whenever results for the same input change.
    pub version: u32,
    /// Model repository whose weights aligned the recording.
    pub model: String,
    /// Decoder attention heads read (`L4H10+L5H10`).
    pub heads: String,
    /// Conditioning source: `recording` or `voice`.
    pub conditioning: String,
}

/// How well the decoder supports the alignment, over units (a Chinese character, a Latin word,
/// or a pronunciation override). Results are returned whatever the quality. On clean speech
/// `confidence` and `coverage` stay above 0.97; below 0.95 the recording most likely does not
/// say the text (omitted, extra, or substituted words). Numbers read in another form that shares
/// syllables (`2026` as 两千零二十六) are not detected; use speech recognition for those.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct AlignQuality {
    /// Mean attention peak of the units (`0..=1`).
    pub confidence: f32,
    /// Fraction of units some frame attends to most (`0..=1`).
    pub coverage: f32,
    /// Indices into `alignment.text` (Unicode scalars; a word's first letter) of units the audio
    /// barely supports: weak attention, or frames the text explains worse than no text.
    pub weak: Vec<usize>,
}

/// Kind of a phone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum PhoneKindWire {
    /// Mandarin initial.
    #[serde(rename = "ini")]
    Initial,
    /// Mandarin final.
    #[serde(rename = "fin")]
    Final,
    /// English `ARPAbet` phone.
    #[serde(rename = "ph")]
    Phone,
}

/// One timed phone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub struct Phone {
    /// Index into `alignment.text` of the character (Chinese) or the word's first letter
    /// (English); for `<字|pinyin>` overrides, the overridden character.
    pub i: usize,
    /// Pinyin initial or final (`ü` as `v`, `y`/`w` spellings as finals), or `ARPAbet` without
    /// stress digits.
    pub p: String,
    /// Tone of a Mandarin final, `1..=5` (`5` neutral).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t: Option<u8>,
    /// `ini`, `fin`, or `ph`.
    pub k: PhoneKindWire,
    /// Start in seconds.
    pub start: f32,
    /// End in seconds.
    pub end: f32,
}

/// How phone boundaries were found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum PhonesMethod {
    /// Cut inside each aligned character or word (initial at the voicing onset, English phones
    /// by typical duration).
    Split,
}

/// Milliseconds spent in each stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct AlignTimings {
    /// Reading and decoding the audio file.
    pub audio_ms: f64,
    /// Resampling and `w2v-BERT` features.
    pub features_ms: f64,
    /// Semantic codec encoder.
    pub encode_ms: f64,
    /// Decoder conditioning.
    pub condition_ms: f64,
    /// Teacher-forced attention, decoding, and phones.
    pub attention_ms: f64,
    /// Engine total, including first-use model loads.
    pub total_ms: f64,
}

/// Result of an [`AlignJob`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct AlignResult {
    /// The aligner and its conditioning.
    pub aligner: AlignerInfo,
    /// Per-character timings of the input with controls removed (same shape and filling rules
    /// as `synth`).
    pub alignment: Alignment,
    /// Sentence timings.
    pub sentences: Vec<SentenceTiming>,
    /// Attention support of the alignment.
    pub quality: AlignQuality,
    /// Phone timings, when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phones: Option<Vec<Phone>>,
    /// How phones were timed, when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phones_method: Option<PhonesMethod>,
    /// Recording duration in seconds.
    pub duration_secs: f32,
    /// Wall-clock seconds spent aligning (audio decoding included).
    pub align_secs: f64,
    /// Per-stage milliseconds.
    pub timings: AlignTimings,
}

/// Aligns a prepared job on `engine`.
///
/// # Errors
///
/// Returns `AUDIO_DECODE_FAILED` for unreadable audio, `INVALID_ARGUMENT` for a recording that
/// is too short or text without pronounced tokens, `TOO_LONG` beyond one decoder pass,
/// `VOICE_NOT_FOUND`, or `MODEL_UNAVAILABLE`.
pub async fn run_align(
    engine: &IndexTtsEngine,
    prepared: PreparedAlign,
) -> Result<AlignResult, ServiceError> {
    let started = Instant::now();
    let request = AlignmentRequest::builder()
        .recording(Recording::File(prepared.audio.clone()))
        .text(SynthesisText::new(prepared.text.as_str())?)
        .language(prepared.language)
        .conditioning(prepared.conditioning.clone())
        .phones(prepared.phones)
        .build();
    let output = engine.align(request).await?;
    Ok(assemble(
        &prepared,
        &output,
        started.elapsed().as_secs_f64(),
    ))
}

/// Builds the wire result from the engine output.
fn assemble(prepared: &PreparedAlign, output: &AlignmentOutput, align_secs: f64) -> AlignResult {
    let duration = output.duration_secs;
    let clamp = |t: f32| t.clamp(0.0, duration);
    let spans: Vec<_> = output
        .spans
        .iter()
        .map(|span| {
            (
                span.chars.clone(),
                clamp(span.start_secs),
                clamp(span.end_secs),
            )
        })
        .collect();
    let alignment = Alignment::from_spans(&prepared.text, &spans, 0.0, duration);
    let chars: Vec<char> = prepared.text.chars().collect();
    let mut weak: Vec<usize> = output
        .quality
        .weak
        .iter()
        .filter_map(|range| {
            range
                .clone()
                .find(|&i| chars.get(i).copied().is_some_and(is_spoken_char))
        })
        .collect();
    weak.dedup();
    let phones = prepared.phones.then(|| {
        output
            .phones
            .iter()
            .map(|phone| wire_phone(phone, clamp))
            .collect()
    });
    let timings = &output.timings;
    AlignResult {
        aligner: AlignerInfo {
            name: ALIGNER_NAME.to_owned(),
            version: ALIGNER_VERSION,
            model: prepared.model.clone(),
            heads: ALIGNMENT_HEADS
                .iter()
                .map(|(layer, head)| format!("L{layer}H{head}"))
                .collect::<Vec<_>>()
                .join("+"),
            conditioning: prepared.conditioning.label().to_owned(),
        },
        sentences: sentences(&alignment),
        alignment,
        quality: AlignQuality {
            confidence: output.quality.confidence,
            coverage: output.quality.coverage,
            weak,
        },
        phones,
        phones_method: prepared.phones.then_some(PhonesMethod::Split),
        duration_secs: duration,
        align_secs,
        timings: AlignTimings {
            audio_ms: timings.audio_ms,
            features_ms: timings.features_ms,
            encode_ms: timings.encode_ms,
            condition_ms: timings.condition_ms,
            attention_ms: timings.attention_ms,
            total_ms: timings.total_ms,
        },
    }
}

fn wire_phone(phone: &PhoneTiming, clamp: impl Fn(f32) -> f32) -> Phone {
    Phone {
        i: phone.char_index,
        p: phone.phone.clone(),
        t: phone.tone,
        k: match phone.kind {
            PhoneKind::Initial => PhoneKindWire::Initial,
            PhoneKind::Final => PhoneKindWire::Final,
            _ => PhoneKindWire::Phone,
        },
        start: clamp(phone.start),
        end: clamp(phone.end),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use index_tts_rs_core::{
        align::{AlignmentOutput, AlignmentQuality, SpanTiming},
        config::EngineConfig,
    };
    use serde_json::{from_value, json, to_value};

    use super::{AlignJob, PreparedAlign, assemble};
    use crate::{ErrorCode, PathPolicy, ServiceError};

    fn job(input: &str) -> AlignJob {
        AlignJob::builder()
            .audio("/tmp/line.wav")
            .input(input)
            .build()
    }

    #[test]
    fn test_should_strip_controls_and_keep_overrides() -> Result<(), ServiceError> {
        let prepared = job("[[speed=0.9]]把<行|hang2>业[[pause=200]]报告。")
            .prepare(&EngineConfig::default(), PathPolicy::LocalAbsolute)?;
        assert_eq!(prepared.text, "把<行|hang2>业报告。");
        assert_eq!(prepared.conditioning.label(), "recording");
        Ok(())
    }

    #[test]
    fn test_should_reject_text_without_pronounced_characters() {
        for input in ["", "  ", "[[pause=300]]", "。，！"] {
            let err = job(input).prepare(&EngineConfig::default(), PathPolicy::LocalAbsolute);
            assert!(
                matches!(&err, Err(e) if e.code == ErrorCode::InvalidArgument),
                "{input:?}: {err:?}"
            );
        }
    }

    #[test]
    fn test_should_reject_conflicting_speakers_and_relative_paths() {
        let config = EngineConfig::default();
        let mut both = job("你好。");
        both.voice = Some("tyr".to_owned());
        both.voice_file = Some(PathBuf::from("/tmp/tyr.itts_voice"));
        assert!(both.prepare(&config, PathPolicy::LocalAbsolute).is_err());
        let relative = AlignJob::builder()
            .audio("line.wav")
            .input("你好。")
            .build();
        assert!(
            relative
                .prepare(&config, PathPolicy::LocalAbsolute)
                .is_err()
        );
        assert!(job("你好。").prepare(&config, PathPolicy::Denied).is_err());
    }

    #[test]
    fn test_should_reject_unknown_fields() {
        let parsed = from_value::<AlignJob>(json!({ "audio": "/a.wav", "input": "x", "seed": 1 }));
        assert!(parsed.is_err());
    }

    fn prepared(text: &str, phones: bool) -> PreparedAlign {
        PreparedAlign {
            audio: PathBuf::from("/tmp/line.wav"),
            text: text.to_owned(),
            language: index_tts_rs_core::config::Language::Zh,
            conditioning: index_tts_rs_core::align::Conditioning::Recording,
            phones,
            model: "yunfengwang/IndexTTS-2.5-mlx".to_owned(),
        }
    }

    fn span(chars: std::ops::Range<usize>, start: f32, end: f32) -> SpanTiming {
        let mut span = SpanTiming::default();
        span.chars = chars;
        span.start_secs = start;
        span.end_secs = end;
        span
    }

    #[test]
    fn test_should_assemble_a_bounded_alignment_with_weak_character_indices()
    -> serde_json::Result<()> {
        let mut output = AlignmentOutput::default();
        output.duration_secs = 1.0;
        output.spans = vec![span(0..1, 0.1, 0.4), span(1..2, 0.4, 1.2)];
        let mut quality = AlignmentQuality::default();
        quality.confidence = 0.8;
        quality.coverage = 0.5;
        quality.weak.push(1..3);
        output.quality = quality;
        let result = assemble(&prepared("你好。", false), &output, 0.05);
        assert_eq!(result.alignment.start.len(), 3);
        assert!(result.alignment.end.iter().all(|&t| t <= 1.0));
        assert!(
            result
                .alignment
                .start
                .windows(2)
                .all(|w| w.first() <= w.last())
        );
        assert_eq!(result.quality.weak, vec![1]);
        assert_eq!(result.aligner.heads, "L4H10+L5H10");
        assert_eq!(result.sentences.len(), 1);
        let value = to_value(&result)?;
        assert!(value.get("phones").is_none() && value.get("phonesMethod").is_none());
        assert_eq!(value.pointer("/aligner/name"), Some(&json!("itts-attn")));
        let with_phones = to_value(assemble(&prepared("你好。", true), &output, 0.05))?;
        assert_eq!(with_phones.get("phonesMethod"), Some(&json!("split")));
        Ok(())
    }
}

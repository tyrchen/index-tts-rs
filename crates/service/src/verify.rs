//! [`VerifyJob`]: check that a recording says its script (`itts verify`, daemon `verify`, MCP
//! `verify`), returning [`VerifyResult`].
//!
//! The recording is force-aligned to the script (the [`AlignJob`] path, on the MLX actor) and
//! recognized verbatim (the [`ListenJob`] path, on the ONNX Runtime actor) at the same time.
//! The script and the recognized units are then compared: each difference becomes a
//! [`Finding`] (repeat, insertion, omission, substitution, number), timed and graded with the
//! aligner's evidence. Running a job needs the `asr` feature; the types are always available.

use std::{
    ops::Range,
    path::{Path, PathBuf},
};

use index_tts_rs_asr::compare::{
    Finding as Difference, FindingKind as DifferenceKind, Severity as DifferenceSeverity,
};
use index_tts_rs_core::config::EngineConfig;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use crate::{
    align::{AlignJob, AlignResult, PreparedAlign},
    error::ServiceError,
    listen::{ListenJob, ListenResult, PreparedListen},
    request::{PathPolicy, resolve_input_path},
};

/// A verification request. `voice` and `voiceFile` (decoder conditioning for the alignment) are
/// mutually exclusive.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TypedBuilder,
)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[builder(field_defaults(default, setter(strip_option)))]
#[non_exhaustive]
pub struct VerifyJob {
    /// Recording (`.wav`/`.mp3`/`.flac`/`.m4a`, any sample rate, at most about 72 s).
    #[builder(!default, setter(!strip_option, into))]
    pub audio: PathBuf,
    /// Script the recording should say (at most 16 KiB and 600 tokens). `<字|pinyin>` /
    /// `<word|ARPAbet>` overrides are honoured; `[[...]]` controls are removed.
    #[builder(!default, setter(!strip_option, into))]
    pub input: String,
    /// Language: `zh`, `zh/en`, `yue`, `ja`, `en` (default from config).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub lang: Option<String>,
    /// Recognizer: `paraformer`, `sense-voice`, or `parakeet` (default per language, as in
    /// [`ListenJob::model`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub model: Option<String>,
    /// Registered voice whose snapshot conditions the aligner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(setter(into))]
    pub voice: Option<String>,
    /// `.itts_voice` snapshot file whose conditioning the aligner uses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_file: Option<PathBuf>,
}

/// A validated [`VerifyJob`]: its alignment and recognition halves.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreparedVerify {
    /// The forced alignment.
    pub align: PreparedAlign,
    /// The recognition.
    pub listen: PreparedListen,
}

impl VerifyJob {
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
    /// Returns `INVALID_ARGUMENT` for anything [`AlignJob::prepare`] or [`ListenJob::prepare`]
    /// rejects.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::config::EngineConfig;
    /// use index_tts_rs_service::{PathPolicy, VerifyJob};
    ///
    /// let job = VerifyJob::builder()
    ///     .audio("/tmp/line.wav")
    ///     .input("你好[[pause=300]]世界。")
    ///     .build();
    /// let prepared = job.prepare(&EngineConfig::default(), PathPolicy::LocalAbsolute)?;
    /// assert_eq!(prepared.align.text, "你好世界。");
    /// assert_eq!(prepared.listen.model.as_str(), "paraformer");
    /// # Ok::<(), index_tts_rs_service::ServiceError>(())
    /// ```
    pub fn prepare(
        &self,
        config: &EngineConfig,
        policy: PathPolicy,
    ) -> Result<PreparedVerify, ServiceError> {
        let mut align = AlignJob::builder()
            .audio(self.audio.clone())
            .input(self.input.clone())
            .build();
        align.lang.clone_from(&self.lang);
        align.voice.clone_from(&self.voice);
        align.voice_file.clone_from(&self.voice_file);
        let mut listen = ListenJob::builder().audio(self.audio.clone()).build();
        listen.lang.clone_from(&self.lang);
        listen.model.clone_from(&self.model);
        Ok(PreparedVerify {
            align: align.prepare(config, policy)?,
            listen: listen.prepare(config, policy)?,
        })
    }
}

/// Kind of a difference between the script and the recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum FindingKind {
    /// Extra units repeating their neighbours: a stutter or a re-read (`北北边`).
    Repeat,
    /// Extra units that are not a repeat.
    Insertion,
    /// Script units that were not heard.
    Omission,
    /// Script units heard as other units.
    Substitution,
    /// A number heard with another value (`3` read as `三十`).
    Number,
}

/// How much a finding matters; ordered `info < warning < error`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum Severity {
    /// One unit differs and the aligner does not doubt it: most likely the recognizer.
    Info,
    /// Two or more units differ, or one the aligner also finds weak.
    Warning,
    /// A repeat, or a number read as another value.
    Error,
}

/// Worst severity of a verification.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum Verdict {
    /// No finding.
    Clean,
    /// Only `info` findings.
    Info,
    /// At least one `warning`, no `error`.
    Warning,
    /// At least one `error`.
    Error,
}

impl FindingKind {
    /// Wire name (`repeat`, `insertion`, `omission`, `substitution`, `number`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Repeat => "repeat",
            Self::Insertion => "insertion",
            Self::Omission => "omission",
            Self::Substitution => "substitution",
            Self::Number => "number",
        }
    }
}

impl Severity {
    /// Wire name (`info`, `warning`, `error`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

impl Verdict {
    /// Wire name (`clean`, `info`, `warning`, `error`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

impl From<Severity> for Verdict {
    fn from(severity: Severity) -> Self {
        match severity {
            Severity::Info => Self::Info,
            Severity::Warning => Self::Warning,
            Severity::Error => Self::Error,
        }
    }
}

/// One difference between the script and the recording.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Finding {
    /// Kind of difference.
    pub kind: FindingKind,
    /// Severity.
    pub severity: Severity,
    /// Script text of the difference (empty for insertions and repeats).
    pub expected: String,
    /// Recognized text of the difference (empty for omissions).
    pub heard: String,
    /// Character range in `align.alignment.text` (`start == end` at an insertion point).
    pub chars: Range<usize>,
    /// Start in seconds.
    pub start: f32,
    /// End in seconds.
    pub end: f32,
    /// Units involved (a CJK character, a number, or a Latin word each count once).
    pub sounds: usize,
    /// Whether the aligner finds a unit in or next to the difference weakly supported.
    pub align_weak: bool,
}

impl From<Difference> for Finding {
    fn from(found: Difference) -> Self {
        Self {
            kind: match found.kind {
                DifferenceKind::Repeat => FindingKind::Repeat,
                DifferenceKind::Insertion => FindingKind::Insertion,
                DifferenceKind::Omission => FindingKind::Omission,
                DifferenceKind::Number => FindingKind::Number,
                _ => FindingKind::Substitution,
            },
            severity: match found.severity {
                DifferenceSeverity::Info => Severity::Info,
                DifferenceSeverity::Warning => Severity::Warning,
                _ => Severity::Error,
            },
            expected: found.expected,
            heard: found.heard,
            chars: found.chars,
            start: found.start,
            end: found.end,
            sounds: found.sounds,
            align_weak: found.align_weak,
        }
    }
}

/// Milliseconds spent in each stage (alignment and recognition run concurrently).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct VerifyTimings {
    /// Forced alignment.
    pub align_ms: f64,
    /// Recognition.
    pub listen_ms: f64,
    /// Script comparison.
    pub compare_ms: f64,
    /// Total.
    pub total_ms: f64,
}

/// Result of a [`VerifyJob`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct VerifyResult {
    /// Worst severity among the findings.
    pub verdict: Verdict,
    /// Differences in script order.
    pub findings: Vec<Finding>,
    /// What the recognizer heard.
    pub heard: ListenResult,
    /// Forced alignment of the script (character timings and quality, no phones).
    pub align: AlignResult,
    /// Recording duration in seconds.
    pub duration_secs: f32,
    /// Wall-clock seconds spent.
    pub verify_secs: f64,
    /// Per-stage milliseconds.
    pub timings: VerifyTimings,
}

impl VerifyResult {
    /// Assembles the result from both halves and the comparison.
    #[must_use]
    pub fn new(
        align: AlignResult,
        heard: ListenResult,
        differences: Vec<Difference>,
        timings: VerifyTimings,
    ) -> Self {
        let findings: Vec<Finding> = differences.into_iter().map(Finding::from).collect();
        let verdict = findings
            .iter()
            .map(|finding| Verdict::from(finding.severity))
            .max()
            .unwrap_or(Verdict::Clean);
        Self {
            verdict,
            findings,
            duration_secs: align.duration_secs,
            heard,
            align,
            verify_secs: timings.total_ms / 1000.0,
            timings,
        }
    }
}

#[cfg(feature = "asr")]
pub use run::run_verify;

#[cfg(feature = "asr")]
mod run {
    use std::time::Instant;

    use index_tts_rs_asr::{
        AsrEngine,
        compare::{AlignEvidence, compare},
    };
    use index_tts_rs_core::engine::IndexTtsEngine;

    use super::{PreparedVerify, VerifyResult, VerifyTimings};
    use crate::{align::run_align, error::ServiceError, listen::listen_tokens};

    /// Verifies a prepared job: aligns on `engine` and recognizes on `asr` concurrently, then
    /// compares.
    ///
    /// # Errors
    ///
    /// Returns any failure of [`run_align`] or [`crate::run_listen`].
    pub async fn run_verify(
        engine: &IndexTtsEngine,
        asr: &AsrEngine,
        prepared: PreparedVerify,
    ) -> Result<VerifyResult, ServiceError> {
        let started = Instant::now();
        let text = prepared.align.text.clone();
        let (aligned, heard) = tokio::join!(
            async {
                let t = Instant::now();
                run_align(engine, prepared.align)
                    .await
                    .map(|r| (r, t.elapsed()))
            },
            async {
                let t = Instant::now();
                listen_tokens(asr, prepared.listen)
                    .await
                    .map(|r| (r, t.elapsed()))
            },
        );
        let (aligned, align_time) = aligned?;
        let ((heard, tokens), listen_time) = heard?;
        let t_compare = Instant::now();
        let evidence = AlignEvidence {
            weak: &aligned.quality.weak,
            starts: &aligned.alignment.start,
            ends: &aligned.alignment.end,
        };
        let differences = compare(&text, &tokens, evidence)?;
        let compare_ms = t_compare.elapsed().as_secs_f64() * 1000.0;
        let timings = VerifyTimings {
            align_ms: align_time.as_secs_f64() * 1000.0,
            listen_ms: listen_time.as_secs_f64() * 1000.0,
            compare_ms,
            total_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
        Ok(VerifyResult::new(aligned, heard, differences, timings))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use index_tts_rs_asr::compare::{Finding as Difference, FindingKind, Severity};
    use index_tts_rs_core::config::EngineConfig;
    use serde_json::{from_value, json, to_value};

    use super::{Verdict, VerifyJob};
    use crate::{ErrorCode, PathPolicy};

    fn job(input: &str) -> VerifyJob {
        VerifyJob::builder()
            .audio("/tmp/line.wav")
            .input(input)
            .build()
    }

    #[test]
    fn test_should_prepare_both_halves_from_one_job() -> Result<(), crate::ServiceError> {
        let mut verify = job("[[speed=0.9]]Chapter 3 begins.");
        verify.lang = Some("en".to_owned());
        let prepared = verify.prepare(&EngineConfig::default(), PathPolicy::LocalAbsolute)?;
        assert_eq!(prepared.align.text, "Chapter 3 begins.");
        assert_eq!(prepared.listen.model.as_str(), "parakeet");
        assert!(!prepared.align.phones);
        Ok(())
    }

    #[test]
    fn test_should_reject_what_either_half_rejects() {
        let config = EngineConfig::default();
        let codes = |verify: VerifyJob| {
            verify
                .prepare(&config, PathPolicy::LocalAbsolute)
                .err()
                .map(|e| e.code)
        };
        assert_eq!(codes(job("")), Some(ErrorCode::InvalidArgument));
        let mut spanish = job("hola.");
        spanish.lang = Some("es".to_owned());
        assert_eq!(codes(spanish), Some(ErrorCode::InvalidArgument));
        let mut both = job("你好。");
        both.voice = Some("tyr".to_owned());
        both.voice_file = Some(PathBuf::from("/tmp/tyr.itts_voice"));
        assert_eq!(codes(both), Some(ErrorCode::InvalidArgument));
        assert!(
            from_value::<VerifyJob>(json!({ "audio": "/a.wav", "input": "x", "phones": true }))
                .is_err()
        );
    }

    #[test]
    fn test_should_take_the_worst_severity_as_verdict() -> serde_json::Result<()> {
        let difference = |kind, severity| Difference {
            kind,
            severity,
            expected: String::new(),
            heard: "北".to_owned(),
            chars: 1..1,
            start: 0.5,
            end: 0.75,
            sounds: 1,
            align_weak: false,
        };
        let verdict = |found: Vec<Difference>| {
            found
                .into_iter()
                .map(super::Finding::from)
                .map(|f| Verdict::from(f.severity))
                .max()
                .unwrap_or(Verdict::Clean)
        };
        assert_eq!(verdict(Vec::new()), Verdict::Clean);
        assert_eq!(
            to_value(Verdict::Warning)?,
            json!(Verdict::Warning.as_str())
        );
        assert_eq!(
            to_value(super::Severity::Info)?,
            json!(super::Severity::Info.as_str())
        );
        assert_eq!(
            to_value(super::FindingKind::Number)?,
            json!(super::FindingKind::Number.as_str())
        );
        assert_eq!(
            verdict(vec![
                difference(FindingKind::Omission, Severity::Info),
                difference(FindingKind::Repeat, Severity::Error),
            ]),
            Verdict::Error
        );
        let finding = super::Finding::from(difference(FindingKind::Repeat, Severity::Error));
        assert_eq!(
            to_value(&finding)?,
            json!({ "kind": "repeat", "severity": "error", "expected": "", "heard": "北",
                    "chars": { "start": 1, "end": 1 }, "start": 0.5, "end": 0.75, "sounds": 1,
                    "alignWeak": false })
        );
        Ok(())
    }
}

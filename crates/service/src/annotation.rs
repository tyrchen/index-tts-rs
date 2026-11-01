//! [`AnnotateJob`]: structured, per-unit delivery direction shared by the daemon, MCP, HTTP, and
//! `itts annotate --units`.
//!
//! The caller supplies units (for example one per script line) with stable IDs; the provider
//! returns only numbers per unit, which are range-checked and mapped back to those IDs. Source
//! text is never rewritten.

use std::{collections::HashSet, fmt};

use index_tts_rs_director::{Direction, Director, DirectorConfig, ReadingStyle};
use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::ServiceError;

/// Longest accepted unit ID, in bytes.
const MAX_UNIT_ID_BYTES: usize = 64;

/// A request to choose a delivery for each unit of a script.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[non_exhaustive]
pub struct AnnotateJob {
    /// Units in reading order (1..=128, at most 16 KiB of text in total).
    pub units: Vec<AnnotationUnit>,
    /// `natural` (narration, prose, dialogue; default) or `poetry`.
    #[serde(default)]
    pub style: ReadingStyle,
    /// Optional description of the piece (genre, audience, tone), at most 1 KiB. Passed to the
    /// provider as data, never as instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brief: Option<String>,
}

/// One unit of text to direct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
#[non_exhaustive]
pub struct AnnotationUnit {
    /// Caller identifier echoed in the result (`^[A-Za-z0-9_.:-]{1,64}$`, unique).
    pub id: String,
    /// Text of the unit, without `[[...]]` controls.
    pub text: String,
}

/// The delivery chosen for one unit, ready to copy into a [`crate::SpeechJob`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Delivery {
    /// The unit's `id`.
    pub id: String,
    /// Emotion mix in `SpeechJob.emo` syntax (`calm:0.4,happy:0.2`); absent for the voice's base
    /// emotion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emo: Option<String>,
    /// Sampling temperature for `SpeechJob.advanced.temperature` (`0.5..=1.2`), if chosen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Speaking rate for `SpeechJob.speed` (`0.8..=1.2`).
    pub speed: f64,
    /// Suggested silence after the unit in milliseconds (`0..=2000`; `0` for the last unit).
    pub pause_after_ms: u32,
    /// The same delivery as inline controls to prefix the unit's text with.
    pub controls: String,
}

/// Result of an [`AnnotateJob`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct AnnotateResult {
    /// One delivery per unit, in the request's order.
    pub deliveries: Vec<Delivery>,
}

/// A provider API key supplied with a request. `Debug` is redacted; the key is only exposed
/// to authenticate the provider call.
#[derive(Clone, JsonSchema)]
#[schemars(with = "String")]
pub struct ApiKey(SecretString);

impl ApiKey {
    /// Wraps a key.
    #[must_use]
    pub fn new(key: impl Into<String>) -> Self {
        Self(SecretString::from(key.into()))
    }

    /// The key as a [`SecretString`] for provider clients.
    #[must_use]
    pub fn into_secret(self) -> SecretString {
        self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey([REDACTED])")
    }
}

impl Serialize for ApiKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0.expose_secret())
    }
}

impl<'de> Deserialize<'de> for ApiKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::new)
    }
}

impl AnnotateJob {
    /// Validates the units and asks the provider for one delivery per unit. `key` overrides
    /// `DEEPSEEK_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ErrorCode::InvalidArgument`] for malformed units or brief,
    /// [`crate::ErrorCode::AnnotationUnavailable`] without a usable key, and
    /// [`crate::ErrorCode::AnnotationFailed`] for provider failures.
    pub async fn run(
        &self,
        config: DirectorConfig,
        key: Option<&ApiKey>,
    ) -> Result<AnnotateResult, ServiceError> {
        self.validate()?;
        let director = match key {
            Some(key) => Director::new(config, key.clone().into_secret())?,
            None => Director::from_env(config)?,
        };
        let texts: Vec<&str> = self.units.iter().map(|unit| unit.text.as_str()).collect();
        let directions = director
            .direct_units(&texts, self.style, self.brief.as_deref())
            .await?;
        Ok(AnnotateResult {
            deliveries: self
                .units
                .iter()
                .zip(directions)
                .map(|(unit, direction)| Delivery::new(unit.id.clone(), &direction))
                .collect(),
        })
    }

    /// Checks unit IDs (pattern and uniqueness); text limits are enforced by the director.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ErrorCode::InvalidArgument`] naming the first offending unit.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_service::AnnotateJob;
    ///
    /// let job: AnnotateJob = serde_json::from_str(
    ///     r#"{"units":[{"id":"s1.l1","text":"你好。"},{"id":"s1.l1","text":"再见。"}]}"#,
    /// )?;
    /// assert!(job.validate().is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn validate(&self) -> Result<(), ServiceError> {
        let mut seen = HashSet::with_capacity(self.units.len());
        for unit in &self.units {
            let valid = (1..=MAX_UNIT_ID_BYTES).contains(&unit.id.len())
                && unit
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-'));
            if !valid {
                return Err(ServiceError::invalid(format!(
                    "unit id `{}` must match ^[A-Za-z0-9_.:-]{{1,64}}$",
                    unit.id.chars().take(64).collect::<String>()
                )));
            }
            if !seen.insert(unit.id.as_str()) {
                return Err(ServiceError::invalid(format!(
                    "unit id `{}` is not unique",
                    unit.id
                )));
            }
        }
        Ok(())
    }
}

impl Delivery {
    fn new(id: String, direction: &Direction) -> Self {
        Self {
            id,
            emo: direction.emotions.map(|emo| emo.format_compact()),
            temperature: direction.temperature.map(hundredths),
            speed: hundredths(direction.speed),
            pause_after_ms: direction.pause_after_ms,
            controls: direction.controls(),
        }
    }
}

/// Rounds to two decimals, the precision the provider works in, so JSON shows `0.94`.
fn hundredths(value: f32) -> f64 {
    (f64::from(value) * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use serde_json::{from_str, to_string};

    use super::{AnnotateJob, ApiKey};
    use crate::ErrorCode;

    #[test]
    fn test_should_redact_api_key_debug_output() {
        let key = ApiKey::new("sk-private");
        assert!(!format!("{key:?}").contains("sk-private"));
    }

    #[test]
    fn test_should_reject_malformed_unit_ids() -> serde_json::Result<()> {
        let job: AnnotateJob = from_str(r#"{"units":[{"id":"bad id","text":"你好。"}]}"#)?;
        let err = job.validate().err();
        assert!(matches!(err, Some(e) if e.code == ErrorCode::InvalidArgument));
        Ok(())
    }

    #[test]
    fn test_should_default_to_natural_style_and_round_trip() -> serde_json::Result<()> {
        let job: AnnotateJob =
            from_str(r#"{"units":[{"id":"a","text":"你好。"}],"brief":"news"}"#)?;
        assert!(job.validate().is_ok());
        assert_eq!(
            to_string(&job)?,
            r#"{"units":[{"id":"a","text":"你好。"}],"style":"natural","brief":"news"}"#
        );
        Ok(())
    }
}

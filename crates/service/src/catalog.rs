//! Self-description for agents: inline control syntax, emotions, languages, parameter ranges,
//! error codes, and JSON Schemas. Served by `itts schema` and the MCP `describe_controls` tool.

use index_tts_rs_core::config::EMOTION_NAMES;
use schemars::{JsonSchema, Schema, schema_for};
use serde::Serialize;
use serde_json::{Value, to_value};

use crate::{
    error::{ErrorCode, ServiceError},
    events::JobEvent,
    request::{MAX_SILENCE_MS, MAX_SPEED, MIN_SPEED, SpeechJob},
};

/// Chinese aliases accepted for each emotion, in [`EMOTION_NAMES`] order.
const EMOTION_ALIASES: [&str; 8] = [
    "高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "平静",
];

/// Inline control tag description.
#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ControlSpec {
    /// Tag syntax.
    pub syntax: &'static str,
    /// Effect and accepted values.
    pub description: &'static str,
}

/// Emotion class description.
#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EmotionSpec {
    /// Canonical English name.
    pub name: &'static str,
    /// Primary Chinese alias.
    pub alias: &'static str,
}

/// Error code description.
#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ErrorSpec {
    /// Wire code.
    pub code: ErrorCode,
    /// CLI exit code.
    pub exit_code: u8,
    /// Meaning.
    pub description: &'static str,
}

/// Everything an agent needs to write a correct [`SpeechJob`].
#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Catalog {
    /// Inline control tags usable inside `input`.
    pub controls: Vec<ControlSpec>,
    /// Emotion classes for `emo` and `[[emo=...]]` (weights `0.0..=1.0`).
    pub emotions: Vec<EmotionSpec>,
    /// Accepted `lang` codes.
    pub languages: Vec<&'static str>,
    /// Speaking rate range for `speed` and `[[speed=...]]`.
    pub speed_range: [f32; 2],
    /// Upper bound for pauses and silences in milliseconds.
    pub max_silence_ms: u32,
    /// Stable error codes.
    pub errors: Vec<ErrorSpec>,
}

/// Builds the control catalog.
///
/// ```
/// let catalog = index_tts_rs_service::catalog();
/// assert_eq!(catalog.emotions.len(), 8);
/// assert!(catalog.controls.iter().any(|c| c.syntax.starts_with("[[pause")));
/// ```
#[must_use]
pub fn catalog() -> Catalog {
    let controls = vec![
        ControlSpec {
            syntax: "[[pause=MS]]",
            description: "Insert MS (0..=5000) milliseconds of silence before the next text.",
        },
        ControlSpec {
            syntax: "[[speed=X]]",
            description: "Speaking rate 0.25..=4.0 (1.0 normal, larger is faster) until changed.",
        },
        ControlSpec {
            syntax: "[[emo=NAME:W,...]]",
            description: "Emotion mix with weights 0.0..=1.0, e.g. melancholic:0.45,calm:0.25; \
                          `none` clears it.",
        },
        ControlSpec {
            syntax: "[[temp=X]]",
            description: "GPT sampling temperature 0.05..=2.0; lower is steadier, higher is more \
                          expressive.",
        },
        ControlSpec {
            syntax: "[[voice=ID]]",
            description: "Switch to another registered voice.",
        },
        ControlSpec {
            syntax: "[[lang=CODE]]",
            description: "Switch language (see `languages`).",
        },
        ControlSpec {
            syntax: "[[reset]]",
            description: "Restore the request's original speed, emotion, temperature, voice, and \
                          language.",
        },
    ];
    let emotions = EMOTION_NAMES
        .iter()
        .zip(EMOTION_ALIASES)
        .map(|(&name, alias)| EmotionSpec { name, alias })
        .collect();
    let errors = ErrorCode::ALL
        .iter()
        .map(|&code| ErrorSpec {
            code,
            exit_code: code.exit_code(),
            description: code.description(),
        })
        .collect();
    Catalog {
        controls,
        emotions,
        languages: vec!["zh", "en", "ja", "yue", "es", "ar", "zh/en"],
        speed_range: [MIN_SPEED, MAX_SPEED],
        max_silence_ms: MAX_SILENCE_MS,
        errors,
    }
}

/// Schemas published by `itts schema`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SchemaKind {
    /// The [`SpeechJob`] request.
    SpeechJob,
    /// The [`JobEvent`] stream.
    Events,
    /// The [`ServiceError`] body.
    Error,
}

/// Returns the JSON Schema for `kind`.
///
/// ```
/// use index_tts_rs_service::{SchemaKind, json_schema};
/// let schema = json_schema(SchemaKind::SpeechJob);
/// assert!(schema.get("properties").and_then(|p| p.get("input")).is_some());
/// ```
#[must_use]
pub fn json_schema(kind: SchemaKind) -> Value {
    let schema: Schema = match kind {
        SchemaKind::SpeechJob => schema_for!(SpeechJob),
        SchemaKind::Events => schema_for!(JobEvent),
        SchemaKind::Error => schema_for!(ServiceError),
    };
    to_value(schema).unwrap_or(Value::Null)
}

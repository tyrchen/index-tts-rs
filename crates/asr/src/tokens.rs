//! Recognizer tokens → heard units: one per CJK character, one per Latin word or number, with
//! approximate start / end times. Punctuation and special tokens are dropped.

use crate::models::AsrModel;

/// Longest time a token is taken to last when the recognizer reports only its start: a pause
/// after a character does not belong to it.
pub const MAX_TOKEN_SECS: f32 = 0.5;

/// One recognized unit: a CJK character, or a Latin word / number.
#[derive(Debug, Clone, PartialEq)]
pub struct HeardToken {
    /// Text as recognized.
    pub text: String,
    /// Start in seconds.
    pub start: f32,
    /// End in seconds.
    pub end: f32,
}

/// One token as the recognizer emits it.
#[derive(Debug, Clone, PartialEq)]
pub struct RawToken {
    /// Token text (BPE pieces carry a leading space or a trailing `@@`).
    pub text: String,
    /// Start in seconds.
    pub start: f32,
    /// Duration in seconds, when the model reports one.
    pub duration: Option<f32>,
}

/// How a model marks Latin word boundaries in its pieces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Joining {
    /// A piece starting with a space (or `▁`) begins a word.
    LeadingSpace,
    /// A piece ending in `@@` continues into the next one.
    ContinuationSuffix,
}

impl Joining {
    const fn of(model: AsrModel) -> Self {
        match model {
            AsrModel::Paraformer => Self::ContinuationSuffix,
            _ => Self::LeadingSpace,
        }
    }
}

/// Whether `c` is a CJK character, kana or hangul (one unit per character).
#[must_use]
pub fn is_cjk(c: char) -> bool {
    matches!(
        u32::from(c),
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF
    )
}

/// Whether `c` belongs to a Latin word or number.
fn is_wordish(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '\'' | '.' | '%')
}

/// Converts the tokens of `model` into heard units within `[0, duration]`.
///
/// # Examples
///
/// ```
/// use index_tts_rs_asr::{AsrModel, RawToken, heard_tokens};
///
/// let raw = [("从", 0.1), ("北", 0.4), (" en", 0.7), ("gine", 0.8)]
///     .map(|(text, start)| RawToken { text: text.to_owned(), start, duration: None });
/// let heard = heard_tokens(AsrModel::SenseVoice, &raw, 2.0);
/// let texts: Vec<_> = heard.iter().map(|t| t.text.as_str()).collect();
/// assert_eq!(texts, ["从", "北", "engine"]);
/// assert!((heard[0].end - 0.4).abs() < 1e-6);
/// ```
#[must_use]
pub fn heard_tokens(model: AsrModel, raw: &[RawToken], duration: f32) -> Vec<HeardToken> {
    let joining = Joining::of(model);
    let ends = raw_ends(raw, duration);
    let mut out: Vec<HeardToken> = Vec::with_capacity(raw.len());
    let mut open_word = false;
    let mut continues = false;
    for (token, &end) in raw.iter().zip(&ends) {
        let start = token.start.clamp(0.0, duration);
        let end = end.clamp(start, duration);
        let (body, starts_word) = match joining {
            Joining::LeadingSpace => {
                let body = token.text.trim_start_matches([' ', '\u{2581}']);
                (body, body.len() != token.text.len())
            }
            Joining::ContinuationSuffix => {
                let starts = !continues;
                continues = token.text.ends_with("@@");
                (token.text.trim_end_matches("@@"), starts)
            }
        };
        if starts_word || body.starts_with('<') {
            open_word = false;
        }
        if body.starts_with('<') && body.ends_with('>') {
            continue;
        }
        push_pieces(&mut out, &mut open_word, body, start, end);
    }
    trim_words(out)
}

/// End of each raw token: start + duration when reported, else the next start, capped at
/// [`MAX_TOKEN_SECS`].
fn raw_ends(raw: &[RawToken], duration: f32) -> Vec<f32> {
    raw.iter()
        .enumerate()
        .map(|(i, token)| match token.duration {
            Some(d) if d > 0.0 => token.start + d,
            _ => {
                let next = raw.get(i + 1).map_or(duration, |t| t.start);
                next.min(token.start + MAX_TOKEN_SECS)
            }
        })
        .collect()
}

/// Splits one token body into CJK characters (sharing its span evenly) and Latin pieces (joined
/// to the open word unless a boundary intervenes).
#[allow(
    clippy::cast_precision_loss,
    reason = "a token holds a handful of characters"
)]
fn push_pieces(out: &mut Vec<HeardToken>, open_word: &mut bool, body: &str, start: f32, end: f32) {
    let cjk = body.chars().filter(|&c| is_cjk(c)).count();
    let step = if cjk > 0 {
        (end - start) / cjk as f32
    } else {
        0.0
    };
    let mut k = 0usize;
    let mut word = String::new();
    for c in body.chars() {
        if is_cjk(c) {
            flush(out, open_word, &mut word, start, end);
            let at = start + step * k as f32;
            out.push(HeardToken {
                text: c.to_string(),
                start: at,
                end: at + step,
            });
            *open_word = false;
            k += 1;
        } else if is_wordish(c) {
            word.push(c);
        } else {
            flush(out, open_word, &mut word, start, end);
            *open_word = false;
        }
    }
    flush(out, open_word, &mut word, start, end);
}

/// Appends the Latin piece in `word` to the open word, or starts a new one. Dots and apostrophes
/// stay inside a word (`5.4.0`, `he's`); [`trim_words`] drops them at its edges.
fn flush(out: &mut Vec<HeardToken>, open_word: &mut bool, word: &mut String, start: f32, end: f32) {
    if word.is_empty() {
        return;
    }
    match out.last_mut() {
        Some(last) if *open_word => {
            last.text.push_str(word);
            last.end = end;
        }
        _ => out.push(HeardToken {
            text: word.clone(),
            start,
            end,
        }),
    }
    *open_word = true;
    word.clear();
}

/// Removes sentence dots and quotes around Latin words, and words left with nothing else.
fn trim_words(tokens: Vec<HeardToken>) -> Vec<HeardToken> {
    tokens
        .into_iter()
        .filter_map(|mut token| {
            let trimmed = token.text.trim_matches(['.', '\'']);
            if trimmed.is_empty() {
                return None;
            }
            if trimmed.len() != token.text.len() {
                token.text = trimmed.to_owned();
            }
            Some(token)
        })
        .collect()
}

/// The heard units as one string: CJK characters run together, Latin words separated by spaces.
#[must_use]
pub fn heard_text(tokens: &[HeardToken]) -> String {
    let mut text = String::new();
    for token in tokens {
        let joins = text
            .chars()
            .last()
            .is_some_and(|c| c.is_ascii_alphanumeric())
            && token
                .text
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric());
        if joins {
            text.push(' ');
        }
        text.push_str(&token.text);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::{AsrModel, MAX_TOKEN_SECS, RawToken, heard_text, heard_tokens};

    fn raw(items: &[(&str, f32)]) -> Vec<RawToken> {
        items
            .iter()
            .map(|&(text, start)| RawToken {
                text: text.to_owned(),
                start,
                duration: None,
            })
            .collect()
    }

    fn texts(model: AsrModel, items: &[(&str, f32)]) -> Vec<String> {
        heard_tokens(model, &raw(items), 10.0)
            .into_iter()
            .map(|t| t.text)
            .collect()
    }

    #[test]
    fn test_should_join_leading_space_pieces_into_words() {
        let got = texts(
            AsrModel::SenseVoice,
            &[
                (" ten", 0.0),
                ("or", 0.1),
                (" r", 0.2),
                (" ", 0.3),
                ("的", 0.4),
            ],
        );
        assert_eq!(got, ["tenor", "r", "的"]);
    }

    #[test]
    fn test_should_join_paraformer_continuations() {
        let got = texts(
            AsrModel::Paraformer,
            &[
                ("ten@@", 0.0),
                ("s@@", 0.1),
                ("or", 0.2),
                ("r", 0.3),
                ("的", 0.4),
            ],
        );
        assert_eq!(got, ["tensor", "r", "的"]);
    }

    #[test]
    fn test_should_drop_punctuation_and_keep_digits_together() {
        let got = texts(
            AsrModel::Parakeet,
            &[
                (" In", 0.0),
                (" ", 0.2),
                ("2", 0.3),
                ("0", 0.35),
                ("2", 0.4),
                ("6", 0.45),
                (",", 0.5),
            ],
        );
        assert_eq!(got, ["In", "2026"]);
        let got = texts(
            AsrModel::Parakeet,
            &[(" by", 0.0), (" 50", 0.2), ("%.", 0.3)],
        );
        assert_eq!(got, ["by", "50%"]);
    }

    #[test]
    fn test_should_split_multi_character_tokens_evenly_and_cap_durations() {
        let heard = heard_tokens(
            AsrModel::SenseVoice,
            &raw(&[("梯田", 1.0), ("水", 3.0)]),
            4.0,
        );
        let [first, second, third] = heard.as_slice() else {
            unreachable!("expected 3 tokens, got {}", heard.len());
        };
        assert!((first.end - (1.0 + MAX_TOKEN_SECS / 2.0)).abs() < 1e-6);
        assert!((second.start - first.end).abs() < 1e-6);
        assert!(third.end <= 3.0 + MAX_TOKEN_SECS + 1e-6);
        assert!(heard.is_sorted_by(|a, b| a.start <= b.start));
    }

    #[test]
    fn test_should_use_reported_durations_and_clamp_to_the_recording() {
        let tokens = [RawToken {
            text: " hot".to_owned(),
            start: 1.9,
            duration: Some(0.4),
        }];
        let heard = heard_tokens(AsrModel::Parakeet, &tokens, 2.0);
        assert!(heard.first().is_some_and(|t| (t.end - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_should_render_heard_text() {
        let heard = heard_tokens(
            AsrModel::SenseVoice,
            &raw(&[("从", 0.0), (" v", 0.1), (" l", 0.2), ("的", 0.3)]),
            1.0,
        );
        assert_eq!(heard_text(&heard), "从v l的");
    }
}

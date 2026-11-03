//! Phone timings inside aligned characters and words (`split`, `specs/itts-align-design.md` §7).
//!
//! The attention aligner only times characters (Chinese) and sub-words (English). Phones are cut
//! inside those intervals: a Mandarin syllable splits into its initial and final, the initial
//! ending where low-frequency (voicing) energy starts for voiceless initials or at a quarter of
//! the syllable for sonorants; an English word's `ARPAbet` phones share its interval by typical
//! duration. Pronunciations come from `<字|pinyin>` / `<word|ARPAbet>` overrides first, then the
//! `pinyin` crate's default reading or CMU Pronouncing Dictionary, then spelling rules.

use std::{collections::HashMap, ops::Range, sync::LazyLock};

use pinyin::ToPinyin;
use serde::{Deserialize, Serialize};

use crate::{
    align::{Interval, TextMap, is_punctuation},
    frontend::has_cjk_ideograph,
};

/// CMU Pronouncing Dictionary (first pronunciation per word, stress digits removed); see
/// `assets/CMUDICT-LICENSE` and `make update-cmudict`.
static CMUDICT_SOURCE: &str = include_str!("../../assets/cmudict.dict");

static CMUDICT: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    CMUDICT_SOURCE
        .lines()
        .filter_map(|line| line.split_once(' '))
        .collect()
});

/// Mandarin initials, longest first so `zh` wins over `z`.
const INITIALS: [&str; 21] = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r",
    "z", "c", "s",
];

/// Finals in the written form used after an initial (`ü` as `v`), plus syllabic nasals.
const FINALS: [&str; 41] = [
    "a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou", "iu", "ie", "ve", "er", "an", "en",
    "in", "un", "vn", "ang", "eng", "ing", "ong", "ia", "iao", "ian", "iang", "iong", "ua", "uo",
    "uai", "uan", "uang", "ueng", "van", "io", "ue", "m", "n", "ng",
];

/// Initials sounded with the vocal folds vibrating; their end is not an energy onset.
const SONORANT_INITIALS: [&str; 4] = ["m", "n", "l", "r"];

/// Low-pass cutoff isolating voicing energy, in Hz.
const LOW_PASS_HZ: f32 = 1_000.0;
/// Envelope hop (10 ms); each frame's RMS window spans two hops.
const ENVELOPE_HOP_SECS: f32 = 0.01;
/// A voiceless initial ends where the low-frequency envelope first exceeds this share of the
/// syllable's peak.
const ONSET_RELATIVE: f32 = 0.3;
/// A sonorant initial takes this share of its syllable.
const SONORANT_SHARE: f32 = 0.25;
/// Longest share of a syllable an initial may take.
const MAX_INITIAL_SHARE: f32 = 0.6;
/// Shortest initial or final, in seconds (capped at a quarter of the syllable).
const MIN_PART_SECS: f32 = 0.02;

/// Kind of a phone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PhoneKind {
    /// Mandarin initial (`ini`).
    #[serde(rename = "ini")]
    Initial,
    /// Mandarin final (`fin`).
    #[serde(rename = "fin")]
    Final,
    /// English `ARPAbet` phone (`ph`).
    #[serde(rename = "ph")]
    Phone,
}

/// One timed phone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PhoneTiming {
    /// Request-text character (Unicode scalar index) the phone belongs to; every phone of an
    /// English word uses the word's first character.
    #[serde(rename = "i")]
    pub char_index: usize,
    /// Initial or final in pinyin (`ü` as `v`), or `ARPAbet` without stress digits.
    #[serde(rename = "p")]
    pub phone: String,
    /// Tone `1..=5` (`5` neutral) of a Mandarin final.
    #[serde(rename = "t", default, skip_serializing_if = "Option::is_none")]
    pub tone: Option<u8>,
    /// Phone kind.
    #[serde(rename = "k")]
    pub kind: PhoneKind,
    /// Start in seconds.
    pub start: f32,
    /// End in seconds.
    pub end: f32,
}

/// A Mandarin syllable split into initial, final, and tone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Syllable {
    /// Initial, absent for zero-initial syllables.
    pub initial: Option<&'static str>,
    /// Final in written form (`ü` as `v`, `y`/`w` spellings restored: `you` → `iu`).
    pub final_: String,
    /// Tone `1..=5`.
    pub tone: u8,
}

/// Splits one pinyin syllable (`zhong1`, `JV4`, `lü4`, `de`) into initial, final, and tone.
///
/// Returns `None` when the syllable is not valid pinyin.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::align::phones::split_syllable;
///
/// let s = split_syllable("you3").ok_or("not pinyin")?;
/// assert_eq!((s.initial, s.final_.as_str(), s.tone), (None, "iu", 3));
/// let s = split_syllable("JV4").ok_or("not pinyin")?;
/// assert_eq!((s.initial, s.final_.as_str()), (Some("j"), "v"));
/// assert_eq!(split_syllable("de").map(|s| s.tone), Some(5));
/// # Ok::<(), &str>(())
/// ```
#[must_use]
pub fn split_syllable(raw: &str) -> Option<Syllable> {
    let lower = raw
        .trim()
        .to_lowercase()
        .replace('ü', "v")
        .replace("u:", "v");
    let (body, tone) = match lower.char_indices().last() {
        Some((at, digit @ '1'..='5')) => {
            (lower.get(..at)?, u8::try_from(digit.to_digit(10)?).ok()?)
        }
        _ => (lower.as_str(), 5),
    };
    if body.is_empty() || !body.bytes().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    let (initial, rest) = INITIALS
        .iter()
        .find_map(|&initial| {
            body.strip_prefix(initial)
                .filter(|rest| !rest.is_empty())
                .map(|rest| (Some(initial), rest))
        })
        .unwrap_or((None, body));
    let final_ = match initial {
        None => zero_initial_final(rest),
        Some("j" | "q" | "x") => match rest.strip_prefix('u') {
            Some(after) => format!("v{after}"),
            None => rest.to_owned(),
        },
        Some(_) => rest.to_owned(),
    };
    FINALS.contains(&final_.as_str()).then_some(Syllable {
        initial,
        final_,
        tone,
    })
}

/// Restores the final of a `y`/`w`-spelled or bare zero-initial syllable.
fn zero_initial_final(body: &str) -> String {
    if let Some(rest) = body.strip_prefix('y') {
        return match rest {
            "u" | "ue" | "uan" | "un" => format!("v{}", rest.get(1..).unwrap_or_default()),
            "ou" => "iu".to_owned(),
            _ if rest.starts_with('i') => rest.to_owned(),
            _ => format!("i{rest}"),
        };
    }
    if let Some(rest) = body.strip_prefix('w') {
        return match rest {
            "u" => "u".to_owned(),
            "ei" => "ui".to_owned(),
            "en" => "un".to_owned(),
            _ => format!("u{rest}"),
        };
    }
    body.to_owned()
}

/// `ARPAbet` phones (no stress digits) of a lower-case English word: CMU Pronouncing Dictionary
/// first, spelling rules otherwise.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::align::phones::word_phones;
///
/// assert_eq!(word_phones("rust"), ["R", "AH", "S", "T"]);
/// assert!(!word_phones("zxqvbn").is_empty());
/// ```
#[must_use]
pub fn word_phones(word: &str) -> Vec<String> {
    if let Some(entry) = CMUDICT.get(word) {
        return entry.split_whitespace().map(str::to_owned).collect();
    }
    spelled_phones(word)
}

/// Approximate letter-to-sound rules for words missing from the dictionary.
fn spelled_phones(word: &str) -> Vec<String> {
    const GRAPHEMES: [(&str, &[&str]); 37] = [
        ("tch", &["CH"]),
        ("igh", &["AY"]),
        ("sch", &["S", "K"]),
        ("th", &["TH"]),
        ("sh", &["SH"]),
        ("ch", &["CH"]),
        ("ph", &["F"]),
        ("ng", &["NG"]),
        ("ck", &["K"]),
        ("qu", &["K", "W"]),
        ("wh", &["W"]),
        ("ee", &["IY"]),
        ("ea", &["IY"]),
        ("oo", &["UW"]),
        ("ou", &["AW"]),
        ("ow", &["OW"]),
        ("oi", &["OY"]),
        ("oy", &["OY"]),
        ("ai", &["EY"]),
        ("ay", &["EY"]),
        ("au", &["AO"]),
        ("aw", &["AO"]),
        ("er", &["ER"]),
        ("ir", &["ER"]),
        ("ur", &["ER"]),
        ("ar", &["AA", "R"]),
        ("or", &["AO", "R"]),
        ("a", &["AE"]),
        ("e", &["EH"]),
        ("i", &["IH"]),
        ("o", &["AA"]),
        ("u", &["AH"]),
        ("c", &["K"]),
        ("q", &["K"]),
        ("x", &["K", "S"]),
        ("j", &["JH"]),
        ("h", &["HH"]),
    ];
    let letters: String = word.chars().filter(char::is_ascii_lowercase).collect();
    // A final silent `e` after a consonant (`make`) is not pronounced.
    let spoken = match letters.strip_suffix('e') {
        Some(stem) if stem.len() > 1 && stem.ends_with(|c: char| !"aeiouy".contains(c)) => stem,
        _ => letters.as_str(),
    };
    let mut phones = Vec::new();
    let mut rest = spoken;
    while let Some(first) = rest.chars().next() {
        if let Some((grapheme, sounds)) = GRAPHEMES.iter().find(|(g, _)| rest.starts_with(g)) {
            phones.extend(sounds.iter().map(|&s| s.to_owned()));
            rest = rest.get(grapheme.len()..).unwrap_or_default();
            continue;
        }
        let sound = match first {
            'y' if rest.len() == spoken.len() => "Y",
            'y' => "IY",
            other => {
                phones.push(other.to_ascii_uppercase().to_string());
                rest = rest.get(1..).unwrap_or_default();
                continue;
            }
        };
        phones.push(sound.to_owned());
        rest = rest.get(1..).unwrap_or_default();
    }
    phones
}

/// Typical relative duration of an `ARPAbet` phone class.
fn phone_weight(phone: &str) -> f32 {
    match phone {
        "AW" | "AY" | "EY" | "OW" | "OY" => 1.3,
        "AA" | "AE" | "AH" | "AO" | "EH" | "ER" | "IH" | "IY" | "UH" | "UW" => 1.0,
        "CH" | "JH" | "F" | "V" | "TH" | "DH" | "S" | "Z" | "SH" | "ZH" | "HH" => 0.8,
        "M" | "N" | "NG" | "L" | "R" | "W" | "Y" => 0.6,
        "P" | "B" | "T" | "D" | "K" | "G" => 0.5,
        _ => 0.7,
    }
}

/// The normalized text a segment was cut from and how it maps onto the request text.
#[derive(Debug, Clone)]
pub struct SpokenText<'a> {
    /// Frontend-normalized text.
    pub normalized: &'a str,
    /// Byte range of the aligned segment within [`Self::normalized`].
    pub body: Range<usize>,
    /// Normalized text onto the request text.
    pub map: &'a TextMap,
    /// The request text.
    pub source: &'a str,
}

/// A pronounceable unit of normalized text.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Unit {
    /// One CJK ideograph.
    Hanzi(Range<usize>),
    /// A run of Latin letters.
    Word(Range<usize>),
    /// A `<|SPECIAL_TOKEN_n|>PRON<|SPECIAL_TOKEN_n|>` override: `whole` includes the markers.
    Override {
        whole: Range<usize>,
        pron: Range<usize>,
        pinyin: bool,
    },
}

/// Splits every aligned syllable and word of `text` into timed phones.
///
/// `tokens` are the normalized byte range and interval of every spoken token; `pcm` is the
/// recording they time.
#[must_use]
pub fn split_phones(
    text: &SpokenText<'_>,
    tokens: &[(Range<usize>, Interval)],
    pcm: &[i16],
    sample_rate: u32,
) -> Vec<PhoneTiming> {
    let timed = char_times(text.normalized, tokens);
    let envelope = low_frequency_envelope(pcm, sample_rate);
    let mut phones = Vec::new();
    for unit in units(text.normalized, text.body.clone()) {
        match unit {
            Unit::Hanzi(bytes) => {
                let (Some(interval), Some(syllable)) = (
                    span_of(&timed, &bytes),
                    text.normalized
                        .get(bytes.clone())
                        .and_then(|s| s.chars().next())
                        .and_then(|ch| ch.to_pinyin())
                        .and_then(|p| split_syllable(p.with_tone_num_end())),
                ) else {
                    continue;
                };
                let at = text.map.source_chars(bytes).start;
                push_syllable(&mut phones, at, &syllable, interval, &envelope);
            }
            Unit::Word(bytes) => {
                let (Some(interval), Some(word)) =
                    (span_of(&timed, &bytes), text.normalized.get(bytes.clone()))
                else {
                    continue;
                };
                let at = text.map.source_chars(bytes).start;
                push_word(&mut phones, at, &word_phones(word), interval);
            }
            Unit::Override {
                whole,
                pron,
                pinyin,
            } => {
                let (Some(interval), Some(spelled)) =
                    (span_of(&timed, &pron), text.normalized.get(pron))
                else {
                    continue;
                };
                let word = override_word(text.source, text.map.source_chars(whole));
                if pinyin {
                    push_override_syllables(&mut phones, &word, spelled, interval, &envelope);
                } else {
                    let arpabet: Vec<String> = spelled
                        .split_whitespace()
                        .map(|p| p.trim_end_matches(|c: char| c.is_ascii_digit()).to_owned())
                        .filter(|p| !p.is_empty())
                        .collect();
                    push_word(&mut phones, word.start, &arpabet, interval);
                }
            }
        }
    }
    phones
}

/// Characters of the overridden word: inside `<word|PRON>` when the request text spells the
/// override, otherwise the whole mapped range.
fn override_word(source: &str, chars: Range<usize>) -> Range<usize> {
    let spelled: Vec<char> = source.chars().skip(chars.start).take(chars.len()).collect();
    match (spelled.first(), spelled.iter().position(|&c| c == '|')) {
        (Some('<'), Some(bar)) if bar > 1 => chars.start + 1..chars.start + bar,
        _ => chars,
    }
}

fn push_override_syllables(
    phones: &mut Vec<PhoneTiming>,
    word: &Range<usize>,
    spelled: &str,
    interval: Interval,
    envelope: &Envelope,
) {
    let syllables: Vec<Syllable> = spelled
        .split_whitespace()
        .filter_map(split_syllable)
        .collect();
    #[allow(clippy::cast_precision_loss, reason = "syllable counts are tiny")]
    let step = (interval.end - interval.start) / syllables.len().max(1) as f32;
    for (k, syllable) in syllables.iter().enumerate() {
        #[allow(clippy::cast_precision_loss, reason = "syllable counts are tiny")]
        let start = interval.start + step * k as f32;
        let at = (word.start + k).min(word.end.saturating_sub(1).max(word.start));
        push_syllable(
            phones,
            at,
            syllable,
            Interval {
                start,
                end: start + step,
            },
            envelope,
        );
    }
}

fn push_syllable(
    phones: &mut Vec<PhoneTiming>,
    at: usize,
    syllable: &Syllable,
    interval: Interval,
    envelope: &Envelope,
) {
    let final_start = syllable.initial.map_or(interval.start, |initial| {
        let cut = initial_end(initial, interval, envelope);
        phones.push(PhoneTiming {
            char_index: at,
            phone: initial.to_owned(),
            tone: None,
            kind: PhoneKind::Initial,
            start: interval.start,
            end: cut,
        });
        cut
    });
    phones.push(PhoneTiming {
        char_index: at,
        phone: syllable.final_.clone(),
        tone: Some(syllable.tone),
        kind: PhoneKind::Final,
        start: final_start,
        end: interval.end,
    });
}

fn push_word(phones: &mut Vec<PhoneTiming>, at: usize, arpabet: &[String], interval: Interval) {
    let total: f32 = arpabet.iter().map(|p| phone_weight(p)).sum();
    if total <= 0.0 {
        return;
    }
    let scale = (interval.end - interval.start) / total;
    let mut cursor = interval.start;
    for phone in arpabet {
        let end = cursor + phone_weight(phone) * scale;
        phones.push(PhoneTiming {
            char_index: at,
            phone: phone.clone(),
            tone: None,
            kind: PhoneKind::Phone,
            start: cursor,
            end,
        });
        cursor = end;
    }
    if let Some(last) = phones.last_mut() {
        last.end = interval.end;
    }
}

/// Where an initial ends inside its syllable `interval`.
fn initial_end(initial: &str, interval: Interval, envelope: &Envelope) -> f32 {
    let duration = interval.end - interval.start;
    let shortest = MIN_PART_SECS.min(duration / 4.0);
    let fallback = interval.start + duration * SONORANT_SHARE;
    let cut = if SONORANT_INITIALS.contains(&initial) {
        fallback
    } else {
        envelope.onset(interval).unwrap_or(fallback)
    };
    cut.clamp(
        interval.start + shortest,
        (interval.start + duration * MAX_INITIAL_SHARE).max(interval.start + shortest),
    )
}

/// Pronounceable units of `normalized[body]` in reading order.
fn units(normalized: &str, body: Range<usize>) -> Vec<Unit> {
    let mut out = Vec::new();
    let mut at = body.start;
    while at < body.end {
        let Some(rest) = normalized.get(at..body.end) else {
            break;
        };
        if let Some((unit, len)) = special_override(rest, at) {
            out.push(unit);
            at += len;
            continue;
        }
        let Some(ch) = rest.chars().next() else {
            break;
        };
        if has_cjk_ideograph(ch.encode_utf8(&mut [0; 4])) {
            out.push(Unit::Hanzi(at..at + ch.len_utf8()));
            at += ch.len_utf8();
        } else if ch.is_ascii_alphabetic() {
            let len = rest
                .find(|c: char| !(c.is_ascii_alphabetic() || c == '\''))
                .unwrap_or(rest.len());
            out.push(Unit::Word(at..at + len));
            at += len;
        } else {
            at += ch.len_utf8();
        }
    }
    out
}

/// Parses `<|SPECIAL_TOKEN_n|>PRON<|SPECIAL_TOKEN_n|>` at the start of `rest` (`offset` is its
/// byte position), returning the unit and its byte length.
fn special_override(rest: &str, offset: usize) -> Option<(Unit, usize)> {
    let tail = rest.strip_prefix("<|SPECIAL_TOKEN_")?;
    let marker_len = rest.len() - tail.len() + tail.find("|>")? + 2;
    let marker = rest.get(..marker_len)?;
    let after = rest.get(marker_len..)?;
    let pron_len = after.find(marker)?;
    let pinyin = marker == "<|SPECIAL_TOKEN_2|>";
    let pron = offset + marker_len..offset + marker_len + pron_len;
    let len = marker_len * 2 + pron_len;
    Some((
        Unit::Override {
            whole: offset..offset + len,
            pron,
            pinyin,
        },
        len,
    ))
}

/// Interval of every spoken normalized character: each token's interval shared evenly by its
/// pronounced characters. Indexed by byte offset.
fn char_times(normalized: &str, tokens: &[(Range<usize>, Interval)]) -> HashMap<usize, Interval> {
    let mut times = HashMap::new();
    for (bytes, interval) in tokens {
        let Some(text) = normalized.get(bytes.clone()) else {
            continue;
        };
        let spoken: Vec<usize> = text
            .char_indices()
            .filter(|(_, ch)| !ch.is_whitespace() && !is_punctuation(*ch))
            .map(|(at, _)| bytes.start + at)
            .collect();
        #[allow(clippy::cast_precision_loss, reason = "token lengths are tiny")]
        let step = (interval.end - interval.start) / spoken.len().max(1) as f32;
        for (k, at) in spoken.into_iter().enumerate() {
            #[allow(clippy::cast_precision_loss, reason = "token lengths are tiny")]
            let start = interval.start + step * k as f32;
            times.insert(
                at,
                Interval {
                    start,
                    end: start + step,
                },
            );
        }
    }
    times
}

/// Union of the timed characters inside `bytes`.
fn span_of(times: &HashMap<usize, Interval>, bytes: &Range<usize>) -> Option<Interval> {
    bytes
        .clone()
        .filter_map(|at| times.get(&at))
        .fold(None, |acc: Option<Interval>, t| {
            Some(acc.map_or(*t, |a| Interval {
                start: a.start.min(t.start),
                end: a.end.max(t.end),
            }))
        })
}

/// RMS envelope of the low-passed recording, one value per 10 ms hop.
#[derive(Debug, Clone, Default)]
struct Envelope {
    values: Vec<f32>,
}

impl Envelope {
    /// Time at which the envelope inside `interval` first exceeds [`ONSET_RELATIVE`] of its
    /// peak (window centre), or `None` when the interval is silent.
    fn onset(&self, interval: Interval) -> Option<f32> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let first = (interval.start / ENVELOPE_HOP_SECS).floor().max(0.0) as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let last = ((interval.end / ENVELOPE_HOP_SECS).ceil().max(0.0) as usize)
            .saturating_sub(1)
            .min(self.values.len());
        let window = self.values.get(first..last.max(first))?;
        let peak = window.iter().copied().fold(0.0_f32, f32::max);
        if peak <= 0.0 {
            return None;
        }
        let index = window.iter().position(|&v| v > peak * ONSET_RELATIVE)?;
        #[allow(clippy::cast_precision_loss)]
        Some((first + index) as f32 * ENVELOPE_HOP_SECS + ENVELOPE_HOP_SECS)
    }
}

/// Low-passes `pcm` at [`LOW_PASS_HZ`] (RBJ biquad, Q = 1/√2) and takes the RMS of 20 ms
/// windows every 10 ms.
fn low_frequency_envelope(pcm: &[i16], sample_rate: u32) -> Envelope {
    if sample_rate == 0 {
        return Envelope::default();
    }
    #[allow(clippy::cast_precision_loss)]
    let rate = sample_rate as f32;
    let omega = std::f32::consts::TAU * LOW_PASS_HZ / rate;
    let alpha = omega.sin() / std::f32::consts::SQRT_2;
    let cos = omega.cos();
    let a0 = 1.0 + alpha;
    let (b0, b1, b2) = (
        (1.0 - cos) / 2.0 / a0,
        (1.0 - cos) / a0,
        (1.0 - cos) / 2.0 / a0,
    );
    let (a1, a2) = (-2.0 * cos / a0, (1.0 - alpha) / a0);
    let (mut x1, mut x2, mut y1, mut y2) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let filtered: Vec<f32> = pcm
        .iter()
        .map(|&s| {
            let x = f32::from(s) / 32_768.0;
            let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            (x2, x1, y2, y1) = (x1, x, y1, y);
            y
        })
        .collect();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let hop = ((rate * ENVELOPE_HOP_SECS) as usize).max(1);
    let values = (0..filtered.len() / hop)
        .map(|i| {
            let window = filtered
                .get(i * hop..((i + 2) * hop).min(filtered.len()))
                .unwrap_or(&[]);
            #[allow(clippy::cast_precision_loss)]
            let mean = window.iter().map(|v| v * v).sum::<f32>() / window.len().max(1) as f32;
            mean.sqrt()
        })
        .collect();
    Envelope { values }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{
        Envelope, Interval, PhoneKind, Unit, initial_end, low_frequency_envelope, override_word,
        split_syllable, units, word_phones,
    };

    #[rstest]
    #[case("kun1", Some("k"), "un", 1)]
    #[case("zhong1", Some("zh"), "ong", 1)]
    #[case("shi4", Some("sh"), "i", 4)]
    #[case("ri4", Some("r"), "i", 4)]
    #[case("si1", Some("s"), "i", 1)]
    #[case("wo3", None, "uo", 3)]
    #[case("yi1", None, "i", 1)]
    #[case("wu3", None, "u", 3)]
    #[case("yu4", None, "v", 4)]
    #[case("ya1", None, "ia", 1)]
    #[case("ye2", None, "ie", 2)]
    #[case("yao4", None, "iao", 4)]
    #[case("you3", None, "iu", 3)]
    #[case("yan2", None, "ian", 2)]
    #[case("yin1", None, "in", 1)]
    #[case("yang2", None, "iang", 2)]
    #[case("ying1", None, "ing", 1)]
    #[case("yong4", None, "iong", 4)]
    #[case("yue4", None, "ve", 4)]
    #[case("yuan3", None, "van", 3)]
    #[case("yun2", None, "vn", 2)]
    #[case("wa1", None, "ua", 1)]
    #[case("wai4", None, "uai", 4)]
    #[case("wei4", None, "ui", 4)]
    #[case("wan3", None, "uan", 3)]
    #[case("wen1", None, "un", 1)]
    #[case("wang2", None, "uang", 2)]
    #[case("weng1", None, "ueng", 1)]
    #[case("ju1", Some("j"), "v", 1)]
    #[case("JV1", Some("j"), "v", 1)]
    #[case("quan2", Some("q"), "van", 2)]
    #[case("xue3", Some("x"), "ve", 3)]
    #[case("xun4", Some("x"), "vn", 4)]
    #[case("lü4", Some("l"), "v", 4)]
    #[case("nv3", Some("n"), "v", 3)]
    #[case("er2", None, "er", 2)]
    #[case("a1", None, "a", 1)]
    #[case("ai4", None, "ai", 4)]
    #[case("de", Some("d"), "e", 5)]
    #[case("n2", None, "n", 2)]
    #[case("liu2", Some("l"), "iu", 2)]
    #[case("gui4", Some("g"), "ui", 4)]
    fn test_should_split_pinyin_syllables(
        #[case] raw: &str,
        #[case] initial: Option<&str>,
        #[case] final_: &str,
        #[case] tone: u8,
    ) {
        let syllable = split_syllable(raw);
        assert_eq!(
            syllable
                .as_ref()
                .map(|s| (s.initial, s.final_.as_str(), s.tone)),
            Some((initial, final_, tone)),
            "{raw}"
        );
    }

    #[rstest]
    #[case("")]
    #[case("hello")]
    #[case("zh")]
    #[case("1")]
    fn test_should_reject_strings_that_are_not_pinyin(#[case] raw: &str) {
        assert_eq!(split_syllable(raw), None);
    }

    #[test]
    fn test_should_look_up_cmudict_and_fall_back_to_spelling() {
        assert_eq!(word_phones("hollow"), ["HH", "AA", "L", "OW"]);
        assert_eq!(word_phones("don't"), ["D", "OW", "N", "T"]);
        assert_eq!(word_phones("thrake"), ["TH", "R", "AE", "K"]);
    }

    #[test]
    fn test_should_find_hanzi_words_and_overrides_as_units() {
        let text = "你好 rust <|SPECIAL_TOKEN_2|>HANG2<|SPECIAL_TOKEN_2|>.";
        let found = units(text, 0..text.len());
        assert_eq!(found.len(), 4, "{found:?}");
        assert_eq!(found.first(), Some(&Unit::Hanzi(0..3)));
        assert_eq!(found.get(2), Some(&Unit::Word(7..11)));
        assert!(matches!(
            found.get(3),
            Some(Unit::Override { pinyin: true, pron, .. })
                if text.get(pron.clone()) == Some("HANG2")
        ));
    }

    #[test]
    fn test_should_point_an_override_at_its_word() {
        assert_eq!(override_word("把<行|hang2>业", 1..10), 2..3);
        assert_eq!(override_word("银行", 0..2), 0..2);
    }

    #[test]
    fn test_should_end_a_voiceless_initial_at_the_voicing_onset() {
        // 16 kHz: 100 ms of silence then a 200 Hz tone; the syllable spans 0-300 ms.
        let pcm: Vec<i16> = (0..4_800)
            .map(|i| {
                if i < 1_600 {
                    0
                } else {
                    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                    let v = (f32::sin(std::f32::consts::TAU * 200.0 * i as f32 / 16_000.0)
                        * 12_000.0) as i16;
                    v
                }
            })
            .collect();
        let envelope = low_frequency_envelope(&pcm, 16_000);
        let syllable = Interval {
            start: 0.0,
            end: 0.3,
        };
        let voiceless = initial_end("sh", syllable, &envelope);
        assert!((voiceless - 0.1).abs() <= 0.011, "{voiceless}");
        let sonorant = initial_end("m", syllable, &envelope);
        assert!((sonorant - 0.075).abs() < 1e-6, "{sonorant}");
        let silent = Envelope::default();
        assert!((initial_end("t", syllable, &silent) - 0.075).abs() < 1e-6);
    }

    #[test]
    fn test_should_serialize_phones_in_the_wire_shape() -> serde_json::Result<()> {
        let phone = super::PhoneTiming {
            char_index: 1,
            phone: "sh".to_owned(),
            tone: None,
            kind: PhoneKind::Initial,
            start: 0.25,
            end: 0.5,
        };
        assert_eq!(
            serde_json::to_value(&phone)?,
            serde_json::json!({ "i": 1, "p": "sh", "k": "ini", "start": 0.25, "end": 0.5 })
        );
        Ok(())
    }
}

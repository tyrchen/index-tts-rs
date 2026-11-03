//! Text-to-audio alignment derived from the GPT decoder's own text attention.
//!
//! Every generated semantic code covers a fixed slice of the output audio (the length regulator
//! expands codes linearly), so knowing which text token each code attends to yields token
//! timings. Two heads ([`ALIGNMENT_HEADS`]) carry a sharp, monotone text alignment; the decoder
//! replays the first [`ALIGNMENT_LAYERS`] layers with teacher forcing to read them, a monotone
//! Viterbi pass turns the attention into token spans, and the waveform's energy envelope moves
//! span edges off pauses. [`TextMap`] then projects normalized token bytes back onto the caller's
//! text. See `docs/research/gpt-attention-alignment-research.md` for the head selection.
//!
//! External recordings ([`AlignmentRequest`], `itts align`) reuse the same aligner: their audio is
//! quantized into semantic codes on a fixed 40 ms grid ([`CODE_FRAME_SECS`]) and replayed through
//! the decoder, and [`align_recording`] also reports how well the decoder supports every
//! character or word ([`AlignmentQuality`]): the attention evidence plus how much the text
//! raises the codes' likelihood over a text-free baseline, so callers can spot text the
//! recording does not say.

mod decode;
pub mod phones;
mod request;
mod textmap;

use std::ops::Range;

pub use decode::{Attention, DecodedTokens, Interval, TokenEvidence, decode_tokens};
use phones::{SpokenText, judged_spans, override_word};
pub use request::{
    ALIGNER_NAME, ALIGNER_VERSION, AlignmentOutput, AlignmentRequest, AlignmentTimings,
    Conditioning, MAX_ALIGN_CODES, MIN_ALIGN_SECS, Recording,
};
use serde::{Deserialize, Serialize};
pub use textmap::{TextMap, is_punctuation};

use crate::frontend::PreparedSegment;

/// `(layer, head)` pairs whose text attention is averaged for alignment.
pub const ALIGNMENT_HEADS: [(usize, usize); 2] = [(4, 10), (5, 10)];

/// Number of leading decoder layers replayed to read [`ALIGNMENT_HEADS`].
pub const ALIGNMENT_LAYERS: usize = 6;

/// Duration of one semantic code of a recording: `w2v-BERT` emits 50 Hz features (10 ms fbank
/// hop, frames stacked in pairs) and the codec encoder halves that rate.
pub const CODE_FRAME_SECS: f32 = 0.04;

/// A token whose renormalized attention peak stays below this is reported as weak.
pub const WEAK_PEAK: f32 = 0.3;

/// A token whose frames the text explains worse than this, relative to a text-free baseline
/// (mean log-probability gain in nats per frame), is reported as weak.
pub const WEAK_GAIN: f32 = -1.0;

/// Timing of a run of characters in the request text.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct SpanTiming {
    /// Character (Unicode scalar) indices in the request text.
    pub chars: Range<usize>,
    /// Start in seconds from the start of the segment audio.
    pub start_secs: f32,
    /// End in seconds from the start of the segment audio.
    pub end_secs: f32,
}

/// How well the decoder supports an alignment of a recording. Measured over judged units: a
/// Chinese character, a Latin word, or a pronunciation override.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct AlignmentQuality {
    /// Mean renormalized attention peak of the units (`0..=1`).
    pub confidence: f32,
    /// Fraction of units that are the most attended on at least one of their frames.
    pub coverage: f32,
    /// Request-text character ranges of units whose attention peak is below [`WEAK_PEAK`], that
    /// no frame attends to most, or whose frames the text explains worse than a text-free
    /// baseline by more than [`WEAK_GAIN`], in reading order.
    pub weak: Vec<Range<usize>>,
}

/// Timings of one recording's tokens plus the evidence behind them.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RecordingAlignment {
    /// Timing of every spoken token on request-text characters.
    pub spans: Vec<SpanTiming>,
    /// Attention support of the alignment.
    pub quality: AlignmentQuality,
    /// Normalized-text byte range and interval of every spoken token, for phone splitting.
    pub tokens: Vec<(Range<usize>, Interval)>,
}

/// Aligns one synthesized segment and projects the result onto request-text characters.
///
/// `attention` is `[frames, tokens + 2]` (columns `<BOS>, tokens..., <EOS>`), `map` maps the
/// frontend's normalized text onto the request text. The segment audio defines the frame length.
#[must_use]
pub fn align_segment(
    segment: &PreparedSegment,
    map: &TextMap,
    attention: &[f32],
    frames: usize,
    pcm: &[i16],
    sample_rate: u32,
) -> Vec<SpanTiming> {
    #[allow(clippy::cast_precision_loss)]
    let frame_secs = pcm.len() as f32 / sample_rate.max(1) as f32 / frames.max(1) as f32;
    let attention = Attention {
        values: attention,
        frames,
        columns: segment.tokens.len() + 2,
    };
    let spoken = spoken_tokens(segment);
    let state_columns: Vec<usize> = spoken.iter().map(|(column, _)| *column).collect();
    let decoded = decode_tokens(attention, &state_columns, frame_secs, &[], pcm, sample_rate);
    spoken
        .into_iter()
        .zip(decoded.intervals)
        .map(|((_, bytes), interval)| SpanTiming {
            chars: map.source_chars(bytes),
            start_secs: interval.start,
            end_secs: interval.end,
        })
        .filter(|span| !span.chars.is_empty())
        .collect()
}

/// Teacher-forced evidence about a recording, one row per semantic code.
#[derive(Debug, Clone, Copy)]
pub struct Replay<'a> {
    /// Row-major `[frames, tokens + 2]` text attention.
    pub attention: &'a [f32],
    /// Per-frame log-probability gain of the text over a text-free baseline (may be empty).
    pub gain: &'a [f32],
    /// Number of codes.
    pub frames: usize,
}

/// Aligns a recording's codes (on the [`CODE_FRAME_SECS`] grid) to `segment` and projects the
/// result onto request-text characters, with the evidence behind every token. Tokens of one
/// Latin word or pronunciation override are judged together (an override is reported on the
/// overridden word).
///
/// # Examples
///
/// ```no_run
/// use index_tts_rs_core::{
///     align::{Replay, TextMap, align_recording, phones::SpokenText},
///     frontend::Frontend,
///     models::resolve_model_dir,
/// };
///
/// let dir = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx")?;
/// let frontend = Frontend::from_model_dir(&dir, 602, true, None)?;
/// let prepared = frontend.prepare_text("你好。", "zh", 600);
/// let map = TextMap::new(&prepared.normalized, "你好。");
/// let segment = &prepared.segments[0];
/// let text = SpokenText {
///     normalized: &prepared.normalized,
///     body: segment.normalized_range(),
///     map: &map,
///     source: "你好。",
/// };
/// // Attention over 10 codes as produced by `Gpt::teacher_forced`.
/// let attention = vec![0.0_f32; 10 * (segment.tokens.len() + 2)];
/// let replay = Replay { attention: &attention, gain: &[], frames: 10 };
/// let aligned = align_recording(segment, &text, replay, &vec![0_i16; 6_400], 16_000);
/// assert!(aligned.quality.confidence <= 1.0);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
#[must_use]
pub fn align_recording(
    segment: &PreparedSegment,
    text: &SpokenText<'_>,
    replay: Replay<'_>,
    pcm: &[i16],
    sample_rate: u32,
) -> RecordingAlignment {
    let attention = Attention {
        values: replay.attention,
        frames: replay.frames,
        columns: segment.tokens.len() + 2,
    };
    let spoken = spoken_tokens(segment);
    let state_columns: Vec<usize> = spoken.iter().map(|(column, _)| *column).collect();
    let decoded = decode_tokens(
        attention,
        &state_columns,
        CODE_FRAME_SECS,
        replay.gain,
        pcm,
        sample_rate,
    );
    let groups = judged_spans(text.normalized, text.body.clone());
    let mut out = RecordingAlignment::default();
    let mut units: Vec<(Range<usize>, TokenEvidence)> = Vec::new();
    let mut current: Option<usize> = None;
    for (((_, bytes), interval), evidence) in spoken
        .into_iter()
        .zip(decoded.intervals)
        .zip(decoded.evidence)
    {
        let chars = text.map.source_chars(bytes.clone());
        // A BPE token may carry a leading space before its word: judge it by its last byte.
        let group = groups
            .iter()
            .position(|span| span.bytes.start < bytes.end && bytes.end <= span.bytes.end);
        match (group, units.last_mut()) {
            (Some(index), Some(last)) if current == Some(index) => {
                last.1 = combine(last.1, evidence);
            }
            (Some(index), _) => {
                let span = groups.get(index).map(|g| (g.bytes.clone(), g.is_override));
                let unit_chars = match span {
                    Some((bytes, true)) => override_word(text.source, text.map.source_chars(bytes)),
                    Some((bytes, false)) => text.map.source_chars(bytes),
                    None => chars.clone(),
                };
                units.push((unit_chars, evidence));
            }
            (None, _) => units.push((chars.clone(), evidence)),
        }
        current = group;
        out.tokens.push((bytes, interval));
        if !chars.is_empty() {
            out.spans.push(SpanTiming {
                chars,
                start_secs: interval.start,
                end_secs: interval.end,
            });
        }
    }
    out.quality = judge(&units);
    out
}

/// Evidence of two tokens of one judged unit: the stronger attention, the mean gain.
fn combine(a: TokenEvidence, b: TokenEvidence) -> TokenEvidence {
    TokenEvidence {
        peak: a.peak.max(b.peak),
        attended: a.attended || b.attended,
        gain: match (a.gain, b.gain) {
            (Some(x), Some(y)) => Some(f32::midpoint(x, y)),
            (x, y) => x.or(y),
        },
    }
}

/// Confidence, coverage, and weak character ranges of judged units.
fn judge(units: &[(Range<usize>, TokenEvidence)]) -> AlignmentQuality {
    #[allow(clippy::cast_precision_loss, reason = "unit counts are small")]
    let count = units.len().max(1) as f32;
    AlignmentQuality {
        confidence: units.iter().map(|(_, e)| e.peak).sum::<f32>() / count,
        #[allow(clippy::cast_precision_loss, reason = "unit counts are small")]
        coverage: units.iter().filter(|(_, e)| e.attended).count() as f32 / count,
        weak: units
            .iter()
            .filter(|(chars, e)| {
                !chars.is_empty()
                    && (e.peak < WEAK_PEAK || !e.attended || e.gain.is_some_and(|g| g < WEAK_GAIN))
            })
            .map(|(chars, _)| chars.clone())
            .collect(),
    }
}

/// `(attention column, normalized bytes)` of every spoken token of `segment`.
fn spoken_tokens(segment: &PreparedSegment) -> Vec<(usize, Range<usize>)> {
    segment
        .tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| segment.is_spoken(token))
        .filter_map(|(index, token)| Some((index + 1, segment.normalized_bytes(token)?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Replay, TextMap, align_recording, phones::SpokenText};
    use crate::frontend::{PreparedSegment, TokenSpan};

    /// `<|en|>` + ` ru` + `st` + `你` over six 40 ms frames: `ru` holds frames 0-3 (its second
    /// piece `st` never gets attention) and `你` frames 4-5.
    fn fixture() -> (PreparedSegment, Vec<f32>) {
        let token = |id, bytes| TokenSpan { id, bytes };
        let segment = PreparedSegment {
            text: "<|en|> rust你".to_owned(),
            tokens: vec![
                token(2, 0..6),
                token(3, 6..9),
                token(4, 9..11),
                token(5, 11..14),
            ],
            prefix_len: 7,
            normalized_offset: 0,
        };
        let columns = 6;
        let mut attention = vec![0.0_f32; 6 * columns];
        for (frame, column) in [(0, 2), (1, 2), (2, 2), (3, 2), (4, 4), (5, 4)] {
            if let Some(cell) = attention.get_mut(frame * columns + column) {
                *cell = 1.0;
            }
        }
        (segment, attention)
    }

    #[test]
    fn test_should_judge_the_pieces_of_a_word_together() {
        let (segment, attention) = fixture();
        let map = TextMap::new("rust你", "rust你");
        let text = SpokenText {
            normalized: "rust你",
            body: 0..7,
            map: &map,
            source: "rust你",
        };
        let pcm = vec![8_000_i16; 16_000 * 6 / 25];
        let replay = Replay {
            attention: &attention,
            gain: &[2.0; 6],
            frames: 6,
        };
        let aligned = align_recording(&segment, &text, replay, &pcm, 16_000);
        assert_eq!(aligned.spans.len(), 3, "{aligned:?}");
        assert!(aligned.quality.weak.is_empty(), "{:?}", aligned.quality);
        assert!((aligned.quality.coverage - 1.0).abs() < 1e-6);

        let replay = Replay {
            attention: &attention,
            gain: &[2.0, 2.0, 2.0, 2.0, -3.0, -3.0],
            frames: 6,
        };
        let aligned = align_recording(&segment, &text, replay, &pcm, 16_000);
        assert_eq!(aligned.quality.weak, vec![4..5]);
    }
}

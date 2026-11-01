//! Text-to-audio alignment derived from the GPT decoder's own text attention.
//!
//! Every generated semantic code covers a fixed slice of the output audio (the length regulator
//! expands codes linearly), so knowing which text token each code attends to yields token
//! timings. Two heads ([`ALIGNMENT_HEADS`]) carry a sharp, monotone text alignment; the decoder
//! replays the first [`ALIGNMENT_LAYERS`] layers with teacher forcing to read them, a monotone
//! Viterbi pass turns the attention into token spans, and the waveform's energy envelope moves
//! span edges off pauses. [`TextMap`] then projects normalized token bytes back onto the caller's
//! text. See `docs/research/gpt-attention-alignment-research.md` for the head selection.

mod decode;
mod textmap;

use std::ops::Range;

pub use decode::{Interval, decode_intervals};
use serde::{Deserialize, Serialize};
pub use textmap::{TextMap, is_punctuation};

use crate::frontend::PreparedSegment;

/// `(layer, head)` pairs whose text attention is averaged for alignment.
pub const ALIGNMENT_HEADS: [(usize, usize); 2] = [(4, 10), (5, 10)];

/// Number of leading decoder layers replayed to read [`ALIGNMENT_HEADS`].
pub const ALIGNMENT_LAYERS: usize = 6;

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

/// Aligns one synthesized segment and projects the result onto request-text characters.
///
/// `attention` is `[frames, tokens + 2]` (columns `<BOS>, tokens..., <EOS>`), `map` maps the
/// frontend's normalized text onto the request text.
#[must_use]
pub fn align_segment(
    segment: &PreparedSegment,
    map: &TextMap,
    attention: &[f32],
    frames: usize,
    pcm: &[i16],
    sample_rate: u32,
) -> Vec<SpanTiming> {
    let columns = segment.tokens.len() + 2;
    let spoken: Vec<(usize, Range<usize>)> = segment
        .tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| segment.is_spoken(token))
        .filter_map(|(index, token)| Some((index + 1, segment.normalized_bytes(token)?)))
        .collect();
    let state_columns: Vec<usize> = spoken.iter().map(|(column, _)| *column).collect();
    let intervals = decode_intervals(attention, frames, columns, &state_columns, pcm, sample_rate);
    spoken
        .into_iter()
        .zip(intervals)
        .map(|((_, bytes), interval)| SpanTiming {
            chars: map.source_chars(bytes),
            start_secs: interval.start,
            end_secs: interval.end,
        })
        .filter(|span| !span.chars.is_empty())
        .collect()
}

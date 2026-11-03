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
//! the decoder, and [`align_recording`] also reports how well the attention supports every token
//! ([`AlignmentQuality`]) so callers can spot text the recording does not say.

mod decode;
pub mod phones;
mod request;
mod textmap;

use std::ops::Range;

pub use decode::{Attention, DecodedTokens, Interval, TokenEvidence, decode_tokens};
pub use request::{
    ALIGNER_NAME, ALIGNER_VERSION, AlignmentOutput, AlignmentRequest, AlignmentTimings,
    Conditioning, MAX_ALIGN_CODES, MIN_ALIGN_SECS,
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

/// How well the attention supports an alignment of a recording.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct AlignmentQuality {
    /// Mean renormalized attention peak of the spoken tokens (`0..=1`).
    pub confidence: f32,
    /// Fraction of spoken tokens that are the most attended token on at least one frame.
    pub coverage: f32,
    /// Request-text character ranges of tokens with a peak below [`WEAK_PEAK`] or no frame
    /// attending to them, in reading order.
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
    let decoded = decode_tokens(attention, &state_columns, frame_secs, pcm, sample_rate);
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

/// Aligns a recording's codes (on the [`CODE_FRAME_SECS`] grid) to `segment` and projects the
/// result onto request-text characters, with the attention evidence of every token.
///
/// # Examples
///
/// ```no_run
/// use index_tts_rs_core::{
///     align::{TextMap, align_recording},
///     frontend::Frontend,
///     models::resolve_model_dir,
/// };
///
/// let dir = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx")?;
/// let frontend = Frontend::from_model_dir(&dir, 602, true, None)?;
/// let prepared = frontend.prepare_text("你好。", "zh", 600);
/// let map = TextMap::new(&prepared.normalized, "你好。");
/// let segment = &prepared.segments[0];
/// // Attention over 10 codes as produced by `Gpt::alignment_attention`.
/// let attention = vec![0.0_f32; 10 * (segment.tokens.len() + 2)];
/// let pcm = vec![0_i16; 6_400];
/// let aligned = align_recording(segment, &map, &attention, 10, &pcm, 16_000);
/// assert!(aligned.quality.confidence <= 1.0);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
#[must_use]
pub fn align_recording(
    segment: &PreparedSegment,
    map: &TextMap,
    attention: &[f32],
    frames: usize,
    pcm: &[i16],
    sample_rate: u32,
) -> RecordingAlignment {
    let attention = Attention {
        values: attention,
        frames,
        columns: segment.tokens.len() + 2,
    };
    let spoken = spoken_tokens(segment);
    let state_columns: Vec<usize> = spoken.iter().map(|(column, _)| *column).collect();
    let decoded = decode_tokens(attention, &state_columns, CODE_FRAME_SECS, pcm, sample_rate);
    let mut out = RecordingAlignment::default();
    let (mut peaks, mut attended) = (0.0_f32, 0_usize);
    for (((_, bytes), interval), evidence) in spoken
        .into_iter()
        .zip(decoded.intervals)
        .zip(decoded.evidence)
    {
        let chars = map.source_chars(bytes.clone());
        peaks += evidence.peak;
        attended += usize::from(evidence.attended);
        if (evidence.peak < WEAK_PEAK || !evidence.attended) && !chars.is_empty() {
            out.quality.weak.push(chars.clone());
        }
        out.tokens.push((bytes, interval));
        if !chars.is_empty() {
            out.spans.push(SpanTiming {
                chars,
                start_secs: interval.start,
                end_secs: interval.end,
            });
        }
    }
    #[allow(clippy::cast_precision_loss, reason = "token counts are small")]
    let count = out.tokens.len().max(1) as f32;
    out.quality.confidence = peaks / count;
    #[allow(clippy::cast_precision_loss, reason = "token counts are small")]
    let covered = attended as f32 / count;
    out.quality.coverage = covered;
    out
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

//! Turns GPT text attention into token timings: a monotone Viterbi path over the spoken tokens,
//! then pause boundaries snapped to the synthesized waveform's energy envelope.

use std::{iter::once, ops::Range};

/// Envelope frame length (10 ms).
const ENVELOPE_FRAME_SECS: f32 = 0.01;
/// Voiced threshold relative to the 95th-percentile envelope level (about -26 dB).
const VOICED_RELATIVE: f32 = 0.05;
/// Absolute voiced floor on the normalized `[-1, 1]` scale.
const VOICED_FLOOR: f32 = 1e-3;
/// Shortest low-energy run treated as a pause rather than a stop closure.
const MIN_PAUSE_SECS: f32 = 0.08;
/// How far a pause may sit from a token boundary and still move it.
const PAUSE_SNAP_SECS: f32 = 0.2;
/// Log-probability floor for attention emissions.
const EMISSION_FLOOR: f32 = 1e-6;

/// Start and end of one spoken token, in seconds from the start of its segment audio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Inclusive start in seconds.
    pub start: f32,
    /// Exclusive end in seconds.
    pub end: f32,
}

/// Aligns `states` spoken tokens to `frames` GPT codes and returns one [`Interval`] per state.
///
/// `attention` is row-major `[frames, columns]`; `state_columns[s]` is the attention column of
/// spoken token `s`. `pcm` is the segment audio (its duration defines the code frame length).
///
/// Returns an empty vector when there are no spoken tokens or no frames.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::align::decode_intervals;
///
/// // Four frames over two tokens: frames 0-1 attend to column 0, frames 2-3 to column 1.
/// let attention = [0.9, 0.1, 0.8, 0.2, 0.1, 0.9, 0.2, 0.8];
/// let pcm = vec![8_000_i16; 400];
/// let spans = decode_intervals(&attention, 4, 2, &[0, 1], &pcm, 400);
/// assert_eq!(spans.len(), 2);
/// assert!((spans[1].start - 0.5).abs() < 1e-6);
/// ```
#[must_use]
pub fn decode_intervals(
    attention: &[f32],
    frames: usize,
    columns: usize,
    state_columns: &[usize],
    pcm: &[i16],
    sample_rate: u32,
) -> Vec<Interval> {
    if frames == 0 || state_columns.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let path = viterbi(attention, frames, columns, state_columns);
    #[allow(clippy::cast_precision_loss)]
    let frame_secs = pcm.len() as f32 / sample_rate as f32 / frames as f32;
    let mut spans = frame_spans(&path, state_columns.len())
        .into_iter()
        .map(|frames| {
            #[allow(clippy::cast_precision_loss)]
            Interval {
                start: frames.start as f32 * frame_secs,
                end: frames.end as f32 * frame_secs,
            }
        })
        .collect::<Vec<_>>();
    snap_to_voicing(&mut spans, pcm, sample_rate);
    share_collapsed(&mut spans);
    spans
}

/// Monotone path where each frame stays on its state or advances by one, from the first state
/// to the last. Returns the state of every frame.
fn viterbi(
    attention: &[f32],
    frames: usize,
    columns: usize,
    state_columns: &[usize],
) -> Vec<usize> {
    let states = state_columns.len();
    let emissions: Vec<f32> = (0..frames)
        .flat_map(|t| {
            let row = attention.get(t * columns..(t + 1) * columns).unwrap_or(&[]);
            let picked: Vec<f32> = state_columns
                .iter()
                .map(|&c| row.get(c).copied().unwrap_or(0.0).max(0.0))
                .collect();
            let total: f32 = picked.iter().sum::<f32>().max(f32::MIN_POSITIVE);
            picked
                .into_iter()
                .map(move |p| (p / total + EMISSION_FLOOR).ln())
        })
        .collect();
    let emission = |t: usize, s: usize| emissions.get(t * states + s).copied().unwrap_or(f32::MIN);

    let mut score = vec![f32::NEG_INFINITY; states];
    if let Some(first) = score.first_mut() {
        *first = emission(0, 0);
    }
    // advanced[t * states + s]: frame t entered state s from s - 1.
    let mut advanced = vec![false; frames * states];
    for t in 1..frames {
        let mut next = vec![f32::NEG_INFINITY; states];
        for s in 0..states {
            let stay = score.get(s).copied().unwrap_or(f32::NEG_INFINITY);
            let advance = if s > 0 {
                score.get(s - 1).copied().unwrap_or(f32::NEG_INFINITY)
            } else {
                f32::NEG_INFINITY
            };
            let (best, moved) = if advance > stay {
                (advance, true)
            } else {
                (stay, false)
            };
            if let Some(slot) = next.get_mut(s) {
                *slot = best + emission(t, s);
            }
            if let Some(flag) = advanced.get_mut(t * states + s) {
                *flag = moved;
            }
        }
        score = next;
    }

    // Fewer frames than states cannot visit every state; end wherever the path can reach.
    let mut state = states.min(frames).saturating_sub(1);
    let mut path = vec![0; frames];
    for t in (0..frames).rev() {
        if let Some(slot) = path.get_mut(t) {
            *slot = state;
        }
        if t > 0 && advanced.get(t * states + state).copied().unwrap_or(false) {
            state = state.saturating_sub(1);
        }
    }
    path
}

/// Frame range covered by each state; unvisited states get an empty range at their position.
fn frame_spans(path: &[usize], states: usize) -> Vec<Range<usize>> {
    let mut spans: Vec<Option<Range<usize>>> = vec![None; states];
    for (t, &s) in path.iter().enumerate() {
        if let Some(slot) = spans.get_mut(s) {
            match slot {
                Some(range) => range.end = t + 1,
                None => *slot = Some(t..t + 1),
            }
        }
    }
    let mut cursor = 0;
    spans
        .into_iter()
        .map(|span| {
            let range = span.unwrap_or(cursor..cursor);
            cursor = range.end;
            range
        })
        .collect()
}

/// Moves token edges off silence: the first token starts at the first voiced frame, the last
/// ends at the last voiced frame, and each pause is given to its nearest token boundary.
fn snap_to_voicing(spans: &mut [Interval], pcm: &[i16], sample_rate: u32) {
    let voiced = voiced_frames(pcm, sample_rate);
    let Some(first) = voiced.iter().position(|&v| v) else {
        return;
    };
    let last = voiced.iter().rposition(|&v| v).unwrap_or(first);
    #[allow(clippy::cast_precision_loss)]
    let to_secs = |frame: usize| frame as f32 * ENVELOPE_FRAME_SECS;
    if let Some(head) = spans.first_mut() {
        head.start = head.start.max(to_secs(first)).min(head.end);
    }
    if let Some(tail) = spans.last_mut() {
        tail.end = tail.end.min(to_secs(last + 1)).max(tail.start);
    }
    for pause in pauses(&voiced) {
        let (p0, p1) = (to_secs(pause.start), to_secs(pause.end));
        let nearest = spans
            .windows(2)
            .enumerate()
            .map(|(j, pair)| {
                let boundary = pair.first().map_or(0.0, |left| left.end);
                let distance = if (p0..=p1).contains(&boundary) {
                    0.0
                } else {
                    (boundary - p0).abs().min((boundary - p1).abs())
                };
                (distance, j)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let Some((distance, j)) = nearest else {
            continue;
        };
        if distance > PAUSE_SNAP_SECS {
            continue;
        }
        // Attention drifts toward the next token around pauses, so the decoded boundary may sit
        // on either side of the silence: speech before the pause belongs to the left token and
        // speech after it to the right one, in both directions.
        if let Some(left) = spans.get_mut(j)
            && p0 > left.start
        {
            left.end = p0;
        }
        if let Some(right) = spans.get_mut(j + 1)
            && p1 < right.end
        {
            right.start = p1;
        }
    }
}

/// Gives every token a positive duration. Snapping can collapse a token whose attention lay
/// entirely in silence; each run of collapsed tokens shares the span of the next intact token
/// (or, at the end, the previous one) evenly.
fn share_collapsed(spans: &mut [Interval]) {
    let collapsed = |span: &Interval| span.end <= span.start;
    let mut i = 0;
    while i < spans.len() {
        if !spans.get(i).is_some_and(collapsed) {
            i += 1;
            continue;
        }
        let run_end = (i..spans.len())
            .find(|&j| spans.get(j).is_some_and(|span| !collapsed(span)))
            .unwrap_or(spans.len());
        // Share with the following intact token, or with the preceding one at the end.
        let group = if run_end < spans.len() {
            i..run_end + 1
        } else {
            i.saturating_sub(1)..run_end
        };
        let (Some(first), Some(last)) = (spans.get(group.start), spans.get(group.end - 1)) else {
            break;
        };
        let (from, to) = (first.start.min(last.start), first.end.max(last.end));
        #[allow(clippy::cast_precision_loss, reason = "token counts are small")]
        let step = (to - from) / group.len() as f32;
        for (k, index) in group.clone().enumerate() {
            if let Some(span) = spans.get_mut(index) {
                #[allow(clippy::cast_precision_loss, reason = "token counts are small")]
                let at = from + step * k as f32;
                *span = Interval {
                    start: at,
                    end: at + step,
                };
            }
        }
        i = group.end;
    }
}

/// Voiced flag per 10 ms frame (20 ms RMS window).
fn voiced_frames(pcm: &[i16], sample_rate: u32) -> Vec<bool> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let hop = ((sample_rate as f32 * ENVELOPE_FRAME_SECS) as usize).max(1);
    let frames = pcm.len() / hop;
    let envelope: Vec<f32> = (0..frames)
        .map(|i| {
            let window = pcm
                .get(i * hop..((i + 2) * hop).min(pcm.len()))
                .unwrap_or(&[]);
            let energy: f32 = window
                .iter()
                .map(|&s| {
                    let v = f32::from(s) / 32_768.0;
                    v * v
                })
                .sum();
            #[allow(clippy::cast_precision_loss)]
            (energy / window.len().max(1) as f32).sqrt()
        })
        .collect();
    let mut sorted = envelope.clone();
    sorted.sort_by(f32::total_cmp);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let p95 = sorted
        .get((sorted.len() as f32 * 0.95) as usize)
        .copied()
        .unwrap_or(0.0);
    let threshold = (p95 * VOICED_RELATIVE).max(VOICED_FLOOR);
    envelope.into_iter().map(|e| e > threshold).collect()
}

/// Runs of unvoiced frames long enough to be pauses, between speech. Leading and trailing
/// silence is not a pause; the first and last token edges are snapped separately.
fn pauses(voiced: &[bool]) -> Vec<Range<usize>> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let min_frames = (MIN_PAUSE_SECS / ENVELOPE_FRAME_SECS).round() as usize;
    let mut runs = Vec::new();
    let mut start = None;
    for (i, &v) in voiced.iter().chain(once(&true)).enumerate() {
        match (v, start) {
            (false, None) => start = Some(i),
            (true, Some(s)) => {
                if s > 0 && i < voiced.len() && i - s >= min_frames {
                    runs.push(s..i);
                }
                start = None;
            }
            _ => {}
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::{Interval, decode_intervals, pauses, share_collapsed, viterbi};

    #[test]
    fn test_should_follow_a_diagonal_attention_path() {
        // 6 frames over 3 tokens, two frames each.
        let mut attention = vec![0.0_f32; 6 * 3];
        for t in 0..6 {
            if let Some(cell) = attention.get_mut(t * 3 + t / 2) {
                *cell = 1.0;
            }
        }
        assert_eq!(
            viterbi(&attention, 6, 3, &[0, 1, 2]),
            vec![0, 0, 1, 1, 2, 2]
        );
    }

    #[test]
    fn test_should_visit_every_state_even_when_attention_skips_one() {
        let attention = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let path = viterbi(&attention, 4, 3, &[0, 1, 2]);
        assert_eq!(path.first(), Some(&0));
        assert_eq!(path.last(), Some(&2));
        assert!(path.contains(&1));
    }

    #[test]
    fn test_should_not_treat_leading_or_trailing_silence_as_pauses() {
        let mut voiced = vec![false; 20];
        for v in voiced.iter_mut().skip(10).take(5) {
            *v = true;
        }
        assert_eq!(pauses(&voiced), Vec::<Range<usize>>::new());
    }

    #[test]
    fn test_should_detect_pauses_longer_than_minimum() {
        let mut voiced = vec![true; 30];
        for v in voiced.iter_mut().skip(5).take(3) {
            *v = false;
        }
        for v in voiced.iter_mut().skip(15).take(10) {
            *v = false;
        }
        assert_eq!(pauses(&voiced), vec![15..25]);
    }

    #[test]
    fn test_should_move_token_start_past_a_pause() {
        // 1 s of audio at 1 kHz: speech 0-300 ms, silence 300-600 ms, speech 600-1000 ms.
        let pcm: Vec<i16> = (0..1000)
            .map(|i| if (300..600).contains(&i) { 0 } else { 10_000 })
            .collect();
        // 10 frames; attention puts token 1 from frame 3 (300 ms) onward.
        let mut attention = vec![0.0_f32; 10 * 2];
        for t in 0..10 {
            if let Some(cell) = attention.get_mut(t * 2 + usize::from(t >= 3)) {
                *cell = 1.0;
            }
        }
        let spans = decode_intervals(&attention, 10, 2, &[0, 1], &pcm, 1000);
        assert_eq!(spans.len(), 2, "{spans:?}");
        if let [first, second] = spans.as_slice() {
            assert!((first.end - 0.3).abs() < 0.011, "{first:?}");
            assert!((second.start - 0.6).abs() < 0.011, "{second:?}");
        }
    }

    #[test]
    fn test_should_share_a_collapsed_token_with_its_neighbour() {
        let mut spans = [
            Interval {
                start: 0.4,
                end: 0.4,
            },
            Interval {
                start: 0.4,
                end: 0.8,
            },
            Interval {
                start: 0.9,
                end: 1.0,
            },
            Interval {
                start: 1.0,
                end: 1.0,
            },
        ];
        share_collapsed(&mut spans);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(close(spans[0].start, 0.4) && close(spans[0].end, 0.6));
        assert!(close(spans[1].start, 0.6) && close(spans[1].end, 0.8));
        assert!(close(spans[2].start, 0.9) && close(spans[2].end, 0.95));
        assert!(close(spans[3].start, 0.95) && close(spans[3].end, 1.0));
    }

    #[test]
    fn test_should_return_nothing_without_states_or_frames() {
        assert_eq!(
            decode_intervals(&[], 0, 0, &[], &[], 22_050),
            Vec::<Interval>::new()
        );
    }
}

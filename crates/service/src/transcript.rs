//! Projects engine segment timings onto the caller's script.
//!
//! The engine reports character ranges of each utterance's request text. A [`Transcript`] knows
//! where every utterance came from in the script, so it maps those ranges onto the *spoken text*
//! (the script with `[[...]]` controls removed) that callers see in [`Alignment::text`].

use std::{iter::once, ops::Range};

use index_tts_rs_core::{
    align::{SpanTiming, TextMap, is_punctuation},
    config::SynthesisRequest,
};
use index_tts_rs_director::ReadingPlan;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Per-character timings over a text, in the shape of `ElevenLabs` `with-timestamps` output.
///
/// `start[i]` and `end[i]` are the seconds at which the `i`-th Unicode scalar of `text` begins
/// and ends (JavaScript: index `Array.from(text)`). Punctuation and whitespace span the pause
/// between their neighbours.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct Alignment {
    /// The aligned text.
    pub text: String,
    /// Start time of each character in seconds.
    pub start: Vec<f32>,
    /// End time of each character in seconds.
    pub end: Vec<f32>,
}

/// One sentence of the spoken text and when it is heard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct SentenceTiming {
    /// Sentence text, trimmed.
    pub text: String,
    /// Character range within [`Alignment::text`].
    pub chars: Range<usize>,
    /// Start of the first spoken character, in seconds.
    pub start: f32,
    /// End of the last spoken character, in seconds.
    pub end: f32,
}

/// Where each utterance of a job came from in its script.
#[derive(Debug, Clone)]
pub struct Transcript {
    spoken: Vec<char>,
    utterances: Vec<UtteranceMap>,
}

#[derive(Debug, Clone)]
struct UtteranceMap {
    /// Byte offset of each request-text character, plus the total length.
    char_bytes: Vec<usize>,
    /// Request text onto the script slice it was read from.
    map: TextMap,
    /// Spoken-text character index where that slice starts.
    offset: usize,
}

impl Transcript {
    /// Builds the projection for `script` compiled into `plan`.
    #[must_use]
    pub fn new(script: &str, plan: &ReadingPlan) -> Self {
        let mut spoken = Vec::with_capacity(script.len());
        // (script byte where a run of spoken text starts, spoken char index at that byte)
        let mut runs: Vec<(usize, usize)> = Vec::with_capacity(plan.controls.len() + 1);
        let mut cursor = 0_usize;
        for control in plan
            .controls
            .iter()
            .chain(once(&(script.len()..script.len())))
        {
            if let Some(run) = script.get(cursor..control.start) {
                runs.push((cursor, spoken.len()));
                spoken.extend(run.chars());
            }
            cursor = control.end.max(cursor);
        }
        let spoken_at = |byte: usize| -> usize {
            let index = runs
                .partition_point(|(start, _)| *start <= byte)
                .saturating_sub(1);
            runs.get(index).map_or(0, |&(start, chars)| {
                chars + script.get(start..byte).map_or(0, |run| run.chars().count())
            })
        };
        let utterances = plan
            .utterances
            .iter()
            .map(|part| {
                UtteranceMap::new(
                    &part.request,
                    script,
                    &part.source,
                    spoken_at(part.source.start),
                )
            })
            .collect();
        Self { spoken, utterances }
    }

    /// The script with controls removed.
    #[must_use]
    pub fn spoken_text(&self) -> String {
        self.spoken.iter().collect()
    }

    /// Number of characters in the spoken text.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spoken.len()
    }

    /// Whether the spoken text is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spoken.is_empty()
    }

    /// Maps request-text characters of `utterance` onto spoken-text characters.
    #[must_use]
    pub fn project(&self, utterance: usize, chars: &Range<usize>) -> Range<usize> {
        let Some(part) = self.utterances.get(utterance) else {
            return 0..0;
        };
        let byte = |ch: usize| {
            part.char_bytes
                .get(ch)
                .or_else(|| part.char_bytes.last())
                .copied()
                .unwrap_or(0)
        };
        let source = part.map.source_chars(byte(chars.start)..byte(chars.end));
        let end = (part.offset + source.end).min(self.spoken.len());
        (part.offset + source.start).min(end)..end
    }

    /// Normalizes a projected segment range so consecutive segments partition the spoken text:
    /// it starts at its first pronounced character and runs up to the next pronounced one,
    /// keeping closing punctuation and the whitespace after it.
    #[must_use]
    pub fn segment_range(&self, chars: Range<usize>) -> Range<usize> {
        let silent_run = |from: usize| {
            self.spoken
                .get(from..)
                .unwrap_or_default()
                .iter()
                .take_while(|&&ch| !is_spoken_char(ch))
                .count()
        };
        let start = (chars.start + silent_run(chars.start)).min(chars.end);
        start..chars.end + silent_run(chars.end)
    }

    /// Text of a spoken-text character range.
    #[must_use]
    pub fn text(&self, chars: &Range<usize>) -> String {
        self.spoken
            .get(chars.clone())
            .unwrap_or_default()
            .iter()
            .collect()
    }

    /// Projects one segment's spans of `utterance` into spoken-text spans shifted by `offset`
    /// seconds.
    #[must_use]
    pub fn project_spans(
        &self,
        utterance: usize,
        spans: &[SpanTiming],
        offset: f32,
    ) -> Vec<(Range<usize>, f32, f32)> {
        spans
            .iter()
            .map(|span| {
                (
                    self.project(utterance, &span.chars),
                    offset + span.start_secs,
                    offset + span.end_secs,
                )
            })
            .filter(|(chars, _, _)| !chars.is_empty())
            .collect()
    }

    /// Builds the [`Alignment`] of `chars` from projected spans, filling punctuation and
    /// whitespace from their neighbours within `[floor, ceil]` seconds.
    #[must_use]
    pub fn alignment(
        &self,
        chars: &Range<usize>,
        spans: &[(Range<usize>, f32, f32)],
        floor: f32,
        ceil: f32,
    ) -> Alignment {
        let len = chars.len();
        let mut start: Vec<Option<f32>> = vec![None; len];
        let mut end: Vec<Option<f32>> = vec![None; len];
        for (range, s, e) in spans {
            let spoken: Vec<usize> = (range.start.max(chars.start)..range.end.min(chars.end))
                .filter(|&i| self.spoken.get(i).copied().is_some_and(is_spoken_char))
                .collect();
            #[allow(clippy::cast_precision_loss, reason = "span lengths are tiny")]
            let step = (e - s) / spoken.len().max(1) as f32;
            for (k, i) in spoken.into_iter().enumerate() {
                #[allow(clippy::cast_precision_loss, reason = "span lengths are tiny")]
                let (cs, ce) = (s + step * k as f32, s + step * (k + 1) as f32);
                let local = i - chars.start;
                if let Some(slot) = start.get_mut(local) {
                    *slot = Some(slot.map_or(cs, |v: f32| v.min(cs)));
                }
                if let Some(slot) = end.get_mut(local) {
                    *slot = Some(slot.map_or(ce, |v: f32| v.max(ce)));
                }
            }
        }
        let (start, end) = fill_gaps(&start, &end, floor, ceil);
        Alignment {
            text: self.text(chars),
            start,
            end,
        }
    }
}

impl UtteranceMap {
    fn new(request: &SynthesisRequest, script: &str, source: &Range<usize>, offset: usize) -> Self {
        let text = request.text.as_str();
        let mut char_bytes: Vec<usize> = text.char_indices().map(|(at, _)| at).collect();
        char_bytes.push(text.len());
        Self {
            char_bytes,
            map: TextMap::new(text, script.get(source.clone()).unwrap_or_default()),
            offset,
        }
    }
}

/// Fills unknown timings from neighbours and enforces monotone, bounded output.
fn fill_gaps(
    start: &[Option<f32>],
    end: &[Option<f32>],
    floor: f32,
    ceil: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = start.len();
    let mut next_start = vec![ceil; len + 1];
    for i in (0..len).rev() {
        let following = next_start.get(i + 1).copied().unwrap_or(ceil);
        if let Some(slot) = next_start.get_mut(i) {
            *slot = start.get(i).copied().flatten().unwrap_or(following);
        }
    }
    let mut out_start = Vec::with_capacity(len);
    let mut out_end = Vec::with_capacity(len);
    let mut previous_end = floor;
    for i in 0..len {
        let following = next_start.get(i + 1).copied().unwrap_or(ceil);
        let s = start
            .get(i)
            .copied()
            .flatten()
            .unwrap_or(previous_end)
            .clamp(floor, ceil);
        let s = s.max(out_start.last().copied().unwrap_or(floor));
        let e = end
            .get(i)
            .copied()
            .flatten()
            .unwrap_or(following)
            .clamp(s, ceil);
        out_start.push(s);
        out_end.push(e);
        previous_end = e;
    }
    (out_start, out_end)
}

/// Splits `alignment` into sentences at terminal punctuation and line breaks.
#[must_use]
pub fn sentences(alignment: &Alignment) -> Vec<SentenceTiming> {
    let chars: Vec<char> = alignment.text.chars().collect();
    let mut out = Vec::new();
    let mut begin = 0_usize;
    for i in 0..chars.len() {
        let ch = chars.get(i).copied().unwrap_or(' ');
        let next = chars.get(i + 1).copied();
        let terminal = matches!(ch, '。' | '！' | '？' | '!' | '?' | '…' | '\n')
            || (ch == '.' && next.is_none_or(char::is_whitespace));
        let closes =
            next.is_some_and(|n| matches!(n, '”' | '’' | '」' | '』' | ')' | '）' | '"' | '\''));
        if terminal && !closes || i + 1 == chars.len() {
            if let Some(sentence) = sentence(&chars, alignment, begin..i + 1) {
                out.push(sentence);
            }
            begin = i + 1;
        }
    }
    out
}

fn sentence(chars: &[char], alignment: &Alignment, range: Range<usize>) -> Option<SentenceTiming> {
    let spoken: Vec<usize> = range
        .clone()
        .filter(|&i| chars.get(i).copied().is_some_and(is_spoken_char))
        .collect();
    let (&first, &last) = (spoken.first()?, spoken.last()?);
    let leading = range
        .clone()
        .take_while(|&i| chars.get(i).is_some_and(|c| c.is_whitespace()))
        .count();
    let trailing = range
        .clone()
        .rev()
        .take_while(|&i| chars.get(i).is_some_and(|c| c.is_whitespace()))
        .count();
    let trimmed = range.start + leading..range.end - trailing;
    Some(SentenceTiming {
        text: chars
            .get(trimmed.clone())
            .unwrap_or_default()
            .iter()
            .collect(),
        chars: trimmed,
        start: alignment.start.get(first).copied().unwrap_or(0.0),
        end: alignment.end.get(last).copied().unwrap_or(0.0),
    })
}

/// Whether `ch` is pronounced (not whitespace or punctuation).
fn is_spoken_char(ch: char) -> bool {
    !ch.is_whitespace() && !is_punctuation(ch)
}

#[cfg(test)]
mod tests {
    use index_tts_rs_core::{
        align::SpanTiming,
        config::{SpeakerSource, SynthesisRequest, SynthesisText, VoiceId},
    };
    use index_tts_rs_director::plan_utterances;

    use super::{Alignment, Transcript, sentences};
    use crate::ServiceError;

    fn transcript(script: &str) -> Result<Transcript, ServiceError> {
        let request = SynthesisRequest::builder()
            .text(SynthesisText::new(script)?)
            .speaker(SpeakerSource::VoiceId(VoiceId::new("tyr")?))
            .build();
        Ok(Transcript::new(script, &plan_utterances(&request, 600)?))
    }

    #[test]
    fn test_should_strip_controls_from_spoken_text() -> Result<(), ServiceError> {
        let t = transcript("你好。[[pause=500]][[emo=calm:0.4]]世界！")?;
        assert_eq!(t.spoken_text(), "你好。世界！");
        // Utterance 1 is "世界！"; its first two characters are spoken chars 3..5.
        assert_eq!(t.project(1, &(0..2)), 3..5);
        Ok(())
    }

    #[test]
    fn test_should_project_joined_lines_back_onto_the_script() -> Result<(), ServiceError> {
        // The reading unit joins the lines as "第一行， 第二行"; "第二行" is request chars 5..8.
        let t = transcript("第一行\n第二行\n\n第三段")?;
        assert_eq!(t.project(0, &(5..8)), 4..7);
        assert_eq!(t.text(&t.project(1, &(0..3))), "第三段");
        Ok(())
    }

    #[test]
    fn test_should_partition_segments_at_pronounced_characters() -> Result<(), ServiceError> {
        let t = transcript("你好！」 然后 ok")?;
        assert_eq!(t.segment_range(0..2), 0..5);
        assert_eq!(t.segment_range(4..7), 5..8);
        Ok(())
    }

    #[test]
    fn test_should_fill_punctuation_from_neighbours() -> Result<(), ServiceError> {
        let t = transcript("你好，世界")?;
        let spans = vec![(0..2, 0.1, 0.5), (3..5, 0.9, 1.3)];
        let a = t.alignment(&(0..5), &spans, 0.0, 2.0);
        assert_eq!(a.text, "你好，世界");
        let close =
            |got: &[f32], want: &[f32]| got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-5);
        assert!(close(&a.start, &[0.1, 0.3, 0.5, 0.9, 1.1]), "{:?}", a.start);
        assert!(close(&a.end, &[0.3, 0.5, 0.9, 1.1, 1.3]), "{:?}", a.end);
        Ok(())
    }

    #[test]
    fn test_should_project_engine_spans_onto_spoken_characters() -> Result<(), ServiceError> {
        let t = transcript("你好，世界。")?;
        let engine = [
            span(0..1, 0.26, 0.44),
            span(1..2, 0.44, 0.64),
            span(3..4, 1.14, 1.40),
            span(4..5, 1.40, 1.62),
        ];
        let spans = t.project_spans(0, &engine, 0.0);
        let a = t.alignment(&(0..6), &spans, 0.0, 2.0);
        assert!(
            (a.start[1] - 0.44).abs() < 1e-6,
            "{:?} from {spans:?}",
            a.start
        );
        Ok(())
    }

    fn span(chars: std::ops::Range<usize>, start: f32, end: f32) -> SpanTiming {
        let mut timing = SpanTiming::default();
        timing.chars = chars;
        timing.start_secs = start;
        timing.end_secs = end;
        timing
    }

    #[test]
    fn test_should_keep_alignment_monotone_and_bounded() -> Result<(), ServiceError> {
        let t = transcript("abc")?;
        let spans = vec![(0..1, 0.5, 0.6), (1..2, 0.2, 0.3), (2..3, 5.0, 6.0)];
        let a = t.alignment(&(0..3), &spans, 0.0, 1.0);
        assert!(a.start.windows(2).all(|w| w[0] <= w[1]));
        assert!(a.start.iter().zip(&a.end).all(|(s, e)| s <= e && *e <= 1.0));
        Ok(())
    }

    #[test]
    fn test_should_split_sentences_at_terminal_marks() {
        let alignment = Alignment {
            text: "Hi there. 3.5 is fine! 好的。".to_owned(),
            start: (0..26).map(|i| i as f32 * 0.1).collect(),
            end: (1..27).map(|i| i as f32 * 0.1).collect(),
        };
        let found = sentences(&alignment);
        let texts: Vec<&str> = found.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["Hi there.", "3.5 is fine!", "好的。"]);
        let first = &found[0];
        assert_eq!(first.chars, 0..9);
        assert!((first.start - 0.0).abs() < 1e-6 && (first.end - 0.8).abs() < 1e-6);
    }
}

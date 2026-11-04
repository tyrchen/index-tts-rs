//! Script verification: what the recognizer heard against what the script says.
//!
//! Both texts become pronounced units (the private `units` module) — CJK characters compared by
//! toneless pinyin, numbers by value, Latin words letter by letter — and are aligned by edit
//! distance. Each gap between matched runs becomes a [`Finding`]: a `repeat` (stutter),
//! `insertion`, `omission`, `substitution`, or `number` read as another value. Severity combines
//! the size of the gap with the forced aligner's evidence ([`AlignEvidence`]): a one-character
//! difference the aligner does not also find weak is most likely the recognizer mishearing a rare
//! word.

pub mod numbers;
mod units;

use std::ops::Range;

use units::{Unit, units};

use crate::{
    error::{AsrError, Result},
    tokens::HeardToken,
};

/// Most `script units × heard units` cells one comparison may fill.
const MAX_CELLS: usize = 4_000_000;

/// What kind of difference a finding is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FindingKind {
    /// Extra units that repeat their neighbours: a stutter or a re-read.
    Repeat,
    /// Extra units that are not a repeat.
    Insertion,
    /// Script units that were not heard.
    Omission,
    /// Script units heard as other units.
    Substitution,
    /// A number heard with another value.
    Number,
}

/// How much a finding matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Severity {
    /// Probably the recognizer, not the recording: one unit the aligner does not doubt.
    Info,
    /// Two or more units differ, or one the aligner also finds weak.
    Warning,
    /// A repeat, or a number read as another value.
    Error,
}

/// One difference between the script and the recording.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    /// Kind of difference.
    pub kind: FindingKind,
    /// Severity.
    pub severity: Severity,
    /// Script text of the gap (empty for insertions and repeats).
    pub expected: String,
    /// Recognized text of the gap (empty for omissions).
    pub heard: String,
    /// Character range in the script (`start == end` at an insertion point).
    pub chars: Range<usize>,
    /// Start in seconds.
    pub start: f32,
    /// End in seconds.
    pub end: f32,
    /// Units involved: the larger side, a Latin word counting once.
    pub sounds: usize,
    /// Whether the forced aligner finds a unit in or next to the gap weak.
    pub align_weak: bool,
}

/// What forced alignment says about the script, indexed by script character.
#[derive(Debug, Clone, Copy, Default)]
pub struct AlignEvidence<'a> {
    /// Characters of units the aligner finds weakly supported (`quality.weak`).
    pub weak: &'a [usize],
    /// Aligned start of every script character.
    pub starts: &'a [f32],
    /// Aligned end of every script character.
    pub ends: &'a [f32],
}

/// Recognized units with their source tokens.
struct Heard<'a> {
    text: Vec<char>,
    units: Vec<Unit>,
    token_of: Vec<usize>,
    tokens: &'a [HeardToken],
}

impl<'a> Heard<'a> {
    fn new(tokens: &'a [HeardToken]) -> Self {
        let mut text = String::new();
        let mut token_of = Vec::new();
        for (k, token) in tokens.iter().enumerate() {
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
                token_of.push(k);
            }
            for c in token.text.chars() {
                text.push(c);
                token_of.push(k);
            }
        }
        Self {
            units: units(&text),
            text: text.chars().collect(),
            token_of,
            tokens,
        }
    }

    fn span(&self, range: Range<usize>) -> Option<(f32, f32)> {
        let first = self.units.get(range.start)?;
        let last = self.units.get(range.end.checked_sub(1)?)?;
        let start = self.token(first.chars.start)?.start;
        let end = self.token(last.chars.end.checked_sub(1)?)?.end;
        Some((start, end.max(start)))
    }

    fn token(&self, char_index: usize) -> Option<&HeardToken> {
        self.tokens.get(*self.token_of.get(char_index)?)
    }

    fn text(&self, range: Range<usize>) -> String {
        slice(&self.text, &self.units, range)
    }
}

/// Source text spanned by `units[range]`.
fn slice(text: &[char], units: &[Unit], range: Range<usize>) -> String {
    let (Some(first), Some(last)) = (
        units.get(range.start),
        range.end.checked_sub(1).and_then(|k| units.get(k)),
    ) else {
        return String::new();
    };
    text.get(first.chars.start..last.chars.end)
        .map(|s| s.iter().collect())
        .unwrap_or_default()
}

/// One step of the edit-distance alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Match(usize, usize),
    Substitute(usize, usize),
    Delete(usize),
    Insert(usize),
}

/// A gap between matched runs: script units `a` against heard units `b`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Gap {
    a: Range<usize>,
    b: Range<usize>,
    sounds_a: Vec<usize>,
    sounds_b: Vec<usize>,
}

/// Compares `script` with what was `heard`, using the aligner's evidence for severity.
///
/// # Errors
///
/// Returns [`AsrError::TooLong`] when the texts are too long to align in one comparison.
///
/// # Examples
///
/// ```
/// use index_tts_rs_asr::{
///     HeardToken,
///     compare::{AlignEvidence, FindingKind, Severity, compare},
/// };
///
/// let heard: Vec<HeardToken> = "从北北边吹来的风"
///     .chars()
///     .enumerate()
///     .map(|(i, c)| HeardToken { text: c.to_string(), start: i as f32 * 0.2, end: (i + 1) as f32 * 0.2 })
///     .collect();
/// let findings = compare("从北边吹来的风。", &heard, AlignEvidence::default())?;
/// assert_eq!(findings.len(), 1);
/// assert_eq!(findings[0].kind, FindingKind::Repeat);
/// assert_eq!(findings[0].severity, Severity::Error);
/// assert_eq!(findings[0].heard, "北");
/// # Ok::<(), index_tts_rs_asr::AsrError>(())
/// ```
pub fn compare(
    script: &str,
    heard: &[HeardToken],
    evidence: AlignEvidence<'_>,
) -> Result<Vec<Finding>> {
    let text: Vec<char> = script.chars().collect();
    let a = units(script);
    let heard = Heard::new(heard);
    let b = &heard.units;
    if a.len().saturating_mul(b.len()) > MAX_CELLS {
        return Err(AsrError::TooLong(format!(
            "comparing {} script units with {} recognized units exceeds one comparison",
            a.len(),
            b.len()
        )));
    }
    let steps = align(&a, b);
    let gaps: Vec<Gap> = merge(widen(gaps(&steps, &a, b), &steps, &a, b))
        .into_iter()
        .map(|gap| trim_equal_words(gap, &a, b))
        .collect();
    Ok(gaps
        .into_iter()
        .filter(|gap| !is_erhua_only(gap, &a, b))
        .map(|gap| finding(&gap, &text, &a, &heard, evidence))
        .collect())
}

/// Edit-distance alignment (unit costs), preferring matches when tracing back.
fn align(a: &[Unit], b: &[Unit]) -> Vec<Step> {
    let table = Table::fill(a, b);
    let same = |i: usize, j: usize| matches!((a.get(i), b.get(j)), (Some(x), Some(y)) if x.same(y));
    let mut steps = Vec::with_capacity(a.len().max(b.len()));
    let (mut i, mut j) = (a.len(), b.len());
    while i > 0 || j > 0 {
        let here = table.get(i, j);
        if i > 0 && j > 0 && same(i - 1, j - 1) && here == table.get(i - 1, j - 1) {
            steps.push(Step::Match(i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && here == table.get(i - 1, j - 1).saturating_add(1) {
            steps.push(Step::Substitute(i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if j > 0 && here == table.get(i, j - 1).saturating_add(1) {
            steps.push(Step::Insert(j - 1));
            j -= 1;
        } else if i > 0 {
            steps.push(Step::Delete(i - 1));
            i -= 1;
        } else {
            steps.push(Step::Insert(j - 1));
            j -= 1;
        }
    }
    steps.reverse();
    steps
}

/// Edit distances `d[i][j]` between `a[..i]` and `b[..j]`, row-major.
struct Table {
    width: usize,
    cells: Vec<u32>,
}

impl Table {
    fn fill(a: &[Unit], b: &[Unit]) -> Self {
        let width = b.len() + 1;
        let mut table = Self {
            width,
            cells: vec![0; (a.len() + 1) * width],
        };
        for j in 0..width {
            table.set(0, j, u32::try_from(j).unwrap_or(u32::MAX));
        }
        for (i, x) in a.iter().enumerate() {
            table.set(i + 1, 0, u32::try_from(i + 1).unwrap_or(u32::MAX));
            for (j, y) in b.iter().enumerate() {
                let diag = table.get(i, j).saturating_add(u32::from(!x.same(y)));
                let up = table.get(i, j + 1).saturating_add(1);
                let left = table.get(i + 1, j).saturating_add(1);
                table.set(i + 1, j + 1, diag.min(up).min(left));
            }
        }
        table
    }

    fn get(&self, i: usize, j: usize) -> u32 {
        self.cells
            .get(i * self.width + j)
            .copied()
            .unwrap_or(u32::MAX)
    }

    fn set(&mut self, i: usize, j: usize, value: u32) {
        if let Some(cell) = self.cells.get_mut(i * self.width + j) {
            *cell = value;
        }
    }
}

/// Maximal runs of non-matching steps, with the sound ids each side contributes.
fn gaps(steps: &[Step], a: &[Unit], b: &[Unit]) -> Vec<Gap> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    let mut open: Option<Gap> = None;
    for step in steps {
        let (ni, nj) = match *step {
            Step::Match(..) => {
                out.extend(open.take());
                (i + 1, j + 1)
            }
            Step::Substitute(..) => (i + 1, j + 1),
            Step::Delete(..) => (i + 1, j),
            Step::Insert(..) => (i, j + 1),
        };
        if !matches!(step, Step::Match(..)) {
            let gap = open.get_or_insert_with(|| Gap {
                a: i..i,
                b: j..j,
                sounds_a: Vec::new(),
                sounds_b: Vec::new(),
            });
            gap.a.end = ni;
            gap.b.end = nj;
            if ni > i {
                gap.sounds_a.push(sound_id(a, i));
            }
            if nj > j {
                gap.sounds_b.push(sound_id(b, j));
            }
        }
        (i, j) = (ni, nj);
    }
    out.extend(open);
    out
}

/// A unit's sound: its own index, or (letters) its word shifted past every index.
fn sound_id(units: &[Unit], k: usize) -> usize {
    units
        .get(k)
        .and_then(Unit::word)
        .map_or(k, |word| usize::MAX / 2 + word)
}

/// Widens every gap touching a Latin word to the whole word on both sides.
fn widen(mut gaps: Vec<Gap>, steps: &[Step], a: &[Unit], b: &[Unit]) -> Vec<Gap> {
    let mut b_of = vec![None; a.len()];
    let mut a_of = vec![None; b.len()];
    for step in steps {
        if let Step::Match(i, j) = *step {
            if let Some(slot) = b_of.get_mut(i) {
                *slot = Some(j);
            }
            if let Some(slot) = a_of.get_mut(j) {
                *slot = Some(i);
            }
        }
    }
    for gap in &mut gaps {
        loop {
            let before = (gap.a.clone(), gap.b.clone());
            gap.a = word_span(a, gap.a.clone());
            gap.b = word_span(b, gap.b.clone());
            for i in gap.a.clone() {
                if let Some(j) = b_of.get(i).copied().flatten() {
                    gap.b = gap.b.start.min(j)..gap.b.end.max(j + 1);
                }
            }
            for j in gap.b.clone() {
                if let Some(i) = a_of.get(j).copied().flatten() {
                    gap.a = gap.a.start.min(i)..gap.a.end.max(i + 1);
                }
            }
            if (gap.a.clone(), gap.b.clone()) == before {
                break;
            }
        }
    }
    gaps
}

/// `range` grown to whole words where it cuts into (or, empty, sits inside) a Latin word.
fn word_span(units: &[Unit], range: Range<usize>) -> Range<usize> {
    let word = |k: usize| units.get(k).and_then(Unit::word);
    let (mut start, mut end) = (range.start, range.end);
    let inside =
        start == end && start > 0 && word(start - 1).is_some() && word(start - 1) == word(start);
    if start < end || inside {
        if let Some(w) = word(start) {
            while start > 0 && word(start - 1) == Some(w) {
                start -= 1;
            }
        }
        let tail = if inside {
            word(end)
        } else {
            end.checked_sub(1).and_then(word)
        };
        if let Some(w) = tail {
            while word(end) == Some(w) {
                end += 1;
            }
        }
    }
    start..end
}

/// Units `[start, end)` of the whole Latin word starting at `start`, if one does.
fn word_at(units: &[Unit], start: usize) -> Option<Range<usize>> {
    let word = units.get(start)?.word()?;
    if start > 0 && units.get(start - 1).and_then(Unit::word) == Some(word) {
        return None;
    }
    let len = units
        .get(start..)?
        .iter()
        .take_while(|u| u.word() == Some(word))
        .count();
    Some(start..start + len)
}

/// Units of the whole Latin word ending at `end`, if one does.
fn word_before(units: &[Unit], end: usize) -> Option<Range<usize>> {
    let word = units.get(end.checked_sub(1)?)?.word()?;
    if units.get(end).and_then(Unit::word) == Some(word) {
        return None;
    }
    let len = units
        .get(..end)?
        .iter()
        .rev()
        .take_while(|u| u.word() == Some(word))
        .count();
    Some(end - len..end)
}

fn same_units(a: &[Unit], b: &[Unit]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same(y))
}

/// Drops whole words heard exactly as written from the edges of a widened gap, so that `the
/// final quarterly report` heard as `the report` is an omission of `final quarterly`.
fn trim_equal_words(mut gap: Gap, a: &[Unit], b: &[Unit]) -> Gap {
    while let (Some(x), Some(y)) = (word_at(a, gap.a.start), word_at(b, gap.b.start))
        && x.end <= gap.a.end
        && y.end <= gap.b.end
        && same_units(
            a.get(x.clone()).unwrap_or_default(),
            b.get(y.clone()).unwrap_or_default(),
        )
    {
        gap.a.start = x.end;
        gap.b.start = y.end;
    }
    while let (Some(x), Some(y)) = (word_before(a, gap.a.end), word_before(b, gap.b.end))
        && x.start >= gap.a.start
        && y.start >= gap.b.start
        && same_units(
            a.get(x.clone()).unwrap_or_default(),
            b.get(y.clone()).unwrap_or_default(),
        )
    {
        gap.a.end = x.start;
        gap.b.end = y.start;
    }
    let inside = |units: &[Unit], range: Range<usize>| -> Vec<usize> {
        range.map(|k| sound_id(units, k)).collect()
    };
    let (keep_a, keep_b) = (inside(a, gap.a.clone()), inside(b, gap.b.clone()));
    gap.sounds_a.retain(|id| keep_a.contains(id));
    gap.sounds_b.retain(|id| keep_b.contains(id));
    gap
}

/// Merges gaps whose ranges overlap after widening.
fn merge(gaps: Vec<Gap>) -> Vec<Gap> {
    let mut out: Vec<Gap> = Vec::with_capacity(gaps.len());
    for gap in gaps {
        match out.last_mut() {
            Some(last) if gap.a.start < last.a.end || gap.b.start < last.b.end => {
                last.a = last.a.start.min(gap.a.start)..last.a.end.max(gap.a.end);
                last.b = last.b.start.min(gap.b.start)..last.b.end.max(gap.b.end);
                last.sounds_a.extend(gap.sounds_a);
                last.sounds_b.extend(gap.sounds_b);
            }
            _ => out.push(gap),
        }
    }
    out
}

fn is_erhua_only(gap: &Gap, a: &[Unit], b: &[Unit]) -> bool {
    let mut involved = units_in(a, gap.a.clone()).chain(units_in(b, gap.b.clone()));
    involved.all(Unit::is_erhua)
}

fn units_in(units: &[Unit], range: Range<usize>) -> impl Iterator<Item = &Unit> {
    units.get(range).into_iter().flatten()
}

/// Whether `extra` equals the `extra.len()` units before or after the gap on either side.
fn is_repeat(gap: &Gap, a: &[Unit], b: &[Unit]) -> bool {
    let extra = b.get(gap.b.clone()).unwrap_or_default();
    let k = extra.len();
    let echoes = |side: &[Unit], range: Range<usize>| {
        side.get(range)
            .is_some_and(|ctx| ctx.len() == k && ctx.iter().zip(extra).all(|(x, y)| x.same(y)))
    };
    k > 0
        && (echoes(a, gap.a.end..gap.a.end + k)
            || echoes(a, gap.a.start.saturating_sub(k)..gap.a.start)
            || echoes(b, gap.b.end..gap.b.end + k)
            || echoes(b, gap.b.start.saturating_sub(k)..gap.b.start))
}

fn kind(gap: &Gap, a: &[Unit], b: &[Unit]) -> FindingKind {
    let (miss, extra) = (
        a.get(gap.a.clone()).unwrap_or_default(),
        b.get(gap.b.clone()).unwrap_or_default(),
    );
    let numbers_both = miss.iter().any(Unit::is_number) && extra.iter().any(Unit::is_number);
    if numbers_both || miss.iter().chain(extra).any(Unit::is_written_number) {
        return FindingKind::Number;
    }
    match (miss.is_empty(), extra.is_empty()) {
        (true, false) if is_repeat(gap, a, b) => FindingKind::Repeat,
        (true, false) => FindingKind::Insertion,
        (false, true) => FindingKind::Omission,
        _ => FindingKind::Substitution,
    }
}

fn distinct(ids: &[usize]) -> usize {
    let mut seen: Vec<usize> = Vec::with_capacity(ids.len());
    for id in ids {
        if !seen.contains(id) {
            seen.push(*id);
        }
    }
    seen.len()
}

fn finding(
    gap: &Gap,
    text: &[char],
    a: &[Unit],
    heard: &Heard<'_>,
    evidence: AlignEvidence<'_>,
) -> Finding {
    let kind = kind(gap, a, &heard.units);
    let sounds = distinct(&gap.sounds_a).max(distinct(&gap.sounds_b)).max(1);
    let chars = script_chars(gap, a, text.len());
    let near = near_chars(gap, a);
    let align_weak = evidence.weak.iter().any(|w| near.contains(w));
    let severity = match kind {
        FindingKind::Repeat | FindingKind::Number => Severity::Error,
        _ if sounds >= 2 || align_weak => Severity::Warning,
        _ => Severity::Info,
    };
    let (start, end) = if gap.b.is_empty() {
        omission_span(gap, &chars, heard, evidence)
    } else {
        heard.span(gap.b.clone()).unwrap_or_default()
    };
    Finding {
        kind,
        severity,
        expected: slice(text, a, gap.a.clone()),
        heard: heard.text(gap.b.clone()),
        chars,
        start,
        end,
        sounds,
        align_weak,
    }
}

/// Script characters of the gap; an empty gap sits before the next script unit.
fn script_chars(gap: &Gap, a: &[Unit], len: usize) -> Range<usize> {
    match (
        a.get(gap.a.start),
        gap.a.end.checked_sub(1).and_then(|k| a.get(k)),
    ) {
        (Some(first), Some(last)) if !gap.a.is_empty() => first.chars.start..last.chars.end,
        (Some(next), _) => next.chars.start..next.chars.start,
        _ => {
            let at = a.last().map_or(len, |u| u.chars.end);
            at..at
        }
    }
}

/// Characters of the gap and one script unit on each side.
fn near_chars(gap: &Gap, a: &[Unit]) -> Range<usize> {
    let lo = gap.a.start.saturating_sub(1);
    let hi = (gap.a.end + 1).min(a.len());
    match (a.get(lo), hi.checked_sub(1).and_then(|k| a.get(k))) {
        (Some(first), Some(last)) => first.chars.start..last.chars.end,
        _ => 0..0,
    }
}

/// Where omitted characters are: their aligned span, else between the heard neighbours.
fn omission_span(
    gap: &Gap,
    chars: &Range<usize>,
    heard: &Heard<'_>,
    evidence: AlignEvidence<'_>,
) -> (f32, f32) {
    let aligned = (
        evidence.starts.get(chars.start).copied(),
        chars
            .end
            .checked_sub(1)
            .and_then(|k| evidence.ends.get(k).copied()),
    );
    if let (Some(start), Some(end)) = aligned
        && end > start
    {
        return (start, end);
    }
    let before = gap
        .b
        .start
        .checked_sub(1)
        .and_then(|k| heard.span(k..k + 1))
        .map_or(0.0, |(_, end)| end);
    let after = heard
        .span(gap.b.start..gap.b.start + 1)
        .map_or(before, |(start, _)| start);
    (before, after.max(before))
}

#[cfg(test)]
mod tests;

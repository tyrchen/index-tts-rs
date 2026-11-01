//! Monotone character mapping between a derived text and the text it was derived from.
//!
//! Text passes through several rewrites before the GPT sees it (control stripping, line joins,
//! punctuation normalization, number verbalization, case folding, pronunciation markers). Rather
//! than instrumenting every rewrite, [`TextMap`] recovers the correspondence with a Myers diff over
//! folded characters: equal runs map one-to-one and each replaced run maps onto its counterpart as
//! a whole, so `2026 -> 二零二六` still lands on the right source characters.

use std::ops::Range;

/// Edit-distance budget for the diff. Beyond it the map degrades to proportional positions,
/// which bounds time and memory (`O(D^2)`) for pathological inputs.
const MAX_EDIT_DISTANCE: usize = 1_024;

/// Character-level correspondence from a `derived` string onto its `source` string.
#[derive(Debug, Clone)]
pub struct TextMap {
    /// Byte offset of each `derived` character, plus a final entry equal to its byte length.
    derived_bytes: Vec<usize>,
    /// Ordered hunks covering both strings completely.
    hunks: Vec<Hunk>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Hunk {
    derived: Range<usize>,
    source: Range<usize>,
}

impl TextMap {
    /// Builds the map from `derived` onto `source`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::align::TextMap;
    ///
    /// let map = TextMap::new("在二零二六年.", "在2026年。");
    /// // "二零二六" (bytes 3..15) came from "2026" (chars 1..5).
    /// assert_eq!(map.source_chars(3..15), 1..5);
    /// // The folded full stop maps back onto "。".
    /// assert_eq!(map.source_chars(18..19), 6..7);
    /// ```
    #[must_use]
    pub fn new(derived: &str, source: &str) -> Self {
        let derived_keys: Vec<char> = derived.chars().map(fold).collect();
        let source_keys: Vec<char> = source.chars().map(fold).collect();
        let mut derived_bytes: Vec<usize> = derived.char_indices().map(|(at, _)| at).collect();
        derived_bytes.push(derived.len());
        let hunks = diff_hunks(&derived_keys, &source_keys).unwrap_or_else(|| {
            vec![Hunk {
                derived: 0..derived_keys.len(),
                source: 0..source_keys.len(),
            }]
        });
        Self {
            derived_bytes,
            hunks,
        }
    }

    /// Maps a byte range of the derived text onto the source characters it came from.
    ///
    /// The result is empty when the derived bytes were inserted by a rewrite (for example a
    /// language prefix or a joined-line comma); it is then positioned where the insertion sits.
    #[must_use]
    pub fn source_chars(&self, derived_bytes: Range<usize>) -> Range<usize> {
        let first = self.char_at_byte(derived_bytes.start);
        if derived_bytes.end <= derived_bytes.start {
            let at = self.source_start(first);
            return at..at;
        }
        let last = self.char_at_byte(derived_bytes.end.saturating_sub(1));
        let start = self.source_start(first);
        start..self.source_end(last).max(start)
    }

    fn char_at_byte(&self, byte: usize) -> usize {
        match self.derived_bytes.binary_search(&byte) {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        }
    }

    /// The hunk containing derived character `ch`; pure source insertions are skipped.
    fn hunk_of(&self, ch: usize) -> Option<&Hunk> {
        let index = self.hunks.partition_point(|hunk| hunk.derived.end <= ch);
        self.hunks.get(index)
    }

    fn source_start(&self, ch: usize) -> usize {
        self.hunk_of(ch).map_or_else(
            || self.hunks.last().map_or(0, |hunk| hunk.source.end),
            |hunk| {
                let offset = ch.saturating_sub(hunk.derived.start);
                hunk.source.start + offset * hunk.source.len() / hunk.derived.len().max(1)
            },
        )
    }

    /// Exclusive source end of derived character `ch`.
    fn source_end(&self, ch: usize) -> usize {
        self.hunk_of(ch).map_or_else(
            || self.hunks.last().map_or(0, |hunk| hunk.source.end),
            |hunk| {
                let offset = ch.saturating_sub(hunk.derived.start).saturating_add(1);
                hunk.source.start + (offset * hunk.source.len()).div_ceil(hunk.derived.len().max(1))
            },
        )
    }
}

/// Folds characters so rewrites that only change case, width, or punctuation style compare
/// equal: letters are lowercased, every punctuation mark becomes one class, every whitespace
/// character another.
fn fold(ch: char) -> char {
    if ch.is_whitespace() {
        ' '
    } else if is_punctuation(ch) {
        '\u{1}'
    } else {
        ch.to_lowercase().next().unwrap_or(ch)
    }
}

/// Whether `ch` is a punctuation mark in ASCII, CJK, or general punctuation blocks.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::align::is_punctuation;
///
/// assert!(is_punctuation('，') && is_punctuation('!') && !is_punctuation('好'));
/// ```
#[must_use]
pub fn is_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(ch,
            '\u{2000}'..='\u{206F}'   // general punctuation (dashes, quotes, ellipsis)
            | '\u{3000}'..='\u{303F}' // CJK symbols and punctuation
            | '\u{FF01}'..='\u{FF0F}' // fullwidth ASCII punctuation
            | '\u{FF1A}'..='\u{FF20}'
            | '\u{FF3B}'..='\u{FF40}'
            | '\u{FF5B}'..='\u{FF65}'
            | '\u{00A1}'..='\u{00BF}'
            | '\u{FE30}'..='\u{FE4F}')
}

/// Myers diff producing alternating equal / replace hunks, or `None` past the edit budget.
fn diff_hunks(a: &[char], b: &[char]) -> Option<Vec<Hunk>> {
    let trace = myers_trace(a, b)?;
    let script = backtrack(&trace, a.len(), b.len());
    Some(group(&script, a.len(), b.len()))
}

/// Forward pass of Myers' O((N+M)D) algorithm. Round `d` stores the furthest `x` reached on
/// each diagonal `k in -d..=d` at index `k + d`.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "indices are bounded by the 16 KiB text cap; names follow Myers' paper"
)]
fn myers_trace(a: &[char], b: &[char]) -> Option<Vec<Vec<isize>>> {
    let n = a.len() as isize;
    let m = b.len() as isize;
    let max = a.len().saturating_add(b.len()).min(MAX_EDIT_DISTANCE);
    let mut trace: Vec<Vec<isize>> = Vec::new();
    for d in 0..=max {
        let d_i = d as isize;
        let mut round = vec![0_isize; 2 * d + 1];
        for k in (-d_i..=d_i).step_by(2) {
            let mut x = match trace.last() {
                None => 0,
                Some(prev) => {
                    let (down, from) = predecessor(prev, d_i, k);
                    if down { from } else { from + 1 }
                }
            };
            let mut y = x - k;
            while x < n && y < m && a.get(x as usize) == b.get(y as usize) {
                x += 1;
                y += 1;
            }
            if let Some(slot) = round.get_mut((k + d_i) as usize) {
                *slot = x;
            }
            if x >= n && y >= m {
                trace.push(round);
                return Some(trace);
            }
        }
        trace.push(round);
    }
    None
}

/// For diagonal `k` in round `d`, whether the path arrives by moving down (from `k + 1`) and the
/// `x` reached on that predecessor diagonal in round `d - 1` (`prev`).
#[allow(clippy::cast_sign_loss)]
fn predecessor(prev: &[isize], d: isize, k: isize) -> (bool, isize) {
    let at = |kk: isize| -> isize {
        let index = kk + d - 1;
        if index < 0 {
            -1
        } else {
            prev.get(index as usize).copied().unwrap_or(-1)
        }
    };
    let down = k == -d || (k != d && at(k - 1) < at(k + 1));
    (down, if down { at(k + 1) } else { at(k - 1) })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Equal,
    Delete,
    Insert,
}

/// Walks the trace backwards into an edit script (`Delete` consumes `a`, `Insert` consumes `b`).
#[allow(
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    reason = "indices are bounded by the 16 KiB text cap; names follow Myers' paper"
)]
fn backtrack(trace: &[Vec<isize>], n: usize, m: usize) -> Vec<Op> {
    let mut ops = Vec::with_capacity(n + m);
    let mut x = n as isize;
    let mut y = m as isize;
    for d in (1..trace.len()).rev() {
        let Some(prev) = trace.get(d - 1) else {
            break;
        };
        let d_i = d as isize;
        let k = x - y;
        let (down, prev_x) = predecessor(prev, d_i, k);
        let prev_k = if down { k + 1 } else { k - 1 };
        let prev_y = prev_x - prev_k;
        let (mid_x, mid_y) = if down {
            (prev_x, prev_y + 1)
        } else {
            (prev_x + 1, prev_y)
        };
        while x > mid_x && y > mid_y {
            ops.push(Op::Equal);
            x -= 1;
            y -= 1;
        }
        ops.push(if down { Op::Insert } else { Op::Delete });
        x = prev_x;
        y = prev_y;
    }
    while x > 0 && y > 0 {
        ops.push(Op::Equal);
        x -= 1;
        y -= 1;
    }
    ops.reverse();
    ops
}

/// Groups an edit script into hunks: each maximal run of non-equal ops becomes one replace hunk.
fn group(script: &[Op], n: usize, m: usize) -> Vec<Hunk> {
    let mut hunks: Vec<Hunk> = Vec::new();
    let (mut x, mut y) = (0_usize, 0_usize);
    let mut pending: Option<(usize, usize)> = None;
    for op in script {
        match op {
            Op::Equal => {
                if let Some((px, py)) = pending.take() {
                    hunks.push(Hunk {
                        derived: px..x,
                        source: py..y,
                    });
                }
                hunks.push(Hunk {
                    derived: x..x + 1,
                    source: y..y + 1,
                });
                x += 1;
                y += 1;
            }
            Op::Delete => {
                pending.get_or_insert((x, y));
                x += 1;
            }
            Op::Insert => {
                pending.get_or_insert((x, y));
                y += 1;
            }
        }
    }
    if let Some((px, py)) = pending {
        hunks.push(Hunk {
            derived: px..n.max(x),
            source: py..m.max(y),
        });
    }
    hunks
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rstest::rstest;

    use super::TextMap;

    fn byte_range(text: &str, chars: std::ops::Range<usize>) -> std::ops::Range<usize> {
        let at = |i: usize| text.char_indices().nth(i).map_or(text.len(), |(b, _)| b);
        at(chars.start)..at(chars.end)
    }

    #[rstest]
    #[case("hello world", "Hello World", 6..11, 6..11)]
    #[case("你好,世界.", "你好，世界。", 3..6, 3..6)]
    #[case("在二零二六年", "在2026年", 1..5, 1..5)]
    #[case("<|zh|> 你好", "你好", 0..7, 0..0)]
    #[case("<|zh|> 你好", "你好", 7..8, 0..1)]
    fn test_should_map_derived_chars_onto_source(
        #[case] derived: &str,
        #[case] source: &str,
        #[case] derived_chars: std::ops::Range<usize>,
        #[case] expected: std::ops::Range<usize>,
    ) {
        let map = TextMap::new(derived, source);
        assert_eq!(
            map.source_chars(byte_range(derived, derived_chars)),
            expected
        );
    }

    #[test]
    fn test_should_spread_a_replaced_run_proportionally() {
        let map = TextMap::new("百分之四十二", "42%");
        assert_eq!(map.source_chars(byte_range("百分之四十二", 0..6)), 0..3);
        let first = map.source_chars(byte_range("百分之四十二", 0..1));
        assert_eq!(first.start, 0);
    }

    #[test]
    fn test_should_degrade_to_proportional_mapping_beyond_edit_budget() {
        let derived: String = "a".repeat(3_000);
        let source: String = "b".repeat(3_000);
        let map = TextMap::new(&derived, &source);
        assert_eq!(map.source_chars(1_500..3_000), 1_500..3_000);
    }

    proptest! {
        #[test]
        fn test_should_map_monotonically_within_bounds(
            derived in "[a-c，。 1-3]{0,40}",
            source in "[a-c,. 1-3x]{0,40}",
        ) {
            let map = TextMap::new(&derived, &source);
            let source_len = source.chars().count();
            let mut last = 0..0;
            for (byte, ch) in derived.char_indices() {
                let range = map.source_chars(byte..byte + ch.len_utf8());
                prop_assert!(range.start <= range.end);
                prop_assert!(range.end <= source_len);
                prop_assert!(range.start >= last.start && range.end >= last.end);
                last = range;
            }
        }
    }
}

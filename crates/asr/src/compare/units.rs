//! Pronounced units of a script or of recognized text, and when two units sound the same.

use std::ops::Range;

use pinyin::ToPinyinMulti;

use super::numbers::{canonical, cn_digit, cn_value, en_values, is_cn_numeral, is_en_number_word};
use crate::tokens::is_cjk;

/// What a unit is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UnitKind {
    /// A CJK character with its toneless readings (several for polyphones).
    Cjk {
        /// The character.
        ch: char,
        /// Toneless pinyin readings (empty for kana and hangul).
        readings: Vec<&'static str>,
    },
    /// A number with the values it may stand for.
    Number {
        /// Canonical values.
        values: Vec<String>,
        /// Readings of a single Chinese numeral character (`一`, `三`), which also equals its
        /// homophones.
        readings: Vec<&'static str>,
    },
    /// One letter (or digit) of a Latin word.
    Letter {
        /// Lowercase letter.
        letter: char,
        /// Word ordinal within the text.
        word: usize,
    },
}

/// One pronounced unit and its character range in the source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unit {
    /// Kind and comparison keys.
    pub kind: UnitKind,
    /// Character (Unicode scalar) range in the source.
    pub chars: Range<usize>,
}

impl Unit {
    /// Whether the two units would be heard the same.
    pub fn same(&self, other: &Self) -> bool {
        match (&self.kind, &other.kind) {
            (
                UnitKind::Cjk {
                    ch: a,
                    readings: ra,
                },
                UnitKind::Cjk {
                    ch: b,
                    readings: rb,
                },
            ) => a == b || overlaps(ra, rb),
            (UnitKind::Number { values: va, .. }, UnitKind::Number { values: vb, .. }) => {
                overlaps(va, vb)
            }
            (UnitKind::Number { readings: rn, .. }, UnitKind::Cjk { readings: rc, .. })
            | (UnitKind::Cjk { readings: rc, .. }, UnitKind::Number { readings: rn, .. }) => {
                overlaps(rn, rc)
            }
            (UnitKind::Letter { letter: a, .. }, UnitKind::Letter { letter: b, .. }) => a == b,
            _ => false,
        }
    }

    /// A number that is more than a single numeral character (`2026`, `三十`, `fifteen`).
    pub const fn is_written_number(&self) -> bool {
        matches!(&self.kind, UnitKind::Number { readings, .. } if readings.is_empty())
    }

    /// Any number unit.
    pub const fn is_number(&self) -> bool {
        matches!(self.kind, UnitKind::Number { .. })
    }

    /// The erhua suffix `儿`, which speakers add or drop freely.
    pub const fn is_erhua(&self) -> bool {
        matches!(self.kind, UnitKind::Cjk { ch: '儿', .. })
    }

    /// Word ordinal of a letter.
    pub const fn word(&self) -> Option<usize> {
        match self.kind {
            UnitKind::Letter { word, .. } => Some(word),
            _ => None,
        }
    }
}

fn overlaps<T: PartialEq>(a: &[T], b: &[T]) -> bool {
    a.iter().any(|x| b.contains(x))
}

/// Toneless pinyin readings of `c` (all of a polyphone's), deduplicated.
fn readings(c: char) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    if let Some(multi) = c.to_pinyin_multi() {
        for reading in multi {
            let plain = reading.plain();
            if !out.contains(&plain) {
                out.push(plain);
            }
        }
    }
    out
}

/// Splits `text` into pronounced units. `<word|pronunciation>` overrides count as `word`.
pub(crate) fn units(text: &str) -> Vec<Unit> {
    let chars: Vec<char> = text.chars().collect();
    let skipped = override_suffixes(&chars);
    let mut scan = Scanner {
        chars: &chars,
        skipped: &skipped,
        out: Vec::new(),
        words: 0,
    };
    let mut i = 0;
    while i < chars.len() {
        i = scan.step(i);
    }
    scan.out
}

/// Marks the `|pronunciation>` part and the `<` of every `<word|pronunciation>` override.
fn override_suffixes(chars: &[char]) -> Vec<bool> {
    let mut skipped = vec![false; chars.len()];
    let mut i = 0;
    while i < chars.len() {
        if chars.get(i) == Some(&'<') {
            let close = (i + 1..chars.len()).find(|&k| matches!(chars.get(k), Some('<' | '>')));
            let bar = (i + 1..close.unwrap_or(i)).find(|&k| chars.get(k) == Some(&'|'));
            if let (Some(close), Some(bar)) = (close, bar)
                && chars.get(close) == Some(&'>')
                && bar > i + 1
            {
                for k in (bar..=close).chain([i]) {
                    if let Some(flag) = skipped.get_mut(k) {
                        *flag = true;
                    }
                }
                i = close + 1;
                continue;
            }
        }
        i += 1;
    }
    skipped
}

struct Scanner<'a> {
    chars: &'a [char],
    skipped: &'a [bool],
    out: Vec<Unit>,
    words: usize,
}

impl Scanner<'_> {
    fn at(&self, i: usize) -> Option<char> {
        if self.skipped.get(i).copied().unwrap_or(true) {
            None
        } else {
            self.chars.get(i).copied()
        }
    }

    fn push(&mut self, kind: UnitKind, chars: Range<usize>) {
        self.out.push(Unit { kind, chars });
    }

    /// Consumes the unit starting at `i`; returns the next index.
    fn step(&mut self, i: usize) -> usize {
        let Some(c) = self.at(i) else {
            return i + 1;
        };
        if c.is_ascii_digit() {
            return self.arabic(i);
        }
        if self.starts_with(i, "百分之")
            && let Some(end) = self.cn_run_end(i + 3)
            && let Some(value) = self.cn_run_value(i + 3, end)
        {
            self.push(
                UnitKind::Number {
                    values: vec![format!("{value}%")],
                    readings: Vec::new(),
                },
                i..end,
            );
            return end;
        }
        if is_cn_numeral(c)
            && let Some(end) = self.cn_run_end(i)
            && let Some(value) = self.cn_run_value(i, end)
        {
            let readings = if end == i + 1 {
                readings(c)
            } else {
                Vec::new()
            };
            self.push(
                UnitKind::Number {
                    values: vec![value],
                    readings,
                },
                i..end,
            );
            return end;
        }
        if is_cjk(c) {
            let after_number = self.out.last().is_some_and(Unit::is_number);
            let readings = if matches!(c, '日' | '号') && after_number {
                vec!["ri", "hao"]
            } else {
                readings(c)
            };
            self.push(UnitKind::Cjk { ch: c, readings }, i..i + 1);
            return i + 1;
        }
        if c.is_ascii_alphabetic() {
            return self.latin(i);
        }
        i + 1
    }

    fn starts_with(&self, i: usize, prefix: &str) -> bool {
        prefix
            .chars()
            .enumerate()
            .all(|(k, p)| self.at(i + k) == Some(p))
    }

    /// End of the Chinese numeral run at `i` (with a `点` decimal part).
    fn cn_run_end(&self, i: usize) -> Option<usize> {
        let mut end = i;
        while let Some(c) = self.at(end) {
            let decimal =
                c == '点' && end > i && self.at(end + 1).is_some_and(|n| cn_digit(n).is_some());
            if !(is_cn_numeral(c) || decimal) {
                break;
            }
            end += 1;
        }
        (end > i).then_some(end)
    }

    /// Value of the run `[i, end)`; `None` for runs that are not numbers (a lone `万`).
    fn cn_run_value(&self, i: usize, end: usize) -> Option<String> {
        let run: String = (i..end).filter_map(|k| self.at(k)).collect();
        let value = cn_value(&run)?;
        let has_zero = run.chars().any(|c| cn_digit(c) == Some(0));
        (value != "0" || has_zero).then_some(value)
    }

    /// An Arabic number (`2026`, `3,000`, `1.5`, `5.4.0`), with a following `%` / `percent`.
    fn arabic(&mut self, i: usize) -> usize {
        let mut end = i;
        while let Some(c) = self.at(end) {
            let joins = matches!(c, '.' | ',')
                && end > i
                && self.at(end + 1).is_some_and(|n| n.is_ascii_digit());
            if !(c.is_ascii_digit() || joins) {
                break;
            }
            end += 1;
        }
        let written: String = (i..end).filter_map(|k| self.at(k)).collect();
        let Some(mut value) = canonical(&written) else {
            return end;
        };
        if let Some(after) = self.percent(end) {
            value.push('%');
            end = after;
        }
        self.push(
            UnitKind::Number {
                values: vec![value],
                readings: Vec::new(),
            },
            i..end,
        );
        end
    }

    /// End of a `%`, `percent` or `per cent` right after `i` (spaces allowed before it).
    fn percent(&self, i: usize) -> Option<usize> {
        let mut k = i;
        while self.at(k) == Some(' ') {
            k += 1;
        }
        if self.at(k) == Some('%') {
            return Some(k + 1);
        }
        let (word, end) = self.word_at(k)?;
        match word.as_str() {
            "percent" => Some(end),
            "per" => {
                let mut j = end;
                while self.at(j) == Some(' ') {
                    j += 1;
                }
                self.word_at(j)
                    .filter(|(next, _)| next == "cent")
                    .map(|(_, e)| e)
            }
            _ => None,
        }
    }

    /// The lowercase ASCII word starting at `i` and its end.
    fn word_at(&self, i: usize) -> Option<(String, usize)> {
        let mut end = i;
        while self
            .at(end)
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '\'')
        {
            end += 1;
        }
        (end > i).then(|| {
            let word: String = (i..end)
                .filter_map(|k| self.at(k))
                .collect::<String>()
                .to_ascii_lowercase();
            (word, end)
        })
    }

    /// A Latin word: a run of number words becomes one number, any other word its letters.
    fn latin(&mut self, i: usize) -> usize {
        let Some((word, end)) = self.word_at(i) else {
            return i + 1;
        };
        if is_en_number_word(&word) {
            return self.en_number(i, word, end);
        }
        self.words += 1;
        for k in i..end {
            if let Some(c) = self.at(k).filter(char::is_ascii_alphanumeric) {
                self.push(
                    UnitKind::Letter {
                        letter: c.to_ascii_lowercase(),
                        word: self.words,
                    },
                    k..k + 1,
                );
            }
        }
        end
    }

    /// A run of English number words (`twenty-six`, `two thousand and five`), with `point`
    /// decimals (`five point four point oh`) and a following `percent`.
    fn en_number(&mut self, i: usize, first: String, first_end: usize) -> usize {
        let mut words = vec![first];
        let mut end = first_end;
        let mut decimals: Vec<char> = Vec::new();
        loop {
            let mut k = end;
            while matches!(self.at(k), Some(' ' | '-')) {
                k += 1;
            }
            let Some((word, next)) = self.word_at(k) else {
                break;
            };
            if is_en_number_word(&word) && decimals.is_empty() {
                words.push(word);
                end = next;
            } else if word == "and" && !words.is_empty() && decimals.is_empty() {
                let mut j = next;
                while self.at(j) == Some(' ') {
                    j += 1;
                }
                match self.word_at(j) {
                    Some((after, after_end)) if is_en_number_word(&after) => {
                        words.push(after);
                        end = after_end;
                    }
                    _ => break,
                }
            } else if word == "point" {
                let mut j = next;
                while self.at(j) == Some(' ') {
                    j += 1;
                }
                let digit = self
                    .word_at(j)
                    .and_then(|(d, e)| digit_word(&d).map(|v| (v, e)));
                let Some((digit, digit_end)) = digit else {
                    break;
                };
                decimals.push(digit);
                end = digit_end;
            } else {
                break;
            }
        }
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let suffix: String = decimals.iter().flat_map(|d| ['.', *d]).collect();
        let mut values: Vec<String> = en_values(&refs)
            .into_iter()
            .map(|v| format!("{v}{suffix}"))
            .collect();
        if let Some(after) = self.percent(end) {
            for value in &mut values {
                value.push('%');
            }
            end = after;
        }
        if let [value] = values.as_slice()
            && decimals.len() == 1
            && let Some(normal) = canonical(value.trim_end_matches('%'))
        {
            let percent = if value.ends_with('%') { "%" } else { "" };
            values = vec![format!("{normal}{percent}")];
        }
        self.push(
            UnitKind::Number {
                values,
                readings: Vec::new(),
            },
            i..end,
        );
        end
    }
}

/// Digit of a decimal word (`four`, `oh`).
fn digit_word(word: &str) -> Option<char> {
    match word {
        "oh" | "o" => Some('0'),
        _ => super::numbers::en_small(word)
            .filter(|v| *v < 10)
            .and_then(|v| char::from_digit(v, 10)),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{Unit, UnitKind, units};

    fn values(text: &str) -> Vec<Vec<String>> {
        units(text)
            .into_iter()
            .filter_map(|u| match u.kind {
                UnitKind::Number { values, .. } => Some(values),
                _ => None,
            })
            .collect()
    }

    fn same(a: &str, b: &str) -> bool {
        let (a, b) = (units(a), units(b));
        a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.same(y))
    }

    #[rstest]
    #[case("2026年，我们", "两千零二十六年我们")]
    #[case("2026年", "二零二六年")]
    #[case("11月3日上午", "十一月三号上午")]
    #[case("速度提升了1.5倍", "速度提升了一点五倍")]
    #[case("大约有85%的人", "大约有百分之八十五的人")]
    #[case("这座山高3000米", "这座山高三千米")]
    #[case("鹏要等", "朋要等")]
    #[case("一共", "依共")]
    #[case("vLLM 的梯田", "v l l m 的梯田")]
    #[case("Homebrew", "home brew")]
    #[case(
        "In 2026, sales grew by 15 percent.",
        "in twenty twenty six sales grew by fifteen %"
    )]
    #[case("in twenty fourteen", "in 2014")]
    #[case("PHP five point four point oh", "PHP 5.4.0")]
    #[case("Chapter 3", "chapter three")]
    #[case("把<行|hang2>业", "把行业")]
    fn test_should_hear_equivalent_readings_as_the_same(#[case] a: &str, #[case] b: &str) {
        assert!(same(a, b), "{a} ≠ {b}: {:?} / {:?}", units(a), units(b));
    }

    #[rstest]
    #[case("第3章", "第三十章")]
    #[case("1.5倍", "十五倍")]
    #[case("85%", "百分之五十八")]
    #[case("北京", "南京")]
    #[case("by 15 percent", "by 50 percent")]
    fn test_should_hear_different_readings_as_different(#[case] a: &str, #[case] b: &str) {
        assert!(!same(a, b), "{a} = {b}");
    }

    #[test]
    fn test_should_keep_a_lone_myriad_as_a_character() {
        assert_eq!(values("万物"), Vec::<Vec<String>>::new());
        assert_eq!(
            values("零下5度"),
            [vec!["0".to_owned()], vec!["5".to_owned()]]
        );
    }

    #[test]
    fn test_should_index_units_by_character_and_skip_override_suffixes() {
        let found: Vec<Unit> = units("<Axon|AE1 K S AA0 N> 用");
        assert_eq!(found.len(), 5);
        assert_eq!(found.first().map(|u| u.chars.clone()), Some(1..2));
        assert_eq!(found.last().map(|u| u.chars.clone()), Some(21..22));
        assert!(found.iter().all(|u| !u.is_number()));
    }
}

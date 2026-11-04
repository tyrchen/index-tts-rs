//! Number values of the ways a number is written or read: Arabic digits, Chinese numerals
//! (positional `两千零二十六` or digit-by-digit `二零二六`), and English number words.
//!
//! Values are canonical strings (`2026`, `1.5`, `5.4.0`, `85%`) so that readings compare by
//! value.

/// Chinese digit characters and their values.
const CN_DIGITS: [(char, u8); 12] = [
    ('零', 0),
    ('〇', 0),
    ('一', 1),
    ('二', 2),
    ('两', 2),
    ('三', 3),
    ('四', 4),
    ('五', 5),
    ('六', 6),
    ('七', 7),
    ('八', 8),
    ('九', 9),
];
/// Chinese place units below ten thousand.
const CN_UNITS: [(char, u32); 3] = [('十', 10), ('百', 100), ('千', 1000)];
/// Chinese myriad units.
const CN_MYRIADS: [(char, u64); 2] = [('万', 10_000), ('亿', 100_000_000)];

const EN_SMALL: [(&str, u32); 28] = [
    ("zero", 0),
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
    ("eight", 8),
    ("nine", 9),
    ("ten", 10),
    ("eleven", 11),
    ("twelve", 12),
    ("thirteen", 13),
    ("fourteen", 14),
    ("fifteen", 15),
    ("sixteen", 16),
    ("seventeen", 17),
    ("eighteen", 18),
    ("nineteen", 19),
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
];
const EN_SCALES: [(&str, u64); 4] = [
    ("hundred", 100),
    ("thousand", 1_000),
    ("million", 1_000_000),
    ("billion", 1_000_000_000),
];

/// Value of a Chinese digit character.
#[must_use]
pub fn cn_digit(c: char) -> Option<u8> {
    CN_DIGITS.iter().find(|(d, _)| *d == c).map(|(_, v)| *v)
}

/// Whether `c` can be part of a Chinese numeral.
#[must_use]
pub fn is_cn_numeral(c: char) -> bool {
    cn_digit(c).is_some()
        || CN_UNITS.iter().any(|(u, _)| *u == c)
        || CN_MYRIADS.iter().any(|(u, _)| *u == c)
}

/// Value of a Chinese numeral run, optionally with a `点` decimal part (`一点五` → `1.5`).
/// Runs of digit characters only read digit by digit (`二零二六` → `2026`).
///
/// ```
/// use index_tts_rs_asr::compare::numbers::cn_value;
/// assert_eq!(cn_value("两千零二十六").as_deref(), Some("2026"));
/// assert_eq!(cn_value("二零二六").as_deref(), Some("2026"));
/// assert_eq!(cn_value("十五").as_deref(), Some("15"));
/// assert_eq!(cn_value("一点五").as_deref(), Some("1.5"));
/// ```
#[must_use]
pub fn cn_value(run: &str) -> Option<String> {
    if let Some((whole, fraction)) = run.split_once('点') {
        let whole = if whole.is_empty() {
            "0".to_owned()
        } else {
            cn_value(whole)?
        };
        let digits: Option<String> = fraction
            .chars()
            .map(|c| cn_digit(c).map(|d| char::from(b'0' + d)))
            .collect();
        return canonical(&format!("{whole}.{}", digits?));
    }
    if run.chars().all(|c| cn_digit(c).is_some()) {
        let digits: Option<String> = run
            .chars()
            .map(|c| cn_digit(c).map(|d| char::from(b'0' + d)))
            .collect();
        return canonical(&digits?);
    }
    positional(run).map(|v| v.to_string())
}

/// Positional value (`三万五千`, `一百零五`, `十二`); `None` on overflow or stray characters.
fn positional(run: &str) -> Option<u64> {
    let (mut total, mut section, mut digit) = (0u64, 0u64, 0u64);
    for c in run.chars() {
        if let Some(d) = cn_digit(c) {
            digit = u64::from(d);
        } else if let Some((_, unit)) = CN_UNITS.iter().find(|(u, _)| *u == c) {
            let lead = if digit == 0 { 1 } else { digit };
            section = section.checked_add(lead.checked_mul(u64::from(*unit))?)?;
            digit = 0;
        } else if let Some((_, myriad)) = CN_MYRIADS.iter().find(|(u, _)| *u == c) {
            total = total
                .checked_add(section)?
                .checked_add(digit)?
                .checked_mul(*myriad)?;
            section = 0;
            digit = 0;
        } else {
            return None;
        }
    }
    total.checked_add(section)?.checked_add(digit)
}

/// Canonical form of an Arabic number: no grouping commas, no leading zeros, no trailing
/// fractional zeros (`1.50` → `1.5`); dotted versions (`5.4.0`) are kept as written.
///
/// ```
/// use index_tts_rs_asr::compare::numbers::canonical;
/// assert_eq!(canonical("3,000").as_deref(), Some("3000"));
/// assert_eq!(canonical("1.50").as_deref(), Some("1.5"));
/// assert_eq!(canonical("5.4.0").as_deref(), Some("5.4.0"));
/// assert_eq!(canonical("007").as_deref(), Some("7"));
/// ```
#[must_use]
pub fn canonical(written: &str) -> Option<String> {
    let plain: String = written.chars().filter(|&c| c != ',').collect();
    if plain.is_empty() || !plain.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }
    if plain.matches('.').count() > 1 {
        return Some(plain);
    }
    let (whole, fraction) = plain.split_once('.').unwrap_or((plain.as_str(), ""));
    let whole = whole.trim_start_matches('0');
    let whole = if whole.is_empty() { "0" } else { whole };
    let fraction = fraction.trim_end_matches('0');
    Some(if fraction.is_empty() {
        whole.to_owned()
    } else {
        format!("{whole}.{fraction}")
    })
}

/// Value of an English number word.
#[must_use]
pub fn en_small(word: &str) -> Option<u32> {
    EN_SMALL.iter().find(|(w, _)| *w == word).map(|(_, v)| *v)
}

/// Whether `word` is an English number word (`seven`, `twenty`, `hundred`).
#[must_use]
pub fn is_en_number_word(word: &str) -> bool {
    en_small(word).is_some() || EN_SCALES.iter().any(|(w, _)| *w == word)
}

/// Values a run of English number words may stand for: the cardinal (`two thousand twenty six`
/// → `2026`) and, for two groups like `twenty fourteen`, the year reading `2014`.
///
/// ```
/// use index_tts_rs_asr::compare::numbers::en_values;
/// assert_eq!(en_values(&["fifteen"]), ["15"]);
/// assert!(en_values(&["twenty", "fourteen"]).contains(&"2014".to_owned()));
/// assert!(en_values(&["two", "hundred", "five"]).contains(&"205".to_owned()));
/// ```
#[must_use]
pub fn en_values(words: &[&str]) -> Vec<String> {
    let (mut total, mut current) = (0u64, 0u64);
    for word in words {
        if let Some(v) = en_small(word) {
            current = current.saturating_add(u64::from(v));
        } else if *word == "hundred" {
            current = current.saturating_mul(100);
        } else if let Some((_, scale)) = EN_SCALES.iter().find(|(w, _)| w == word) {
            total = total.saturating_add(current.saturating_mul(*scale));
            current = 0;
        }
    }
    let mut values = vec![total.saturating_add(current).to_string()];
    if let [first, second] = groups(words).as_slice()
        && (10..=99).contains(first)
        && *second <= 99
    {
        values.push(format!("{first}{second:02}"));
    }
    values
}

/// Splits number words into groups below one hundred (`twenty fourteen` → `[20, 14]`, `twenty
/// six` → `[26]`); empty when a scale word appears.
fn groups(words: &[&str]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let mut tens_open = false;
    for word in words {
        let Some(v) = en_small(word) else {
            return Vec::new();
        };
        match out.last_mut() {
            Some(last) if tens_open && v < 10 => {
                *last += v;
                tens_open = false;
            }
            _ => {
                out.push(v);
                tens_open = v >= 20 && v % 10 == 0;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{canonical, cn_value, en_values};

    #[rstest]
    #[case("两千零二十六", "2026")]
    #[case("二千零二十六", "2026")]
    #[case("一千八百二十四", "1824")]
    #[case("一八二四", "1824")]
    #[case("三千", "3000")]
    #[case("三万五千", "35000")]
    #[case("一百零五", "105")]
    #[case("十", "10")]
    #[case("二十一", "21")]
    #[case("一一零", "110")]
    #[case("零", "0")]
    #[case("一点五", "1.5")]
    #[case("零点零五", "0.05")]
    #[case("一百五十五万亿", "155000000000000")]
    fn test_should_read_chinese_numerals(#[case] run: &str, #[case] value: &str) {
        assert_eq!(cn_value(run).as_deref(), Some(value));
    }

    #[test]
    fn test_should_reject_overflowing_or_foreign_runs() {
        assert_eq!(cn_value("九亿亿亿亿亿亿亿亿亿"), None);
        assert_eq!(canonical("1a"), None);
        assert_eq!(canonical(""), None);
    }

    #[rstest]
    #[case(&["three"], &["3"])]
    #[case(&["twenty", "six"], &["26"])]
    #[case(&["two", "thousand", "twenty", "six"], &["2026"])]
    #[case(&["nineteen", "eighty", "four"], &["103", "1984"])]
    fn test_should_read_english_number_words(#[case] words: &[&str], #[case] values: &[&str]) {
        assert_eq!(en_values(words), values);
    }
}

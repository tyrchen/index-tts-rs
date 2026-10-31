//! Multilingual text normalizer, punctuation mapper, and YAML glossary engine.
//!
//! Replicates `core/frontend.py::TextNormalizer` in pure Rust without `wetext`,
//! handling punctuation cleanup, glossary term substitution, English contractions,
//! technical term hyphen protection, Pinyin tone preservation, and Chinese/English
//! number/decimal/percentage verbalization.

use std::{collections::HashMap, fs, path::Path, sync::LazyLock};

use config::{Config, File, FileFormat};
use regex::{Captures, Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

use crate::{
    config::validate_safe_file_path,
    error::{IndexTtsError, Result},
    frontend::parser::{correct_pinyin, has_cjk_ideograph},
};

static EMAIL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$").unwrap_or_else(|_| unreachable!())
});

static PINYIN_TONE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:^|[^a-zA-Z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])",
    )
    .unwrap_or_else(|_| unreachable!())
});

static NAME_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}").unwrap_or_else(|_| unreachable!())
});

static TECH_TERM_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+").unwrap_or_else(|_| unreachable!())
});

static TECH_RESTORE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s*<H>\s*").unwrap_or_else(|_| unreachable!()));

static CONTRACTION_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(what|where|who|which|how|t?here|it|s?he|that|this)'s\b")
        .unwrap_or_else(|_| unreachable!())
});

static ANNOTATION_SPAN_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<[^|>\n]+\|[^>\n]+>").unwrap_or_else(|_| unreachable!()));

static NUMBER_TOKEN_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\d+(?:\.\d+)?%?").unwrap_or_else(|_| unreachable!()));

/// A single glossary entry supporting either a universal string replacement or
/// per-language replacements (`zh`, `en`, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum GlossaryEntry {
    /// Universal replacement string across all languages.
    Universal(String),
    /// Per-language replacement table keyed by language code (`"zh"`, `"en"`).
    Localized(HashMap<String, String>),
}

impl GlossaryEntry {
    /// Resolves the replacement text for `lang`, falling back to `original_term`.
    #[must_use]
    pub fn resolve<'a>(&'a self, lang: &str, original_term: &'a str) -> &'a str {
        match self {
            Self::Universal(val) => val.as_str(),
            Self::Localized(map) => map.get(lang).map_or(original_term, String::as_str),
        }
    }
}

/// Pure-Rust multilingual text normalizer.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct TextNormalizer {
    /// Whether glossary term replacement is active.
    pub enable_glossary: bool,
    /// Loaded glossary dictionary.
    pub glossary: HashMap<String, GlossaryEntry>,
}

impl TextNormalizer {
    /// Creates a new [`TextNormalizer`].
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::frontend::normalizer::TextNormalizer;
    ///
    /// let norm = TextNormalizer::new(true);
    /// assert!(norm.enable_glossary);
    /// ```
    #[must_use]
    pub fn new(enable_glossary: bool) -> Self {
        Self {
            enable_glossary,
            glossary: HashMap::new(),
        }
    }

    /// Loads glossary entries from a YAML string.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Config`] if the YAML is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::frontend::normalizer::TextNormalizer;
    ///
    /// let mut norm = TextNormalizer::new(true);
    /// norm.load_glossary_from_yaml_str("LLM:\n  zh: 大语言模型\n  en: large language model\n")?;
    /// assert_eq!(norm.normalize("我们训练了LLM"), "我们训练了大语言模型");
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn load_glossary_from_yaml_str(&mut self, yaml_str: &str) -> Result<()> {
        if yaml_str.trim().is_empty() {
            return Ok(());
        }
        let cfg = Config::builder()
            .add_source(File::from_str(yaml_str, FileFormat::Yaml))
            .build()
            .map_err(|err| IndexTtsError::Config {
                message: format!("failed to parse glossary YAML: {err}"),
                source: Some(err),
            })?;
        let map: HashMap<String, GlossaryEntry> =
            cfg.try_deserialize().map_err(|err| IndexTtsError::Config {
                message: format!("failed to deserialize glossary entries: {err}"),
                source: Some(err),
            })?;
        self.glossary = map;
        Ok(())
    }

    /// Loads glossary entries from a YAML file on disk if it exists.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if the path is unsafe, [`IndexTtsError::Io`] on
    /// read failure, or [`IndexTtsError::Config`] if the YAML content is invalid.
    #[allow(clippy::disallowed_methods)]
    pub fn load_glossary_from_path(&mut self, path: impl AsRef<Path>) -> Result<bool> {
        let path_ref = path.as_ref();
        validate_safe_file_path(path_ref, "glossary_path")?;
        if !path_ref.exists() {
            return Ok(false);
        }
        let content =
            fs::read_to_string(path_ref).map_err(|err| IndexTtsError::io(path_ref, err))?;
        self.load_glossary_from_yaml_str(&content)?;
        Ok(true)
    }

    /// Determines whether `text` should use the Chinese normalization pipeline.
    ///
    /// Mirrors `TextNormalizer.use_chinese` in `core/frontend.py`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::frontend::normalizer::TextNormalizer;
    ///
    /// let norm = TextNormalizer::default();
    /// assert!(norm.use_chinese("大家好"));
    /// assert!(norm.use_chinese("hang2"));
    /// assert!(norm.use_chinese("12345"));
    /// assert!(!norm.use_chinese("Hello world"));
    /// ```
    #[must_use]
    pub fn use_chinese(&self, s: &str) -> bool {
        let has_chinese = has_cjk_ideograph(s);
        let has_alpha = s.chars().any(|c| c.is_ascii_alphabetic());
        let is_email = EMAIL_REGEX.is_match(s);
        if has_chinese || !has_alpha || is_email {
            return true;
        }
        PINYIN_TONE_REGEX.is_match(s)
    }

    /// Normalizes `text` according to detected language rules.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::frontend::normalizer::TextNormalizer;
    ///
    /// let norm = TextNormalizer::new(false);
    /// assert_eq!(
    ///     norm.normalize("欢迎使用 IndexTTS-2.5！"),
    ///     "欢迎使用 IndexTTS-二点五!"
    /// );
    /// assert_eq!(
    ///     norm.normalize("We have 3 apples and 25% gain."),
    ///     "We have three apples and twenty five percent gain."
    /// );
    /// ```
    #[must_use]
    pub fn normalize(&self, text: &str) -> String {
        if self.use_chinese(text) {
            let expanded = CONTRACTION_REGEX.replace_all(text, "${1} is");
            let mut working = if self.enable_glossary {
                self.apply_glossary_terms(&expanded, "zh")
            } else {
                expanded.into_owned()
            };
            working = working.trim_end().to_owned();

            let (working, saved_annotations) = save_annotations(&working);
            let (working, tech_list) = save_tech_terms(&working);
            let (working, pinyin_list) = save_pinyin_tones(&working);
            let (working, name_list) = save_names(&working);

            let normalized = verbalize_chinese_numbers(&working);

            let restored = restore_names(&normalized, &name_list);
            let restored = restore_pinyin_tones(&restored, &pinyin_list);
            let restored = restore_tech_terms(&restored, &tech_list);
            let restored = restore_annotations(&restored, &saved_annotations);
            clean_punctuation(&restored, true)
        } else {
            let expanded = CONTRACTION_REGEX.replace_all(text, "${1} is");
            let working = if self.enable_glossary {
                self.apply_glossary_terms(&expanded, "en")
            } else {
                expanded.into_owned()
            };

            let (working, saved_annotations) = save_annotations(&working);
            let (working, tech_list) = save_tech_terms(&working);
            let normalized = verbalize_english_numbers(&working);
            let restored = restore_tech_terms(&normalized, &tech_list);
            let restored = restore_annotations(&restored, &saved_annotations);
            clean_punctuation(&restored, false)
        }
    }

    /// Applies longest-first case-insensitive glossary replacements for `lang`.
    #[must_use]
    pub fn apply_glossary_terms(&self, text: &str, lang: &str) -> String {
        if self.glossary.is_empty() {
            return text.to_owned();
        }
        let mut terms: Vec<&String> = self.glossary.keys().collect();
        terms.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

        let mut transformed = text.to_owned();
        for term in terms {
            let Some(entry) = self.glossary.get(term) else {
                continue;
            };
            let replacement = entry.resolve(lang, term);
            let escaped = regex::escape(term);
            if let Ok(re) = RegexBuilder::new(&escaped)
                .case_insensitive(true)
                .size_limit(1 << 20)
                .build()
            {
                transformed = re.replace_all(&transformed, replacement).into_owned();
            }
        }
        transformed
    }
}

/// Cleans punctuation characters according to `CHAR_REP_MAP` (or `ZH_CHAR_REP_MAP` when `zh_mode`
/// is `true`).
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::frontend::normalizer::clean_punctuation;
///
/// assert_eq!(clean_punctuation("你好，世界！...", false), "你好,世界!…");
/// assert_eq!(clean_punctuation("价格$10", true), "价格.10");
/// ```
#[must_use]
pub fn clean_punctuation(text: &str, zh_mode: bool) -> String {
    let mut s = text
        .replace("...", "…")
        .replace(",,,", "…")
        .replace("，，，", "…")
        .replace("……", "…");

    let mut out = String::with_capacity(s.len());
    for ch in s.drain(..) {
        let mapped = match ch {
            '：' | '；' | ';' | '，' | '、' | ':' => ',',
            '。' => '.',
            '$' if zh_mode => '.',
            '！' => '!',
            '？' => '?',
            '\n' => ' ',
            '·' | '—' | '～' | '~' => '-',
            '“' | '”' | '"' | '‘' | '’' | '（' | '）' | '(' | ')' | '《' | '》' | '【' | '】'
            | '[' | ']' | '「' | '」' => '\'',
            other => other,
        };
        out.push(mapped);
    }
    out
}

fn save_annotations(text: &str) -> (String, Vec<String>) {
    let mut saved = Vec::new();
    for m in ANNOTATION_SPAN_REGEX.find_iter(text) {
        let span = m.as_str().to_owned();
        if !saved.contains(&span) {
            saved.push(span);
        }
    }
    if saved.is_empty() {
        return (text.to_owned(), saved);
    }
    let mut out = text.to_owned();
    for (i, span) in saved.iter().enumerate() {
        let tag = placeholder_tag("ann", i);
        out = out.replace(span, &tag);
    }
    (out, saved)
}

fn restore_annotations(text: &str, saved: &[String]) -> String {
    if saved.is_empty() {
        return text.to_owned();
    }
    let mut out = text.to_owned();
    for (i, span) in saved.iter().enumerate() {
        let tag = placeholder_tag("ann", i);
        out = out.replace(&tag, span);
    }
    out
}

fn save_tech_terms(text: &str) -> (String, Vec<String>) {
    let mut terms: Vec<String> = TECH_TERM_REGEX
        .find_iter(text)
        .map(|m| m.as_str().to_owned())
        .collect();
    if terms.is_empty() {
        return (text.to_owned(), terms);
    }
    terms.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    terms.dedup();

    let mut out = text.to_owned();
    for term in &terms {
        let escaped = term.replace('-', "<H>");
        out = out.replace(term, &escaped);
    }
    (out, terms)
}

fn restore_tech_terms(text: &str, terms: &[String]) -> String {
    if terms.is_empty() {
        return text.to_owned();
    }
    TECH_RESTORE_REGEX.replace_all(text, "-").into_owned()
}

fn save_pinyin_tones(text: &str) -> (String, Vec<String>) {
    let mut pinyins = Vec::new();
    for caps in PINYIN_TONE_REGEX.captures_iter(text) {
        let base = caps.get(1).map_or("", |m| m.as_str());
        let tone = caps.get(2).map_or("", |m| m.as_str());
        let combined = format!("{base}{tone}");
        if !pinyins.contains(&combined) {
            pinyins.push(combined);
        }
    }
    if pinyins.is_empty() {
        return (text.to_owned(), pinyins);
    }
    pinyins.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let mut out = text.to_owned();
    for (i, p) in pinyins.iter().enumerate() {
        let tag = placeholder_tag("pinyin", i);
        out = out.replace(p, &tag);
    }
    (out, pinyins)
}

fn restore_pinyin_tones(text: &str, pinyins: &[String]) -> String {
    if pinyins.is_empty() {
        return text.to_owned();
    }
    let mut out = text.to_owned();
    for (i, p) in pinyins.iter().enumerate() {
        let tag = placeholder_tag("pinyin", i);
        let corrected = correct_pinyin(p);
        out = out.replace(&tag, &corrected);
    }
    out
}

fn save_names(text: &str) -> (String, Vec<String>) {
    let mut names = Vec::new();
    for m in NAME_REGEX.find_iter(text) {
        let n = m.as_str().to_owned();
        if !names.contains(&n) {
            names.push(n);
        }
    }
    if names.is_empty() {
        return (text.to_owned(), names);
    }
    let mut out = text.to_owned();
    for (i, name) in names.iter().enumerate() {
        let tag = placeholder_tag("n", i);
        out = out.replace(name, &tag);
    }
    (out, names)
}

fn restore_names(text: &str, names: &[String]) -> String {
    if names.is_empty() {
        return text.to_owned();
    }
    let mut out = text.to_owned();
    for (i, name) in names.iter().enumerate() {
        let tag = placeholder_tag("n", i);
        out = out.replace(&tag, name);
    }
    out
}

fn placeholder_tag(prefix: &str, idx: usize) -> String {
    let mut letters = String::new();
    let mut n = idx;
    loop {
        #[allow(clippy::cast_possible_truncation)]
        let ch = (b'a' + (n % 26) as u8) as char;
        letters.push(ch);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }
    format!("<{prefix}_{letters}>")
}

fn verbalize_chinese_numbers(text: &str) -> String {
    NUMBER_TOKEN_REGEX
        .replace_all(text, |caps: &Captures<'_>| {
            let token = caps.get(0).map_or("", |m| m.as_str());
            format_chinese_number_token(token)
        })
        .into_owned()
}

fn format_chinese_number_token(token: &str) -> String {
    if let Some(without_pct) = token.strip_suffix('%') {
        return format!("百分之{}", format_chinese_number_token(without_pct));
    }
    if let Some((int_part, frac_part)) = token.split_once('.') {
        let int_zh = format_chinese_integer(int_part);
        let frac_zh: String = frac_part.chars().map(zh_digit_char).collect();
        return format!("{int_zh}点{frac_zh}");
    }
    format_chinese_integer(token)
}

fn format_chinese_integer(digits: &str) -> String {
    if digits.len() > 1 && digits.starts_with('0') {
        return digits.chars().map(zh_digit_char).collect();
    }
    let Ok(val) = digits.parse::<u64>() else {
        return digits.chars().map(zh_digit_char).collect();
    };
    if val == 0 {
        return "零".to_owned();
    }
    if val >= 100_000_000 {
        return digits.chars().map(zh_digit_char).collect();
    }
    if val >= 10_000 {
        let wan = val / 10_000;
        let rem = val % 10_000;
        let wan_str = format_zh_under_10000(wan, false);
        if rem == 0 {
            return format!("{wan_str}万");
        }
        let need_zero = rem < 1_000;
        let rem_str = format_zh_under_10000(rem, true);
        if need_zero {
            return format!("{wan_str}万零{rem_str}");
        }
        return format!("{wan_str}万{rem_str}");
    }
    format_zh_under_10000(val, false)
}

#[allow(clippy::cast_possible_truncation)]
fn format_zh_under_10000(val: u64, full_tens: bool) -> String {
    const DIGITS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
    if val < 10 {
        return DIGITS.get(val as usize).copied().unwrap_or("零").to_owned();
    }
    if val < 20 && !full_tens {
        let ones = val % 10;
        if ones == 0 {
            return "十".to_owned();
        }
        return format!("十{}", DIGITS.get(ones as usize).copied().unwrap_or(""));
    }

    let units = [(1_000_u64, "千"), (100, "百"), (10, "十"), (1, "")];
    let mut out = String::new();
    let mut rem = val;
    let mut prev_zero = false;

    for (divisor, unit_name) in units {
        let d = rem / divisor;
        rem %= divisor;
        if d > 0 {
            if prev_zero && !out.is_empty() {
                out.push('零');
            }
            out.push_str(DIGITS.get(d as usize).copied().unwrap_or(""));
            out.push_str(unit_name);
            prev_zero = false;
        } else if !out.is_empty() {
            prev_zero = true;
        }
    }
    out
}

const fn zh_digit_char(c: char) -> char {
    match c {
        '0' => '零',
        '1' => '一',
        '2' => '二',
        '3' => '三',
        '4' => '四',
        '5' => '五',
        '6' => '六',
        '7' => '七',
        '8' => '八',
        '9' => '九',
        other => other,
    }
}

fn verbalize_english_numbers(text: &str) -> String {
    NUMBER_TOKEN_REGEX
        .replace_all(text, |caps: &Captures<'_>| {
            let token = caps.get(0).map_or("", |m| m.as_str());
            format_english_number_token(token)
        })
        .into_owned()
}

fn format_english_number_token(token: &str) -> String {
    if let Some(without_pct) = token.strip_suffix('%') {
        return format!("{} percent", format_english_number_token(without_pct));
    }
    if let Some((int_part, frac_part)) = token.split_once('.') {
        let int_en = format_english_integer(int_part);
        let frac_words: Vec<&str> = frac_part.chars().map(en_digit_word).collect();
        return format!("{int_en} point {}", frac_words.join(" "));
    }
    format_english_integer(token)
}

fn format_english_integer(digits: &str) -> String {
    let Ok(val) = digits.parse::<u64>() else {
        return digits
            .chars()
            .map(en_digit_word)
            .collect::<Vec<_>>()
            .join(" ");
    };
    if val == 0 {
        return "zero".to_owned();
    }
    if val >= 1_000_000 {
        return digits
            .chars()
            .map(en_digit_word)
            .collect::<Vec<_>>()
            .join(" ");
    }
    if val >= 1_000 {
        let thousands = val / 1_000;
        let rem = val % 1_000;
        let t_str = format_en_under_1000(thousands);
        if rem == 0 {
            return format!("{t_str} thousand");
        }
        return format!("{t_str} thousand {}", format_en_under_1000(rem));
    }
    format_en_under_1000(val)
}

#[allow(clippy::cast_possible_truncation)]
fn format_en_under_1000(val: u64) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    if val < 20 {
        return ONES.get(val as usize).copied().unwrap_or("zero").to_owned();
    }
    if val < 100 {
        let t = val / 10;
        let o = val % 10;
        let t_word = TENS.get(t as usize).copied().unwrap_or("");
        if o == 0 {
            return t_word.to_owned();
        }
        let o_word = ONES.get(o as usize).copied().unwrap_or("");
        return format!("{t_word} {o_word}");
    }
    let h = val / 100;
    let rem = val % 100;
    let h_word = ONES.get(h as usize).copied().unwrap_or("");
    if rem == 0 {
        return format!("{h_word} hundred");
    }
    format!("{h_word} hundred {}", format_en_under_1000(rem))
}

const fn en_digit_word(c: char) -> &'static str {
    match c {
        '0' => "zero",
        '1' => "one",
        '2' => "two",
        '3' => "three",
        '4' => "four",
        '5' => "five",
        '6' => "six",
        '7' => "seven",
        '8' => "eight",
        '9' => "nine",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_normalize_chinese_tech_terms_and_decimals() {
        let norm = TextNormalizer::new(false);
        let out = norm.normalize("大家好，欢迎使用 IndexTTS-2.5 语音合成引擎！");
        assert_eq!(out, "大家好,欢迎使用 IndexTTS-二点五 语音合成引擎!");
    }

    #[test]
    fn test_should_preserve_pronunciation_annotations_and_correct_jqx() {
        let norm = TextNormalizer::new(false);
        let out = norm.normalize("请把<行|hang2>业报告发给<行|xing2>长，并检查<居|ju1>民区。");
        assert_eq!(
            out,
            "请把<行|hang2>业报告发给<行|xing2>长,并检查<居|ju1>民区."
        );
    }

    #[test]
    fn test_should_normalize_english_numbers_and_contractions() {
        let norm = TextNormalizer::new(false);
        let out = norm.normalize("What's new? We have 3 apples and 25% improvement!");
        assert_eq!(
            out,
            "What is new? We have three apples and twenty five percent improvement!"
        );
    }
}

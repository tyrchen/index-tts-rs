//! `winnow`-based parser for `<word|PRON>` pronunciation annotations and
//! `<|SPECIAL_TOKEN_*|>` atomic span extraction.

use std::sync::LazyLock;

use regex::Regex;
use winnow::{
    Parser,
    error::ContextError,
    token::{literal, take_while},
};

static JQX_UMLAUT_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b([jqx])[uü](n|e|an)*([1-5])").unwrap_or_else(|_| unreachable!())
});

/// Normalizes Chinese Pinyin syllables starting with `j`, `q`, or `x` followed by
/// `u` or `ü` and a tone digit (`1..=5`) into `JV` / `QV` / `XV` uppercase form.
///
/// Mirrors `TextNormalizer.correct_pinyin` in `core/frontend.py`.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::frontend::parser::correct_pinyin;
///
/// assert_eq!(correct_pinyin("ju1"), "JV1");
/// assert_eq!(correct_pinyin("qu2"), "QV2");
/// assert_eq!(correct_pinyin("xu3"), "XV3");
/// assert_eq!(correct_pinyin("jü4"), "JV4");
/// assert_eq!(correct_pinyin("juan1"), "JVAN1");
/// assert_eq!(correct_pinyin("hang2"), "HANG2");
/// ```
#[must_use]
pub fn correct_pinyin(pinyin: &str) -> String {
    let replaced = JQX_UMLAUT_REGEX.replace_all(pinyin, "${1}v${2}${3}");
    replaced.to_uppercase()
}

/// Returns `true` if `s` is non-empty and consists exclusively of Hiragana
/// (`\u{3040}..=\u{309F}`) or exclusively of Katakana (`\u{30A0}..=\u{30FF}`).
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::frontend::parser::is_kana;
///
/// assert!(is_kana("あした"));
/// assert!(is_kana("トウキョウ"));
/// assert!(!is_kana("あア"));
/// assert!(!is_kana("HANG2"));
/// ```
#[must_use]
pub fn is_kana(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let all_hiragana = s.chars().all(|c| ('\u{3040}'..='\u{309F}').contains(&c));
    let all_katakana = s.chars().all(|c| ('\u{30A0}'..='\u{30FF}').contains(&c));
    all_hiragana || all_katakana
}

/// Returns `true` if `s` contains any CJK Unified Ideograph (`\u{4E00}..=\u{9FFF}`).
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::frontend::parser::has_cjk_ideograph;
///
/// assert!(has_cjk_ideograph("行业"));
/// assert!(!has_cjk_ideograph("read"));
/// ```
#[must_use]
pub fn has_cjk_ideograph(s: &str) -> bool {
    s.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))
}

/// Parses and expands all `<word|pronunciation>` annotations in `text` using `winnow`:
///
/// - Japanese Hiragana or Katakana pronunciations become `" {PRON} "`.
/// - Words containing CJK ideographs become `"<|SPECIAL_TOKEN_2|>{PRON}<|SPECIAL_TOKEN_2|>"`.
/// - All other words (e.g. English phonemes) become
///   `"<|SPECIAL_TOKEN_1|>{PRON}<|SPECIAL_TOKEN_1|>"`.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::frontend::parser::apply_pronunciation_annotations;
///
/// assert_eq!(
///     apply_pronunciation_annotations("<行|hang2>业"),
///     "<|SPECIAL_TOKEN_2|>HANG2<|SPECIAL_TOKEN_2|>业"
/// );
/// assert_eq!(
///     apply_pronunciation_annotations("<read|R EH1 D>"),
///     "<|SPECIAL_TOKEN_1|>R EH1 D<|SPECIAL_TOKEN_1|>"
/// );
/// assert_eq!(
///     apply_pronunciation_annotations("<明日|あした>"),
///     " あした "
/// );
/// ```
#[must_use]
pub fn apply_pronunciation_annotations(text: &str) -> String {
    let mut out = String::with_capacity(text.len().saturating_add(32));
    let mut input = text;

    while !input.is_empty() {
        if input.starts_with('<') && !input.starts_with("<|") {
            let mut probe = input;
            if let Ok((word, pron)) = parse_annotation_tag(&mut probe) {
                let normalized_pron = if has_cjk_ideograph(word) {
                    correct_pinyin(pron)
                } else {
                    pron.to_uppercase()
                };
                if is_kana(&normalized_pron) {
                    out.push(' ');
                    out.push_str(&normalized_pron);
                    out.push(' ');
                } else if has_cjk_ideograph(word) {
                    out.push_str("<|SPECIAL_TOKEN_2|>");
                    out.push_str(&normalized_pron);
                    out.push_str("<|SPECIAL_TOKEN_2|>");
                } else {
                    out.push_str("<|SPECIAL_TOKEN_1|>");
                    out.push_str(&normalized_pron);
                    out.push_str("<|SPECIAL_TOKEN_1|>");
                }
                input = probe;
                continue;
            }
        }

        // Advance to next `<` or consume one char if starting at `<`
        if let Some(ch) = input.chars().next() {
            out.push(ch);
            input = input.get(ch.len_utf8()..).unwrap_or("");
        }
        let plain = take_while::<_, _, ContextError>(0.., |c: char| c != '<')
            .parse_next(&mut input)
            .unwrap_or("");
        out.push_str(plain);
    }

    out
}

/// Uppercases all `<|tag|>` special token delimiters in `text` using `winnow`.
///
/// Mirrors `re.sub(r'<\|([^|]+)\|>', lambda m: f'<|{m.group(1).upper()}|>', text)`.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::frontend::parser::uppercase_special_tokens;
///
/// assert_eq!(
///     uppercase_special_tokens("hello <|special_token_1|>R<|special_token_1|>"),
///     "hello <|SPECIAL_TOKEN_1|>R<|SPECIAL_TOKEN_1|>"
/// );
/// ```
#[must_use]
pub fn uppercase_special_tokens(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut input = text;

    while !input.is_empty() {
        if input.starts_with("<|") {
            let mut probe = input;
            if let Ok(inner) = parse_special_token_tag(&mut probe) {
                out.push_str("<|");
                out.push_str(&inner.to_uppercase());
                out.push_str("|>");
                input = probe;
                continue;
            }
        }
        if let Some(ch) = input.chars().next() {
            out.push(ch);
            input = input.get(ch.len_utf8()..).unwrap_or("");
        }
        let plain = take_while::<_, _, ContextError>(0.., |c: char| c != '<')
            .parse_next(&mut input)
            .unwrap_or("");
        out.push_str(plain);
    }

    out
}

/// Splits `text` into a sequence of `(segment_str, is_atomic)` pairs where atomic
/// spans match `<|SPECIAL_TOKEN_\d+|>.*?<|SPECIAL_TOKEN_\d+|>`.
///
/// Atomic spans must never be split across sentence segments by the token splitter.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::frontend::parser::split_atomic_pieces;
///
/// let pieces = split_atomic_pieces("请把<|SPECIAL_TOKEN_2|>HANG2<|SPECIAL_TOKEN_2|>业");
/// assert_eq!(
///     pieces,
///     vec![
///         ("请把".to_owned(), false),
///         ("<|SPECIAL_TOKEN_2|>HANG2<|SPECIAL_TOKEN_2|>".to_owned(), true),
///         ("业".to_owned(), false),
///     ]
/// );
/// ```
#[must_use]
pub fn split_atomic_pieces(text: &str) -> Vec<(String, bool)> {
    let mut pieces = Vec::new();
    let mut plain_buf = String::new();
    let mut input = text;

    while !input.is_empty() {
        if input.starts_with("<|SPECIAL_TOKEN_")
            && let Some(span_len) = match_atomic_special_span(input)
            && let Some(atomic_slice) = input.get(..span_len)
        {
            if !plain_buf.is_empty() {
                pieces.push((std::mem::take(&mut plain_buf), false));
            }
            pieces.push((atomic_slice.to_owned(), true));
            input = input.get(span_len..).unwrap_or("");
            continue;
        }

        if let Some(ch) = input.chars().next() {
            plain_buf.push(ch);
            input = input.get(ch.len_utf8()..).unwrap_or("");
        }
    }

    if !plain_buf.is_empty() {
        pieces.push((plain_buf, false));
    }

    pieces
}

fn parse_annotation_tag<'a>(input: &mut &'a str) -> Result<(&'a str, &'a str), ContextError> {
    literal("<").parse_next(input)?;
    let word = take_while(1.., |c: char| !matches!(c, '|' | '>' | '\n' | '<')).parse_next(input)?;
    literal("|").parse_next(input)?;
    let pron = take_while(1.., |c: char| !matches!(c, '>' | '\n' | '<')).parse_next(input)?;
    literal(">").parse_next(input)?;
    Ok((word, pron))
}

fn parse_special_token_tag<'a>(input: &mut &'a str) -> Result<&'a str, ContextError> {
    literal("<|").parse_next(input)?;
    let inner = take_while(1.., |c: char| c != '|').parse_next(input)?;
    literal("|>").parse_next(input)?;
    Ok(inner)
}

fn parse_special_token_number_tag(input: &mut &str) -> Result<(), ContextError> {
    literal("<|SPECIAL_TOKEN_").parse_next(input)?;
    take_while(1.., |c: char| c.is_ascii_digit()).parse_next(input)?;
    literal("|>").parse_next(input)?;
    Ok(())
}

fn match_atomic_special_span(input: &str) -> Option<usize> {
    let mut probe = input;
    parse_special_token_number_tag(&mut probe).ok()?;
    let open_len = input.len().saturating_sub(probe.len());

    let mut search_offset = open_len;
    while let Some(rel_pos) = input.get(search_offset..)?.find("<|SPECIAL_TOKEN_") {
        let candidate_start = search_offset.saturating_add(rel_pos);
        let mut close_probe = input.get(candidate_start..)?;
        if parse_special_token_number_tag(&mut close_probe).is_ok() {
            let total_len = input.len().saturating_sub(close_probe.len());
            return Some(total_len);
        }
        search_offset = candidate_start.saturating_add("<|SPECIAL_TOKEN_".len());
    }
    None
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn test_should_wrap_chinese_pinyin_with_special_token_2() {
        let out = apply_pronunciation_annotations("<行|hang2>业");
        assert_eq!(out, "<|SPECIAL_TOKEN_2|>HANG2<|SPECIAL_TOKEN_2|>业");
    }

    #[test]
    fn test_should_wrap_english_phoneme_with_special_token_1() {
        let out = apply_pronunciation_annotations("<read|R EH1 D>");
        assert_eq!(out, "<|SPECIAL_TOKEN_1|>R EH1 D<|SPECIAL_TOKEN_1|>");
    }

    #[test]
    fn test_should_inline_japanese_kana_with_spaces() {
        assert_eq!(apply_pronunciation_annotations("<明日|あした>"), " あした ");
        assert_eq!(
            apply_pronunciation_annotations("<東京|トウキョウ>"),
            " トウキョウ "
        );
    }

    #[rstest]
    #[case("ju1", "JV1")]
    #[case("qu2", "QV2")]
    #[case("xu3", "XV3")]
    #[case("jü4", "JV4")]
    #[case("juan1", "JVAN1")]
    #[case("xun2", "XVN2")]
    #[case("que4", "QVE4")]
    fn test_should_normalize_jqx_umlaut_to_v(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(correct_pinyin(input), expected);
    }
}

//! Token-budget sentence splitter and high-level [`Frontend`] pipeline.
//!
//! Splits normalized text along punctuation boundaries (`[，。！？、；：,\.!\?;:\n]`)
//! while preserving `<|SPECIAL_TOKEN_*|>` atomic blocks and stripping reserved
//! `START_TEXT_TOKEN` (`0`) and `STOP_TEXT_TOKEN` (`1`) IDs.

use std::{ops::Range, path::Path};

use crate::{
    align::is_punctuation,
    config::{DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT, Language},
    error::Result,
    frontend::{
        normalizer::{TextNormalizer, clean_punctuation},
        parser::{apply_pronunciation_annotations, split_atomic_pieces, uppercase_special_tokens},
        tokenizer::{TIKTOKEN_VOCAB_FILENAME, TiktokenTokenizer, TokenSpan},
    },
};

/// High-level multilingual text frontend combining normalization, pronunciation
/// override parsing, sentence splitting, and Tiktoken BPE encoding.
#[derive(Debug, Clone)]
pub struct Frontend {
    tokenizer: TiktokenTokenizer,
    normalizer: TextNormalizer,
    text_pos_capacity: usize,
    use_normalization: bool,
}

impl Frontend {
    /// Initializes a [`Frontend`] from a model snapshot directory containing
    /// `multilingual_zh_ja_yue_char_del.tiktoken` and an optional `glossary.yaml`.
    ///
    /// # Errors
    ///
    /// Returns [` crate::error::IndexTtsError`] if loading the `.tiktoken` vocabulary
    /// or `glossary.yaml` fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{
    ///     config::Language,
    ///     frontend::Frontend,
    ///     models::resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let fe = Frontend::from_model_dir(&dir, 602, true, None)?;
    ///     let segments = fe.prepare("大家好，欢迎使用 IndexTTS-2.5！", Language::Zh, 120);
    ///     assert_eq!(segments.len(), 1);
    ///     assert!(!segments[0].is_empty());
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        text_pos_capacity: usize,
        use_normalization: bool,
        glossary_path: Option<&Path>,
    ) -> Result<Self> {
        let dir_ref = model_dir.as_ref();
        let vocab_path = dir_ref.join(TIKTOKEN_VOCAB_FILENAME);
        let tokenizer = TiktokenTokenizer::from_path(vocab_path)?;

        let mut normalizer = TextNormalizer::new(true);
        let effective_glossary =
            glossary_path.map_or_else(|| dir_ref.join("glossary.yaml"), Path::to_path_buf);
        let _ = normalizer.load_glossary_from_path(&effective_glossary)?;

        Ok(Self::new(
            tokenizer,
            normalizer,
            text_pos_capacity,
            use_normalization,
        ))
    }

    /// Constructs a [`Frontend`] from an already-loaded [`TiktokenTokenizer`] and
    /// [`TextNormalizer`].
    #[must_use]
    pub fn new(
        tokenizer: TiktokenTokenizer,
        normalizer: TextNormalizer,
        text_pos_capacity: usize,
        use_normalization: bool,
    ) -> Self {
        Self {
            tokenizer,
            normalizer,
            text_pos_capacity: text_pos_capacity.max(4),
            use_normalization,
        }
    }

    /// Returns a reference to the underlying [`TiktokenTokenizer`].
    #[must_use]
    pub const fn tokenizer(&self) -> &TiktokenTokenizer {
        &self.tokenizer
    }

    /// Splits `text` into segments whose BPE token length (plus `lang_prefix`) does
    /// not exceed `min(max_tokens, text_pos_capacity - 2)`.
    ///
    /// Mirrors `Frontend.split_text_by_tokens` in `core/frontend.py`, ensuring
    /// `<|SPECIAL_TOKEN_\d+|>...<|SPECIAL_TOKEN_\d+|>` atomic spans are never broken.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{
    ///     frontend::Frontend,
    ///     models::resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let fe = Frontend::from_model_dir(&dir, 602, true, None)?;
    ///     let segs = fe.split_text_by_tokens("第一句。第二句。", 6, "<|zh|> ");
    ///     assert_eq!(segs.len(), 2);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[must_use]
    pub fn split_text_by_tokens(
        &self,
        text: &str,
        max_tokens: usize,
        lang_prefix: &str,
    ) -> Vec<String> {
        let cap_limit = self.text_pos_capacity.saturating_sub(2);
        let effective_max = if max_tokens == 0 {
            DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT
        } else {
            max_tokens
        };
        let prefix_len = self.tokenizer.token_len(lang_prefix);
        let budget = effective_max
            .min(cap_limit)
            .saturating_sub(prefix_len)
            .max(1);

        if self.tokenizer.token_len(text) <= budget {
            return vec![text.to_owned()];
        }

        let mut chunks = Vec::new();
        for (piece, is_atomic) in split_atomic_pieces(text) {
            if is_atomic {
                chunks.push(piece);
                continue;
            }
            for part in split_after_punctuation(&piece) {
                if part.is_empty() {
                    continue;
                }
                if self.tokenizer.token_len(part) <= budget {
                    chunks.push(part.to_owned());
                    continue;
                }
                let mut current = String::new();
                for ch in part.chars() {
                    let mut candidate = current.clone();
                    candidate.push(ch);
                    if !current.is_empty() && self.tokenizer.token_len(&candidate) > budget {
                        chunks.push(current);
                        current = ch.to_string();
                    } else {
                        current.push(ch);
                    }
                }
                if !current.is_empty() {
                    chunks.push(current);
                }
            }
        }

        let mut segments = Vec::new();
        let mut current = String::new();
        for chunk in chunks {
            let mut candidate = current.clone();
            candidate.push_str(&chunk);
            if !current.is_empty() && self.tokenizer.token_len(&candidate) > budget {
                segments.push(current);
                current = chunk;
            } else {
                current.push_str(&chunk);
            }
        }
        if !current.is_empty() {
            segments.push(current);
        }
        if segments.is_empty() {
            vec![text.to_owned()]
        } else {
            segments
        }
    }

    /// Runs the full text preparation pipeline (`clean_punctuation` -> `normalize` ->
    /// case fold -> `apply_pronunciation_annotations` -> `uppercase_special_tokens` ->
    /// `split_text_by_tokens` -> `encode_with_special` -> strip token IDs `<= 1`).
    ///
    /// Returns one `Vec<u32>` of token IDs per sentence segment, guaranteed to have
    /// length `<= text_pos_capacity - 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{
    ///     config::Language,
    ///     frontend::Frontend,
    ///     models::resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let fe = Frontend::from_model_dir(&dir, 602, true, None)?;
    ///     let ids = fe.prepare("请把<行|hang2>业报告发给<行|xing2>长。", Language::Zh, 120);
    ///     assert_eq!(ids.len(), 1);
    ///     assert!(ids[0].contains(&58959)); // <|SPECIAL_TOKEN_2|>
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[must_use]
    pub fn prepare(
        &self,
        text: &str,
        lang: Language,
        max_text_tokens_per_segment: usize,
    ) -> Vec<Vec<u32>> {
        self.prepare_with_lang_str(text, lang.as_code(), max_text_tokens_per_segment)
    }

    /// Runs the full text preparation pipeline using a raw language code string
    /// (e.g. `"zh"`, `"en"`, `"ja"`, `"yue"`, `"zh/en"`).
    #[must_use]
    pub fn prepare_with_lang_str(
        &self,
        text: &str,
        lang_raw: &str,
        max_text_tokens_per_segment: usize,
    ) -> Vec<Vec<u32>> {
        self.prepare_text(text, lang_raw, max_text_tokens_per_segment)
            .segments
            .into_iter()
            .map(|segment| segment.tokens.into_iter().map(|token| token.id).collect())
            .collect()
    }

    /// Runs the full text preparation pipeline and keeps the normalized text plus, for each
    /// segment, the encoded text and every token's byte range. Token IDs are identical to
    /// [`Self::prepare_with_lang_str`].
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{frontend::Frontend, models::resolve_model_dir};
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let fe = Frontend::from_model_dir(&dir, 602, true, None)?;
    ///     let prepared = fe.prepare_text("第一句。第二句。", "zh", 6);
    ///     assert_eq!(prepared.segments.len(), 2);
    ///     let second = &prepared.segments[1];
    ///     assert_eq!(&prepared.normalized[second.normalized_range()], "第二句.");
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[must_use]
    pub fn prepare_text(
        &self,
        text: &str,
        lang_raw: &str,
        max_text_tokens_per_segment: usize,
    ) -> PreparedText {
        let lang = lang_raw.to_ascii_lowercase();
        let lang_prefix = format!("<|{lang}|> ");

        let mut working = clean_punctuation(text, false);
        if self.use_normalization && matches!(lang.as_str(), "zh" | "zhen" | "en") {
            working = self.normalizer.normalize(&working);
        }
        if matches!(lang.as_str(), "ja" | "zh" | "zhen" | "en") {
            working = working.to_lowercase();
        } else if lang == "es" {
            working = working.to_uppercase();
        }

        working = apply_pronunciation_annotations(&working);
        working = uppercase_special_tokens(&working);

        let pieces = self.split_text_by_tokens(&working, max_text_tokens_per_segment, &lang_prefix);
        let max_cap = self.text_pos_capacity.saturating_sub(2).max(1);

        let mut cursor = 0_usize;
        let segments = pieces
            .into_iter()
            .map(|piece| {
                // Segments partition `working` in order; locate defensively all the same.
                let normalized_offset = working
                    .get(cursor..)
                    .and_then(|rest| rest.find(piece.as_str()))
                    .map_or(cursor, |found| cursor + found);
                cursor = normalized_offset + piece.len();
                let text = format!("{lang_prefix}{piece}");
                let mut tokens: Vec<TokenSpan> = self
                    .tokenizer
                    .encode_with_offsets(&text)
                    .into_iter()
                    .filter(|token| token.id > 1)
                    .collect();
                tokens.truncate(max_cap);
                PreparedSegment {
                    text,
                    tokens,
                    prefix_len: lang_prefix.len(),
                    normalized_offset,
                }
            })
            .collect();
        PreparedText {
            normalized: working,
            segments,
        }
    }
}

/// Output of [`Frontend::prepare_text`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedText {
    /// Normalized text the segments were cut from (no language prefix).
    pub normalized: String,
    /// Tokenized segments in reading order.
    pub segments: Vec<PreparedSegment>,
}

/// One tokenized sentence segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSegment {
    /// Encoded text: the `<|lang|> ` prefix followed by the segment's normalized text.
    pub text: String,
    /// Tokens with byte ranges into [`Self::text`].
    pub tokens: Vec<TokenSpan>,
    /// Byte length of the language prefix at the start of [`Self::text`].
    pub prefix_len: usize,
    /// Byte offset of the segment body within [`PreparedText::normalized`].
    pub normalized_offset: usize,
}

impl PreparedSegment {
    /// Token IDs as fed to the GPT decoder.
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn token_ids(&self) -> Vec<i32> {
        self.tokens.iter().map(|token| token.id as i32).collect()
    }

    /// Byte range of the segment body within [`PreparedText::normalized`].
    #[must_use]
    pub fn normalized_range(&self) -> Range<usize> {
        let body = self.text.len().saturating_sub(self.prefix_len);
        self.normalized_offset..self.normalized_offset + body
    }

    /// Byte range of `token` within [`PreparedText::normalized`], or `None` for tokens wholly
    /// inside the language prefix. A token that merged the prefix's trailing space with the
    /// first word (` honestly`) is clipped to its body part.
    #[must_use]
    pub fn normalized_bytes(&self, token: &TokenSpan) -> Option<Range<usize>> {
        let end = token
            .bytes
            .end
            .checked_sub(self.prefix_len)
            .filter(|&end| end > 0)?;
        let start = token.bytes.start.saturating_sub(self.prefix_len);
        Some(self.normalized_offset + start..self.normalized_offset + end)
    }

    /// Whether `token` is pronounced: not a special token, whitespace, or punctuation.
    #[must_use]
    pub fn is_spoken(&self, token: &TokenSpan) -> bool {
        let Some(bytes) = self.text.as_bytes().get(token.bytes.clone()) else {
            return false;
        };
        if bytes.starts_with(b"<|") && bytes.ends_with(b"|>") {
            return false;
        }
        String::from_utf8_lossy(bytes)
            .chars()
            .any(|ch| !ch.is_whitespace() && !is_punctuation(ch))
    }
}

/// Splits `piece` immediately after any sentence punctuation character
/// (`[，。！？、；：,\.!\?;:\n]`), replicating `re.split(r'(?<=[，。！？、；：,\.!\?;:\n])',
/// piece)`.
fn split_after_punctuation(piece: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0_usize;
    for (idx, ch) in piece.char_indices() {
        if is_split_punctuation(ch) {
            let end = idx.saturating_add(ch.len_utf8());
            if let Some(sub) = piece.get(start..end) {
                parts.push(sub);
            }
            start = end;
        }
    }
    if start < piece.len()
        && let Some(tail) = piece.get(start..)
    {
        parts.push(tail);
    }
    parts
}

const fn is_split_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '，' | '。' | '！' | '？' | '、' | '；' | '：' | ',' | '.' | '!' | '?' | ';' | ':' | '\n'
    )
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf, sync::LazyLock};

    use proptest::prelude::*;
    use safetensors::SafeTensors;

    use super::*;
    use crate::{
        config::{DEFAULT_MODEL_REPO, DEFAULT_TEXT_POS_CAPACITY},
        models::resolve_model_dir,
    };

    static SHARED_FRONTEND: LazyLock<Option<Frontend>> = LazyLock::new(|| {
        let dir = resolve_model_dir(None, DEFAULT_MODEL_REPO).ok()?;
        Frontend::from_model_dir(&dir, DEFAULT_TEXT_POS_CAPACITY, true, None).ok()
    });

    #[test]
    #[ignore = "requires local IndexTTS model weights"]
    fn test_should_match_python_reference_cp05_bit_exact_tokenizer_ids() {
        assert!(
            SHARED_FRONTEND.is_some(),
            "SHARED_FRONTEND must initialize from local model dir"
        );
        let Some(fe) = SHARED_FRONTEND.as_ref() else {
            return;
        };
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("golden_ref.safetensors");
        #[allow(clippy::disallowed_methods)]
        let bytes = fs::read(&fixture_path).unwrap_or_default();
        let Ok((_, header)) = SafeTensors::read_metadata(&bytes) else {
            return;
        };
        let Ok(st) = SafeTensors::deserialize(&bytes) else {
            return;
        };
        let Some(meta) = header.metadata() else {
            return;
        };

        for idx in 0..6 {
            let lang_key = format!("sample_{idx}_lang");
            let text_key = format!("sample_{idx}_text");
            let tensor_key = format!("token_ids_{idx}");

            let lang_str = meta.get(&lang_key).map_or("zh", String::as_str);
            let text_str = meta.get(&text_key).map_or("", String::as_str);
            let expected_ids = extract_u32(&st, &tensor_key);

            let rust_segments = fe.prepare_with_lang_str(text_str, lang_str, 120);
            assert_eq!(
                rust_segments.len(),
                1,
                "sample {idx} segment count mismatch"
            );
            let rust_ids = rust_segments.first().cloned().unwrap_or_default();
            assert_eq!(
                rust_ids, expected_ids,
                "CP-05 sample {idx} ({lang_str}: {text_str}) token IDs are not bit-exact!"
            );
        }
    }

    #[test]
    fn test_should_never_split_inside_special_token_span() {
        let Some(fe) = SHARED_FRONTEND.as_ref() else {
            return;
        };
        let text = "请把<|SPECIAL_TOKEN_2|>HANG2<|SPECIAL_TOKEN_2|>业报告发给<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>长";
        // Use a tiny token budget so split occurs right around the atomic span boundary
        let segments = fe.split_text_by_tokens(text, 8, "<|zh|> ");
        assert!(segments.len() >= 2);
        for seg in &segments {
            let count = seg.matches("<|SPECIAL_TOKEN_2|>").count();
            assert_eq!(
                count % 2,
                0,
                "segment `{seg}` split inside a <|SPECIAL_TOKEN_2|> atomic span!"
            );
        }
    }

    #[test]
    fn test_should_strip_literal_zero_and_one_tokens() {
        let Some(fe) = SHARED_FRONTEND.as_ref() else {
            return;
        };
        // In tiktoken vocab, `!` has rank 0 (`START_TEXT_TOKEN`) and `"` has rank 1
        // (`STOP_TEXT_TOKEN`).
        let segments = fe.prepare("Hello, surprise!", Language::En, 120);
        for seg in segments {
            assert!(!seg.contains(&0), "found unstripped token 0 in {seg:?}");
            assert!(!seg.contains(&1), "found unstripped token 1 in {seg:?}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn proptest_text_frontend_never_panics_on_arbitrary_utf8(
            s in ".*",
            max_toks in 1_usize..200_usize,
        ) {
            if let Some(fe) = SHARED_FRONTEND.as_ref() {
                let segments = fe.prepare(&s, Language::ZhEn, max_toks);
                for seg in segments {
                    prop_assert!(seg.len() <= DEFAULT_TEXT_POS_CAPACITY - 2);
                    prop_assert!(seg.iter().all(|&id| id > 1));
                }
            }
        }
    }

    fn extract_u32(st: &SafeTensors<'_>, name: &str) -> Vec<u32> {
        let Ok(view) = st.tensor(name) else {
            return Vec::new();
        };
        view.data()
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .filter_map(|c| u32::try_from(i32::from_le_bytes(c)).ok())
            .collect()
    }
}

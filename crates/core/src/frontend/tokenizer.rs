//! Pure-Rust Tiktoken BPE tokenizer for `multilingual_zh_ja_yue_char_del.tiktoken`.
//!
//! Parses the 58,836 base BPE merge ranks using `winnow` and `base64`, registers
//! the exact 1,673 special tokens (`58836..=60508`) defined in `core/frontend.py`,
//! and executes linear-time regex pre-tokenization + byte-pair merging with 100%
//! bit-exact parity against Python `tiktoken`.

use std::{collections::HashMap, fs, path::Path, sync::LazyLock};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use regex::Regex;
use winnow::{
    Parser,
    ascii::{digit1, space1},
    error::ContextError,
    token::take_while,
};

use crate::{
    config::validate_safe_file_path,
    error::{IndexTtsError, Result},
};

/// Filename of the Tiktoken BPE vocabulary inside the model directory.
pub const TIKTOKEN_VOCAB_FILENAME: &str = "multilingual_zh_ja_yue_char_del.tiktoken";

/// Number of base BPE tokens in `multilingual_zh_ja_yue_char_del.tiktoken` (`58836`).
pub const EXPECTED_BASE_VOCAB_SIZE: usize = 58_836;

/// Number of languages registered as `<|lang|>` special tokens (`99`).
pub const NUM_REGISTERED_LANGUAGES: usize = 99;

/// Ordered language list from `core/frontend.py::LANGUAGES`.
pub const LANGUAGES_ORDER: [&str; 104] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en",
];

const AUDIO_EVENTS: [&str; 11] = [
    "ASR",
    "AED",
    "SER",
    "Speech",
    "/Speech",
    "BGM",
    "/BGM",
    "Laughter",
    "/Laughter",
    "Applause",
    "/Applause",
];

const EMOTIONS: [&str; 4] = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"];

const TTS_VOCAL_PREFIXES: [&str; 7] = [
    "TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H",
];

/// Linear-time equivalent of `PAT_STR` (with whitespace lookahead handled in
/// `find_pretoken_pieces`).
static PRETOKEN_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
        .unwrap_or_else(|_| unreachable!())
});

/// Pure-Rust Tiktoken BPE tokenizer for IndexTTS-2.5.
#[derive(Debug, Clone)]
pub struct TiktokenTokenizer {
    /// Mapping from byte sequence to BPE merge rank (`0..58836`).
    encoder: HashMap<Vec<u8>, u32>,
    /// Mapping from special token string (`"<|zh|>"`, `"<|SPECIAL_TOKEN_2|>"`, etc.) to token ID.
    special_tokens: HashMap<String, u32>,
    /// Compiled alternation regex matching all registered special tokens.
    special_regex: Regex,
}

impl TiktokenTokenizer {
    /// Parses a `.tiktoken` vocabulary string and registers all 1,673 special tokens.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Tokenize`] if any line in the `.tiktoken` file fails
    /// `winnow` or `base64` decoding.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{
    ///     frontend::tokenizer::{TIKTOKEN_VOCAB_FILENAME, TiktokenTokenizer},
    ///     models::resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let tok = TiktokenTokenizer::from_path(dir.join(TIKTOKEN_VOCAB_FILENAME))?;
    ///     assert_eq!(tok.base_vocab_size(), 58_836);
    ///     assert_eq!(tok.total_vocab_size(), 60_509);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_vocab_str(content: &str) -> Result<Self> {
        let mut encoder = HashMap::with_capacity(EXPECTED_BASE_VOCAB_SIZE);

        for (line_idx, raw_line) in content.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let (b64_part, rank) =
                parse_tiktoken_line(line).map_err(|()| IndexTtsError::Tokenize {
                    message: format!("invalid tiktoken line {} (`{line}`)", line_idx + 1),
                })?;
            let token_bytes = if b64_part.bytes().all(|b| b == b'=') {
                Vec::new()
            } else {
                BASE64_STANDARD
                    .decode(b64_part)
                    .map_err(|err| IndexTtsError::Tokenize {
                        message: format!("base64 decode error on line {}: {err}", line_idx + 1),
                    })?
            };
            encoder.insert(token_bytes, rank);
        }

        if encoder.is_empty() {
            return Err(IndexTtsError::Tokenize {
                message: "tiktoken vocabulary is empty".to_owned(),
            });
        }

        let mut next_id = encoder.len() as u32;
        let specials_list = build_special_tokens_list(NUM_REGISTERED_LANGUAGES);
        let mut special_tokens = HashMap::with_capacity(specials_list.len());
        let mut escaped_patterns = Vec::with_capacity(specials_list.len());

        for spec in specials_list {
            escaped_patterns.push(regex::escape(&spec));
            special_tokens.insert(spec, next_id);
            next_id = next_id.saturating_add(1);
        }

        let pattern = escaped_patterns.join("|");
        let special_regex = Regex::new(&pattern).map_err(|err| IndexTtsError::Tokenize {
            message: format!("failed to compile special token regex: {err}"),
        })?;

        Ok(Self {
            encoder,
            special_tokens,
            special_regex,
        })
    }

    /// Loads [`TiktokenTokenizer`] from a `.tiktoken` file path on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if the path is unsafe, [`IndexTtsError::Io`] on
    /// read failure, or [`IndexTtsError::Tokenize`] if parsing fails.
    #[allow(clippy::disallowed_methods)]
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        validate_safe_file_path(path_ref, "tiktoken_vocab_path")?;
        let content =
            fs::read_to_string(path_ref).map_err(|err| IndexTtsError::io(path_ref, err))?;
        Self::from_vocab_str(&content)
    }

    /// Returns the number of base BPE mergeable ranks (`58836`).
    #[must_use]
    pub fn base_vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Returns the total vocabulary size including special tokens (`60509`).
    #[must_use]
    pub fn total_vocab_size(&self) -> usize {
        self.encoder.len().saturating_add(self.special_tokens.len())
    }

    /// Encodes `text` with `allowed_special = "all"`, matching `encoding.encode(text,
    /// allowed_special="all")`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{
    ///     frontend::tokenizer::{TIKTOKEN_VOCAB_FILENAME, TiktokenTokenizer},
    ///     models::resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let tok = TiktokenTokenizer::from_path(dir.join(TIKTOKEN_VOCAB_FILENAME))?;
    ///     let ids = tok.encode_with_special("<|zh|> 大家好");
    ///     assert_eq!(ids.first().copied(), Some(58839));
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[must_use]
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut last_end = 0_usize;

        for m in self.special_regex.find_iter(text) {
            if m.start() > last_end
                && let Some(ordinary) = text.get(last_end..m.start())
            {
                self.encode_ordinary_into(ordinary, &mut out);
            }
            if let Some(&spec_id) = self.special_tokens.get(m.as_str()) {
                out.push(spec_id);
            }
            last_end = m.end();
        }

        if last_end < text.len()
            && let Some(tail) = text.get(last_end..)
        {
            self.encode_ordinary_into(tail, &mut out);
        }

        out
    }

    /// Returns the token count of `text` when encoded with `allowed_special = "all"`.
    #[must_use]
    pub fn token_len(&self, text: &str) -> usize {
        self.encode_with_special(text).len()
    }

    fn encode_ordinary_into(&self, text: &str, out: &mut Vec<u32>) {
        let mut pos = 0_usize;
        while pos < text.len() {
            let Some(rest) = text.get(pos..) else {
                break;
            };
            let Some(m) = PRETOKEN_REGEX.find(rest) else {
                break;
            };
            let matched = m.as_str();
            let advance_bytes = if is_multi_whitespace_before_non_ws(matched, rest.len()) {
                // Equivalent to `\s+(?!\S)`: leave the final whitespace character for the next
                // match
                let last_ch_len = matched.chars().last().map_or(1, char::len_utf8);
                matched.len().saturating_sub(last_ch_len)
            } else {
                matched.len()
            };

            if let Some(piece) = rest.get(..advance_bytes) {
                self.bpe_encode_piece(piece.as_bytes(), out);
            }
            pos = pos.saturating_add(advance_bytes.max(1));
        }
    }

    fn bpe_encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        if let Some(&rank) = self.encoder.get(piece) {
            out.push(rank);
            return;
        }
        if piece.len() == 1 {
            return;
        }

        // Initialize parts: each entry is (start_byte_idx, rank_with_next)
        let mut parts: Vec<(usize, u32)> = Vec::with_capacity(piece.len() + 1);
        for idx in 0..piece.len() {
            let rank = if idx + 1 < piece.len() {
                piece
                    .get(idx..=idx + 1)
                    .and_then(|pair| self.encoder.get(pair).copied())
                    .unwrap_or(u32::MAX)
            } else {
                u32::MAX
            };
            parts.push((idx, rank));
        }
        parts.push((piece.len(), u32::MAX));

        loop {
            if parts.len() <= 2 {
                break;
            }
            // Find the pair with minimum rank
            let mut min_rank = u32::MAX;
            let mut min_idx = 0_usize;
            for (i, &(_, rank)) in parts.iter().take(parts.len().saturating_sub(1)).enumerate() {
                if rank < min_rank {
                    min_rank = rank;
                    min_idx = i;
                }
            }
            if min_rank == u32::MAX {
                break;
            }

            // Remove parts[min_idx + 1] and recompute ranks for min_idx and min_idx - 1
            parts.remove(min_idx + 1);
            let new_rank = get_pair_rank(&self.encoder, piece, &parts, min_idx);
            if let Some(slot) = parts.get_mut(min_idx) {
                slot.1 = new_rank;
            }
            if min_idx > 0 {
                let prev_rank = get_pair_rank(&self.encoder, piece, &parts, min_idx - 1);
                if let Some(prev_slot) = parts.get_mut(min_idx - 1) {
                    prev_slot.1 = prev_rank;
                }
            }
        }

        for window in parts.windows(2) {
            if let [(start, _), (end, _)] = *window
                && let Some(sub) = piece.get(start..end)
                && let Some(&rank) = self.encoder.get(sub)
            {
                out.push(rank);
            }
        }
    }
}

fn get_pair_rank(
    encoder: &HashMap<Vec<u8>, u32>,
    piece: &[u8],
    parts: &[(usize, u32)],
    idx: usize,
) -> u32 {
    if idx + 2 >= parts.len() {
        return u32::MAX;
    }
    let start = parts.get(idx).map_or(0, |p| p.0);
    let end = parts.get(idx + 2).map_or(0, |p| p.0);
    piece
        .get(start..end)
        .and_then(|slice| encoder.get(slice).copied())
        .unwrap_or(u32::MAX)
}

fn is_multi_whitespace_before_non_ws(matched: &str, rest_len: usize) -> bool {
    if matched.len() == rest_len {
        // Reaches EOF: `(?!\S)` succeeds on the entire whitespace run
        return false;
    }
    let mut chars = matched.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_whitespace() {
        return false;
    }
    // Check if there are >= 2 chars and all are whitespace
    let Some(second) = chars.next() else {
        return false;
    };
    second.is_whitespace() && chars.all(char::is_whitespace)
}

fn parse_tiktoken_line(mut input: &str) -> std::result::Result<(&str, u32), ()> {
    let b64_part = take_while::<_, _, ContextError>(1.., |c: char| !c.is_ascii_whitespace())
        .parse_next(&mut input)
        .map_err(|_| ())?;
    space1::<_, ContextError>
        .parse_next(&mut input)
        .map_err(|_| ())?;
    let rank_str = digit1::<_, ContextError>
        .parse_next(&mut input)
        .map_err(|_| ())?;
    let rank = rank_str.parse::<u32>().map_err(|_| ())?;
    Ok((b64_part, rank))
}

#[allow(clippy::cast_precision_loss)]
fn build_special_tokens_list(num_languages: usize) -> Vec<String> {
    let mut specials = Vec::with_capacity(1_673);
    specials.push("<|endoftext|>".to_owned());
    specials.push("<|startoftranscript|>".to_owned());

    for &lang in LANGUAGES_ORDER.iter().take(num_languages) {
        specials.push(format!("<|{lang}|>"));
    }
    for &ae in &AUDIO_EVENTS {
        specials.push(format!("<|{ae}|>"));
    }
    for &emo in &EMOTIONS {
        specials.push(format!("<|{emo}|>"));
    }
    for &ctrl in &[
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ] {
        specials.push(ctrl.to_owned());
    }
    for i in 1..=30 {
        specials.push(format!("<|SPECIAL_TOKEN_{i}|>"));
    }
    for &tts in &TTS_VOCAL_PREFIXES {
        specials.push(format!("<|{tts}|>"));
    }
    for i in 1..=13 {
        specials.push(format!("<|TTS/SP{i:02}|>"));
    }
    for i in 0..1_501 {
        let secs = f64::from(i) * 0.02;
        specials.push(format!("<|{secs:.2}|>"));
    }
    specials
}

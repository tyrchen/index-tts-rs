//! Multilingual text frontend for IndexTTS-2.5.
//!
//! Combines `winnow`-based pronunciation override parsing (`<word|PRON>`),
//! YAML glossary & number/punctuation normalization, Tiktoken BPE encoding,
//! and token-budget sentence splitting.

pub mod normalizer;
pub mod parser;
pub mod splitter;
pub mod tokenizer;

pub use normalizer::{GlossaryEntry, TextNormalizer, clean_punctuation};
pub use parser::{
    apply_pronunciation_annotations, correct_pinyin, has_cjk_ideograph, is_kana,
    split_atomic_pieces, uppercase_special_tokens,
};
pub use splitter::Frontend;
pub use tokenizer::{TIKTOKEN_VOCAB_FILENAME, TiktokenTokenizer};

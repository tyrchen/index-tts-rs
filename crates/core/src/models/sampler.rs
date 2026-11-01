//! Stage 1 autoregressive GPT token sampler (`SeenTokenBitSet`, `RepetitionPenalty`,
//! `Temperature`, $O(V)$ `TopK` via `select_nth_unstable_by`, `TopP` nucleus filtering, and
//! `greedy` argmax decoding).

use std::cmp::Ordering;

use mlx_rs::Array;

use crate::{
    config::SamplingParams,
    error::{IndexTtsError, Result},
    models::gpt::{
        GPT_MEL_POS_LEN, GPT_MEL_VOCAB, Gpt, KvCache, START_MEL_TOKEN, STOP_MEL_TOKEN,
        STOP_TEXT_TOKEN,
    },
};

/// Number of 64-bit words required to cover `0..8194` (`129 * 64 = 8256`).
pub const BITSET_WORDS: usize = 129;

/// Lightweight, deterministic `SplitMix64` pseudo-random number generator for multinomial token
/// sampling without external crate dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerRng {
    state: u64,
}

impl SamplerRng {
    /// Creates a new [`SamplerRng`] seeded with `seed`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::SamplerRng;
    ///
    /// let mut rng = SamplerRng::new(42);
    /// let u = rng.next_f32();
    /// assert!((0.0..1.0).contains(&u));
    /// ```
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Generates the next pseudo-random `u64`.
    pub const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Generates a uniform `f32` in `[0.0, 1.0)`.
    #[allow(clippy::cast_precision_loss)]
    pub fn next_f32(&mut self) -> f32 {
        let upper_24 = (self.next_u64() >> 40) as u32;
        (upper_24 as f32) * (1.0_f32 / ((1_u32 << 24) as f32))
    }
}

/// Fixed-size `[u64; 129]` bitset tracking generated Mel token IDs for $O(\text{popcount})$
/// repetition penalty application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeenTokenBitSet {
    words: [u64; BITSET_WORDS],
}

impl Default for SeenTokenBitSet {
    fn default() -> Self {
        Self::new_with_initial_tokens()
    }
}

impl SeenTokenBitSet {
    /// Creates an empty [`SeenTokenBitSet`] with no bits set.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::SeenTokenBitSet;
    ///
    /// let set = SeenTokenBitSet::empty();
    /// assert!(!set.contains(1));
    /// ```
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            words: [0_u64; BITSET_WORDS],
        }
    }

    /// Creates a [`SeenTokenBitSet`] pre-populated with `{STOP_TEXT_TOKEN (1), START_MEL_TOKEN
    /// (8192)}` matching the Python reference `seen = {1, START_MEL_TOKEN}`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::SeenTokenBitSet;
    ///
    /// let set = SeenTokenBitSet::new_with_initial_tokens();
    /// assert!(set.contains(1));
    /// assert!(set.contains(8192));
    /// assert!(!set.contains(100));
    /// ```
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn new_with_initial_tokens() -> Self {
        let mut set = Self::empty();
        set.insert(STOP_TEXT_TOKEN as usize);
        set.insert(START_MEL_TOKEN as usize);
        set
    }

    /// Inserts `token_id` into the bitset if `token_id < 8256`.
    pub fn insert(&mut self, token_id: usize) {
        let word_idx = token_id / 64;
        let bit_idx = token_id % 64;
        if let Some(word) = self.words.get_mut(word_idx) {
            *word |= 1_u64 << bit_idx;
        }
    }

    /// Returns `true` if `token_id` is present in the bitset.
    #[must_use]
    pub fn contains(&self, token_id: usize) -> bool {
        let word_idx = token_id / 64;
        let bit_idx = token_id % 64;
        self.words
            .get(word_idx)
            .is_some_and(|&w| (w & (1_u64 << bit_idx)) != 0)
    }

    /// Applies HuggingFace-style repetition penalty in-place to all seen token IDs:
    /// negative logits are multiplied by `penalty`, positive logits are divided by `penalty`.
    pub fn apply_repetition_penalty(&self, logits: &mut [f32], penalty: f32) {
        if (penalty - 1.0_f32).abs() <= f32::EPSILON {
            return;
        }
        for (word_idx, &word_val) in self.words.iter().enumerate() {
            let mut w = word_val;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let tid = word_idx * 64 + bit;
                if let Some(logit) = logits.get_mut(tid) {
                    if *logit < 0.0_f32 {
                        *logit *= penalty;
                    } else {
                        *logit /= penalty;
                    }
                }
                w &= w - 1;
            }
        }
    }
}

/// Samples a single next Mel token ID (`0..8194`) from raw `logits` (`[8194]`).
///
/// Order of operations matches `index_tts_2_5_mlx/core/gpt_sampler.py`:
/// 1. In-place `RepetitionPenalty` over all tokens in `seen`.
/// 2. If `params.greedy`, returns `argmax(logits)`.
/// 3. Scales logits by `1.0 / params.temperature`.
/// 4. Filters to the top `K` candidates in $O(V)$ via `select_nth_unstable_by`.
/// 5. Sorts those `K` candidates descending, applies `TopP` nucleus masking (always retaining at
///    least the top-1 token), computes softmax probabilities, and draws from `rng`.
///
/// # Errors
///
/// Returns [`IndexTtsError::Validation`] if `logits` is empty.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::{
///     config::SamplingParams,
///     models::{SamplerRng, SeenTokenBitSet, sample_next_token},
/// };
///
/// let mut logits = vec![-10.0_f32; 8194];
/// logits[42] = 25.0;
/// let seen = SeenTokenBitSet::new_with_initial_tokens();
/// let mut rng = SamplerRng::new(7);
/// let tok = sample_next_token(&mut logits, &seen, &SamplingParams::default(), &mut rng)?;
/// assert_eq!(tok, 42);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn sample_next_token(
    logits: &mut [f32],
    seen: &SeenTokenBitSet,
    params: &SamplingParams,
    rng: &mut SamplerRng,
) -> Result<i32> {
    if logits.is_empty() {
        return Err(IndexTtsError::validation(
            "logits",
            "logits slice must not be empty",
        ));
    }

    seen.apply_repetition_penalty(logits, params.repetition_penalty.get());

    if params.greedy {
        return Ok(argmax_finite(logits));
    }

    let inv_temp = 1.0_f32 / params.temperature.get();
    let mut candidates: Vec<(i32, f32)> = Vec::with_capacity(logits.len());
    for (idx, &val) in logits.iter().enumerate() {
        let scaled = if val.is_finite() {
            val * inv_temp
        } else {
            f32::NEG_INFINITY
        };
        candidates.push((idx as i32, scaled));
    }

    let raw_k = params.top_k.get() as usize;
    let k = if raw_k == 0 {
        candidates.len()
    } else {
        raw_k.min(candidates.len())
    };
    if k < candidates.len() {
        candidates.select_nth_unstable_by(k - 1, |a, b| desc_f32_cmp(a.1, b.1));
        candidates.truncate(k);
    }
    candidates.sort_unstable_by(|a, b| desc_f32_cmp(a.1, b.1));

    let max_logit = candidates
        .first()
        .map_or(0.0_f32, |&(_, v)| if v.is_finite() { v } else { 0.0_f32 });

    // First softmax over Top-K candidates for Top-P cumulative sum
    let mut probs: Vec<f32> = Vec::with_capacity(candidates.len());
    let mut prob_sum = 0.0_f32;
    for &(_, logit) in &candidates {
        let p = if logit.is_finite() {
            (logit - max_logit).exp()
        } else {
            0.0_f32
        };
        probs.push(p);
        prob_sum += p;
    }

    if prob_sum <= 0.0_f32 || !prob_sum.is_finite() {
        return Ok(candidates.first().map_or(0, |&(id, _)| id));
    }

    let top_p = params.top_p.get();
    let mut cum_prob = 0.0_f32;
    let mut active_len = candidates.len();
    for (idx, &p) in probs.iter().enumerate() {
        let norm_p = p / prob_sum;
        if idx > 0 && cum_prob > top_p {
            active_len = idx;
            break;
        }
        cum_prob += norm_p;
    }

    candidates.truncate(active_len);
    probs.truncate(active_len);

    let active_sum: f32 = probs.iter().copied().sum();
    if active_sum <= 0.0_f32 || !active_sum.is_finite() {
        return Ok(candidates.first().map_or(0, |&(id, _)| id));
    }

    let draw: f32 = rng.next_f32() * active_sum;
    let mut acc = 0.0_f32;
    for (&(id, _), &p) in candidates.iter().zip(probs.iter()) {
        acc += p;
        if draw <= acc {
            return Ok(id);
        }
    }

    Ok(candidates.last().map_or(0, |&(id, _)| id))
}

/// Inputs that condition one run of Mel code generation.
#[derive(Debug, Clone, Copy)]
pub struct CodePrompt<'a> {
    /// Text token IDs (language prefix included).
    pub text_tokens: &'a [i32],
    /// `[1, 3, 1280]` speaker/emotion conditioning.
    pub conds: &'a Array,
    /// Language token ID.
    pub lang_id: i32,
    /// Codes already spoken for the beginning of `text_tokens`; generation continues after
    /// them. Empty for a fresh utterance.
    pub prior_codes: &'a [i32],
    /// Fewest new codes before [`STOP_MEL_TOKEN`] may be sampled. Continuations need it: the
    /// prior codes end where an utterance ended, so the model is inclined to stop at once.
    pub min_codes: usize,
}

/// Generates discrete semantic Mel codes (`0..8192`) autoregressively using [`Gpt`] until
/// [`STOP_MEL_TOKEN`] (`8193`) or `params.max_mel_tokens` is reached.
///
/// # Errors
///
/// Returns [`IndexTtsError`] if prefill or step decoding fails.
pub fn generate_codes(
    gpt: &Gpt,
    text_tokens: &[i32],
    conds: &Array,
    lang_id: i32,
    cache: &mut KvCache,
    params: &SamplingParams,
    rng: &mut SamplerRng,
) -> Result<Vec<i32>> {
    let prompt = CodePrompt {
        text_tokens,
        conds,
        lang_id,
        prior_codes: &[],
        min_codes: 0,
    };
    generate_codes_after(gpt, &prompt, cache, params, rng)
}

/// Generates the Mel codes that follow `prompt.prior_codes` (only the new codes are returned).
/// Prior codes count toward the positional capacity but not toward the repetition penalty.
///
/// # Errors
///
/// Returns [`IndexTtsError::Validation`] if no positions remain, or [`IndexTtsError`] if
/// prefill or step decoding fails.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub fn generate_codes_after(
    gpt: &Gpt,
    prompt: &CodePrompt<'_>,
    cache: &mut KvCache,
    params: &SamplingParams,
    rng: &mut SamplerRng,
) -> Result<Vec<i32>> {
    let prior = prompt.prior_codes.len();
    let max_steps = (params.max_mel_tokens as usize)
        .min(GPT_MEL_POS_LEN.saturating_sub(3).saturating_sub(prior));
    if max_steps == 0 {
        return Err(IndexTtsError::validation(
            "max_mel_tokens",
            "no Mel positions left to generate (max_mel_tokens is 0 or the prior codes fill the \
             capacity)",
        ));
    }

    let mut logits = gpt.prefill_after(
        prompt.text_tokens,
        prompt.conds,
        prompt.lang_id,
        prompt.prior_codes,
        cache,
    )?;
    // The repetition penalty guards the new codes against loops; prior codes were already
    // spoken, and penalizing them pushes the continuation toward unusual phone codes (an audible
    // accent, ~1.6x the error rate in testing).
    let mut seen = SeenTokenBitSet::new_with_initial_tokens();
    let mut codes = Vec::with_capacity(max_steps.min(512));

    for step in 0..max_steps {
        if codes.len() < prompt.min_codes
            && let Some(stop) = logits.get_mut(STOP_MEL_TOKEN as usize)
        {
            *stop = f32::NEG_INFINITY;
        }
        let tok = sample_next_token(&mut logits, &seen, params, rng)?;
        if tok == STOP_MEL_TOKEN {
            break;
        }
        codes.push(tok);
        if tok >= 0 && (tok as usize) < GPT_MEL_VOCAB {
            seen.insert(tok as usize);
        }
        if step + 1 < max_steps {
            let pos_idx = (prior + step) as i32 + 2;
            logits = gpt.step(cache, tok, pos_idx)?;
        }
    }

    Ok(codes)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn argmax_finite(logits: &[f32]) -> i32 {
    let mut best_idx = 0_i32;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val.is_finite() && val > best_val {
            best_val = val;
            best_idx = idx as i32;
        }
    }
    best_idx
}

fn desc_f32_cmp(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use proptest::prelude::*;
    use safetensors::SafeTensors;

    use super::*;
    use crate::{
        config::{DEFAULT_MODEL_REPO, Language, RepetitionPenalty, Temperature, TopK, TopP},
        models::{DEFAULT_KV_MAX_SEQ, resolve_model_dir},
    };

    #[test]
    fn test_should_apply_repetition_penalty_on_seen_tokens() {
        let mut seen = SeenTokenBitSet::new_with_initial_tokens();
        seen.insert(10);
        seen.insert(20);

        let mut logits = vec![0.0_f32; 32];
        if let Some(slot) = logits.get_mut(1) {
            *slot = 2.0;
        }
        if let Some(slot) = logits.get_mut(10) {
            *slot = 4.0;
        }
        if let Some(slot) = logits.get_mut(20) {
            *slot = -3.0;
        }
        if let Some(slot) = logits.get_mut(25) {
            *slot = 4.0;
        }

        seen.apply_repetition_penalty(&mut logits, 2.0);
        assert!((logits.get(1).copied().unwrap_or(0.0) - 1.0).abs() < 1e-6);
        assert!((logits.get(10).copied().unwrap_or(0.0) - 2.0).abs() < 1e-6);
        assert!((logits.get(20).copied().unwrap_or(0.0) - (-6.0)).abs() < 1e-6);
        assert!((logits.get(25).copied().unwrap_or(0.0) - 4.0).abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn proptest_sampler_always_returns_valid_mel_token_id(
            raw_logits in prop::collection::vec(prop::num::f32::ANY, GPT_MEL_VOCAB),
            temp in 0.05_f32..=2.0_f32,
            top_p in 0.05_f32..=1.0_f32,
            top_k in 1_u32..=500_u32,
            rep_pen in 1.0_f32..=2.5_f32,
            greedy in prop::bool::ANY,
            seed in any::<u64>(),
        ) {
            let mut logits = raw_logits;
            let seen = SeenTokenBitSet::new_with_initial_tokens();
            let params = SamplingParams::builder()
                .temperature(Temperature::new(temp).unwrap_or_default())
                .top_p(TopP::new(top_p).unwrap_or_default())
                .top_k(TopK::new(top_k).unwrap_or_default())
                .repetition_penalty(RepetitionPenalty::new(rep_pen).unwrap_or_default())
                .max_mel_tokens(32)
                .greedy(greedy)
                .build();
            let mut rng = SamplerRng::new(seed);
            let res = sample_next_token(&mut logits, &seen, &params, &mut rng);
            prop_assert!(res.is_ok());
            if let Ok(tok) = res {
                let vocab_i32 = i32::try_from(GPT_MEL_VOCAB).unwrap_or(8194);
                prop_assert!(tok >= 0 && tok < vocab_i32);
            }
        }
    }

    #[test]
    fn test_should_match_python_reference_cp11_greedy_decode_codes() {
        let Ok(model_dir) = resolve_model_dir(None, DEFAULT_MODEL_REPO) else {
            return;
        };
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("golden_ref.safetensors");
        #[allow(clippy::disallowed_methods)]
        let Ok(bytes) = fs::read(&fixture_path) else {
            return;
        };
        let Ok(st) = SafeTensors::deserialize(&bytes) else {
            return;
        };

        let Ok(gpt) = Gpt::load_from_path(model_dir.join("gpt_int8.safetensors")) else {
            return;
        };

        let token_ids_0 = extract_i32(&st, "token_ids_0");
        let conds = extract_array(&st, "conds");
        let expected_codes = extract_i32(&st, "greedy_codes");

        let Ok(mut cache) = KvCache::new(DEFAULT_KV_MAX_SEQ) else {
            return;
        };
        let params = SamplingParams::builder()
            .max_mel_tokens(32)
            .greedy(true)
            .build();
        let mut rng = SamplerRng::new(0);

        let codes_res = generate_codes(
            &gpt,
            &token_ids_0,
            &conds,
            Language::Zh.token_id(),
            &mut cache,
            &params,
            &mut rng,
        );
        assert!(codes_res.is_ok());
        if let Ok(rust_codes) = codes_res {
            assert_eq!(
                rust_codes, expected_codes,
                "CP-11 32-step greedy Mel codes must match Python reference 100% bit-exact"
            );
        }
    }

    fn extract_f32(st: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<f32>) {
        let Ok(view) = st.tensor(name) else {
            return (Vec::new(), Vec::new());
        };
        let mut out = Vec::with_capacity(view.data().len() / 4);
        for &b in view.data().as_chunks::<4>().0 {
            out.push(f32::from_le_bytes(b));
        }
        (view.shape().to_vec(), out)
    }

    fn extract_i32(st: &SafeTensors<'_>, name: &str) -> Vec<i32> {
        let Ok(view) = st.tensor(name) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(view.data().len() / 4);
        for &b in view.data().as_chunks::<4>().0 {
            out.push(i32::from_le_bytes(b));
        }
        out
    }

    fn extract_array(st: &SafeTensors<'_>, name: &str) -> Array {
        let (shape, data) = extract_f32(st, name);
        let shape_i32: Vec<i32> = shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        Array::from_slice(&data, &shape_i32)
    }
}

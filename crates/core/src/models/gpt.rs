//! Stage 1 GPT-2 `UnifiedVoice` autoregressive decoder (`gpt_int8.safetensors`) with 8-bit group
//! quantization (`group_size=64, bits=8`) and preallocated in-place KV cache.
//!
//! Architecture:
//! - `text_embedding` (`[60510, 1280]`), `lang_embedding` (`[107, 1280]`), `mel_embedding` (`[8194,
//!   1280]`), `mel_pos_embedding` (`[1818, 1280]`), `text_pos_embedding` (`[602, 1280]`).
//! - 24-layer GPT-2 core (`1280` dim, `20` heads, `64` head dim, `5120` MLP inner dim with
//!   `gelu_approximate`).
//! - `final_norm` (`LayerNorm(1280, eps=1e-5)`) + quantized `mel_head` (`1280 -> 8194`).

use std::{iter::once, path::Path};

use mlx_rs::{
    Array,
    fast::{ScaledDotProductAttentionMask, layer_norm, scaled_dot_product_attention},
    nn::gelu_approximate,
    ops::{
        concatenate,
        indexing::{TryIndexMutOp, TryIndexOp},
        quantized_matmul, split_equal, zeros,
    },
    transforms::eval,
};

use crate::{
    error::{IndexTtsError, Result},
    models::{FilteredWeights, ModelWeightKind, SafeTensorsLoader},
};

/// GPT hidden model dimension (`1280`).
pub const GPT_DIM: usize = 1_280;
/// Number of attention heads (`20`).
pub const GPT_NUM_HEADS: usize = 20;
/// Per-head dimension (`64`).
pub const GPT_HEAD_DIM: usize = 64;
/// Number of GPT-2 transformer layers (`24`).
pub const GPT_NUM_LAYERS: usize = 24;
/// MLP intermediate dimension (`5120`).
pub const GPT_INNER_DIM: usize = 5_120;
/// `LayerNorm` epsilon (`1e-5`).
pub const GPT_NORM_EPS: f32 = 1e-5;
/// Mel token vocabulary size (`8194`).
pub const GPT_MEL_VOCAB: usize = 8_194;
/// Text token vocabulary size (`60510`).
pub const GPT_TEXT_VOCAB: usize = 60_510;
/// Language token vocabulary size (`107`).
pub const GPT_LANG_VOCAB: usize = 107;
/// Maximum Mel positional embedding length (`1818`).
pub const GPT_MEL_POS_LEN: usize = 1_818;
/// Maximum text positional embedding length (`602`).
pub const GPT_TEXT_POS_LEN: usize = 602;
/// Default preallocated KV cache sequence capacity (`2048`).
pub const DEFAULT_KV_MAX_SEQ: usize = 2_048;

/// Start-of-text token ID (`0`).
pub const START_TEXT_TOKEN: i32 = 0;
/// End-of-text token ID (`1`).
pub const STOP_TEXT_TOKEN: i32 = 1;
/// Start-of-Mel token ID (`8192`).
pub const START_MEL_TOKEN: i32 = 8_192;
/// Stop-of-Mel token ID (`8193`).
pub const STOP_MEL_TOKEN: i32 = 8_193;

const QUANT_GROUP_SIZE: i32 = 64;
const QUANT_BITS: i32 = 8;
const ATTN_SCALE: f32 = 0.125; // 64^(-0.5)
const KV_ALLOC_STEP: usize = 256;

/// Preallocated in-place Key/Value cache across all 24 GPT-2 attention layers.
#[derive(Debug, Clone)]
pub struct KvCache {
    keys: Vec<Array>,
    values: Vec<Array>,
    valid_len: usize,
    max_seq: usize,
}

impl KvCache {
    /// Allocates a new zero-initialized [`KvCache`] with capacity for `max_seq` tokens.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `max_seq == 0` or exceeds `i32::MAX`, or
    /// [`IndexTtsError::Mlx`] if buffer allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::KvCache;
    ///
    /// let cache = KvCache::new(512)?;
    /// assert_eq!(cache.valid_len(), 0);
    /// assert_eq!(cache.max_seq(), 512);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn new(max_seq: usize) -> Result<Self> {
        if max_seq == 0 {
            return Err(IndexTtsError::validation(
                "max_seq",
                "KV cache max_seq must be >= 1",
            ));
        }
        let _max_seq_i32 = i32::try_from(max_seq)
            .map_err(|_| IndexTtsError::validation("max_seq", "max_seq exceeds i32::MAX"))?;

        let init_cap = max_seq.min(KV_ALLOC_STEP) as i32;
        let shape = [1, GPT_NUM_HEADS as i32, init_cap, GPT_HEAD_DIM as i32];
        let mut keys = Vec::with_capacity(GPT_NUM_LAYERS);
        let mut values = Vec::with_capacity(GPT_NUM_LAYERS);
        for _ in 0..GPT_NUM_LAYERS {
            keys.push(zeros::<f32>(&shape)?);
            values.push(zeros::<f32>(&shape)?);
        }
        eval(keys.iter().chain(values.iter()))?;

        Ok(Self {
            keys,
            values,
            valid_len: 0,
            max_seq,
        })
    }

    /// Resets the valid sequence length to `0`, retaining preallocated layer buffers for reuse.
    pub fn reset(&mut self) {
        self.valid_len = 0;
    }

    /// Returns the number of currently populated sequence positions in the cache.
    #[must_use]
    pub const fn valid_len(&self) -> usize {
        self.valid_len
    }

    /// Returns the maximum sequence capacity of the cache.
    #[must_use]
    pub const fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Writes `k_new` and `v_new` (`[1, 20, step_len, 64]`) in-place into layer `layer_idx` at
    /// `self.valid_len..new_valid_len` and returns `(k_all, v_all)` sliced over `0..new_valid_len`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn update_layer(
        &mut self,
        layer_idx: usize,
        k_new: &Array,
        v_new: &Array,
        new_valid_len: usize,
    ) -> Result<(Array, Array)> {
        if new_valid_len > self.max_seq {
            return Err(IndexTtsError::validation(
                "kv_cache",
                format!(
                    "sequence length {new_valid_len} exceeds KV cache capacity {}",
                    self.max_seq
                ),
            ));
        }

        let prev_i32 = self.valid_len as i32;
        let new_i32 = new_valid_len as i32;

        let k_slot = self
            .keys
            .get_mut(layer_idx)
            .ok_or_else(|| IndexTtsError::mlx("KV cache key layer index out of bounds"))?;
        let v_slot = self
            .values
            .get_mut(layer_idx)
            .ok_or_else(|| IndexTtsError::mlx("KV cache value layer index out of bounds"))?;

        let cur_cap = usize::try_from(k_slot.dim(2)).unwrap_or(0);
        if new_valid_len > cur_cap {
            let target_cap = new_valid_len
                .next_multiple_of(KV_ALLOC_STEP)
                .min(self.max_seq);
            let grow_i32 = (target_cap - cur_cap) as i32;
            let pad_zeros =
                zeros::<f32>(&[1, GPT_NUM_HEADS as i32, grow_i32, GPT_HEAD_DIM as i32])?;
            *k_slot = concatenate(&[&*k_slot, &pad_zeros], 2)?;
            *v_slot = concatenate(&[&*v_slot, &pad_zeros], 2)?;
        }

        k_slot
            .try_index_mut((.., .., prev_i32..new_i32, ..), k_new)
            .map_err(|err| {
                IndexTtsError::mlx(format!("KV cache key slice update failed: {err}"))
            })?;
        v_slot
            .try_index_mut((.., .., prev_i32..new_i32, ..), v_new)
            .map_err(|err| {
                IndexTtsError::mlx(format!("KV cache value slice update failed: {err}"))
            })?;

        let k_all = k_slot.try_index((.., .., ..new_i32, ..))?;
        let v_all = v_slot.try_index((.., .., ..new_i32, ..))?;
        Ok((k_all, v_all))
    }

    /// Evaluates `logits` together with all layer KV buffers in a single batched Metal command
    /// submission so the computation graph stays shallow across decode steps.
    fn eval_with_logits(&self, logits: &Array) -> Result<()> {
        eval(
            once(logits)
                .chain(self.keys.iter())
                .chain(self.values.iter()),
        )?;
        Ok(())
    }
}

/// 8-bit group-quantized linear layer (`group_size=64, bits=8`).
#[derive(Debug, Clone)]
struct QuantizedLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    bias: Array,
}

impl QuantizedLinear {
    fn from_weights(
        weights: &FilteredWeights,
        prefix: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let packed_in = in_features / 4;
        let groups = in_features / (QUANT_GROUP_SIZE as usize);
        let weight =
            weights.tensor_array_shaped(&format!("{prefix}.weight"), &[out_features, packed_in])?;
        let scales =
            weights.tensor_array_shaped(&format!("{prefix}.scales"), &[out_features, groups])?;
        let biases =
            weights.tensor_array_shaped(&format!("{prefix}.biases"), &[out_features, groups])?;
        let bias = weights.tensor_array_shaped(&format!("{prefix}.bias"), &[out_features])?;
        Ok(Self {
            weight,
            scales,
            biases,
            bias,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mm = quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            QUANT_GROUP_SIZE,
            QUANT_BITS,
        )?;
        Ok(mm.add(&self.bias)?)
    }
}

#[derive(Debug, Clone)]
struct Gpt2Block {
    ln1_weight: Array,
    ln1_bias: Array,
    c_attn: QuantizedLinear,
    c_proj: QuantizedLinear,
    ln2_weight: Array,
    ln2_bias: Array,
    c_fc: QuantizedLinear,
    mlp_c_proj: QuantizedLinear,
}

impl Gpt2Block {
    fn from_weights(weights: &FilteredWeights, layer_idx: usize) -> Result<Self> {
        let prefix = format!("gpt.h.{layer_idx}");
        let ln1_weight =
            weights.tensor_array_shaped(&format!("{prefix}.ln_1.weight"), &[GPT_DIM])?;
        let ln1_bias = weights.tensor_array_shaped(&format!("{prefix}.ln_1.bias"), &[GPT_DIM])?;
        let c_attn = QuantizedLinear::from_weights(
            weights,
            &format!("{prefix}.attn.c_attn"),
            GPT_DIM,
            GPT_DIM * 3,
        )?;
        let c_proj = QuantizedLinear::from_weights(
            weights,
            &format!("{prefix}.attn.c_proj"),
            GPT_DIM,
            GPT_DIM,
        )?;
        let ln2_weight =
            weights.tensor_array_shaped(&format!("{prefix}.ln_2.weight"), &[GPT_DIM])?;
        let ln2_bias = weights.tensor_array_shaped(&format!("{prefix}.ln_2.bias"), &[GPT_DIM])?;
        let c_fc = QuantizedLinear::from_weights(
            weights,
            &format!("{prefix}.mlp.c_fc"),
            GPT_DIM,
            GPT_INNER_DIM,
        )?;
        let mlp_c_proj = QuantizedLinear::from_weights(
            weights,
            &format!("{prefix}.mlp.c_proj"),
            GPT_INNER_DIM,
            GPT_DIM,
        )?;

        Ok(Self {
            ln1_weight,
            ln1_bias,
            c_attn,
            c_proj,
            ln2_weight,
            ln2_bias,
            c_fc,
            mlp_c_proj,
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn forward(
        &self,
        x: &Array,
        layer_idx: usize,
        cache: &mut KvCache,
        new_valid_len: usize,
        causal: bool,
    ) -> Result<Array> {
        let batch = x.dim(0);
        let time = x.dim(1);
        let n_head = GPT_NUM_HEADS as i32;
        let head_dim = GPT_HEAD_DIM as i32;

        // 1. Pre-LN + QKV projection
        let n1 = layer_norm(
            x,
            Some(&self.ln1_weight),
            Some(&self.ln1_bias),
            GPT_NORM_EPS,
        )?;
        let qkv = self.c_attn.forward(&n1)?;
        let parts = split_equal(&qkv, 3, -1)?;
        let (Some(q_raw), Some(k_raw), Some(v_raw)) = (parts.first(), parts.get(1), parts.get(2))
        else {
            return Err(IndexTtsError::mlx("failed to split GPT-2 QKV projection"));
        };

        let q_heads = q_raw
            .reshape(&[batch, time, n_head, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k_new = k_raw
            .reshape(&[batch, time, n_head, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v_new = v_raw
            .reshape(&[batch, time, n_head, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let (k_all, v_all) = cache.update_layer(layer_idx, &k_new, &v_new, new_valid_len)?;
        let mask = if causal && time > 1 {
            Some(ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let attn = scaled_dot_product_attention(q_heads, k_all, v_all, ATTN_SCALE, mask, None)?;
        let attn_flat =
            attn.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[batch, time, GPT_DIM as i32])?;
        let hidden = x.add(&self.c_proj.forward(&attn_flat)?)?;

        // 2. Pre-LN + GELU MLP
        let n2 = layer_norm(
            &hidden,
            Some(&self.ln2_weight),
            Some(&self.ln2_bias),
            GPT_NORM_EPS,
        )?;
        let fc = gelu_approximate(&self.c_fc.forward(&n2)?)?;
        let mlp_out = self.mlp_c_proj.forward(&fc)?;
        Ok(hidden.add(&mlp_out)?)
    }
}

/// Stage 1 Int8-quantized GPT-2 `UnifiedVoice` autoregressive decoder (`gpt_int8.safetensors`).
#[derive(Debug, Clone)]
pub struct Gpt {
    text_embedding: Array,
    lang_embedding: Array,
    mel_embedding: Array,
    mel_pos_embedding: Array,
    text_pos_embedding: Array,
    blocks: Vec<Gpt2Block>,
    ln_f_weight: Array,
    ln_f_bias: Array,
    final_norm_weight: Array,
    final_norm_bias: Array,
    mel_head: QuantizedLinear,
}

impl Gpt {
    /// Constructs a [`Gpt`] decoder from pre-filtered weights.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if any required parameter tensor is missing or has
    /// an unexpected shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::{
    ///     Gpt, ModelWeightKind, SafeTensorsLoader, resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let weights = SafeTensorsLoader::load_from_path(
    ///         ModelWeightKind::GptInt8,
    ///         dir.join("gpt_int8.safetensors"),
    ///     )?;
    ///     let gpt = Gpt::from_weights(&weights)?;
    ///     assert_eq!(gpt.num_layers(), 24);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let text_embedding =
            weights.tensor_array_shaped("text_embedding.weight", &[GPT_TEXT_VOCAB, GPT_DIM])?;
        let lang_embedding =
            weights.tensor_array_shaped("lang_embedding.weight", &[GPT_LANG_VOCAB, GPT_DIM])?;
        let mel_embedding =
            weights.tensor_array_shaped("mel_embedding.weight", &[GPT_MEL_VOCAB, GPT_DIM])?;
        let mel_pos_embedding = weights
            .tensor_array_shaped("mel_pos_embedding.emb.weight", &[GPT_MEL_POS_LEN, GPT_DIM])?;
        let text_pos_embedding = weights.tensor_array_shaped(
            "text_pos_embedding.emb.weight",
            &[GPT_TEXT_POS_LEN, GPT_DIM],
        )?;

        let mut blocks = Vec::with_capacity(GPT_NUM_LAYERS);
        for i in 0..GPT_NUM_LAYERS {
            blocks.push(Gpt2Block::from_weights(weights, i)?);
        }

        let ln_f_weight = weights.tensor_array_shaped("gpt.ln_f.weight", &[GPT_DIM])?;
        let ln_f_bias = weights.tensor_array_shaped("gpt.ln_f.bias", &[GPT_DIM])?;
        let final_norm_weight = weights.tensor_array_shaped("final_norm.weight", &[GPT_DIM])?;
        let final_norm_bias = weights.tensor_array_shaped("final_norm.bias", &[GPT_DIM])?;
        let mel_head = QuantizedLinear::from_weights(weights, "mel_head", GPT_DIM, GPT_MEL_VOCAB)?;

        Ok(Self {
            text_embedding,
            lang_embedding,
            mel_embedding,
            mel_pos_embedding,
            text_pos_embedding,
            blocks,
            ln_f_weight,
            ln_f_bias,
            final_norm_weight,
            final_norm_bias,
            mel_head,
        })
    }

    /// Loads [`Gpt`] weights from `gpt_int8.safetensors` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if the file cannot be read or tensors fail validation.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let weights = SafeTensorsLoader::load_from_path(ModelWeightKind::GptInt8, path)?;
        Self::from_weights(&weights)
    }

    /// Returns the number of GPT-2 layers (`24`).
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Prefills the KV cache with `[conds (3), <BOS>, text_tokens..., <EOS>, <START_MEL>]` and
    /// returns the `[8194]` next-token logits as a CPU `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `text_tokens` is empty or exceeds positional
    /// limits, or [`IndexTtsError::Mlx`] if GPU execution fails.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn prefill(
        &self,
        text_tokens: &[i32],
        conds: &Array,
        lang_id: i32,
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        if text_tokens.is_empty() {
            return Err(IndexTtsError::validation(
                "text_tokens",
                "text_tokens must not be empty",
            ));
        }
        let full_text_len = text_tokens.len() + 2;
        if full_text_len > GPT_TEXT_POS_LEN {
            return Err(IndexTtsError::validation(
                "text_tokens",
                format!("text sequence length {full_text_len} exceeds {GPT_TEXT_POS_LEN}"),
            ));
        }
        if conds.shape() != [1, 3, GPT_DIM as i32] {
            return Err(IndexTtsError::validation(
                "conds",
                format!("expected shape [1, 3, 1280], got {:?}", conds.shape()),
            ));
        }
        if !(0..GPT_LANG_VOCAB as i32).contains(&lang_id) {
            return Err(IndexTtsError::validation(
                "lang_id",
                format!("lang_id {lang_id} out of range 0..{GPT_LANG_VOCAB}"),
            ));
        }

        cache.reset();

        // Build full text sequence: [0, text_tokens..., 1]
        let mut full_text = Vec::with_capacity(full_text_len);
        full_text.push(START_TEXT_TOKEN);
        full_text.extend_from_slice(text_tokens);
        full_text.push(STOP_TEXT_TOKEN);

        let full_text_i32 = full_text_len as i32;
        let text_arr = Array::from_slice(&full_text, &[1, full_text_i32]);
        let pos_indices: Vec<i32> = (0..full_text_i32).collect();
        let pos_arr = Array::from_slice(&pos_indices, &[1, full_text_i32]);

        let lang_arr = Array::from_slice(&[lang_id], &[1]);
        let lang_emb =
            self.lang_embedding
                .try_index(&lang_arr)?
                .reshape(&[1, 1, GPT_DIM as i32])?;

        let t_emb = self
            .text_embedding
            .try_index(&text_arr)?
            .add(&lang_emb)?
            .add(&self.text_pos_embedding.try_index(&pos_arr)?)?;

        let start_mel_arr = Array::from_slice(&[START_MEL_TOKEN], &[1, 1]);
        let start_pos_arr = Array::from_slice(&[0_i32], &[1, 1]);
        let m_start = self
            .mel_embedding
            .try_index(&start_mel_arr)?
            .add(&self.mel_pos_embedding.try_index(&start_pos_arr)?)?;

        let emb = concatenate(&[conds, &t_emb, &m_start], 1)?;
        let seq_len = usize::try_from(emb.dim(1)).unwrap_or(0);

        let mut h = emb;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, layer_idx, cache, seq_len, true)?;
        }
        cache.valid_len = seq_len;

        let h_ln_f = layer_norm(
            &h,
            Some(&self.ln_f_weight),
            Some(&self.ln_f_bias),
            GPT_NORM_EPS,
        )?;
        let last_idx = (seq_len as i32) - 1;
        let h_last_in = h_ln_f
            .try_index((.., last_idx, ..))?
            .reshape(&[1, GPT_DIM as i32])?;
        let h_final = layer_norm(
            &h_last_in,
            Some(&self.final_norm_weight),
            Some(&self.final_norm_bias),
            GPT_NORM_EPS,
        )?;
        let logits = self
            .mel_head
            .forward(&h_final)?
            .reshape(&[GPT_MEL_VOCAB as i32])?;
        cache.eval_with_logits(&logits)?;

        Ok(logits.try_as_slice::<f32>()?.to_vec())
    }

    /// Executes a single autoregressive decode step for `mel_token_id` at positional index
    /// `pos_idx` (`step + 2`), updating `cache` in-place and returning `[8194]` logits.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `mel_token_id` or `pos_idx` is out of bounds, or
    /// [`IndexTtsError::Mlx`] if GPU execution fails.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn step(&self, cache: &mut KvCache, mel_token_id: i32, pos_idx: i32) -> Result<Vec<f32>> {
        if !(0..GPT_MEL_VOCAB as i32).contains(&mel_token_id) {
            return Err(IndexTtsError::validation(
                "mel_token_id",
                format!("mel_token_id {mel_token_id} out of range 0..{GPT_MEL_VOCAB}"),
            ));
        }
        if !(0..GPT_MEL_POS_LEN as i32).contains(&pos_idx) {
            return Err(IndexTtsError::validation(
                "pos_idx",
                format!("pos_idx {pos_idx} out of range 0..{GPT_MEL_POS_LEN}"),
            ));
        }

        let tok_arr = Array::from_slice(&[mel_token_id], &[1, 1]);
        let pos_arr = Array::from_slice(&[pos_idx], &[1, 1]);
        let emb = self
            .mel_embedding
            .try_index(&tok_arr)?
            .add(&self.mel_pos_embedding.try_index(&pos_arr)?)?;

        let new_valid_len = cache.valid_len + 1;
        let mut h = emb;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, layer_idx, cache, new_valid_len, false)?;
        }
        cache.valid_len = new_valid_len;

        let h_ln_f = layer_norm(
            &h,
            Some(&self.ln_f_weight),
            Some(&self.ln_f_bias),
            GPT_NORM_EPS,
        )?;
        let h_last_in = h_ln_f.reshape(&[1, GPT_DIM as i32])?;
        let h_final = layer_norm(
            &h_last_in,
            Some(&self.final_norm_weight),
            Some(&self.final_norm_bias),
            GPT_NORM_EPS,
        )?;
        let logits = self
            .mel_head
            .forward(&h_final)?
            .reshape(&[GPT_MEL_VOCAB as i32])?;
        cache.eval_with_logits(&logits)?;

        Ok(logits.try_as_slice::<f32>()?.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp::Ordering, fs, path::PathBuf};

    use safetensors::SafeTensors;

    use super::*;
    use crate::{
        config::{DEFAULT_MODEL_REPO, Language},
        dsp::cosine_similarity_and_max_diff,
        models::resolve_model_dir,
    };

    #[test]
    fn test_should_match_python_reference_cp10_gpt_prefill_logits() {
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

        let gpt_res = Gpt::load_from_path(model_dir.join("gpt_int8.safetensors"));
        assert!(gpt_res.is_ok());
        let Ok(gpt) = gpt_res else {
            return;
        };

        let token_ids_0 = extract_i32(&st, "token_ids_0");
        let conds = extract_array(&st, "conds");
        let (_, expected_logits) = extract_f32(&st, "prefill_logits");

        let Ok(mut cache) = KvCache::new(DEFAULT_KV_MAX_SEQ) else {
            return;
        };
        let logits_res = gpt.prefill(&token_ids_0, &conds, Language::Zh.token_id(), &mut cache);
        assert!(logits_res.is_ok());
        let Ok(rust_logits) = logits_res else {
            return;
        };

        assert_eq!(rust_logits.len(), expected_logits.len());
        let (cos, _max_diff) = cosine_similarity_and_max_diff(&rust_logits, &expected_logits);
        assert!(
            cos >= 0.999_9,
            "CP-10 prefill_logits cosine similarity {cos} < 0.9999"
        );

        let rust_top10 = top_k_indices(&rust_logits, 10);
        let ref_top10 = top_k_indices(&expected_logits, 10);
        assert_eq!(
            rust_top10, ref_top10,
            "CP-10 prefill_logits Top-10 token IDs mismatch"
        );
    }

    fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
        let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        pairs.into_iter().take(k).map(|(idx, _)| idx).collect()
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

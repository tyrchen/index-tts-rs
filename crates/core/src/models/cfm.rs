//! Conditional Flow Matching (CFM) `DiT` estimator (`cfm_estimator.safetensors`) and all-GPU
//! 25-step Euler ODE solver with Classifier-Free Guidance (CFG).
//!
//! Architecture (`index_tts_2_5_mlx/models/cfm.py`):
//! - Dual sinusoidal `TimestepEmbedder` (`t_embedder` for transformer `AdaLN` & `FinalLayer`,
//!   `t_embedder2` for `WaveNet`, with `1000.0 * t * freqs`).
//! - 13-layer U-ViT `Transformer` (`512` dim, `8` heads, `64` head dim, `RoPE` `base=10000`,
//!   `AdaLN` with `w * RMSNorm(x) + b`, `SwiGLU` MLP `1536`, long skip connections `0..=5 ->
//!   7..=12`).
//! - Post-transformer `skip_linear(cat(h, x))` -> `conv1(x_res)` -> 8-layer `WaveNet(h, t2) +
//!   res_projection(x_res)`.
//! - `FinalLayer(h, t1)` (`LayerNorm(eps=1e-6, affine=false)` + `AdaLN` modulation + `Linear(512 ->
//!   512)`)
//!   + `conv2` (`Conv1d(512 -> 80, k=1)`).

use std::path::Path;

use mlx_rs::{
    Array, array,
    fast::{
        ScaledDotProductAttentionMask, layer_norm, rms_norm, rope, scaled_dot_product_attention,
    },
    nn::silu,
    ops::{
        addmm, broadcast_to, concatenate, conv1d, indexing::TryIndexOp, sigmoid, split_equal, tanh,
        zeros,
    },
};

use crate::{
    error::{IndexTtsError, Result},
    models::{FilteredWeights, ModelWeightKind, SafeTensorsLoader},
};

/// Transformer & `WaveNet` model dimension (`512`).
pub const CFM_DIM: usize = 512;
/// Number of U-ViT Transformer layers (`13`).
pub const CFM_NUM_LAYERS: usize = 13;
/// Number of attention heads (`8`).
pub const CFM_NUM_HEADS: usize = 8;
/// Per-head dimension (`64`).
pub const CFM_HEAD_DIM: usize = 64;
/// `RoPE` theta base (`10000.0`).
pub const CFM_ROPE_BASE: f32 = 10_000.0;
/// `RMSNorm` epsilon (`1e-5`).
pub const CFM_NORM_EPS: f32 = 1e-5;
/// `SwiGLU` intermediate dimension (`1536`).
pub const CFM_INTERMEDIATE: usize = 1_536;
/// Mel spectrogram channel dimension (`80`).
pub const CFM_IN_CHANNELS: usize = 80;
/// Speaker style vector dimension (`192`).
pub const CFM_STYLE_DIM: usize = 192;
/// Number of `WaveNet` residual layers (`8`).
pub const CFM_WN_LAYERS: usize = 8;
/// `WaveNet` convolution kernel size (`5`).
pub const CFM_WN_KERNEL: usize = 5;
/// Merged input dimension before `cond_x_merge_linear` (`80 + 80 + 512 + 192 = 864`).
pub const CFM_MERGED_IN_DIM: usize = CFM_IN_CHANNELS * 2 + CFM_DIM + CFM_STYLE_DIM;

const SKIP_SPLIT_LAYER: usize = CFM_NUM_LAYERS / 2;
const FINAL_NORM_EPS: f32 = 1e-6;

fn linear_bias(x: &Array, weight: &Array, bias: &Array) -> Result<Array> {
    Ok(addmm(bias, x, weight.t(), None, None)?)
}

/// Precomputes rotary position embedding cosine/sine tables `[seq_len, 32, 2]`.
///
/// # Errors
///
/// Returns [`IndexTtsError::Validation`] if `seq_len == 0` or exceeds `i32::MAX`.
#[allow(clippy::cast_precision_loss)]
pub fn compute_rope_freqs(seq_len: usize) -> Result<Array> {
    if seq_len == 0 {
        return Err(IndexTtsError::validation(
            "seq_len",
            "seq_len must be >= 1 for RoPE frequency table",
        ));
    }
    let seq_len_i32 = i32::try_from(seq_len)
        .map_err(|_| IndexTtsError::validation("seq_len", "seq_len exceeds i32::MAX"))?;
    let half_dim = CFM_HEAD_DIM / 2;
    let half_dim_i32 = i32::try_from(half_dim).unwrap_or(32);

    let mut inv_freqs = Vec::with_capacity(half_dim);
    for k in 0..half_dim {
        let exponent = (2 * k) as f32 / (CFM_HEAD_DIM as f32);
        inv_freqs.push(1.0_f32 / CFM_ROPE_BASE.powf(exponent));
    }

    let mut table = Vec::with_capacity(seq_len * half_dim * 2);
    for t in 0..seq_len {
        let t_f32 = t as f32;
        for &freq in &inv_freqs {
            let angle = t_f32 * freq;
            table.push(angle.cos());
            table.push(angle.sin());
        }
    }
    Ok(Array::from_slice(&table, &[seq_len_i32, half_dim_i32, 2]))
}

#[derive(Debug, Clone)]
struct TimestepEmbedder {
    freqs_1000: Array,
    mlp0_weight: Array,
    mlp0_bias: Array,
    mlp2_weight: Array,
    mlp2_bias: Array,
}

impl TimestepEmbedder {
    #[allow(clippy::cast_precision_loss)]
    fn from_weights(weights: &FilteredWeights, prefix: &str) -> Result<Self> {
        let half = 128_usize;
        let ln_10000 = 10_000.0_f32.ln();
        let mut freq_vals = Vec::with_capacity(half);
        for k in 0..half {
            freq_vals.push(1000.0_f32 * (-ln_10000 * (k as f32) / (half as f32)).exp());
        }
        let freqs_1000 = Array::from_slice(&freq_vals, &[1, 128]);
        let mlp0_weight = weights
            .tensor_array_shaped(&format!("{prefix}.mlp.layers.0.weight"), &[CFM_DIM, 256])?;
        let mlp0_bias =
            weights.tensor_array_shaped(&format!("{prefix}.mlp.layers.0.bias"), &[CFM_DIM])?;
        let mlp2_weight = weights.tensor_array_shaped(
            &format!("{prefix}.mlp.layers.2.weight"),
            &[CFM_DIM, CFM_DIM],
        )?;
        let mlp2_bias =
            weights.tensor_array_shaped(&format!("{prefix}.mlp.layers.2.bias"), &[CFM_DIM])?;
        Ok(Self {
            freqs_1000,
            mlp0_weight,
            mlp0_bias,
            mlp2_weight,
            mlp2_bias,
        })
    }

    fn forward(&self, t: &Array) -> Result<Array> {
        let args = t.reshape(&[-1, 1])?.multiply(&self.freqs_1000)?;
        let emb = concatenate(&[&args.cos()?, &args.sin()?], -1)?;
        let h = silu(&linear_bias(&emb, &self.mlp0_weight, &self.mlp0_bias)?)?;
        linear_bias(&h, &self.mlp2_weight, &self.mlp2_bias)
    }
}

#[derive(Debug, Clone)]
struct AdaLn {
    proj_weight: Array,
    proj_bias: Array,
    norm_weight: Array,
}

impl AdaLn {
    fn from_weights(weights: &FilteredWeights, prefix: &str) -> Result<Self> {
        let proj_weight = weights.tensor_array_shaped(
            &format!("{prefix}.project_layer.weight"),
            &[CFM_DIM * 2, CFM_DIM],
        )?;
        let proj_bias =
            weights.tensor_array_shaped(&format!("{prefix}.project_layer.bias"), &[CFM_DIM * 2])?;
        let norm_weight =
            weights.tensor_array_shaped(&format!("{prefix}.norm.weight"), &[CFM_DIM])?;
        Ok(Self {
            proj_weight,
            proj_bias,
            norm_weight,
        })
    }

    fn forward(&self, x: &Array, c_expanded: &Array) -> Result<Array> {
        let wb = linear_bias(c_expanded, &self.proj_weight, &self.proj_bias)?;
        let parts = split_equal(&wb, 2, -1)?;
        let (Some(w), Some(b)) = (parts.first(), parts.get(1)) else {
            return Err(IndexTtsError::mlx("failed to split AdaLN modulation"));
        };
        let normed = rms_norm(x, Some(&self.norm_weight), CFM_NORM_EPS)?;
        Ok(w.multiply(&normed)?.add(b)?)
    }
}

#[derive(Debug, Clone)]
struct TransformerBlock {
    norm1: AdaLn,
    wqkv_weight_t: Array,
    wo_weight_t: Array,
    norm2: AdaLn,
    w13_weight_t: Array,
    w2_weight_t: Array,
    skip_in_weight: Array,
    skip_in_bias: Array,
}

impl TransformerBlock {
    #[allow(clippy::similar_names)]
    fn from_weights(weights: &FilteredWeights, layer_idx: usize) -> Result<Self> {
        let prefix = format!("transformer.layers.{layer_idx}");
        let norm1 = AdaLn::from_weights(weights, &format!("{prefix}.attention_norm"))?;
        let wqkv_weight = weights.tensor_array_shaped(
            &format!("{prefix}.attention.wqkv.weight"),
            &[CFM_DIM * 3, CFM_DIM],
        )?;
        let wo_weight = weights.tensor_array_shaped(
            &format!("{prefix}.attention.wo.weight"),
            &[CFM_DIM, CFM_DIM],
        )?;
        let norm2 = AdaLn::from_weights(weights, &format!("{prefix}.ffn_norm"))?;
        let w1_weight = weights.tensor_array_shaped(
            &format!("{prefix}.feed_forward.w1.weight"),
            &[CFM_INTERMEDIATE, CFM_DIM],
        )?;
        let w3_weight = weights.tensor_array_shaped(
            &format!("{prefix}.feed_forward.w3.weight"),
            &[CFM_INTERMEDIATE, CFM_DIM],
        )?;
        let w2_weight = weights.tensor_array_shaped(
            &format!("{prefix}.feed_forward.w2.weight"),
            &[CFM_DIM, CFM_INTERMEDIATE],
        )?;
        let skip_in_weight = weights.tensor_array_shaped(
            &format!("{prefix}.skip_in_linear.weight"),
            &[CFM_DIM, CFM_DIM * 2],
        )?;
        let skip_in_bias =
            weights.tensor_array_shaped(&format!("{prefix}.skip_in_linear.bias"), &[CFM_DIM])?;

        let wqkv_weight_t = wqkv_weight.t().contiguous()?;
        let wo_weight_t = wo_weight.t().contiguous()?;
        let w13_weight_t = concatenate(&[&w1_weight, &w3_weight], 0)?
            .t()
            .contiguous()?;
        let w2_weight_t = w2_weight.t().contiguous()?;
        wqkv_weight_t.eval()?;
        wo_weight_t.eval()?;
        w13_weight_t.eval()?;
        w2_weight_t.eval()?;

        Ok(Self {
            norm1,
            wqkv_weight_t,
            wo_weight_t,
            norm2,
            w13_weight_t,
            w2_weight_t,
            skip_in_weight,
            skip_in_bias,
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn forward(&self, x: &Array, c_expanded: &Array, skip_in: Option<&Array>) -> Result<Array> {
        let x_in = if let Some(skip) = skip_in {
            let cat = concatenate(&[x, skip], -1)?;
            linear_bias(&cat, &self.skip_in_weight, &self.skip_in_bias)?
        } else {
            x.clone()
        };

        let batch = x_in.dim(0);
        let time = x_in.dim(1);
        let n_head = CFM_NUM_HEADS as i32;
        let head_dim = CFM_HEAD_DIM as i32;

        // 1. Attention with fused Metal RoPE (`traditional = true` interleaved complex rotation)
        let n1 = self.norm1.forward(&x_in, c_expanded)?;
        let qkv = n1.matmul(&self.wqkv_weight_t)?;
        let qkv_parts = split_equal(&qkv, 3, -1)?;
        let (Some(q_raw), Some(k_raw), Some(v_raw)) =
            (qkv_parts.first(), qkv_parts.get(1), qkv_parts.get(2))
        else {
            return Err(IndexTtsError::mlx("failed to split QKV tensor"));
        };

        let q_heads = q_raw
            .reshape(&[batch, time, n_head, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k_heads = k_raw
            .reshape(&[batch, time, n_head, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v_heads = v_raw
            .reshape(&[batch, time, n_head, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q_rope = rope(&q_heads, head_dim, true, CFM_ROPE_BASE, 1.0, 0, None)?;
        let k_rope = rope(&k_heads, head_dim, true, CFM_ROPE_BASE, 1.0, 0, None)?;

        let attn = scaled_dot_product_attention(
            q_rope,
            k_rope,
            v_heads,
            0.125_f32,
            None::<ScaledDotProductAttentionMask>,
            None,
        )?;
        let attn_out = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, time, CFM_DIM as i32])?
            .matmul(&self.wo_weight_t)?;
        let hidden = x_in.add(&attn_out)?;

        // 2. Fused SwiGLU FFN (`w1` and `w3` concatenated)
        let n2 = self.norm2.forward(&hidden, c_expanded)?;
        let gate_up = n2.matmul(&self.w13_weight_t)?;
        let gu_parts = split_equal(&gate_up, 2, -1)?;
        let (Some(gate_raw), Some(up)) = (gu_parts.first(), gu_parts.get(1)) else {
            return Err(IndexTtsError::mlx("failed to split SwiGLU gate/up tensor"));
        };
        let ffn_out = silu(gate_raw)?.multiply(up)?.matmul(&self.w2_weight_t)?;
        Ok(hidden.add(&ffn_out)?)
    }
}

#[derive(Debug, Clone)]
struct WaveNetLayer {
    in_weight: Array,
    in_bias: Array,
    res_skip_weight: Array,
    res_skip_bias: Array,
    is_last: bool,
}

#[derive(Debug, Clone)]
struct WaveNet {
    cond_weight: Array,
    cond_bias: Array,
    layers: Vec<WaveNetLayer>,
}

impl WaveNet {
    fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let cond_weight = weights.tensor_array_shaped(
            "wavenet.cond_layer.weight",
            &[CFM_DIM * 2 * CFM_WN_LAYERS, 1, CFM_DIM],
        )?;
        let cond_bias = weights
            .tensor_array_shaped("wavenet.cond_layer.bias", &[CFM_DIM * 2 * CFM_WN_LAYERS])?;

        let mut layers = Vec::with_capacity(CFM_WN_LAYERS);
        for i in 0..CFM_WN_LAYERS {
            let is_last = i + 1 == CFM_WN_LAYERS;
            let in_weight = weights.tensor_array_shaped(
                &format!("wavenet.in_layers.{i}.weight"),
                &[CFM_DIM * 2, CFM_WN_KERNEL, CFM_DIM],
            )?;
            let in_bias = weights
                .tensor_array_shaped(&format!("wavenet.in_layers.{i}.bias"), &[CFM_DIM * 2])?;
            let rs_out_ch = if is_last { CFM_DIM } else { CFM_DIM * 2 };
            let res_skip_weight = weights.tensor_array_shaped(
                &format!("wavenet.res_skip_layers.{i}.weight"),
                &[rs_out_ch, 1, CFM_DIM],
            )?;
            let res_skip_bias = weights
                .tensor_array_shaped(&format!("wavenet.res_skip_layers.{i}.bias"), &[rs_out_ch])?;
            layers.push(WaveNetLayer {
                in_weight,
                in_bias,
                res_skip_weight,
                res_skip_bias,
                is_last,
            });
        }

        Ok(Self {
            cond_weight,
            cond_bias,
            layers,
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn forward(&self, x: &Array, c_expanded: &Array) -> Result<Array> {
        let g_all = conv1d(c_expanded, &self.cond_weight, 1, 0, 1, 1)?.add(&self.cond_bias)?;
        let g_parts = split_equal(&g_all, CFM_WN_LAYERS as i32, -1)?;
        let mut output = zeros::<f32>(x.shape())?;
        let mut h = x.clone();

        for (layer, g) in self.layers.iter().zip(g_parts.iter()) {
            let acts = conv1d(&h, &layer.in_weight, 1, 2, 1, 1)?
                .add(&layer.in_bias)?
                .add(g)?;
            let ab = split_equal(&acts, 2, -1)?;
            let (Some(a), Some(b)) = (ab.first(), ab.get(1)) else {
                return Err(IndexTtsError::mlx(
                    "failed to split WaveNet gated activations",
                ));
            };
            let act = tanh(a)?.multiply(&sigmoid(b)?)?;
            let rs = conv1d(&act, &layer.res_skip_weight, 1, 0, 1, 1)?.add(&layer.res_skip_bias)?;

            if layer.is_last {
                output = output.add(&rs)?;
            } else {
                let rs_parts = split_equal(&rs, 2, -1)?;
                let (Some(res_part), Some(skip_part)) = (rs_parts.first(), rs_parts.get(1)) else {
                    return Err(IndexTtsError::mlx(
                        "failed to split WaveNet res/skip tensor",
                    ));
                };
                h = h.add(res_part)?;
                output = output.add(skip_part)?;
            }
        }
        Ok(output)
    }
}

/// Conditional Flow Matching `DiT` estimator (`cfm_estimator.safetensors`).
#[derive(Debug, Clone)]
pub struct CfmDit {
    cond_proj_weight: Array,
    cond_proj_bias: Array,
    merge_weight: Array,
    merge_bias: Array,
    t_embedder: TimestepEmbedder,
    t_embedder2: TimestepEmbedder,
    blocks: Vec<TransformerBlock>,
    transformer_norm: AdaLn,
    skip_linear_weight: Array,
    skip_linear_bias: Array,
    conv1_weight: Array,
    conv1_bias: Array,
    res_proj_weight: Array,
    res_proj_bias: Array,
    wavenet: WaveNet,
    final_ada_weight: Array,
    final_ada_bias: Array,
    final_linear_weight: Array,
    final_linear_bias: Array,
    conv2_weight: Array,
    conv2_bias: Array,
}

impl CfmDit {
    /// Constructs a [`CfmDit`] from pre-filtered weights.
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
    ///     CfmDit, ModelWeightKind, SafeTensorsLoader, resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let weights = SafeTensorsLoader::load_from_path(
    ///         ModelWeightKind::CfmEstimator,
    ///         dir.join("cfm_estimator.safetensors"),
    ///     )?;
    ///     let dit = CfmDit::from_weights(&weights)?;
    ///     assert_eq!(dit.num_layers(), 13);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let cond_proj_weight =
            weights.tensor_array_shaped("cond_projection.weight", &[CFM_DIM, CFM_DIM])?;
        let cond_proj_bias = weights.tensor_array_shaped("cond_projection.bias", &[CFM_DIM])?;
        let merge_weight = weights
            .tensor_array_shaped("cond_x_merge_linear.weight", &[CFM_DIM, CFM_MERGED_IN_DIM])?;
        let merge_bias = weights.tensor_array_shaped("cond_x_merge_linear.bias", &[CFM_DIM])?;

        let t_embedder = TimestepEmbedder::from_weights(weights, "t_embedder")?;
        let t_embedder2 = TimestepEmbedder::from_weights(weights, "t_embedder2")?;

        let mut blocks = Vec::with_capacity(CFM_NUM_LAYERS);
        for i in 0..CFM_NUM_LAYERS {
            blocks.push(TransformerBlock::from_weights(weights, i)?);
        }
        let transformer_norm = AdaLn::from_weights(weights, "transformer.norm")?;

        let skip_linear_weight = weights
            .tensor_array_shaped("skip_linear.weight", &[CFM_DIM, CFM_DIM + CFM_IN_CHANNELS])?;
        let skip_linear_bias = weights.tensor_array_shaped("skip_linear.bias", &[CFM_DIM])?;
        let conv1_weight = weights.tensor_array_shaped("conv1.weight", &[CFM_DIM, CFM_DIM])?;
        let conv1_bias = weights.tensor_array_shaped("conv1.bias", &[CFM_DIM])?;
        let res_proj_weight =
            weights.tensor_array_shaped("res_projection.weight", &[CFM_DIM, CFM_DIM])?;
        let res_proj_bias = weights.tensor_array_shaped("res_projection.bias", &[CFM_DIM])?;

        let wavenet = WaveNet::from_weights(weights)?;

        let final_ada_weight = weights.tensor_array_shaped(
            "final_layer.adaLN_modulation.layers.1.weight",
            &[CFM_DIM * 2, CFM_DIM],
        )?;
        let final_ada_bias = weights
            .tensor_array_shaped("final_layer.adaLN_modulation.layers.1.bias", &[CFM_DIM * 2])?;
        let final_linear_weight =
            weights.tensor_array_shaped("final_layer.linear.weight", &[CFM_DIM, CFM_DIM])?;
        let final_linear_bias =
            weights.tensor_array_shaped("final_layer.linear.bias", &[CFM_DIM])?;

        let conv2_weight =
            weights.tensor_array_shaped("conv2.weight", &[CFM_IN_CHANNELS, 1, CFM_DIM])?;
        let conv2_bias = weights.tensor_array_shaped("conv2.bias", &[CFM_IN_CHANNELS])?;

        Ok(Self {
            cond_proj_weight,
            cond_proj_bias,
            merge_weight,
            merge_bias,
            t_embedder,
            t_embedder2,
            blocks,
            transformer_norm,
            skip_linear_weight,
            skip_linear_bias,
            conv1_weight,
            conv1_bias,
            res_proj_weight,
            res_proj_bias,
            wavenet,
            final_ada_weight,
            final_ada_bias,
            final_linear_weight,
            final_linear_bias,
            conv2_weight,
            conv2_bias,
        })
    }

    /// Loads [`CfmDit`] weights from `cfm_estimator.safetensors` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if the file cannot be read or tensors fail validation.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let weights = SafeTensorsLoader::load_from_path(ModelWeightKind::CfmEstimator, path)?;
        Self::from_weights(&weights)
    }

    /// Returns the number of U-ViT Transformer layers (`13`).
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Evaluates a single `DiT` step on `[B, T, 80]` `x_t` and `[B, T, 864]` `x_merged`.
    fn forward_merged(&self, x_t: &Array, x_merged: &Array, t: &Array) -> Result<Array> {
        let mut h = linear_bias(x_merged, &self.merge_weight, &self.merge_bias)?;
        let t1 = self.t_embedder.forward(t)?;
        let batch = h.dim(0);
        let dim = h.dim(2);
        let t1_expanded = t1.reshape(&[batch, 1, dim])?;

        let mut skips: Vec<Array> = Vec::with_capacity(SKIP_SPLIT_LAYER);
        for (i, block) in self.blocks.iter().enumerate() {
            let skip_in = if i > SKIP_SPLIT_LAYER {
                skips.pop()
            } else {
                None
            };
            h = block.forward(&h, &t1_expanded, skip_in.as_ref())?;
            if i < SKIP_SPLIT_LAYER {
                skips.push(h.clone());
            }
        }
        h = self.transformer_norm.forward(&h, &t1_expanded)?;

        let hx = concatenate(&[&h, x_t], -1)?;
        let x_res = linear_bias(&hx, &self.skip_linear_weight, &self.skip_linear_bias)?;
        h = linear_bias(&x_res, &self.conv1_weight, &self.conv1_bias)?;

        let t2 = self.t_embedder2.forward(t)?;
        let t2_expanded = t2.reshape(&[batch, 1, dim])?;
        let wn_out = self.wavenet.forward(&h, &t2_expanded)?;
        let res_proj = linear_bias(&x_res, &self.res_proj_weight, &self.res_proj_bias)?;
        h = wn_out.add(&res_proj)?;

        // FinalLayer(h, t1)
        let mod_wb = linear_bias(&silu(&t1)?, &self.final_ada_weight, &self.final_ada_bias)?;
        let parts = split_equal(&mod_wb, 2, -1)?;
        let (Some(shift), Some(scale)) = (parts.first(), parts.get(1)) else {
            return Err(IndexTtsError::mlx("failed to split FinalLayer modulation"));
        };
        let shift = shift.reshape(&[batch, 1, dim])?;
        let scale = scale.reshape(&[batch, 1, dim])?;
        let normed = layer_norm(&h, None, None, FINAL_NORM_EPS)?;
        let mod_h = normed
            .multiply(&array!(1.0_f32).add(&scale)?)?
            .add(&shift)?;
        let final_h = linear_bias(&mod_h, &self.final_linear_weight, &self.final_linear_bias)?;

        let out = conv1d(&final_h, &self.conv2_weight, 1, 0, 1, 1)?.add(&self.conv2_bias)?;
        Ok(out.transpose_axes(&[0, 2, 1])?)
    }

    /// Runs the 25-step all-GPU Euler ODE solver with Classifier-Free Guidance (CFG).
    ///
    /// Static conditioning tensors (`prompt_x`, `cond_projection(cond)`, `style`, `RoPE` frequency
    /// table, and the prompt-region zero mask) are precomputed once on GPU outside the 25-step
    /// Euler loop.
    ///
    /// # Arguments
    ///
    /// * `cat_condition` - Regulated prompt + target semantic conditioning `[1, T_total, 512]`.
    /// * `ref_mel` - Reference speaker Mel spectrogram `[1, 80, T_ref]`.
    /// * `style` - Reference speaker `CAMPPlus` style vector `[1, 192]`.
    /// * `noise_z` - Initial standard normal noise tensor `[1, 80, T_total]`.
    /// * `n_timesteps` - Number of Euler integration steps (typically `25`).
    /// * `cfg_rate` - Classifier-Free Guidance rate (typically `0.7`).
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if input tensor shapes are incompatible or
    /// `n_timesteps == 0`, or [`IndexTtsError::Mlx`] if GPU execution fails.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines
    )]
    pub fn solve_euler(
        &self,
        cat_condition: &Array,
        ref_mel: &Array,
        style: &Array,
        noise_z: &Array,
        n_timesteps: usize,
        cfg_rate: f32,
    ) -> Result<Array> {
        if n_timesteps == 0 {
            return Err(IndexTtsError::validation(
                "n_timesteps",
                "n_timesteps must be >= 1",
            ));
        }
        if cat_condition.ndim() != 3 || cat_condition.dim(0) != 1 {
            return Err(IndexTtsError::validation(
                "cat_condition",
                format!(
                    "expected [1, T_total, 512], got shape {:?}",
                    cat_condition.shape()
                ),
            ));
        }
        if ref_mel.ndim() != 3 || ref_mel.dim(0) != 1 || ref_mel.dim(1) != CFM_IN_CHANNELS as i32 {
            return Err(IndexTtsError::validation(
                "ref_mel",
                format!("expected [1, 80, T_ref], got shape {:?}", ref_mel.shape()),
            ));
        }
        if style.ndim() != 2 || style.dim(0) != 1 || style.dim(1) != CFM_STYLE_DIM as i32 {
            return Err(IndexTtsError::validation(
                "style",
                format!("expected [1, 192], got shape {:?}", style.shape()),
            ));
        }

        let t_total_i32 = cat_condition.dim(1);
        let prompt_len_i32 = ref_mel.dim(2);
        if prompt_len_i32 >= t_total_i32 {
            return Err(IndexTtsError::validation(
                "ref_mel",
                format!(
                    "ref_mel length ({prompt_len_i32}) must be strictly less than cat_condition \
                     length ({t_total_i32})"
                ),
            ));
        }
        if noise_z.shape() != [1, CFM_IN_CHANNELS as i32, t_total_i32] {
            return Err(IndexTtsError::validation(
                "noise_z",
                format!(
                    "expected shape [1, 80, {t_total_i32}], got {:?}",
                    noise_z.shape()
                ),
            ));
        }

        let t_total = usize::try_from(t_total_i32).unwrap_or(0);
        let prompt_len = usize::try_from(prompt_len_i32).unwrap_or(0);
        let gen_len_i32 = t_total_i32 - prompt_len_i32;

        // 1. Precompute static CFG tail [2, T_total, 80 + 512 + 192 = 784]
        let zeros_gen_mel = zeros::<f32>(&[1, CFM_IN_CHANNELS as i32, gen_len_i32])?;
        let prompt_x_pos = concatenate(&[ref_mel, &zeros_gen_mel], 2)?;
        let prompt_x_neg = zeros::<f32>(&[1, CFM_IN_CHANNELS as i32, t_total_i32])?;
        let prompt_x_cfg_t =
            concatenate(&[&prompt_x_pos, &prompt_x_neg], 0)?.transpose_axes(&[0, 2, 1])?;

        let zeros_cond = zeros::<f32>(&[1, t_total_i32, CFM_DIM as i32])?;
        let cond_cfg = concatenate(&[cat_condition, &zeros_cond], 0)?;
        let cond_proj_cfg = linear_bias(&cond_cfg, &self.cond_proj_weight, &self.cond_proj_bias)?;

        let zeros_style = zeros::<f32>(&[1, CFM_STYLE_DIM as i32])?;
        let style_cfg = concatenate(&[style, &zeros_style], 0)?;
        let style_cfg_bcast = broadcast_to(
            &style_cfg.reshape(&[2, 1, CFM_STYLE_DIM as i32])?,
            &[2, t_total_i32, CFM_STYLE_DIM as i32],
        )?;

        let static_tail_cfg =
            concatenate(&[&prompt_x_cfg_t, &cond_proj_cfg, &style_cfg_bcast], -1)?;
        static_tail_cfg.eval()?;

        // 2. Precompute prompt-region zero mask [1, 1, T_total]
        let mut mask_vals = vec![1.0_f32; t_total];
        for slot in mask_vals.iter_mut().take(prompt_len) {
            *slot = 0.0_f32;
        }
        let prompt_mask = Array::from_slice(&mask_vals, &[1, 1, t_total_i32]);

        // 3. 25-step all-GPU Euler integration
        let dt_val = 1.0_f32 / (n_timesteps as f32);
        let dt_arr = array!(dt_val);
        let pos_weight = array!(1.0_f32 + cfg_rate);
        let neg_weight = array!(cfg_rate);

        let mut x = noise_z.clone();
        for step in 0..n_timesteps {
            let t_val = (step as f32) * dt_val;
            let t_arr = Array::from_slice(&[t_val, t_val], &[2]);

            x = x.multiply(&prompt_mask)?;
            let x_cfg_t = concatenate(&[&x, &x], 0)?.transpose_axes(&[0, 2, 1])?;
            let x_merged = concatenate(&[&x_cfg_t, &static_tail_cfg], -1)?;

            let dphi_dt = self.forward_merged(&x_cfg_t, &x_merged, &t_arr)?;
            let parts = split_equal(&dphi_dt, 2, 0)?;
            let (Some(pos), Some(neg)) = (parts.first(), parts.get(1)) else {
                return Err(IndexTtsError::mlx("failed to split CFG pos/neg velocity"));
            };

            let guided = if cfg_rate > 0.0 {
                pos.multiply(&pos_weight)?
                    .subtract(&neg.multiply(&neg_weight)?)?
            } else {
                pos.clone()
            };
            x = x.add(&guided.multiply(&dt_arr)?)?;
            if (step + 1) % 5 == 0 {
                x.eval()?;
            }
        }

        // 5. Slice generated frames [1, 80, prompt_len..T_total]
        let generated = x
            .try_index((.., .., prompt_len_i32..t_total_i32))?
            .contiguous()?;
        generated.eval()?;
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use safetensors::SafeTensors;

    use super::*;
    use crate::{
        config::DEFAULT_MODEL_REPO, dsp::cosine_similarity_and_max_diff, models::resolve_model_dir,
    };

    #[test]
    fn test_should_match_python_reference_cp13_cfm_euler_solver() {
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

        let dit_res = CfmDit::load_from_path(model_dir.join("cfm_estimator.safetensors"));
        assert!(dit_res.is_ok(), "CfmDit load error: {:?}", dit_res.err());
        let Ok(dit) = dit_res else {
            return;
        };

        let cat_condition = extract_array(&st, "cat_condition");
        let ref_mel = extract_array(&st, "ref_mel");
        let style = extract_array(&st, "style");
        let cfm_noise_z = extract_array(&st, "cfm_noise_z");
        let (expected_shape, expected_mel) = extract_f32(&st, "generated_mel");

        let mel_res = dit.solve_euler(&cat_condition, &ref_mel, &style, &cfm_noise_z, 25, 0.7);
        assert!(mel_res.is_ok());
        let Ok(mel) = mel_res else {
            return;
        };

        let expected_shape_i32: Vec<i32> = expected_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        assert_eq!(mel.shape(), expected_shape_i32.as_slice());

        let rust_mel = mel.try_as_slice::<f32>().unwrap_or(&[]);
        let (cos, _max_diff) = cosine_similarity_and_max_diff(rust_mel, &expected_mel);
        let rel_rmse = compute_relative_rmse(rust_mel, &expected_mel);
        assert!(
            cos >= 0.999_5,
            "CP-13 generated_mel cosine similarity {cos} < 0.9995"
        );
        assert!(
            rel_rmse < 3e-3,
            "CP-13 generated_mel relative RMSE {rel_rmse} >= 3e-3"
        );
    }

    fn compute_relative_rmse(a: &[f32], b: &[f32]) -> f64 {
        let mut diff_sq = 0.0_f64;
        let mut ref_sq = 0.0_f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = f64::from(x) - f64::from(y);
            diff_sq += d * d;
            ref_sq += f64::from(y) * f64::from(y);
        }
        (diff_sq / ref_sq.max(1e-12)).sqrt()
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

    fn extract_array(st: &SafeTensors<'_>, name: &str) -> Array {
        let (shape, data) = extract_f32(st, name);
        let shape_i32: Vec<i32> = shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        Array::from_slice(&data, &shape_i32)
    }
}

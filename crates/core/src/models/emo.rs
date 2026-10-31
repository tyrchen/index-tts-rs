//! Stage 0 Emotion Vector Extractor (`EmoVec`, `raw/emo.safetensors`).
//!
//! Takes `w2v-bert-2.0` features `[B, T, 1024]` and sequence lengths `ilens`, runs a
//! `Conv2dSubsampling2` front-end + 4-layer `ConformerEncoder` (with `pos_bias_u`/`pos_bias_v`
//! relative positional attention) + 2-layer `PerceiverResampler` (`GEGLU` + `PercRMSNorm`) +
//! `emovec_layer` + `emo_layer` to produce `[B, 1280]` emotion conditioning vectors.

use std::path::Path;

use mlx_rs::{
    Array,
    fast::{ScaledDotProductAttentionMask, layer_norm, scaled_dot_product_attention},
    nn::{gelu, relu, silu},
    ops::{
        addmm, broadcast_to, concatenate, conv1d, conv2d, indexing::IndexOp, maximum, sigmoid,
        softmax_axis,
    },
};

use crate::{
    error::{IndexTtsError, Result},
    models::{FilteredWeights, ModelWeightKind, SafeTensorsLoader},
};

/// Input feature dimension from `SemanticModel` (`1024`).
pub const EMO_ENC_IN_DIM: usize = 1_024;
/// Conformer encoder hidden dimension (`512`).
pub const EMO_ENC_DIM: usize = 512;
/// Number of Conformer attention heads (`4`).
pub const EMO_ENC_HEADS: usize = 4;
/// Number of Conformer encoder blocks (`4`).
pub const EMO_ENC_BLOCKS: usize = 4;
/// Conformer feed-forward intermediate units (`1024`).
pub const EMO_ENC_FF_UNITS: usize = 1_024;
/// Conformer attention head dimension (`128`).
pub const EMO_ENC_HEAD_DIM: usize = EMO_ENC_DIM / EMO_ENC_HEADS;
/// Conformer depthwise convolution kernel size (`15`).
pub const EMO_CONV_KERNEL: usize = 15;
/// Conformer depthwise convolution symmetric padding (`7`).
pub const EMO_CONV_PADDING: i32 = 7;
/// Flattened channel-frequency dimension after `Conv2dSubsampling2` (`512 * 511 = 261632`).
pub const EMO_SUBSAMPLE_FLATTENED: usize = EMO_ENC_DIM * ((EMO_ENC_IN_DIM - 1) / 2);
/// Perceiver resampler hidden dimension (`1024`).
pub const EMO_PERC_DIM: usize = 1_024;
/// Perceiver cross-attention heads (`4`).
pub const EMO_PERC_HEADS: usize = 4;
/// Perceiver cross-attention head dimension (`64`).
pub const EMO_PERC_HEAD_DIM: usize = 64;
/// Perceiver inner attention dimension (`256`).
pub const EMO_PERC_INNER: usize = EMO_PERC_HEADS * EMO_PERC_HEAD_DIM;
/// Perceiver resampler depth (`2`).
pub const EMO_PERC_DEPTH: usize = 2;
/// Perceiver `GEGLU` feed-forward inner dimension (`1365`).
pub const EMO_PERC_FF_INNER: usize = (EMO_PERC_DIM * 2 * 2) / 3;
/// Output emotion vector dimension (`1280`).
pub const EMO_MODEL_DIM: usize = 1_280;
/// `LayerNorm` epsilon (`1e-5`).
pub const EMO_NORM_EPS: f32 = 1e-5;
/// Additive mask penalty for padded positions (`-1e30`).
const MASK_NEG: f32 = -1e30;
/// Perceiver attention scale (`64^-0.5 = 0.125`).
const PERC_ATTN_SCALE: f32 = 0.125;

#[derive(Debug, Clone)]
struct RelPosAttention {
    q_weight: Array,
    q_bias: Array,
    k_weight: Array,
    k_bias: Array,
    v_weight: Array,
    v_bias: Array,
    out_weight: Array,
    out_bias: Array,
    pos_weight: Array,
    pos_bias_u: Array,
    pos_bias_v: Array,
}

impl RelPosAttention {
    fn from_weights(weights: &FilteredWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            q_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear_q.weight"),
                &[EMO_ENC_DIM, EMO_ENC_DIM],
            )?,
            q_bias: weights
                .tensor_array_shaped(&format!("{prefix}.linear_q.bias"), &[EMO_ENC_DIM])?,
            k_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear_k.weight"),
                &[EMO_ENC_DIM, EMO_ENC_DIM],
            )?,
            k_bias: weights
                .tensor_array_shaped(&format!("{prefix}.linear_k.bias"), &[EMO_ENC_DIM])?,
            v_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear_v.weight"),
                &[EMO_ENC_DIM, EMO_ENC_DIM],
            )?,
            v_bias: weights
                .tensor_array_shaped(&format!("{prefix}.linear_v.bias"), &[EMO_ENC_DIM])?,
            out_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear_out.weight"),
                &[EMO_ENC_DIM, EMO_ENC_DIM],
            )?,
            out_bias: weights
                .tensor_array_shaped(&format!("{prefix}.linear_out.bias"), &[EMO_ENC_DIM])?,
            pos_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear_pos.weight"),
                &[EMO_ENC_DIM, EMO_ENC_DIM],
            )?,
            pos_bias_u: weights.tensor_array_shaped(
                &format!("{prefix}.pos_bias_u"),
                &[EMO_ENC_HEADS, EMO_ENC_HEAD_DIM],
            )?,
            pos_bias_v: weights.tensor_array_shaped(
                &format!("{prefix}.pos_bias_v"),
                &[EMO_ENC_HEADS, EMO_ENC_HEAD_DIM],
            )?,
        })
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::many_single_char_names
    )]
    fn forward(&self, x: &Array, pos_emb: &Array, amask: &Array) -> Result<Array> {
        let b = x.dim(0);
        let t = x.dim(1);
        let heads = EMO_ENC_HEADS as i32;
        let head_dim = EMO_ENC_HEAD_DIM as i32;
        let dim = EMO_ENC_DIM as i32;

        let q = addmm(&self.q_bias, x, self.q_weight.t(), None, None)?
            .reshape(&[b, t, heads, head_dim])?;
        let k = addmm(&self.k_bias, x, self.k_weight.t(), None, None)?
            .reshape(&[b, t, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = addmm(&self.v_bias, x, self.v_weight.t(), None, None)?
            .reshape(&[b, t, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let p = pos_emb
            .matmul(self.pos_weight.t())?
            .reshape(&[1, t, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q_u = q.add(&self.pos_bias_u)?.transpose_axes(&[0, 2, 1, 3])?;
        let q_v = q.add(&self.pos_bias_v)?.transpose_axes(&[0, 2, 1, 3])?;

        let matrix_ac = q_u.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
        let matrix_bd = q_v.matmul(&p.transpose_axes(&[0, 1, 3, 2])?)?;

        let scale = 1.0_f32 / (EMO_ENC_HEAD_DIM as f32).sqrt();
        let scores = matrix_ac
            .add(&matrix_bd)?
            .multiply(Array::from_f32(scale))?
            .add(amask)?;
        let attn = softmax_axis(&scores, -1, None)?;
        let o = attn
            .matmul(&v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, t, dim])?;

        Ok(addmm(&self.out_bias, &o, self.out_weight.t(), None, None)?)
    }
}

#[derive(Debug, Clone)]
struct PositionwiseFeedForward {
    w1_weight: Array,
    w1_bias: Array,
    w2_weight: Array,
    w2_bias: Array,
}

impl PositionwiseFeedForward {
    fn from_weights(weights: &FilteredWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1_weight: weights.tensor_array_shaped(
                &format!("{prefix}.w_1.weight"),
                &[EMO_ENC_FF_UNITS, EMO_ENC_DIM],
            )?,
            w1_bias: weights
                .tensor_array_shaped(&format!("{prefix}.w_1.bias"), &[EMO_ENC_FF_UNITS])?,
            w2_weight: weights.tensor_array_shaped(
                &format!("{prefix}.w_2.weight"),
                &[EMO_ENC_DIM, EMO_ENC_FF_UNITS],
            )?,
            w2_bias: weights.tensor_array_shaped(&format!("{prefix}.w_2.bias"), &[EMO_ENC_DIM])?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = addmm(&self.w1_bias, x, self.w1_weight.t(), None, None)?;
        let h = silu(&h)?;
        Ok(addmm(&self.w2_bias, &h, self.w2_weight.t(), None, None)?)
    }
}

#[derive(Debug, Clone)]
struct EmoConvolutionModule {
    pw1_weight: Array,
    pw1_bias: Array,
    dw_weight: Array,
    dw_bias: Array,
    norm_weight: Array,
    norm_bias: Array,
    pw2_weight: Array,
    pw2_bias: Array,
}

impl EmoConvolutionModule {
    fn from_weights(weights: &FilteredWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            pw1_weight: weights.tensor_array_shaped(
                &format!("{prefix}.pointwise_conv1.weight"),
                &[2 * EMO_ENC_DIM, 1, EMO_ENC_DIM],
            )?,
            pw1_bias: weights.tensor_array_shaped(
                &format!("{prefix}.pointwise_conv1.bias"),
                &[2 * EMO_ENC_DIM],
            )?,
            dw_weight: weights.tensor_array_shaped(
                &format!("{prefix}.depthwise_conv.weight"),
                &[EMO_ENC_DIM, EMO_CONV_KERNEL, 1],
            )?,
            dw_bias: weights
                .tensor_array_shaped(&format!("{prefix}.depthwise_conv.bias"), &[EMO_ENC_DIM])?,
            norm_weight: weights
                .tensor_array_shaped(&format!("{prefix}.norm.weight"), &[EMO_ENC_DIM])?,
            norm_bias: weights
                .tensor_array_shaped(&format!("{prefix}.norm.bias"), &[EMO_ENC_DIM])?,
            pw2_weight: weights.tensor_array_shaped(
                &format!("{prefix}.pointwise_conv2.weight"),
                &[EMO_ENC_DIM, 1, EMO_ENC_DIM],
            )?,
            pw2_bias: weights
                .tensor_array_shaped(&format!("{prefix}.pointwise_conv2.bias"), &[EMO_ENC_DIM])?,
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn forward(&self, x: &Array, vmask: &Array) -> Result<Array> {
        let h = x.multiply(vmask)?;
        let h = conv1d(&h, &self.pw1_weight, 1, 0, 1, 1)?.add(&self.pw1_bias)?;
        let parts = h.split_equal(2, -1)?;
        let (Some(a), Some(b)) = (parts.first(), parts.get(1)) else {
            return Err(IndexTtsError::mlx(
                "GLU split failed in EmoConvolutionModule",
            ));
        };
        let h = a.multiply(sigmoid(b)?)?;
        let h = conv1d(
            &h,
            &self.dw_weight,
            1,
            EMO_CONV_PADDING,
            1,
            EMO_ENC_DIM as i32,
        )?
        .add(&self.dw_bias)?;
        let h = layer_norm(
            &h,
            Some(&self.norm_weight),
            Some(&self.norm_bias),
            EMO_NORM_EPS,
        )?;
        let h = silu(&h)?;
        let h = conv1d(&h, &self.pw2_weight, 1, 0, 1, 1)?.add(&self.pw2_bias)?;
        Ok(h.multiply(vmask)?)
    }
}

#[derive(Debug, Clone)]
struct ConformerEncoderLayer {
    self_attn: RelPosAttention,
    feed_forward: PositionwiseFeedForward,
    conv_module: EmoConvolutionModule,
    norm_ff_weight: Array,
    norm_ff_bias: Array,
    norm_mha_weight: Array,
    norm_mha_bias: Array,
    norm_conv_weight: Array,
    norm_conv_bias: Array,
    norm_final_weight: Array,
    norm_final_bias: Array,
}

impl ConformerEncoderLayer {
    fn from_weights(weights: &FilteredWeights, idx: usize) -> Result<Self> {
        let prefix = format!("enc.encoders.{idx}");
        Ok(Self {
            self_attn: RelPosAttention::from_weights(weights, &format!("{prefix}.self_attn"))?,
            feed_forward: PositionwiseFeedForward::from_weights(
                weights,
                &format!("{prefix}.feed_forward"),
            )?,
            conv_module: EmoConvolutionModule::from_weights(
                weights,
                &format!("{prefix}.conv_module"),
            )?,
            norm_ff_weight: weights
                .tensor_array_shaped(&format!("{prefix}.norm_ff.weight"), &[EMO_ENC_DIM])?,
            norm_ff_bias: weights
                .tensor_array_shaped(&format!("{prefix}.norm_ff.bias"), &[EMO_ENC_DIM])?,
            norm_mha_weight: weights
                .tensor_array_shaped(&format!("{prefix}.norm_mha.weight"), &[EMO_ENC_DIM])?,
            norm_mha_bias: weights
                .tensor_array_shaped(&format!("{prefix}.norm_mha.bias"), &[EMO_ENC_DIM])?,
            norm_conv_weight: weights
                .tensor_array_shaped(&format!("{prefix}.norm_conv.weight"), &[EMO_ENC_DIM])?,
            norm_conv_bias: weights
                .tensor_array_shaped(&format!("{prefix}.norm_conv.bias"), &[EMO_ENC_DIM])?,
            norm_final_weight: weights
                .tensor_array_shaped(&format!("{prefix}.norm_final.weight"), &[EMO_ENC_DIM])?,
            norm_final_bias: weights
                .tensor_array_shaped(&format!("{prefix}.norm_final.bias"), &[EMO_ENC_DIM])?,
        })
    }

    fn forward(&self, x: &Array, pos_emb: &Array, amask: &Array, vmask: &Array) -> Result<Array> {
        let mha_in = layer_norm(
            x,
            Some(&self.norm_mha_weight),
            Some(&self.norm_mha_bias),
            EMO_NORM_EPS,
        )?;
        let x = x.add(&self.self_attn.forward(&mha_in, pos_emb, amask)?)?;

        let conv_in = layer_norm(
            &x,
            Some(&self.norm_conv_weight),
            Some(&self.norm_conv_bias),
            EMO_NORM_EPS,
        )?;
        let x = x.add(&self.conv_module.forward(&conv_in, vmask)?)?;

        let ff_in = layer_norm(
            &x,
            Some(&self.norm_ff_weight),
            Some(&self.norm_ff_bias),
            EMO_NORM_EPS,
        )?;
        let x = x.add(&self.feed_forward.forward(&ff_in)?)?;

        Ok(layer_norm(
            &x,
            Some(&self.norm_final_weight),
            Some(&self.norm_final_bias),
            EMO_NORM_EPS,
        )?)
    }
}

#[derive(Debug, Clone)]
struct PercLayer {
    to_q_weight: Array,
    to_kv_weight: Array,
    to_out_weight: Array,
    ff0_weight: Array,
    ff0_bias: Array,
    ff2_weight: Array,
    ff2_bias: Array,
}

impl PercLayer {
    fn from_weights(weights: &FilteredWeights, idx: usize) -> Result<Self> {
        let prefix = format!("perc.layers.{idx}");
        Ok(Self {
            to_q_weight: weights.tensor_array_shaped(
                &format!("{prefix}.0.to_q.weight"),
                &[EMO_PERC_INNER, EMO_PERC_DIM],
            )?,
            to_kv_weight: weights.tensor_array_shaped(
                &format!("{prefix}.0.to_kv.weight"),
                &[2 * EMO_PERC_INNER, EMO_PERC_DIM],
            )?,
            to_out_weight: weights.tensor_array_shaped(
                &format!("{prefix}.0.to_out.weight"),
                &[EMO_PERC_DIM, EMO_PERC_INNER],
            )?,
            ff0_weight: weights.tensor_array_shaped(
                &format!("{prefix}.1.0.weight"),
                &[2 * EMO_PERC_FF_INNER, EMO_PERC_DIM],
            )?,
            ff0_bias: weights
                .tensor_array_shaped(&format!("{prefix}.1.0.bias"), &[2 * EMO_PERC_FF_INNER])?,
            ff2_weight: weights.tensor_array_shaped(
                &format!("{prefix}.1.2.weight"),
                &[EMO_PERC_DIM, EMO_PERC_FF_INNER],
            )?,
            ff2_bias: weights
                .tensor_array_shaped(&format!("{prefix}.1.2.bias"), &[EMO_PERC_DIM])?,
        })
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::many_single_char_names
    )]
    fn forward(&self, latents: &Array, context: &Array, amask: &Array) -> Result<Array> {
        let b = latents.dim(0);
        let n = latents.dim(1);
        let ctx = concatenate(&[latents, context], 1)?;
        let s = ctx.dim(1);
        let heads = EMO_PERC_HEADS as i32;
        let head_dim = EMO_PERC_HEAD_DIM as i32;
        let inner = EMO_PERC_INNER as i32;

        let q = latents
            .matmul(self.to_q_weight.t())?
            .reshape(&[b, n, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let kv = ctx.matmul(self.to_kv_weight.t())?;
        let kv_parts = kv.split_equal(2, -1)?;
        let (Some(k_raw), Some(v_raw)) = (kv_parts.first(), kv_parts.get(1)) else {
            return Err(IndexTtsError::mlx("Perceiver KV split failed"));
        };
        let k = k_raw
            .reshape(&[b, s, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v_raw
            .reshape(&[b, s, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let attn_out = scaled_dot_product_attention(
            q,
            k,
            v,
            PERC_ATTN_SCALE,
            Some(ScaledDotProductAttentionMask::Array(amask)),
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, n, inner])?
        .matmul(self.to_out_weight.t())?;

        let latents = attn_out.add(latents)?;

        let ff_hidden = addmm(&self.ff0_bias, &latents, self.ff0_weight.t(), None, None)?;
        let ff_parts = ff_hidden.split_equal(2, -1)?;
        let (Some(a), Some(gate)) = (ff_parts.first(), ff_parts.get(1)) else {
            return Err(IndexTtsError::mlx("GEGLU split failed in Perceiver"));
        };
        let geglu = gelu(gate)?.multiply(a)?;
        let ff_out = addmm(&self.ff2_bias, &geglu, self.ff2_weight.t(), None, None)?;

        Ok(ff_out.add(&latents)?)
    }
}

/// Stage 0 Emotion Vector Extractor (`raw/emo.safetensors`).
#[derive(Debug, Clone)]
pub struct EmoVec {
    embed_conv_weight: Array,
    embed_conv_bias: Array,
    embed_out_weight: Array,
    embed_out_bias: Array,
    pe: Array,
    encoders: Vec<ConformerEncoderLayer>,
    after_norm_weight: Array,
    after_norm_bias: Array,
    proj_context_weight: Array,
    proj_context_bias: Array,
    latents: Array,
    perc_layers: Vec<PercLayer>,
    perc_norm_gamma: Array,
    emovec_weight: Array,
    emovec_bias: Array,
    emo_weight: Array,
    emo_bias: Array,
}

impl EmoVec {
    /// Constructs an [`EmoVec`] model from pre-filtered weights.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if any required tensor or positional buffer is
    /// missing or has an unexpected shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::{
    ///     EmoVec, ModelWeightKind, SafeTensorsLoader, resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let weights = SafeTensorsLoader::load_from_path(
    ///         ModelWeightKind::EmoVec,
    ///         dir.join("raw/emo.safetensors"),
    ///     )?;
    ///     let emo = EmoVec::from_weights(&weights)?;
    ///     assert_eq!(emo.num_encoder_blocks(), 4);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let embed_conv_weight =
            weights.tensor_array_shaped("enc.embed_conv.weight", &[EMO_ENC_DIM, 3, 3, 1])?;
        let embed_conv_bias = weights.tensor_array_shaped("enc.embed_conv.bias", &[EMO_ENC_DIM])?;
        let embed_out_weight = weights.tensor_array_shaped(
            "enc.embed_out.weight",
            &[EMO_ENC_DIM, EMO_SUBSAMPLE_FLATTENED],
        )?;
        let embed_out_bias = weights.tensor_array_shaped("enc.embed_out.bias", &[EMO_ENC_DIM])?;
        let pe = weights.buffer_array("enc.embed.pos_enc.pe")?;

        let mut encoders = Vec::with_capacity(EMO_ENC_BLOCKS);
        for idx in 0..EMO_ENC_BLOCKS {
            encoders.push(ConformerEncoderLayer::from_weights(weights, idx)?);
        }

        let after_norm_weight =
            weights.tensor_array_shaped("enc.after_norm.weight", &[EMO_ENC_DIM])?;
        let after_norm_bias = weights.tensor_array_shaped("enc.after_norm.bias", &[EMO_ENC_DIM])?;

        let proj_context_weight = weights
            .tensor_array_shaped("perc.proj_context.weight", &[EMO_PERC_DIM, EMO_ENC_DIM])?;
        let proj_context_bias =
            weights.tensor_array_shaped("perc.proj_context.bias", &[EMO_PERC_DIM])?;
        let latents = weights.tensor_array_shaped("perc.latents", &[1, EMO_PERC_DIM])?;

        let mut perc_layers = Vec::with_capacity(EMO_PERC_DEPTH);
        for idx in 0..EMO_PERC_DEPTH {
            perc_layers.push(PercLayer::from_weights(weights, idx)?);
        }

        let perc_norm_gamma = weights.tensor_array_shaped("perc.norm.gamma", &[EMO_PERC_DIM])?;
        let emovec_weight =
            weights.tensor_array_shaped("emovec_layer.weight", &[EMO_MODEL_DIM, EMO_PERC_DIM])?;
        let emovec_bias = weights.tensor_array_shaped("emovec_layer.bias", &[EMO_MODEL_DIM])?;
        let emo_weight =
            weights.tensor_array_shaped("emo_layer.weight", &[EMO_MODEL_DIM, EMO_MODEL_DIM])?;
        let emo_bias = weights.tensor_array_shaped("emo_layer.bias", &[EMO_MODEL_DIM])?;

        Ok(Self {
            embed_conv_weight,
            embed_conv_bias,
            embed_out_weight,
            embed_out_bias,
            pe,
            encoders,
            after_norm_weight,
            after_norm_bias,
            proj_context_weight,
            proj_context_bias,
            latents,
            perc_layers,
            perc_norm_gamma,
            emovec_weight,
            emovec_bias,
            emo_weight,
            emo_bias,
        })
    }

    /// Loads [`EmoVec`] weights directly from `raw/emo.safetensors` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if the file cannot be read or tensors fail validation.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let weights = SafeTensorsLoader::load_from_path(ModelWeightKind::EmoVec, path)?;
        Self::from_weights(&weights)
    }

    /// Returns the number of Conformer encoder blocks (`4`).
    #[must_use]
    pub fn num_encoder_blocks(&self) -> usize {
        self.encoders.len()
    }

    /// Extracts `[B, 1280]` emotion conditioning vectors from `feat` (`[B, T, 1024]`) and
    /// per-utterance frame lengths `ilens` (`[B]`).
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `feat` is not `[B, T, 1024]` with `T >= 3` or
    /// `ilens.len() != B`, or [`IndexTtsError::Mlx`] if GPU execution fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_rs::Array;
    /// use index_tts_rs_core::models::{EmoVec, resolve_model_dir};
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let emo = EmoVec::load_from_path(dir.join("raw/emo.safetensors"))?;
    ///     let feat = Array::zeros::<f32>(&[1, 16, 1024])?;
    ///     let out = emo.forward(&feat, &[16])?;
    ///     assert_eq!(out.shape(), &[1, 1280]);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        clippy::similar_names
    )]
    pub fn forward(&self, feat: &Array, ilens: &[usize]) -> Result<Array> {
        if feat.ndim() != 3 {
            return Err(IndexTtsError::validation(
                "feat",
                format!("expected 3D tensor [B, T, 1024], got {:?}", feat.shape()),
            ));
        }
        let b = feat.dim(0);
        let t = feat.dim(1);
        let f_dim = feat.dim(2);
        let b_usize = usize::try_from(b).unwrap_or(0);
        let t_usize = usize::try_from(t).unwrap_or(0);

        if b_usize == 0 || t_usize < 3 {
            return Err(IndexTtsError::validation(
                "feat",
                format!("batch must be >= 1 and sequence length T >= 3, got B={b}, T={t}"),
            ));
        }
        if usize::try_from(f_dim).unwrap_or(0) != EMO_ENC_IN_DIM {
            return Err(IndexTtsError::validation(
                "feat",
                format!("expected {EMO_ENC_IN_DIM} channels, got {f_dim}"),
            ));
        }
        if ilens.len() != b_usize {
            return Err(IndexTtsError::validation(
                "ilens",
                format!("expected {b_usize} lengths, got {}", ilens.len()),
            ));
        }

        // Subsampled length after Conv2d(k=3, s=2): valid[:, 2::2] has length (T - 1) / 2
        let t_sub_usize = (t_usize - 1) / 2;
        let t_sub = t_sub_usize as i32;

        // Build masks:
        // vmask: [B, T', 1] (1.0 for valid, 0.0 for pad)
        // amask: [B, 1, 1, T'] (0.0 for valid, -1e30 for pad)
        // amask_perc: [B, 1, 1, T' + 1] (prepend 0.0 for latent query)
        let mut vmask_vec = Vec::with_capacity(b_usize * t_sub_usize);
        let mut amask_vec = Vec::with_capacity(b_usize * t_sub_usize);
        let mut amask_perc_vec = Vec::with_capacity(b_usize * (t_sub_usize + 1));

        for &len in ilens {
            amask_perc_vec.push(0.0_f32);
            for s in 0..t_sub_usize {
                let orig_idx = 2 + 2 * s;
                if orig_idx < len {
                    vmask_vec.push(1.0_f32);
                    amask_vec.push(0.0_f32);
                    amask_perc_vec.push(0.0_f32);
                } else {
                    vmask_vec.push(0.0_f32);
                    amask_vec.push(MASK_NEG);
                    amask_perc_vec.push(MASK_NEG);
                }
            }
        }

        let vmask = Array::from_slice(&vmask_vec, &[b, t_sub, 1]);
        let amask = Array::from_slice(&amask_vec, &[b, 1, 1, t_sub]);
        let amask_perc = Array::from_slice(&amask_perc_vec, &[b, 1, 1, t_sub + 1]);

        // Conv2dSubsampling2: [B, T, F, 1] -> Conv2d(k=3, s=2) -> ReLU -> [B, T', F', C]
        let x_4d = feat.reshape(&[b, t, f_dim, 1])?;
        let conv_out = relu(
            &conv2d(&x_4d, &self.embed_conv_weight, (2, 2), (0, 0), (1, 1), 1)?
                .add(&self.embed_conv_bias)?,
        )?;
        // Transpose [B, T', F', C] -> [B, T', C, F'] and flatten (C, F') to 261632
        let flat = conv_out.transpose_axes(&[0, 1, 3, 2])?.reshape(&[
            b,
            t_sub,
            EMO_SUBSAMPLE_FLATTENED as i32,
        ])?;

        let xscale = (EMO_ENC_DIM as f32).sqrt();
        let mut x = addmm(
            &self.embed_out_bias,
            &flat,
            self.embed_out_weight.t(),
            None,
            None,
        )?
        .multiply(Array::from_f32(xscale))?;

        let pos_emb = self.pe.index((.., ..t_sub, ..));
        for layer in &self.encoders {
            x = layer.forward(&x, &pos_emb, &amask, &vmask)?;
        }
        x = layer_norm(
            &x,
            Some(&self.after_norm_weight),
            Some(&self.after_norm_bias),
            EMO_NORM_EPS,
        )?;

        // PerceiverResampler
        let ctx = addmm(
            &self.proj_context_bias,
            &x,
            self.proj_context_weight.t(),
            None,
            None,
        )?;
        let latents_3d = self.latents.reshape(&[1, 1, EMO_PERC_DIM as i32])?;
        let mut latents = broadcast_to(&latents_3d, &[b, 1, EMO_PERC_DIM as i32])?;

        for layer in &self.perc_layers {
            latents = layer.forward(&latents, &ctx, &amask_perc)?;
        }

        // PercRMSNorm: x / max(||x||_2, 1e-12) * sqrt(1024) * gamma
        let norm = latents.multiply(&latents)?.sum_axes(&[-1], true)?.sqrt()?;
        let denom = maximum(&norm, Array::from_f32(1e-12))?;
        let perc_scale = (EMO_PERC_DIM as f32).sqrt();
        let conds = latents
            .divide(&denom)?
            .multiply(Array::from_f32(perc_scale))?
            .multiply(&self.perc_norm_gamma)?
            .reshape(&[b, EMO_PERC_DIM as i32])?;

        let h = addmm(
            &self.emovec_bias,
            &conds,
            self.emovec_weight.t(),
            None,
            None,
        )?;
        let out = addmm(&self.emo_bias, &h, self.emo_weight.t(), None, None)?;
        out.eval()?;
        Ok(out)
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
    fn test_should_match_python_reference_cp07_emo_vec() {
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

        let emo_res = EmoVec::load_from_path(model_dir.join("raw/emo.safetensors"));
        assert!(emo_res.is_ok(), "failed to load EmoVec: {emo_res:?}");
        let Ok(emo) = emo_res else {
            return;
        };

        let (spk_shape, spk_data) = extract_f32(&st, "spk_cond_emb");
        let (expected_shape, expected_emo) = extract_f32(&st, "emovec");
        let t_len = spk_shape.get(1).copied().unwrap_or(0);

        let spk_shape_i32: Vec<i32> = spk_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        let feat_arr = Array::from_slice(&spk_data, &spk_shape_i32);

        let out_res = emo.forward(&feat_arr, &[t_len]);
        assert!(out_res.is_ok(), "EmoVec forward failed: {out_res:?}");
        let Ok(out) = out_res else {
            return;
        };

        let expected_shape_i32: Vec<i32> = expected_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        assert_eq!(out.shape(), expected_shape_i32.as_slice());

        let rust_vals = out.try_as_slice::<f32>().unwrap_or(&[]);
        assert_eq!(rust_vals.len(), expected_emo.len());
        let (cos, max_diff) = cosine_similarity_and_max_diff(rust_vals, &expected_emo);
        assert!(
            cos >= 0.999_9,
            "CP-07 emovec cosine similarity {cos} < 0.9999"
        );
        assert!(
            max_diff < 1e-3,
            "CP-07 emovec max_abs_diff {max_diff} >= 1e-3"
        );
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
}

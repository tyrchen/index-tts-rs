//! Stage 0 Semantic Encoder (`facebook/w2v-bert-2.0`, `raw/semantic_model.safetensors`).
//!
//! Runs the `FeatureProjection(160 -> 1024)` and the first 17 Conformer encoder layers
//! (`0..=16`, corresponding to `hidden_states[17]`), followed by per-channel normalization
//! `(h - _mean) / _std`. Layers `17..=23` and `masked_spec_embed` are stripped at load time,
//! saving 677.1 MB of memory.

use std::path::Path;

use mlx_rs::{
    Array,
    fast::{ScaledDotProductAttentionMask, layer_norm, scaled_dot_product_attention},
    nn::silu,
    ops::{addmm, conv1d, einsum, indexing::TryIndexOp, pad, select, sigmoid},
};

use crate::{
    error::{IndexTtsError, Result},
    models::{FilteredWeights, ModelWeightKind, SEMANTIC_ACTIVE_LAYERS, SafeTensorsLoader},
};

/// Input acoustic feature dimension (`160`, from `w2v_features`).
pub const SEMANTIC_FEAT_DIM: usize = 160;
/// Hidden dimension of the `w2v-bert-2.0` encoder (`1024`).
pub const SEMANTIC_DIM: usize = 1_024;
/// Number of self-attention heads (`16`).
pub const SEMANTIC_NUM_HEADS: usize = 16;
/// Per-head dimension (`64`).
pub const SEMANTIC_HEAD_DIM: usize = SEMANTIC_DIM / SEMANTIC_NUM_HEADS;
/// Macaron feed-forward intermediate dimension (`4096`).
pub const SEMANTIC_INTERMEDIATE: usize = 4_096;
/// Causal depthwise convolution kernel size (`31`).
pub const SEMANTIC_CONV_KERNEL: usize = 31;
/// Left causal padding for the `k=31` depthwise convolution (`30`).
pub const SEMANTIC_CONV_PAD_LEFT: i32 = 30;
/// Maximum left relative distance (`64`).
pub const SEMANTIC_LEFT_POS: i32 = 64;
/// Maximum right relative distance (`8`).
pub const SEMANTIC_RIGHT_POS: i32 = 8;
/// Relative distance embedding table rows (`64 + 8 + 1 = 73`).
pub const SEMANTIC_DIST_TABLE_SIZE: usize = 73;
/// `LayerNorm` epsilon (`1e-5`).
pub const SEMANTIC_NORM_EPS: f32 = 1e-5;
/// Scaled dot-product attention scale (`64^-0.5 = 0.125`).
const ATTN_SCALE: f32 = 0.125;

#[derive(Debug, Clone)]
struct FeatureProjection {
    ln_weight: Array,
    ln_bias: Array,
    proj_weight: Array,
    proj_bias: Array,
}

impl FeatureProjection {
    fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        Ok(Self {
            ln_weight: weights.tensor_array_shaped(
                "feature_projection.layer_norm.weight",
                &[SEMANTIC_FEAT_DIM],
            )?,
            ln_bias: weights
                .tensor_array_shaped("feature_projection.layer_norm.bias", &[SEMANTIC_FEAT_DIM])?,
            proj_weight: weights.tensor_array_shaped(
                "feature_projection.projection.weight",
                &[SEMANTIC_DIM, SEMANTIC_FEAT_DIM],
            )?,
            proj_bias: weights
                .tensor_array_shaped("feature_projection.projection.bias", &[SEMANTIC_DIM])?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let normed = layer_norm(
            x,
            Some(&self.ln_weight),
            Some(&self.ln_bias),
            SEMANTIC_NORM_EPS,
        )?;
        Ok(addmm(
            &self.proj_bias,
            &normed,
            self.proj_weight.t(),
            None,
            None,
        )?)
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    inter_weight: Array,
    inter_bias: Array,
    out_weight: Array,
    out_bias: Array,
}

impl FeedForward {
    fn from_weights(weights: &FilteredWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            inter_weight: weights.tensor_array_shaped(
                &format!("{prefix}.intermediate_dense.weight"),
                &[SEMANTIC_INTERMEDIATE, SEMANTIC_DIM],
            )?,
            inter_bias: weights.tensor_array_shaped(
                &format!("{prefix}.intermediate_dense.bias"),
                &[SEMANTIC_INTERMEDIATE],
            )?,
            out_weight: weights.tensor_array_shaped(
                &format!("{prefix}.output_dense.weight"),
                &[SEMANTIC_DIM, SEMANTIC_INTERMEDIATE],
            )?,
            out_bias: weights
                .tensor_array_shaped(&format!("{prefix}.output_dense.bias"), &[SEMANTIC_DIM])?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = addmm(&self.inter_bias, x, self.inter_weight.t(), None, None)?;
        let h = silu(&h)?;
        Ok(addmm(&self.out_bias, &h, self.out_weight.t(), None, None)?)
    }
}

#[derive(Debug, Clone)]
struct RelativeKeySelfAttention {
    q_weight: Array,
    q_bias: Array,
    k_weight: Array,
    k_bias: Array,
    v_weight: Array,
    v_bias: Array,
    out_weight: Array,
    out_bias: Array,
    distance_embedding: Array,
}

impl RelativeKeySelfAttention {
    fn from_weights(weights: &FilteredWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            q_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear_q.weight"),
                &[SEMANTIC_DIM, SEMANTIC_DIM],
            )?,
            q_bias: weights
                .tensor_array_shaped(&format!("{prefix}.linear_q.bias"), &[SEMANTIC_DIM])?,
            k_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear_k.weight"),
                &[SEMANTIC_DIM, SEMANTIC_DIM],
            )?,
            k_bias: weights
                .tensor_array_shaped(&format!("{prefix}.linear_k.bias"), &[SEMANTIC_DIM])?,
            v_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear_v.weight"),
                &[SEMANTIC_DIM, SEMANTIC_DIM],
            )?,
            v_bias: weights
                .tensor_array_shaped(&format!("{prefix}.linear_v.bias"), &[SEMANTIC_DIM])?,
            out_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear_out.weight"),
                &[SEMANTIC_DIM, SEMANTIC_DIM],
            )?,
            out_bias: weights
                .tensor_array_shaped(&format!("{prefix}.linear_out.bias"), &[SEMANTIC_DIM])?,
            distance_embedding: weights.tensor_array_shaped(
                &format!("{prefix}.distance_embedding.weight"),
                &[SEMANTIC_DIST_TABLE_SIZE, SEMANTIC_HEAD_DIM],
            )?,
        })
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::many_single_char_names
    )]
    fn forward(&self, h: &Array, dist_idx: &Array, attn_bias: Option<&Array>) -> Result<Array> {
        let b = h.dim(0);
        let t = h.dim(1);
        let heads = SEMANTIC_NUM_HEADS as i32;
        let head_dim = SEMANTIC_HEAD_DIM as i32;
        let dim = SEMANTIC_DIM as i32;

        let q = addmm(&self.q_bias, h, self.q_weight.t(), None, None)?
            .reshape(&[b, t, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = addmm(&self.k_bias, h, self.k_weight.t(), None, None)?
            .reshape(&[b, t, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = addmm(&self.v_bias, h, self.v_weight.t(), None, None)?
            .reshape(&[b, t, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // pos_emb: [T, T, D]
        let pos_emb = self.distance_embedding.try_index(dist_idx)?;
        let rel =
            einsum("bhld,lrd->bhlr", [&q, &pos_emb])?.multiply(Array::from_f32(ATTN_SCALE))?;

        let mask = match attn_bias {
            Some(bias) => rel.add(bias)?,
            None => rel,
        };

        let out = scaled_dot_product_attention(
            q,
            k,
            v,
            ATTN_SCALE,
            Some(ScaledDotProductAttentionMask::Array(&mask)),
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, t, dim])?;

        Ok(addmm(
            &self.out_bias,
            &out,
            self.out_weight.t(),
            None,
            None,
        )?)
    }
}

#[derive(Debug, Clone)]
struct ConvolutionModule {
    ln_weight: Array,
    ln_bias: Array,
    pw_conv1_weight: Array,
    dw_conv_weight: Array,
    dw_ln_weight: Array,
    dw_ln_bias: Array,
    pw_conv2_weight: Array,
}

impl ConvolutionModule {
    fn from_weights(weights: &FilteredWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            ln_weight: weights
                .tensor_array_shaped(&format!("{prefix}.layer_norm.weight"), &[SEMANTIC_DIM])?,
            ln_bias: weights
                .tensor_array_shaped(&format!("{prefix}.layer_norm.bias"), &[SEMANTIC_DIM])?,
            pw_conv1_weight: weights.tensor_array_shaped(
                &format!("{prefix}.pointwise_conv1.weight"),
                &[2 * SEMANTIC_DIM, 1, SEMANTIC_DIM],
            )?,
            dw_conv_weight: weights.tensor_array_shaped(
                &format!("{prefix}.depthwise_conv.weight"),
                &[SEMANTIC_DIM, SEMANTIC_CONV_KERNEL, 1],
            )?,
            dw_ln_weight: weights.tensor_array_shaped(
                &format!("{prefix}.depthwise_layer_norm.weight"),
                &[SEMANTIC_DIM],
            )?,
            dw_ln_bias: weights.tensor_array_shaped(
                &format!("{prefix}.depthwise_layer_norm.bias"),
                &[SEMANTIC_DIM],
            )?,
            pw_conv2_weight: weights.tensor_array_shaped(
                &format!("{prefix}.pointwise_conv2.weight"),
                &[SEMANTIC_DIM, 1, SEMANTIC_DIM],
            )?,
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn forward(&self, h: &Array, conv_mask: Option<&Array>) -> Result<Array> {
        let mut x = layer_norm(
            h,
            Some(&self.ln_weight),
            Some(&self.ln_bias),
            SEMANTIC_NORM_EPS,
        )?;
        if let Some(mask) = conv_mask {
            x = select(mask, &x, Array::from_f32(0.0))?;
        }
        let x = conv1d(&x, &self.pw_conv1_weight, 1, 0, 1, 1)?;
        let parts = x.split_equal(2, -1)?;
        let (Some(a), Some(b)) = (parts.first(), parts.get(1)) else {
            return Err(IndexTtsError::mlx("GLU split failed in ConvolutionModule"));
        };
        let x = a.multiply(sigmoid(b)?)?;
        let x = pad(
            &x,
            &[(0, 0), (SEMANTIC_CONV_PAD_LEFT, 0), (0, 0)],
            None,
            None,
        )?;
        let x = conv1d(&x, &self.dw_conv_weight, 1, 0, 1, SEMANTIC_DIM as i32)?;
        let x = layer_norm(
            &x,
            Some(&self.dw_ln_weight),
            Some(&self.dw_ln_bias),
            SEMANTIC_NORM_EPS,
        )?;
        let x = silu(&x)?;
        Ok(conv1d(&x, &self.pw_conv2_weight, 1, 0, 1, 1)?)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    ffn1_ln_weight: Array,
    ffn1_ln_bias: Array,
    ffn1: FeedForward,
    attn_ln_weight: Array,
    attn_ln_bias: Array,
    self_attn: RelativeKeySelfAttention,
    conv_module: ConvolutionModule,
    ffn2_ln_weight: Array,
    ffn2_ln_bias: Array,
    ffn2: FeedForward,
    final_ln_weight: Array,
    final_ln_bias: Array,
}

impl EncoderLayer {
    fn from_weights(weights: &FilteredWeights, layer_idx: usize) -> Result<Self> {
        let prefix = format!("encoder.layers.{layer_idx}");
        Ok(Self {
            ffn1_ln_weight: weights.tensor_array_shaped(
                &format!("{prefix}.ffn1_layer_norm.weight"),
                &[SEMANTIC_DIM],
            )?,
            ffn1_ln_bias: weights
                .tensor_array_shaped(&format!("{prefix}.ffn1_layer_norm.bias"), &[SEMANTIC_DIM])?,
            ffn1: FeedForward::from_weights(weights, &format!("{prefix}.ffn1"))?,
            attn_ln_weight: weights.tensor_array_shaped(
                &format!("{prefix}.self_attn_layer_norm.weight"),
                &[SEMANTIC_DIM],
            )?,
            attn_ln_bias: weights.tensor_array_shaped(
                &format!("{prefix}.self_attn_layer_norm.bias"),
                &[SEMANTIC_DIM],
            )?,
            self_attn: RelativeKeySelfAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
            )?,
            conv_module: ConvolutionModule::from_weights(
                weights,
                &format!("{prefix}.conv_module"),
            )?,
            ffn2_ln_weight: weights.tensor_array_shaped(
                &format!("{prefix}.ffn2_layer_norm.weight"),
                &[SEMANTIC_DIM],
            )?,
            ffn2_ln_bias: weights
                .tensor_array_shaped(&format!("{prefix}.ffn2_layer_norm.bias"), &[SEMANTIC_DIM])?,
            ffn2: FeedForward::from_weights(weights, &format!("{prefix}.ffn2"))?,
            final_ln_weight: weights.tensor_array_shaped(
                &format!("{prefix}.final_layer_norm.weight"),
                &[SEMANTIC_DIM],
            )?,
            final_ln_bias: weights
                .tensor_array_shaped(&format!("{prefix}.final_layer_norm.bias"), &[SEMANTIC_DIM])?,
        })
    }

    fn forward(
        &self,
        h: &Array,
        dist_idx: &Array,
        attn_bias: Option<&Array>,
        conv_mask: Option<&Array>,
    ) -> Result<Array> {
        let half = Array::from_f32(0.5);
        let f1_norm = layer_norm(
            h,
            Some(&self.ffn1_ln_weight),
            Some(&self.ffn1_ln_bias),
            SEMANTIC_NORM_EPS,
        )?;
        let h = self.ffn1.forward(&f1_norm)?.multiply(&half)?.add(h)?;

        let attn_norm = layer_norm(
            &h,
            Some(&self.attn_ln_weight),
            Some(&self.attn_ln_bias),
            SEMANTIC_NORM_EPS,
        )?;
        let h = self
            .self_attn
            .forward(&attn_norm, dist_idx, attn_bias)?
            .add(&h)?;

        let h = self.conv_module.forward(&h, conv_mask)?.add(&h)?;

        let f2_norm = layer_norm(
            &h,
            Some(&self.ffn2_ln_weight),
            Some(&self.ffn2_ln_bias),
            SEMANTIC_NORM_EPS,
        )?;
        let h = self.ffn2.forward(&f2_norm)?.multiply(&half)?.add(&h)?;

        Ok(layer_norm(
            &h,
            Some(&self.final_ln_weight),
            Some(&self.final_ln_bias),
            SEMANTIC_NORM_EPS,
        )?)
    }
}

/// Stage 0 `w2v-bert-2.0` Semantic Encoder (`raw/semantic_model.safetensors`, layers `0..17`).
#[derive(Debug, Clone)]
pub struct SemanticModel {
    feature_projection: FeatureProjection,
    layers: Vec<EncoderLayer>,
    mean: Array,
    std: Array,
}

impl SemanticModel {
    /// Constructs a [`SemanticModel`] from pre-filtered weights.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if any required tensor or buffer is missing or has
    /// an unexpected shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::{
    ///     ModelWeightKind, SafeTensorsLoader, SemanticModel, resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let weights = SafeTensorsLoader::load_from_path(
    ///         ModelWeightKind::SemanticModel,
    ///         dir.join("raw/semantic_model.safetensors"),
    ///     )?;
    ///     let model = SemanticModel::from_weights(&weights)?;
    ///     assert_eq!(model.num_layers(), 17);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let feature_projection = FeatureProjection::from_weights(weights)?;
        let mut layers = Vec::with_capacity(SEMANTIC_ACTIVE_LAYERS);
        for idx in 0..SEMANTIC_ACTIVE_LAYERS {
            layers.push(EncoderLayer::from_weights(weights, idx)?);
        }
        let mean = weights.buffer_array("_mean")?;
        let std = weights.buffer_array("_std")?;
        Ok(Self {
            feature_projection,
            layers,
            mean,
            std,
        })
    }

    /// Loads [`SemanticModel`] weights directly from `raw/semantic_model.safetensors` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if the file cannot be read or tensors fail validation.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let weights = SafeTensorsLoader::load_from_path(ModelWeightKind::SemanticModel, path)?;
        Self::from_weights(&weights)
    }

    /// Returns the number of active Conformer encoder layers (`17`).
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Encodes `[B, T, 160]` `w2v_features` and optional `[B, T]` `attention_mask` (`1` = keep,
    /// `0` = pad) into normalized `[B, T, 1024]` semantic features (`spk_cond_emb`).
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if shapes are invalid or [`IndexTtsError::Mlx`] if
    /// GPU execution fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_rs::Array;
    /// use index_tts_rs_core::models::{SemanticModel, resolve_model_dir};
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let model = SemanticModel::load_from_path(dir.join("raw/semantic_model.safetensors"))?;
    ///     let feat = Array::zeros::<f32>(&[1, 8, 160])?;
    ///     let mask = Array::ones::<i32>(&[1, 8])?;
    ///     let out = model.forward(&feat, Some(&mask))?;
    ///     assert_eq!(out.shape(), &[1, 8, 1024]);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn forward(&self, input_features: &Array, attention_mask: Option<&Array>) -> Result<Array> {
        if input_features.ndim() != 3 {
            return Err(IndexTtsError::validation(
                "input_features",
                format!(
                    "expected 3D tensor [B, T, 160], got {:?}",
                    input_features.shape()
                ),
            ));
        }
        let b = input_features.dim(0);
        let t = input_features.dim(1);
        let c = input_features.dim(2);
        if b <= 0 || t <= 0 {
            return Err(IndexTtsError::validation(
                "input_features",
                "batch and sequence dimensions must be positive",
            ));
        }
        if usize::try_from(c).unwrap_or(0) != SEMANTIC_FEAT_DIM {
            return Err(IndexTtsError::validation(
                "input_features",
                format!("expected {SEMANTIC_FEAT_DIM} feature channels, got {c}"),
            ));
        }

        let mut h = self.feature_projection.forward(input_features)?;

        let (attn_bias, conv_mask) = if let Some(mask) = attention_mask {
            if mask.shape() != [b, t] {
                return Err(IndexTtsError::validation(
                    "attention_mask",
                    format!("expected shape [{b}, {t}], got {:?}", mask.shape()),
                ));
            }
            let bool_mask = mask.ne(Array::from_int(0))?;
            let conv_m = bool_mask.reshape(&[b, t, 1])?;
            h = select(&conv_m, &h, Array::from_f32(0.0))?;

            let mask_f32 = mask.as_type::<f32>()?.reshape(&[b, 1, 1, t])?;
            let bias = Array::from_f32(1.0)
                .subtract(&mask_f32)?
                .multiply(Array::from_f32(f32::MIN))?;
            (Some(bias), Some(conv_m))
        } else {
            (None, None)
        };

        let t_usize = usize::try_from(t).unwrap_or(0);
        let dist_idx = build_distance_index_matrix(t_usize);

        for layer in &self.layers {
            h = layer.forward(&h, &dist_idx, attn_bias.as_ref(), conv_mask.as_ref())?;
        }

        let out = h.subtract(&self.mean)?.divide(&self.std)?;
        out.eval()?;
        Ok(out)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn build_distance_index_matrix(t: usize) -> Array {
    let mut indices = Vec::with_capacity(t * t);
    for q in 0..t {
        let q_i32 = q as i32;
        for k in 0..t {
            let k_i32 = k as i32;
            let d =
                (k_i32 - q_i32).clamp(-SEMANTIC_LEFT_POS, SEMANTIC_RIGHT_POS) + SEMANTIC_LEFT_POS;
            indices.push(d);
        }
    }
    let t_i32 = t as i32;
    Array::from_slice(&indices, &[t_i32, t_i32])
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
    fn test_should_match_python_reference_cp06_semantic_model() {
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

        let model_res =
            SemanticModel::load_from_path(model_dir.join("raw/semantic_model.safetensors"));
        assert!(
            model_res.is_ok(),
            "failed to load SemanticModel: {model_res:?}"
        );
        let Ok(model) = model_res else {
            return;
        };

        let (feat_shape, feat_data) = extract_f32(&st, "w2v_input_features");
        let (mask_shape, mask_data) = extract_i32(&st, "w2v_attention_mask");
        let (expected_shape, expected_spk) = extract_f32(&st, "spk_cond_emb");

        let feat_shape_i32: Vec<i32> = feat_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        let mask_shape_i32: Vec<i32> = mask_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();

        let feat_arr = Array::from_slice(&feat_data, &feat_shape_i32);
        let mask_arr = Array::from_slice(&mask_data, &mask_shape_i32);

        let out_res = model.forward(&feat_arr, Some(&mask_arr));
        assert!(out_res.is_ok(), "SemanticModel forward failed: {out_res:?}");
        let Ok(out) = out_res else {
            return;
        };

        let expected_shape_i32: Vec<i32> = expected_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        assert_eq!(out.shape(), expected_shape_i32.as_slice());

        let rust_vals = out.try_as_slice::<f32>().unwrap_or(&[]);
        assert_eq!(rust_vals.len(), expected_spk.len());
        let (cos, max_diff) = cosine_similarity_and_max_diff(rust_vals, &expected_spk);
        let rel_rmse = relative_rmse(rust_vals, &expected_spk);
        assert!(
            cos >= 0.999_9,
            "CP-06 spk_cond_emb cosine similarity {cos} < 0.9999"
        );
        assert!(
            rel_rmse < 1.5e-3,
            "CP-06 spk_cond_emb relative RMSE {rel_rmse} >= 1.5e-3"
        );
        assert!(
            max_diff < 3e-2,
            "CP-06 spk_cond_emb max_abs_diff {max_diff} >= 3e-2"
        );
    }

    #[allow(clippy::cast_possible_truncation)]
    fn relative_rmse(actual: &[f32], expected: &[f32]) -> f32 {
        let mut diff_sq = 0.0_f64;
        let mut ref_sq = 0.0_f64;
        for (&a, &b) in actual.iter().zip(expected.iter()) {
            let d = f64::from(a) - f64::from(b);
            diff_sq += d * d;
            ref_sq += f64::from(b) * f64::from(b);
        }
        if ref_sq <= 0.0 {
            return 0.0;
        }
        (diff_sq.sqrt() / ref_sq.sqrt()) as f32
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

    fn extract_i32(st: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<i32>) {
        let Ok(view) = st.tensor(name) else {
            return (Vec::new(), Vec::new());
        };
        let mut out = Vec::with_capacity(view.data().len() / 4);
        for &b in view.data().as_chunks::<4>().0 {
            out.push(i32::from_le_bytes(b));
        }
        (view.shape().to_vec(), out)
    }
}

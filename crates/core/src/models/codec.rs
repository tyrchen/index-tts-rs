//! `RepCodec` (`raw/semantic_codec.safetensors`) decoder-only MLX implementation.
//!
//! Decodes discrete semantic tokens `[B, T]` (`0..8192`) into `[B, 2*T, 1024]` continuous
//! representations via factorized vector dequantization (`8 -> 1024`), a 12-layer Vocos `ConvNeXt`
//! backbone (`384` channels), linear projection (`384 -> 1024`), 2x nearest-neighbor repetition,
//! and a final `Conv1d(1024 -> 1024, k=3)` smoothing layer.

use std::path::Path;

use mlx_rs::{
    Array,
    fast::layer_norm,
    nn::gelu,
    ops::{addmm, conv1d, indexing::TryIndexOp},
};

use crate::{
    error::{IndexTtsError, Result},
    models::{FilteredWeights, ModelWeightKind, SafeTensorsLoader},
};

/// Output feature dimension of the semantic codec (`1024`).
pub const CODEC_HIDDEN: usize = 1_024;
/// Codebook vocabulary size (`8192`).
pub const CODEC_CODEBOOK_SIZE: usize = 8_192;
/// Factorized codebook embedding dimension (`8`).
pub const CODEC_CODEBOOK_DIM: usize = 8;
/// Vocos `ConvNeXt` backbone hidden dimension (`384`).
pub const CODEC_VOCOS_DIM: usize = 384;
/// Vocos `ConvNeXt` pointwise intermediate dimension (`2048`).
pub const CODEC_INTERMEDIATE: usize = 2_048;
/// Number of `ConvNeXt` blocks in the Vocos backbone (`12`).
pub const CODEC_NUM_LAYERS: usize = 12;
/// `LayerNorm` epsilon for the Vocos backbone (`1e-6`).
pub const CODEC_NORM_EPS: f32 = 1e-6;
/// Temporal upsampling repetition factor (`2`).
pub const CODEC_UPSAMPLE_FACTOR: i32 = 2;

#[derive(Debug, Clone)]
struct ConvNeXtBlock {
    dwconv_weight: Array,
    dwconv_bias: Array,
    norm_weight: Array,
    norm_bias: Array,
    pwconv1_weight: Array,
    pwconv1_bias: Array,
    pwconv2_weight: Array,
    pwconv2_bias: Array,
    gamma: Array,
}

impl ConvNeXtBlock {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn forward(&self, x: &Array) -> Result<Array> {
        let residual = x;
        let h = conv1d(x, &self.dwconv_weight, 1, 3, 1, CODEC_VOCOS_DIM as i32)?
            .add(&self.dwconv_bias)?;
        let h = layer_norm(
            &h,
            Some(&self.norm_weight),
            Some(&self.norm_bias),
            CODEC_NORM_EPS,
        )?;
        let h = addmm(&self.pwconv1_bias, &h, self.pwconv1_weight.t(), None, None)?;
        let h = gelu(&h)?;
        let h = addmm(&self.pwconv2_bias, &h, self.pwconv2_weight.t(), None, None)?;
        let h = self.gamma.multiply(&h)?;
        Ok(residual.add(&h)?)
    }
}

/// Decoder-only `SemanticCodec` (`raw/semantic_codec.safetensors`).
#[derive(Debug, Clone)]
pub struct SemanticCodec {
    codebook_weight: Array,
    out_proj_weight: Array,
    out_proj_bias: Array,
    embed_weight: Array,
    embed_bias: Array,
    norm_weight: Array,
    norm_bias: Array,
    convnext: Vec<ConvNeXtBlock>,
    final_norm_weight: Array,
    final_norm_bias: Array,
    proj_out_weight: Array,
    proj_out_bias: Array,
    up_weight: Array,
    up_bias: Array,
}

impl SemanticCodec {
    /// Constructs a [`SemanticCodec`] from pre-filtered weights.
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
    ///     ModelWeightKind, SafeTensorsLoader, SemanticCodec, resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let weights = SafeTensorsLoader::load_from_path(
    ///         ModelWeightKind::SemanticCodec,
    ///         dir.join("raw/semantic_codec.safetensors"),
    ///     )?;
    ///     let codec = SemanticCodec::from_weights(&weights)?;
    ///     assert_eq!(codec.num_layers(), 12);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let codebook_weight = weights.tensor_array_shaped(
            "quantizer.quantizers.0.codebook.weight",
            &[CODEC_CODEBOOK_SIZE, CODEC_CODEBOOK_DIM],
        )?;
        let out_proj_weight = weights.tensor_array_shaped(
            "quantizer.quantizers.0.out_project.weight",
            &[CODEC_HIDDEN, 1, CODEC_CODEBOOK_DIM],
        )?;
        let out_proj_bias = weights
            .tensor_array_shaped("quantizer.quantizers.0.out_project.bias", &[CODEC_HIDDEN])?;

        let embed_weight = weights.tensor_array_shaped(
            "decoder.0.embed.weight",
            &[CODEC_VOCOS_DIM, 7, CODEC_HIDDEN],
        )?;
        let embed_bias = weights.tensor_array_shaped("decoder.0.embed.bias", &[CODEC_VOCOS_DIM])?;
        let norm_weight =
            weights.tensor_array_shaped("decoder.0.norm.weight", &[CODEC_VOCOS_DIM])?;
        let norm_bias = weights.tensor_array_shaped("decoder.0.norm.bias", &[CODEC_VOCOS_DIM])?;

        let mut convnext = Vec::with_capacity(CODEC_NUM_LAYERS);
        for i in 0..CODEC_NUM_LAYERS {
            let prefix = format!("decoder.0.convnext.{i}");
            let dwconv_weight = weights.tensor_array_shaped(
                &format!("{prefix}.dwconv.weight"),
                &[CODEC_VOCOS_DIM, 7, 1],
            )?;
            let dwconv_bias = weights
                .tensor_array_shaped(&format!("{prefix}.dwconv.bias"), &[CODEC_VOCOS_DIM])?;
            let block_norm_weight = weights
                .tensor_array_shaped(&format!("{prefix}.norm.weight"), &[CODEC_VOCOS_DIM])?;
            let block_norm_bias =
                weights.tensor_array_shaped(&format!("{prefix}.norm.bias"), &[CODEC_VOCOS_DIM])?;
            let pwconv1_weight = weights.tensor_array_shaped(
                &format!("{prefix}.pwconv1.weight"),
                &[CODEC_INTERMEDIATE, CODEC_VOCOS_DIM],
            )?;
            let pwconv1_bias = weights
                .tensor_array_shaped(&format!("{prefix}.pwconv1.bias"), &[CODEC_INTERMEDIATE])?;
            let pwconv2_weight = weights.tensor_array_shaped(
                &format!("{prefix}.pwconv2.weight"),
                &[CODEC_VOCOS_DIM, CODEC_INTERMEDIATE],
            )?;
            let pwconv2_bias = weights
                .tensor_array_shaped(&format!("{prefix}.pwconv2.bias"), &[CODEC_VOCOS_DIM])?;
            let gamma =
                weights.tensor_array_shaped(&format!("{prefix}.gamma"), &[CODEC_VOCOS_DIM])?;

            convnext.push(ConvNeXtBlock {
                dwconv_weight,
                dwconv_bias,
                norm_weight: block_norm_weight,
                norm_bias: block_norm_bias,
                pwconv1_weight,
                pwconv1_bias,
                pwconv2_weight,
                pwconv2_bias,
                gamma,
            });
        }

        let final_norm_weight =
            weights.tensor_array_shaped("decoder.0.final_layer_norm.weight", &[CODEC_VOCOS_DIM])?;
        let final_norm_bias =
            weights.tensor_array_shaped("decoder.0.final_layer_norm.bias", &[CODEC_VOCOS_DIM])?;
        let proj_out_weight =
            weights.tensor_array_shaped("decoder.1.weight", &[CODEC_HIDDEN, CODEC_VOCOS_DIM])?;
        let proj_out_bias = weights.tensor_array_shaped("decoder.1.bias", &[CODEC_HIDDEN])?;
        let up_weight =
            weights.tensor_array_shaped("up.weight", &[CODEC_HIDDEN, 3, CODEC_HIDDEN])?;
        let up_bias = weights.tensor_array_shaped("up.bias", &[CODEC_HIDDEN])?;

        Ok(Self {
            codebook_weight,
            out_proj_weight,
            out_proj_bias,
            embed_weight,
            embed_bias,
            norm_weight,
            norm_bias,
            convnext,
            final_norm_weight,
            final_norm_bias,
            proj_out_weight,
            proj_out_bias,
            up_weight,
            up_bias,
        })
    }

    /// Loads [`SemanticCodec`] weights from `raw/semantic_codec.safetensors` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if the file cannot be read or tensors fail validation.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let weights = SafeTensorsLoader::load_from_path(ModelWeightKind::SemanticCodec, path)?;
        Self::from_weights(&weights)
    }

    /// Returns the number of `ConvNeXt` layers (`12`).
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.convnext.len()
    }

    /// Decodes a `[B, T]` integer token tensor (`0..8192`) into `[B, 2*T, 1024]` continuous
    /// features.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `codes` is not 2D `[B, T]` with `T > 0`, or
    /// [`IndexTtsError::Mlx`] if GPU execution fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_rs::Array;
    /// use index_tts_rs_core::models::{SemanticCodec, resolve_model_dir};
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let codec = SemanticCodec::load_from_path(dir.join("raw/semantic_codec.safetensors"))?;
    ///     let codes = Array::from_slice(&[100_i32, 200, 300, 400], &[1, 4]);
    ///     let s_infer = codec.decode(&codes)?;
    ///     assert_eq!(s_infer.shape(), &[1, 8, 1024]);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn decode(&self, codes: &Array) -> Result<Array> {
        if codes.ndim() != 2 {
            return Err(IndexTtsError::validation(
                "codes",
                format!(
                    "expected 2D token tensor [B, T], got shape {:?}",
                    codes.shape()
                ),
            ));
        }
        if codes.dim(1) <= 0 {
            return Err(IndexTtsError::validation(
                "codes",
                "token sequence length T must be >= 1",
            ));
        }

        // 1. VQ codebook lookup [B, T] -> [B, T, 8] -> out_project Conv1d(k=1) -> [B, T, 1024]
        let z_q = self.codebook_weight.try_index(codes)?;
        let x = conv1d(&z_q, &self.out_proj_weight, 1, 0, 1, 1)?.add(&self.out_proj_bias)?;

        // 2. VocosBackbone (decoder.0): embed -> norm -> 12x ConvNeXtBlock -> final_layer_norm
        let mut h = conv1d(&x, &self.embed_weight, 1, 3, 1, 1)?.add(&self.embed_bias)?;
        h = layer_norm(
            &h,
            Some(&self.norm_weight),
            Some(&self.norm_bias),
            CODEC_NORM_EPS,
        )?;
        for block in &self.convnext {
            h = block.forward(&h)?;
        }
        h = layer_norm(
            &h,
            Some(&self.final_norm_weight),
            Some(&self.final_norm_bias),
            CODEC_NORM_EPS,
        )?;

        // 3. Linear projection (decoder.1): [B, T, 384] -> [B, T, 1024]
        let h = addmm(
            &self.proj_out_bias,
            &h,
            self.proj_out_weight.t(),
            None,
            None,
        )?;

        // 4. 2x nearest-neighbor upsample + Conv1d(k=3, pad=1)
        let h_up = Array::repeat_axis::<f32>(h, CODEC_UPSAMPLE_FACTOR, 1)?;
        let out = conv1d(&h_up, &self.up_weight, 1, 1, 1, 1)?.add(&self.up_bias)?;
        out.eval()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use mlx_rs::ops::concatenate;
    use safetensors::SafeTensors;

    use super::*;
    use crate::{
        config::DEFAULT_MODEL_REPO,
        dsp::cosine_similarity_and_max_diff,
        models::{LengthRegulator, resolve_model_dir},
    };

    #[test]
    fn test_should_match_python_reference_cp12_semantic_codec_and_cat_condition() {
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

        let codec_res =
            SemanticCodec::load_from_path(model_dir.join("raw/semantic_codec.safetensors"));
        assert!(codec_res.is_ok());
        let Ok(codec) = codec_res else {
            return;
        };

        let greedy_codes = extract_i32(&st, "greedy_codes");
        let (expected_s_shape, expected_s_infer) = extract_f32(&st, "s_infer");
        let code_len_i32 = i32::try_from(greedy_codes.len()).unwrap_or(0);
        let codes_arr = Array::from_slice(&greedy_codes, &[1, code_len_i32]);

        let s_infer_res = codec.decode(&codes_arr);
        assert!(s_infer_res.is_ok());
        let Ok(s_infer) = s_infer_res else {
            return;
        };

        let expected_shape_i32: Vec<i32> = expected_s_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        assert_eq!(s_infer.shape(), expected_shape_i32.as_slice());

        let rust_s_vals = s_infer.try_as_slice::<f32>().unwrap_or(&[]);
        let (cos, max_diff) = cosine_similarity_and_max_diff(rust_s_vals, &expected_s_infer);
        let rel_rmse = relative_rmse(rust_s_vals, &expected_s_infer);
        assert!(
            cos >= 0.999_9,
            "CP-12 s_infer cosine similarity {cos} < 0.9999"
        );
        assert!(
            rel_rmse < 1e-3,
            "CP-12 s_infer relative RMSE {rel_rmse} >= 1e-3"
        );
        assert!(
            max_diff < 5e-3,
            "CP-12 s_infer max_abs_diff {max_diff} >= 5e-3"
        );

        // Also verify LengthRegulator(s_infer) + prompt_condition == cat_condition
        let Ok(lr) =
            LengthRegulator::load_from_path(model_dir.join("raw/length_regulator.safetensors"))
        else {
            return;
        };
        let (prompt_shape, prompt_data) = extract_f32(&st, "prompt_condition");
        let (_, expected_cat) = extract_f32(&st, "cat_condition");

        let s_len = usize::try_from(s_infer.dim(1)).unwrap_or(0);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let target_len = ((s_len as f64) * 1.72).floor() as usize;
        let Ok(cond) = lr.forward(&s_infer, target_len) else {
            return;
        };
        let prompt_shape_i32: Vec<i32> = prompt_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        let prompt_arr = Array::from_slice(&prompt_data, &prompt_shape_i32);
        let Ok(cat_cond) = concatenate(&[&prompt_arr, &cond], 1) else {
            return;
        };
        let _ = cat_cond.eval();
        let rust_cat = cat_cond.try_as_slice::<f32>().unwrap_or(&[]);
        let (cat_cos, cat_diff) = cosine_similarity_and_max_diff(rust_cat, &expected_cat);
        assert!(
            cat_cos >= 0.999_9,
            "CP-12 cat_condition cosine similarity {cat_cos} < 0.9999"
        );
        assert!(
            cat_diff < 1e-3,
            "CP-12 cat_condition max_abs_diff {cat_diff} >= 1e-3"
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
}

//! Length Regulator (`raw/length_regulator.safetensors`) for nearest-neighbor duration
//! interpolation and 4-block Conv1d + `GroupNorm` + Mish refinement.
//!
//! Converts `w2v-bert-2.0` or `SemanticCodec` `[B, T_in, 1024]` features into `[B, T_out, 512]`
//! frame-aligned conditioning tensors for the CFM `DiT` estimator.

use std::path::Path;

use mlx_rs::{Array, fast::layer_norm, nn::mish, ops::conv1d};

use crate::{
    error::{IndexTtsError, Result},
    models::{FilteredWeights, ModelWeightKind, SafeTensorsLoader},
};

/// Input feature dimension (`1024`).
pub const LR_IN_CHANNELS: usize = 1_024;
/// Regulated conditioning feature dimension (`512`).
pub const LR_CHANNELS: usize = 512;
/// Number of `Conv1d + GroupNorm + Mish` blocks (`4`).
pub const LR_NUM_BLOCKS: usize = 4;
/// `GroupNorm` epsilon (`1e-5`).
pub const LR_NORM_EPS: f32 = 1e-5;

const BLOCK_POSITIONS: [usize; LR_NUM_BLOCKS] = [0, 3, 6, 9];

/// Computes nearest-neighbor interpolation source indices mapping `t_in` frames to `t_out` frames.
///
/// Matches `torch.nn.functional.interpolate(..., size=t_out, mode="nearest")` using `f64`
/// precision:
/// `src_idx[dst] = min(floor(dst * (t_in / t_out)), t_in - 1)`.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::models::nearest_expand_indices;
///
/// let idx = nearest_expand_indices(3, 6);
/// assert_eq!(idx, vec![0, 0, 1, 1, 2, 2]);
/// ```
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
pub fn nearest_expand_indices(t_in: usize, t_out: usize) -> Vec<i32> {
    if t_in == 0 || t_out == 0 {
        return Vec::new();
    }
    let scale = (t_in as f64) / (t_out as f64);
    let max_src = (t_in.saturating_sub(1)) as f64;
    let mut indices = Vec::with_capacity(t_out);
    for dst in 0..t_out {
        let src = ((dst as f64) * scale).floor().min(max_src) as i32;
        indices.push(src);
    }
    indices
}

#[derive(Debug, Clone)]
struct RegulatorBlock {
    conv_weight: Array,
    conv_bias: Array,
    norm_weight: Array,
    norm_bias: Array,
}

impl RegulatorBlock {
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = conv1d(x, &self.conv_weight, 1, 1, 1, 1)?.add(&self.conv_bias)?;
        let h = pytorch_group_norm_1(&h, &self.norm_weight, &self.norm_bias, LR_NORM_EPS)?;
        Ok(mish(&h)?)
    }
}

/// Evaluates `nn.GroupNorm(1, C, eps, pytorch_compatible=True)` on a `[B, T, C]` tensor.
fn pytorch_group_norm_1(x: &Array, weight: &Array, bias: &Array, eps: f32) -> Result<Array> {
    let batch = x.dim(0);
    let time = x.dim(1);
    let channels = x.dim(2);
    let flat = x.reshape(&[batch, 1, -1])?;
    let normed = layer_norm(&flat, None, None, eps)?.reshape(&[batch, time, channels])?;
    Ok(normed.multiply(weight)?.add(bias)?)
}

/// MLX Length Regulator model (`raw/length_regulator.safetensors`).
#[derive(Debug, Clone)]
pub struct LengthRegulator {
    content_in_weight: Array,
    content_in_bias: Array,
    blocks: Vec<RegulatorBlock>,
    final_conv_weight: Array,
    final_conv_bias: Array,
}

impl LengthRegulator {
    /// Constructs a [`LengthRegulator`] from pre-filtered weights.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if any required tensor is missing or has an
    /// unexpected shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::{
    ///     LengthRegulator, ModelWeightKind, SafeTensorsLoader, resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let weights = SafeTensorsLoader::load_from_path(
    ///         ModelWeightKind::LengthRegulator,
    ///         dir.join("raw/length_regulator.safetensors"),
    ///     )?;
    ///     let lr = LengthRegulator::from_weights(&weights)?;
    ///     assert_eq!(lr.num_blocks(), 4);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let content_in_weight = weights
            .tensor_array_shaped("content_in_proj.weight", &[LR_CHANNELS, LR_IN_CHANNELS])?;
        let content_in_bias =
            weights.tensor_array_shaped("content_in_proj.bias", &[LR_CHANNELS])?;

        let mut blocks = Vec::with_capacity(LR_NUM_BLOCKS);
        for pos in BLOCK_POSITIONS {
            let conv_weight = weights.tensor_array_shaped(
                &format!("model.layers.{pos}.weight"),
                &[LR_CHANNELS, 3, LR_CHANNELS],
            )?;
            let conv_bias =
                weights.tensor_array_shaped(&format!("model.layers.{pos}.bias"), &[LR_CHANNELS])?;
            let norm_pos = pos + 1;
            let norm_weight = weights
                .tensor_array_shaped(&format!("model.layers.{norm_pos}.weight"), &[LR_CHANNELS])?;
            let norm_bias = weights
                .tensor_array_shaped(&format!("model.layers.{norm_pos}.bias"), &[LR_CHANNELS])?;
            blocks.push(RegulatorBlock {
                conv_weight,
                conv_bias,
                norm_weight,
                norm_bias,
            });
        }

        let final_conv_weight = weights
            .tensor_array_shaped("model.layers.12.weight", &[LR_CHANNELS, 1, LR_CHANNELS])?;
        let final_conv_bias =
            weights.tensor_array_shaped("model.layers.12.bias", &[LR_CHANNELS])?;

        Ok(Self {
            content_in_weight,
            content_in_bias,
            blocks,
            final_conv_weight,
            final_conv_bias,
        })
    }

    /// Loads [`LengthRegulator`] weights directly from `raw/length_regulator.safetensors` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if the file cannot be read or tensors fail validation.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let weights = SafeTensorsLoader::load_from_path(ModelWeightKind::LengthRegulator, path)?;
        Self::from_weights(&weights)
    }

    /// Returns the number of convolutional refinement blocks (`4`).
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Projects and interpolates `[B, T_in, 1024]` features to `[B, target_len, 512]`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `x` is not 3D `[B, T_in, 1024]` or `target_len ==
    /// 0`, or [`IndexTtsError::Mlx`] if GPU execution fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_rs::Array;
    /// use index_tts_rs_core::models::{LengthRegulator, resolve_model_dir};
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let lr = LengthRegulator::load_from_path(dir.join("raw/length_regulator.safetensors"))?;
    ///     let x = Array::zeros::<f32>(&[1, 10, 1024])?;
    ///     let y = lr.forward(&x, 17)?;
    ///     assert_eq!(y.shape(), &[1, 17, 512]);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn forward(&self, x: &Array, target_len: usize) -> Result<Array> {
        if x.ndim() != 3 {
            return Err(IndexTtsError::validation(
                "x",
                format!(
                    "expected 3D tensor [B, T_in, 1024], got shape {:?}",
                    x.shape()
                ),
            ));
        }
        let t_in_i32 = x.dim(1);
        let c_in_i32 = x.dim(2);
        if t_in_i32 <= 0 {
            return Err(IndexTtsError::validation(
                "x",
                format!("input sequence length must be positive, got {t_in_i32}"),
            ));
        }
        if usize::try_from(c_in_i32).unwrap_or(0) != LR_IN_CHANNELS {
            return Err(IndexTtsError::validation(
                "x",
                format!("expected {LR_IN_CHANNELS} input channels, got {c_in_i32}"),
            ));
        }
        if target_len == 0 {
            return Err(IndexTtsError::validation(
                "target_len",
                "target_len must be >= 1",
            ));
        }

        let t_in = usize::try_from(t_in_i32).unwrap_or(0);
        let mut h = x
            .matmul(self.content_in_weight.t())?
            .add(&self.content_in_bias)?;

        if t_in != target_len {
            let indices = nearest_expand_indices(t_in, target_len);
            let target_len_i32 = i32::try_from(target_len).map_err(|_| {
                IndexTtsError::validation("target_len", "target_len exceeds i32::MAX")
            })?;
            let idx_arr = Array::from_slice(&indices, &[target_len_i32]);
            h = h.take_axis(&idx_arr, 1)?;
        }

        for block in &self.blocks {
            h = block.forward(&h)?;
        }

        let out = conv1d(&h, &self.final_conv_weight, 1, 0, 1, 1)?.add(&self.final_conv_bias)?;
        out.eval()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use proptest::prelude::*;
    use safetensors::SafeTensors;

    use super::*;
    use crate::{
        config::DEFAULT_MODEL_REPO, dsp::cosine_similarity_and_max_diff, models::resolve_model_dir,
    };

    #[test]
    fn test_should_compute_nearest_expand_indices_correctly() {
        assert_eq!(nearest_expand_indices(0, 5), Vec::<i32>::new());
        assert_eq!(nearest_expand_indices(5, 0), Vec::<i32>::new());
        assert_eq!(nearest_expand_indices(4, 4), vec![0, 1, 2, 3]);
        assert_eq!(nearest_expand_indices(2, 5), vec![0, 0, 0, 1, 1]);
    }

    proptest! {
        #[test]
        fn proptest_nearest_expand_indices_monotonic_and_in_bounds(
            t_in in 1_usize..=500,
            t_out in 1_usize..=1000,
        ) {
            let idx = nearest_expand_indices(t_in, t_out);
            prop_assert_eq!(idx.len(), t_out);
            let t_in_i32 = i32::try_from(t_in).unwrap_or(1);
            for (pos, &val) in idx.iter().enumerate() {
                prop_assert!(val >= 0 && val < t_in_i32);
                if pos > 0 {
                    let prev = idx.get(pos - 1).copied().unwrap_or(0);
                    prop_assert!(val >= prev, "indices must be monotonically non-decreasing");
                }
            }
        }
    }

    #[test]
    fn test_should_match_python_reference_cp09_length_regulator() {
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

        let lr_res =
            LengthRegulator::load_from_path(model_dir.join("raw/length_regulator.safetensors"));
        assert!(lr_res.is_ok());
        let Ok(lr) = lr_res else {
            return;
        };

        let (spk_shape, spk_data) = extract_f32(&st, "spk_cond_emb");
        let (expected_shape, expected_prompt) = extract_f32(&st, "prompt_condition");
        let target_len = expected_shape.get(1).copied().unwrap_or(172);

        let shape_i32: Vec<i32> = spk_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        let x = Array::from_slice(&spk_data, &shape_i32);

        let out_res = lr.forward(&x, target_len);
        assert!(out_res.is_ok());
        if let Ok(out) = out_res {
            assert_eq!(
                out.shape(),
                &[
                    1,
                    i32::try_from(target_len).unwrap_or(0),
                    i32::try_from(LR_CHANNELS).unwrap_or(0)
                ]
            );
            let rust_vals = out.try_as_slice::<f32>().unwrap_or(&[]);
            assert_eq!(rust_vals.len(), expected_prompt.len());
            let (cos, max_diff) = cosine_similarity_and_max_diff(rust_vals, &expected_prompt);
            assert!(
                cos >= 0.999_9,
                "CP-09 prompt_condition cosine similarity {cos} < 0.9999"
            );
            assert!(
                max_diff < 1e-3,
                "CP-09 prompt_condition max_abs_diff {max_diff} >= 1e-3"
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
}

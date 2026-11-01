//! Vocos `ConvNeXt` block (depthwise `Conv1d(k=7)` → `LayerNorm` → pointwise MLP with GELU →
//! layer scale → residual), shared by the semantic codec decoder and the bandwidth extender.

use mlx_rs::{
    Array,
    fast::layer_norm,
    nn::gelu,
    ops::{addmm, conv1d},
};

use crate::{error::Result, models::FilteredWeights};

/// Depthwise convolution kernel size (`7`).
const KERNEL: usize = 7;
/// Depthwise convolution padding (`3`).
const PADDING: i32 = 3;
/// `LayerNorm` epsilon (`1e-6`).
const NORM_EPS: f32 = 1e-6;

/// One `ConvNeXt` block operating on `[B, T, dim]` (MLX channels-last) activations.
#[derive(Debug, Clone)]
pub(crate) struct ConvNeXtBlock {
    dim: i32,
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
    /// Loads `{prefix}.{dwconv,norm,pwconv1,pwconv2,gamma}` with the depthwise weight already in
    /// MLX `[dim, 7, 1]` layout.
    pub(crate) fn from_weights(
        weights: &FilteredWeights,
        prefix: &str,
        dim: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let tensor = |name: &str, shape: &[usize]| {
            weights.tensor_array_shaped(&format!("{prefix}.{name}"), shape)
        };
        Ok(Self {
            dim: i32::try_from(dim).unwrap_or(i32::MAX),
            dwconv_weight: tensor("dwconv.weight", &[dim, KERNEL, 1])?,
            dwconv_bias: tensor("dwconv.bias", &[dim])?,
            norm_weight: tensor("norm.weight", &[dim])?,
            norm_bias: tensor("norm.bias", &[dim])?,
            pwconv1_weight: tensor("pwconv1.weight", &[intermediate, dim])?,
            pwconv1_bias: tensor("pwconv1.bias", &[intermediate])?,
            pwconv2_weight: tensor("pwconv2.weight", &[dim, intermediate])?,
            pwconv2_bias: tensor("pwconv2.bias", &[dim])?,
            gamma: tensor("gamma", &[dim])?,
        })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        let h = conv1d(x, &self.dwconv_weight, 1, PADDING, 1, self.dim)?.add(&self.dwconv_bias)?;
        let h = layer_norm(&h, Some(&self.norm_weight), Some(&self.norm_bias), NORM_EPS)?;
        let h = addmm(&self.pwconv1_bias, &h, self.pwconv1_weight.t(), None, None)?;
        let h = gelu(&h)?;
        let h = addmm(&self.pwconv2_bias, &h, self.pwconv2_weight.t(), None, None)?;
        Ok(x.add(&self.gamma.multiply(&h)?)?)
    }
}

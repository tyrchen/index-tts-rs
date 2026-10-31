//! Stage 0 `CAMPPlus` Speaker Encoder (`raw/campplus.safetensors`).
//!
//! Takes 16 kHz Kaldi filterbank features `[B, T, 80]` (`campplus_fbank`) and produces `[B, 192]`
//! speaker style embeddings (`style`) via a 2D `ResBlock` front-end (`FCM`) and a densely connected
//! time-delay neural network (`XVector` with 3 `CAMDenseTDNNBlock` stages `[12, 24, 16]`,
//! 100-frame segment pooling, unbiased `StatsPool(ddof=1)`, and final `DenseLayer`).
//! All `BatchNorm` layers execute strictly in inference (`eval`) mode using `running_mean` and
//! `running_var`.

use std::path::Path;

use mlx_rs::{
    Array,
    nn::relu,
    ops::{
        broadcast_to, concatenate, conv1d, conv2d, expand_dims, indexing::IndexOp, mean_axis, pad,
        rsqrt, sigmoid, std_axis,
    },
};

use crate::{
    error::{IndexTtsError, Result},
    models::{FilteredWeights, ModelWeightKind, SafeTensorsLoader},
};

/// Input filterbank dimension (`80`).
pub const CAMP_FEAT_DIM: usize = 80;
/// Output speaker embedding dimension (`192`).
pub const CAMP_EMB_DIM: usize = 192;
/// `FCM` 2D residual channel count (`32`).
pub const CAMP_M_CHANNELS: usize = 32;
/// `FCM` flattened output channel count (`32 * (80 / 8) = 320`).
pub const CAMP_FCM_OUT_CHANNELS: usize = CAMP_M_CHANNELS * (CAMP_FEAT_DIM / 8);
/// Dense TDNN growth rate (`32`).
pub const CAMP_GROWTH_RATE: usize = 32;
/// Bottleneck multiplier (`4`, so `bn_channels = 128`).
pub const CAMP_BN_SIZE: usize = 4;
/// Initial `XVector` TDNN channel count (`128`).
pub const CAMP_INIT_CHANNELS: usize = 128;
/// Layer counts in the 3 `CAMDenseTDNNBlock` stages (`[12, 24, 16]`).
pub const CAMP_BLOCK_LAYERS: [usize; 3] = [12, 24, 16];
/// Kernel sizes in the 3 `CAMDenseTDNNBlock` stages (`[3, 3, 3]`).
pub const CAMP_BLOCK_KERNELS: [usize; 3] = [3, 3, 3];
/// Dilations in the 3 `CAMDenseTDNNBlock` stages (`[1, 2, 2]`).
pub const CAMP_BLOCK_DILATIONS: [usize; 3] = [1, 2, 2];
/// `BatchNorm` epsilon (`1e-5`).
pub const CAMP_BN_EPS: f32 = 1e-5;
/// Segment pooling window length in frames (`100`).
pub const CAMP_SEG_LEN: usize = 100;

#[derive(Debug, Clone)]
struct EvalBatchNorm {
    running_mean: Array,
    running_var: Array,
    affine: Option<(Array, Array)>,
    apply_relu: bool,
}

impl EvalBatchNorm {
    fn from_weights(
        weights: &FilteredWeights,
        prefix: &str,
        channels: usize,
        has_affine: bool,
        apply_relu: bool,
    ) -> Result<Self> {
        let running_mean =
            weights.tensor_array_shaped(&format!("{prefix}.running_mean"), &[channels])?;
        let running_var =
            weights.tensor_array_shaped(&format!("{prefix}.running_var"), &[channels])?;
        let affine = if has_affine {
            let weight = weights.tensor_array_shaped(&format!("{prefix}.weight"), &[channels])?;
            let bias = weights.tensor_array_shaped(&format!("{prefix}.bias"), &[channels])?;
            Some((weight, bias))
        } else {
            None
        };
        Ok(Self {
            running_mean,
            running_var,
            affine,
            apply_relu,
        })
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(&self, x: &Array) -> Result<Array> {
        let inv_std = rsqrt(&self.running_var.add(Array::from_f32(CAMP_BN_EPS))?)?;
        let mut h = x.subtract(&self.running_mean)?.multiply(&inv_std)?;
        if let Some((weight, bias)) = &self.affine {
            h = weight.multiply(&h)?.add(bias)?;
        }
        if self.apply_relu {
            Ok(relu(&h)?)
        } else {
            Ok(h)
        }
    }
}

#[derive(Debug, Clone)]
struct BasicResBlock {
    conv1_weight: Array,
    bn1: EvalBatchNorm,
    conv2_weight: Array,
    bn2: EvalBatchNorm,
    shortcut: Option<(Array, EvalBatchNorm)>,
    stride: i32,
}

impl BasicResBlock {
    fn from_weights(
        weights: &FilteredWeights,
        prefix: &str,
        channels: usize,
        stride: i32,
    ) -> Result<Self> {
        let conv1_weight = weights.tensor_array_shaped(
            &format!("{prefix}.conv1.weight"),
            &[channels, 3, 3, channels],
        )?;
        let bn1 =
            EvalBatchNorm::from_weights(weights, &format!("{prefix}.bn1"), channels, true, true)?;
        let conv2_weight = weights.tensor_array_shaped(
            &format!("{prefix}.conv2.weight"),
            &[channels, 3, 3, channels],
        )?;
        let bn2 =
            EvalBatchNorm::from_weights(weights, &format!("{prefix}.bn2"), channels, true, false)?;

        let shortcut = if stride == 1 {
            None
        } else {
            let sc_conv = weights.tensor_array_shaped(
                &format!("{prefix}.shortcut.0.weight"),
                &[channels, 1, 1, channels],
            )?;
            let sc_bn = EvalBatchNorm::from_weights(
                weights,
                &format!("{prefix}.shortcut.1"),
                channels,
                true,
                false,
            )?;
            Some((sc_conv, sc_bn))
        };

        Ok(Self {
            conv1_weight,
            bn1,
            conv2_weight,
            bn2,
            shortcut,
            stride,
        })
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(&self, x: &Array) -> Result<Array> {
        let out = conv2d(x, &self.conv1_weight, (self.stride, 1), (1, 1), (1, 1), 1)?;
        let out = self.bn1.forward(&out)?;
        let out = conv2d(&out, &self.conv2_weight, (1, 1), (1, 1), (1, 1), 1)?;
        let out = self.bn2.forward(&out)?;

        let residual = match &self.shortcut {
            Some((sc_conv, sc_bn)) => {
                let sc = conv2d(x, sc_conv, (self.stride, 1), (0, 0), (1, 1), 1)?;
                sc_bn.forward(&sc)?
            }
            None => x.clone(),
        };

        Ok(relu(&out.add(&residual)?)?)
    }
}

#[derive(Debug, Clone)]
struct FcmFrontEnd {
    conv1_weight: Array,
    bn1: EvalBatchNorm,
    layer1: [BasicResBlock; 2],
    layer2: [BasicResBlock; 2],
    conv2_weight: Array,
    bn2: EvalBatchNorm,
}

impl FcmFrontEnd {
    fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let conv1_weight =
            weights.tensor_array_shaped("head.conv1.weight", &[CAMP_M_CHANNELS, 3, 3, 1])?;
        let bn1 = EvalBatchNorm::from_weights(weights, "head.bn1", CAMP_M_CHANNELS, true, true)?;
        let layer1 = [
            BasicResBlock::from_weights(weights, "head.layer1.0", CAMP_M_CHANNELS, 2)?,
            BasicResBlock::from_weights(weights, "head.layer1.1", CAMP_M_CHANNELS, 1)?,
        ];
        let layer2 = [
            BasicResBlock::from_weights(weights, "head.layer2.0", CAMP_M_CHANNELS, 2)?,
            BasicResBlock::from_weights(weights, "head.layer2.1", CAMP_M_CHANNELS, 1)?,
        ];
        let conv2_weight = weights.tensor_array_shaped(
            "head.conv2.weight",
            &[CAMP_M_CHANNELS, 3, 3, CAMP_M_CHANNELS],
        )?;
        let bn2 = EvalBatchNorm::from_weights(weights, "head.bn2", CAMP_M_CHANNELS, true, true)?;

        Ok(Self {
            conv1_weight,
            bn1,
            layer1,
            layer2,
            conv2_weight,
            bn2,
        })
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(&self, x: &Array) -> Result<Array> {
        // x: [B, T, F] -> transpose [B, F, T] -> expand_dims [B, F, T, 1]
        let mut out = expand_dims(&x.transpose_axes(&[0, 2, 1])?, -1)?;
        out = conv2d(&out, &self.conv1_weight, (1, 1), (1, 1), (1, 1), 1)?;
        out = self.bn1.forward(&out)?;
        for block in &self.layer1 {
            out = block.forward(&out)?;
        }
        for block in &self.layer2 {
            out = block.forward(&out)?;
        }
        out = conv2d(&out, &self.conv2_weight, (2, 1), (1, 1), (1, 1), 1)?;
        out = self.bn2.forward(&out)?;

        // out: [B, H=F/8, T, C] -> transpose(0, 3, 1, 2) [B, C, H, T] -> reshape [B, C*H, T] ->
        // transpose [B, T, C*H]
        let b = out.dim(0);
        let h = out.dim(1);
        let t = out.dim(2);
        let c = out.dim(3);
        Ok(out
            .transpose_axes(&[0, 3, 1, 2])?
            .reshape(&[b, c * h, t])?
            .transpose_axes(&[0, 2, 1])?)
    }
}

/// Computes `CAMLayer.seg_pooling`: 100-frame average pooling (handling partial tail segment)
/// broadcast back to `[B, T, C]`.
///
/// # Errors
///
/// Returns [`IndexTtsError::Mlx`] if tensor reshape/pad operations fail.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
pub fn seg_pool_100(x: &Array) -> Result<Array> {
    let b = x.dim(0);
    let t = x.dim(1);
    let c = x.dim(2);
    let t_usize = usize::try_from(t).unwrap_or(0);
    let seg_len = CAMP_SEG_LEN;
    let nseg = t_usize.div_ceil(seg_len);
    let padded_len = nseg * seg_len;
    let pad_right = (padded_len - t_usize) as i32;

    let x_padded = if pad_right > 0 {
        pad(x, &[(0, 0), (0, pad_right), (0, 0)], None, None)?
    } else {
        x.clone()
    };

    let nseg_i32 = nseg as i32;
    let seg_i32 = seg_len as i32;
    let sums = x_padded
        .reshape(&[b, nseg_i32, seg_i32, c])?
        .sum_axis(2, false)?;

    let mut counts_vec = Vec::with_capacity(nseg);
    for seg_idx in 0..nseg {
        let start = seg_idx * seg_len;
        let valid = (t_usize - start).min(seg_len);
        counts_vec.push(valid as f32);
    }
    let counts = Array::from_slice(&counts_vec, &[1, nseg_i32, 1]);
    let means = sums.divide(&counts)?.reshape(&[b, nseg_i32, 1, c])?;
    let broadcasted =
        broadcast_to(&means, &[b, nseg_i32, seg_i32, c])?.reshape(&[b, nseg_i32 * seg_i32, c])?;

    Ok(broadcasted.index((.., ..t, ..)))
}

#[derive(Debug, Clone)]
struct CamLayer {
    linear_local_weight: Array,
    linear1_weight: Array,
    linear1_bias: Array,
    linear2_weight: Array,
    linear2_bias: Array,
    padding: i32,
    dilation: i32,
}

impl CamLayer {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn from_weights(
        weights: &FilteredWeights,
        prefix: &str,
        bn_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Result<Self> {
        let reduced = bn_channels / 2;
        let padding = (((kernel_size - 1) / 2) * dilation) as i32;
        Ok(Self {
            linear_local_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear_local.weight"),
                &[out_channels, kernel_size, bn_channels],
            )?,
            linear1_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear1.weight"),
                &[reduced, 1, bn_channels],
            )?,
            linear1_bias: weights
                .tensor_array_shaped(&format!("{prefix}.linear1.bias"), &[reduced])?,
            linear2_weight: weights.tensor_array_shaped(
                &format!("{prefix}.linear2.weight"),
                &[out_channels, 1, reduced],
            )?,
            linear2_bias: weights
                .tensor_array_shaped(&format!("{prefix}.linear2.bias"), &[out_channels])?,
            padding,
            dilation: dilation as i32,
        })
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(&self, x: &Array) -> Result<Array> {
        let y = conv1d(
            x,
            &self.linear_local_weight,
            1,
            self.padding,
            self.dilation,
            1,
        )?;
        let global_mean = mean_axis(x, 1, true)?;
        let seg_means = seg_pool_100(x)?;
        let context = global_mean.add(&seg_means)?;
        let context =
            relu(&conv1d(&context, &self.linear1_weight, 1, 0, 1, 1)?.add(&self.linear1_bias)?)?;
        let mask =
            sigmoid(&conv1d(&context, &self.linear2_weight, 1, 0, 1, 1)?.add(&self.linear2_bias)?)?;
        Ok(y.multiply(&mask)?)
    }
}

#[derive(Debug, Clone)]
struct CamDenseTdnnLayer {
    nonlinear1: EvalBatchNorm,
    linear1_weight: Array,
    nonlinear2: EvalBatchNorm,
    cam_layer: CamLayer,
}

impl CamDenseTdnnLayer {
    fn from_weights(
        weights: &FilteredWeights,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        bn_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Result<Self> {
        let nonlinear1 = EvalBatchNorm::from_weights(
            weights,
            &format!("{prefix}.nonlinear1.batchnorm"),
            in_channels,
            true,
            true,
        )?;
        let linear1_weight = weights.tensor_array_shaped(
            &format!("{prefix}.linear1.weight"),
            &[bn_channels, 1, in_channels],
        )?;
        let nonlinear2 = EvalBatchNorm::from_weights(
            weights,
            &format!("{prefix}.nonlinear2.batchnorm"),
            bn_channels,
            true,
            true,
        )?;
        let cam_layer = CamLayer::from_weights(
            weights,
            &format!("{prefix}.cam_layer"),
            bn_channels,
            out_channels,
            kernel_size,
            dilation,
        )?;
        Ok(Self {
            nonlinear1,
            linear1_weight,
            nonlinear2,
            cam_layer,
        })
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.nonlinear1.forward(x)?;
        let h = conv1d(&h, &self.linear1_weight, 1, 0, 1, 1)?;
        let h = self.nonlinear2.forward(&h)?;
        self.cam_layer.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct CamDenseTdnnBlock {
    layers: Vec<CamDenseTdnnLayer>,
}

impl CamDenseTdnnBlock {
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        weights: &FilteredWeights,
        prefix: &str,
        num_layers: usize,
        in_channels: usize,
        out_channels: usize,
        bn_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_prefix = format!("{prefix}.tdnnd{}", i + 1);
            let ch_in = in_channels + i * out_channels;
            layers.push(CamDenseTdnnLayer::from_weights(
                weights,
                &layer_prefix,
                ch_in,
                out_channels,
                bn_channels,
                kernel_size,
                dilation,
            )?);
        }
        Ok(Self { layers })
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut cur = x.clone();
        for layer in &self.layers {
            let y = layer.forward(&cur)?;
            cur = concatenate(&[&cur, &y], -1)?;
        }
        Ok(cur)
    }
}

#[derive(Debug, Clone)]
struct TransitLayer {
    nonlinear: EvalBatchNorm,
    linear_weight: Array,
}

impl TransitLayer {
    fn from_weights(
        weights: &FilteredWeights,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        let nonlinear = EvalBatchNorm::from_weights(
            weights,
            &format!("{prefix}.nonlinear.batchnorm"),
            in_channels,
            true,
            true,
        )?;
        let linear_weight = weights.tensor_array_shaped(
            &format!("{prefix}.linear.weight"),
            &[out_channels, 1, in_channels],
        )?;
        Ok(Self {
            nonlinear,
            linear_weight,
        })
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.nonlinear.forward(x)?;
        Ok(conv1d(&h, &self.linear_weight, 1, 0, 1, 1)?)
    }
}

/// Stage 0 `CAMPPlus` Speaker Encoder (`raw/campplus.safetensors`).
#[derive(Debug, Clone)]
pub struct CampPlus {
    head: FcmFrontEnd,
    tdnn_linear_weight: Array,
    tdnn_nonlinear: EvalBatchNorm,
    blocks: Vec<(CamDenseTdnnBlock, TransitLayer)>,
    out_nonlinear: EvalBatchNorm,
    dense_linear_weight: Array,
    dense_nonlinear: EvalBatchNorm,
}

impl CampPlus {
    /// Constructs a [`CampPlus`] speaker encoder from pre-filtered weights.
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
    ///     CampPlus, ModelWeightKind, SafeTensorsLoader, resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let weights = SafeTensorsLoader::load_from_path(
    ///         ModelWeightKind::CampPlus,
    ///         dir.join("raw/campplus.safetensors"),
    ///     )?;
    ///     let camp = CampPlus::from_weights(&weights)?;
    ///     assert_eq!(camp.num_dense_blocks(), 3);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let head = FcmFrontEnd::from_weights(weights)?;

        let tdnn_linear_weight = weights.tensor_array_shaped(
            "xvector.tdnn.linear.weight",
            &[CAMP_INIT_CHANNELS, 5, CAMP_FCM_OUT_CHANNELS],
        )?;
        let tdnn_nonlinear = EvalBatchNorm::from_weights(
            weights,
            "xvector.tdnn.nonlinear.batchnorm",
            CAMP_INIT_CHANNELS,
            true,
            true,
        )?;

        let mut channels = CAMP_INIT_CHANNELS;
        let bn_channels = CAMP_BN_SIZE * CAMP_GROWTH_RATE;
        let mut blocks = Vec::with_capacity(CAMP_BLOCK_LAYERS.len());

        for (idx, ((&num_layers, &kernel_size), &dilation)) in CAMP_BLOCK_LAYERS
            .iter()
            .zip(CAMP_BLOCK_KERNELS.iter())
            .zip(CAMP_BLOCK_DILATIONS.iter())
            .enumerate()
        {
            let block_idx = idx + 1;
            let block = CamDenseTdnnBlock::from_weights(
                weights,
                &format!("xvector.block{block_idx}"),
                num_layers,
                channels,
                CAMP_GROWTH_RATE,
                bn_channels,
                kernel_size,
                dilation,
            )?;
            channels += num_layers * CAMP_GROWTH_RATE;
            let out_ch = channels / 2;
            let transit = TransitLayer::from_weights(
                weights,
                &format!("xvector.transit{block_idx}"),
                channels,
                out_ch,
            )?;
            channels = out_ch;
            blocks.push((block, transit));
        }

        let out_nonlinear = EvalBatchNorm::from_weights(
            weights,
            "xvector.out_nonlinear.batchnorm",
            channels,
            true,
            true,
        )?;

        let stats_channels = channels * 2;
        let dense_linear_weight = weights.tensor_array_shaped(
            "xvector.dense.linear.weight",
            &[CAMP_EMB_DIM, 1, stats_channels],
        )?;
        let dense_nonlinear = EvalBatchNorm::from_weights(
            weights,
            "xvector.dense.nonlinear.batchnorm",
            CAMP_EMB_DIM,
            false,
            false,
        )?;

        Ok(Self {
            head,
            tdnn_linear_weight,
            tdnn_nonlinear,
            blocks,
            out_nonlinear,
            dense_linear_weight,
            dense_nonlinear,
        })
    }

    /// Loads [`CampPlus`] weights directly from `raw/campplus.safetensors` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if the file cannot be read or tensors fail validation.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let weights = SafeTensorsLoader::load_from_path(ModelWeightKind::CampPlus, path)?;
        Self::from_weights(&weights)
    }

    /// Returns the number of dense TDNN blocks (`3`).
    #[must_use]
    pub fn num_dense_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Encodes `[B, T, 80]` Kaldi filterbank features into `[B, 192]` speaker style vectors.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `fbank` is not `[B, T, 80]` with `T >= 4`, or
    /// [`IndexTtsError::Mlx`] if GPU execution fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_rs::Array;
    /// use index_tts_rs_core::models::{CampPlus, resolve_model_dir};
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let camp = CampPlus::load_from_path(dir.join("raw/campplus.safetensors"))?;
    ///     let fbank = Array::zeros::<f32>(&[1, 32, 80])?;
    ///     let style = camp.forward(&fbank)?;
    ///     assert_eq!(style.shape(), &[1, 192]);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn forward(&self, fbank: &Array) -> Result<Array> {
        if fbank.ndim() != 3 {
            return Err(IndexTtsError::validation(
                "fbank",
                format!("expected 3D tensor [B, T, 80], got {:?}", fbank.shape()),
            ));
        }
        let b = fbank.dim(0);
        let t = fbank.dim(1);
        let f_dim = fbank.dim(2);
        if b <= 0 || t < 4 {
            return Err(IndexTtsError::validation(
                "fbank",
                format!("batch must be >= 1 and T >= 4, got B={b}, T={t}"),
            ));
        }
        if usize::try_from(f_dim).unwrap_or(0) != CAMP_FEAT_DIM {
            return Err(IndexTtsError::validation(
                "fbank",
                format!("expected {CAMP_FEAT_DIM} mel bins, got {f_dim}"),
            ));
        }

        let mut x = self.head.forward(fbank)?;
        // TDNN: Conv1d(in=320, out=128, k=5, stride=2, pad=2) + BatchNorm + ReLU
        x = conv1d(&x, &self.tdnn_linear_weight, 2, 2, 1, 1)?;
        x = self.tdnn_nonlinear.forward(&x)?;

        for (block, transit) in &self.blocks {
            x = block.forward(&x)?;
            x = transit.forward(&x)?;
        }

        x = self.out_nonlinear.forward(&x)?;

        // StatsPool: mean(axis=1) + std(axis=1, ddof=1) -> [B, 2*C]
        let mean = mean_axis(&x, 1, false)?;
        let std = std_axis(&x, 1, false, 1)?;
        let stats = concatenate(&[&mean, &std], -1)?;

        // DenseLayer: expand_dims(axis=1) -> Conv1d(k=1) -> squeeze(axis=1) ->
        // BatchNorm(affine=False)
        let stats_3d = expand_dims(&stats, 1)?;
        let dense_3d = conv1d(&stats_3d, &self.dense_linear_weight, 1, 0, 1, 1)?;
        let dense_2d = dense_3d.index((.., 0, ..));
        let out = self.dense_nonlinear.forward(&dense_2d)?;
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
        config::{DEFAULT_MODEL_REPO, EmoAlpha},
        dsp::cosine_similarity_and_max_diff,
        models::{SpkProjWeights, resolve_model_dir},
        voice::SpeakerContext,
    };

    #[test]
    fn test_should_match_python_reference_cp08_campplus_and_conds() {
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

        let camp_res = CampPlus::load_from_path(model_dir.join("raw/campplus.safetensors"));
        assert!(camp_res.is_ok(), "failed to load CampPlus: {camp_res:?}");
        let Ok(camp) = camp_res else {
            return;
        };

        let (fbank_shape, fbank_data) = extract_f32(&st, "campplus_fbank");
        let (expected_style_shape, expected_style) = extract_f32(&st, "style");
        let (_, expected_emovec) = extract_f32(&st, "emovec");
        let (_, expected_conds) = extract_f32(&st, "conds");

        let fbank_shape_i32: Vec<i32> = fbank_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        let fbank_arr = Array::from_slice(&fbank_data, &fbank_shape_i32);

        let style_res = camp.forward(&fbank_arr);
        assert!(style_res.is_ok(), "CampPlus forward failed: {style_res:?}");
        let Ok(style_arr) = style_res else {
            return;
        };

        let expected_shape_i32: Vec<i32> = expected_style_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        assert_eq!(style_arr.shape(), expected_shape_i32.as_slice());

        let rust_style = style_arr.try_as_slice::<f32>().unwrap_or(&[]);
        let (style_cos, style_diff) = cosine_similarity_and_max_diff(rust_style, &expected_style);
        assert!(
            style_cos >= 0.999_9,
            "CP-08 style cosine similarity {style_cos} < 0.9999"
        );
        assert!(
            style_diff < 1e-3,
            "CP-08 style max_abs_diff {style_diff} >= 1e-3"
        );

        // Verify spk_proj + emovec -> conds [1, 3, 1280]
        let Ok(spk_proj) = SpkProjWeights::load_from_path(model_dir.join("spk_proj.npz")) else {
            return;
        };
        let Ok(spk_cond) = spk_proj.project(rust_style) else {
            return;
        };
        let Ok(rust_conds) =
            SpeakerContext::build_conds(&spk_cond, &expected_emovec, EmoAlpha::default())
        else {
            return;
        };
        let (conds_cos, conds_diff) = cosine_similarity_and_max_diff(&rust_conds, &expected_conds);
        assert!(
            conds_cos >= 0.999_9,
            "CP-08 conds cosine similarity {conds_cos} < 0.9999"
        );
        assert!(
            conds_diff < 1e-3,
            "CP-08 conds max_abs_diff {conds_diff} >= 1e-3"
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

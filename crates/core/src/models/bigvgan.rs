//! `BigVGANv2` neural vocoder (`raw/bigvgan.safetensors`) with anti-aliased `SnakeBeta`
//! activations, deduplicated Kaiser-Sinc filters, phase-decomposed `ConvTranspose1d`, and
//! 16-channel-aligned `F32` Metal convolution kernels.
//!
//! Converts `[B, 80, T_mel]` Mel spectrograms into `[B, 1, 256 * T_mel]` (or `[256 * T_mel]`)
//! 22.05 kHz audio waveforms.

use std::path::Path;

use mlx_rs::{
    Array, array,
    ops::{
        PadMode, broadcast_to, clip, concatenate, conv1d, indexing::TryIndexOp, pad, sin, zeros,
    },
};

use crate::{
    error::{IndexTtsError, Result},
    models::{FilteredWeights, ModelWeightKind, SafeTensorsLoader},
};

/// Input Mel channel count (`80`).
pub const BIGVGAN_NUM_MELS: usize = 80;
/// Initial upsampling channel dimension (`1536`).
pub const BIGVGAN_UP_INIT_CH: usize = 1_536;
/// Upsampling stride rates per stage (`[4, 4, 2, 2, 2, 2]`, product = `256`).
pub const BIGVGAN_UP_RATES: [usize; 6] = [4, 4, 2, 2, 2, 2];
/// Upsampling `ConvTranspose1d` kernel sizes per stage (`[8, 8, 4, 4, 4, 4]`).
pub const BIGVGAN_UP_KERNELS: [usize; 6] = [8, 8, 4, 4, 4, 4];
/// AMP residual block kernel sizes (`[3, 7, 11]`).
pub const BIGVGAN_RES_KERNELS: [usize; 3] = [3, 7, 11];
/// AMP residual block dilation factors (`[1, 3, 5]`).
pub const BIGVGAN_RES_DILATIONS: [usize; 3] = [1, 3, 5];
/// Deduplicated Kaiser-Sinc filter tap count (`12`).
pub const BIGVGAN_FILTER_LEN: usize = 12;
/// Total hop length / temporal upsampling factor (`256`).
pub const BIGVGAN_HOP_LENGTH: usize = 256;

const SNAKE_EPS: f32 = 1e-9;

/// `Conv1d` wrapper that pads channel dimensions to a multiple of 16 so MLX's Metal backend
/// accumulates in full `F32` precision instead of falling back to reduced precision on odd
/// channel counts (e.g. `C=24`, `C=80`, `C=1`).
#[derive(Debug, Clone)]
struct Conv1dF32 {
    weight_padded: Array,
    bias: Option<Array>,
    out_channels: i32,
    pad_in: i32,
    pad_out: i32,
    stride: i32,
    padding: i32,
    dilation: i32,
}

impl Conv1dF32 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn new(
        weight: Array,
        bias: Option<Array>,
        stride: i32,
        padding: i32,
        dilation: i32,
    ) -> Result<Self> {
        let co = weight.dim(0);
        let k = weight.dim(1);
        let ci = weight.dim(2);
        let pad_in = (16 - (ci % 16)) % 16;
        let pad_out = (16 - (co % 16)) % 16;

        let mut weight_padded = weight;
        if pad_in > 0 {
            let z = zeros::<f32>(&[co, k, pad_in])?;
            weight_padded = concatenate(&[&weight_padded, &z], 2)?;
        }
        if pad_out > 0 {
            let z = zeros::<f32>(&[pad_out, k, ci + pad_in])?;
            weight_padded = concatenate(&[&weight_padded, &z], 0)?;
        }
        weight_padded.eval()?;

        Ok(Self {
            weight_padded,
            bias,
            out_channels: co,
            pad_in,
            pad_out,
            stride,
            padding,
            dilation,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x_in = if self.pad_in > 0 {
            pad(
                x,
                &[(0, 0), (0, 0), (0, self.pad_in)],
                None,
                PadMode::Constant,
            )?
        } else {
            x.clone()
        };

        let mut y = conv1d(
            &x_in,
            &self.weight_padded,
            self.stride,
            self.padding,
            self.dilation,
            1,
        )?;
        if self.pad_out > 0 {
            y = y.try_index((.., .., ..self.out_channels))?;
        }
        if let Some(bias) = &self.bias {
            y = y.add(bias)?;
        }
        Ok(y)
    }
}

/// Phase-decomposed `ConvTranspose1d` evaluated via `stride` stride-1 [`Conv1dF32`] phase
/// convolutions.
#[derive(Debug, Clone)]
struct ConvTranspose1d {
    phases: Vec<(i32, i32, Conv1dF32)>,
    bias: Array,
    stride: i32,
    out_channels: i32,
}

impl ConvTranspose1d {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn new(weight: &Array, bias: Array, kernel_size: usize, stride: usize) -> Result<Self> {
        let s = stride as i32;
        let k = kernel_size as i32;
        let p_pad = (k - s) / 2;
        let out_channels = weight.dim(0);

        let mut phases = Vec::with_capacity(stride);
        for p in 0..s {
            let r = (p + p_pad) % s;
            let q = (p + p_pad - r) / s;
            let mut ks: Vec<i32> = (r..k).step_by(stride).collect();
            ks.reverse();
            let ks_len = i32::try_from(ks.len()).unwrap_or(0);
            let left = ks_len - 1 - q;
            let right = q;
            let ks_arr = Array::from_slice(&ks, &[ks_len]);
            let hp = weight.take_axis(&ks_arr, 1)?;
            let phase_conv = Conv1dF32::new(hp, None, 1, 0, 1)?;
            phases.push((left, right, phase_conv));
        }

        Ok(Self {
            phases,
            bias,
            stride: s,
            out_channels,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let b = x.dim(0);
        let t = x.dim(1);

        let mut outs = Vec::with_capacity(self.phases.len());
        for (left, right, phase_conv) in &self.phases {
            let xp = pad(
                x,
                &[(0, 0), (*left, *right), (0, 0)],
                None,
                PadMode::Constant,
            )?;
            let y_p = phase_conv
                .forward(&xp)?
                .reshape(&[b, t, 1, self.out_channels])?;
            outs.push(y_p);
        }

        let refs: Vec<&Array> = outs.iter().collect();
        let stacked = concatenate(&refs, 2)?.reshape(&[b, t * self.stride, self.out_channels])?;
        Ok(stacked.add(&self.bias)?)
    }
}

/// Precomputed 2x Kaiser-Sinc anti-aliased upsampling (`w_up_even`, `w_up_odd` 6-tap polyphase
/// filters) and downsampling (`w_dn` 12-tap filter) depthwise filters for a given channel count
/// `C`.
#[derive(Debug, Clone)]
struct KaiserResampleFilters {
    channels: i32,
    w_up_even: Array,
    w_up_odd: Array,
    w_dn: Array,
}

impl KaiserResampleFilters {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn new(base_filter_12: &[f32], channels: usize) -> Result<Self> {
        let c_i32 = channels as i32;
        let half_len = BIGVGAN_FILTER_LEN / 2;
        let mut up_12 = [0.0_f32; BIGVGAN_FILTER_LEN];
        for (dst, &src) in up_12.iter_mut().zip(base_filter_12.iter().rev()) {
            *dst = 2.0_f32 * src;
        }
        let mut up_even = [0.0_f32; BIGVGAN_FILTER_LEN / 2];
        let mut up_odd = [0.0_f32; BIGVGAN_FILTER_LEN / 2];
        for idx in 0..half_len {
            if let Some(&ev) = up_12.get(2 * idx)
                && let Some(slot) = up_even.get_mut(idx)
            {
                *slot = ev;
            }
            if let Some(&od) = up_12.get(2 * idx + 1)
                && let Some(slot) = up_odd.get_mut(idx)
            {
                *slot = od;
            }
        }
        let w_up_even_1 = Array::from_slice(&up_even, &[1, half_len as i32, 1]);
        let w_up_odd_1 = Array::from_slice(&up_odd, &[1, half_len as i32, 1]);
        let w_dn_1 = Array::from_slice(base_filter_12, &[1, BIGVGAN_FILTER_LEN as i32, 1]);

        let w_up_even = broadcast_to(&w_up_even_1, &[c_i32, half_len as i32, 1])?.contiguous()?;
        let w_up_odd = broadcast_to(&w_up_odd_1, &[c_i32, half_len as i32, 1])?.contiguous()?;
        let w_dn = broadcast_to(&w_dn_1, &[c_i32, BIGVGAN_FILTER_LEN as i32, 1])?.contiguous()?;
        w_up_even.eval()?;
        w_up_odd.eval()?;
        w_dn.eval()?;

        Ok(Self {
            channels: c_i32,
            w_up_even,
            w_up_odd,
            w_dn,
        })
    }

    fn upsample_2x(&self, x: &Array) -> Result<Array> {
        let batch = x.dim(0);
        let time = x.dim(1);
        let channels = self.channels;

        let xp = pad(x, &[(0, 0), (3, 3), (0, 0)], None, PadMode::Edge)?;
        let xp_even = xp.try_index((.., ..(time + 5), ..))?;
        let xp_odd = xp.try_index((.., 1..(time + 6), ..))?;

        let y_even = conv1d(&xp_even, &self.w_up_even, 1, 0, 1, channels)?
            .reshape(&[batch, time, 1, channels])?;
        let y_odd = conv1d(&xp_odd, &self.w_up_odd, 1, 0, 1, channels)?
            .reshape(&[batch, time, 1, channels])?;

        Ok(concatenate(&[&y_even, &y_odd], 2)?.reshape(&[batch, 2 * time, channels])?)
    }

    fn downsample_2x(&self, x: &Array) -> Result<Array> {
        let xp = pad(x, &[(0, 0), (5, 6), (0, 0)], None, PadMode::Edge)?;
        Ok(conv1d(&xp, &self.w_dn, 2, 0, 1, self.channels)?)
    }
}

/// Anti-aliased `Activation1d(SnakeBeta)`: 2x Kaiser-Sinc upsample -> `SnakeBeta` -> 2x
/// Kaiser-Sinc downsample.
#[derive(Debug, Clone)]
struct Activation1d {
    exp_alpha: Array,
    inv_beta: Array,
}

impl Activation1d {
    fn from_weights(weights: &FilteredWeights, prefix: &str, channels: usize) -> Result<Self> {
        let alpha = weights.tensor_array_shaped(&format!("{prefix}.act.alpha"), &[channels])?;
        let beta = weights.tensor_array_shaped(&format!("{prefix}.act.beta"), &[channels])?;
        let exp_alpha = alpha.exp()?;
        let inv_beta = array!(1.0_f32).divide(&beta.exp()?.add(&array!(SNAKE_EPS))?)?;
        exp_alpha.eval()?;
        inv_beta.eval()?;
        Ok(Self {
            exp_alpha,
            inv_beta,
        })
    }

    fn forward(&self, x: &Array, resample: &KaiserResampleFilters) -> Result<Array> {
        let x_up = resample.upsample_2x(x)?;
        let sin_xa = sin(&x_up.multiply(&self.exp_alpha)?)?;
        let sin_sq = sin_xa.square()?;
        let activated = x_up.add(&sin_sq.multiply(&self.inv_beta)?)?;
        resample.downsample_2x(&activated)
    }
}

/// Single `AMPBlock1` residual block with 3 dilated `(Activation1d + Conv1d, Activation1d +
/// Conv1d)` branches.
#[derive(Debug, Clone)]
struct AmpBlock1 {
    branches: Vec<(Activation1d, Conv1dF32, Activation1d, Conv1dF32)>,
}

impl AmpBlock1 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn from_weights(
        weights: &FilteredWeights,
        prefix: &str,
        channels: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        let k_i32 = kernel_size as i32;
        let mut branches = Vec::with_capacity(BIGVGAN_RES_DILATIONS.len());
        for (idx, &dil) in BIGVGAN_RES_DILATIONS.iter().enumerate() {
            let d_i32 = dil as i32;
            let pad_d = (k_i32 * d_i32 - d_i32) / 2;
            let pad_1 = (k_i32 - 1) / 2;

            let act1 = Activation1d::from_weights(
                weights,
                &format!("{prefix}.activations.{}", 2 * idx),
                channels,
            )?;
            let c1_w = weights.tensor_array_shaped(
                &format!("{prefix}.convs1.{idx}.weight"),
                &[channels, kernel_size, channels],
            )?;
            let c1_b =
                weights.tensor_array_shaped(&format!("{prefix}.convs1.{idx}.bias"), &[channels])?;
            let conv1 = Conv1dF32::new(c1_w, Some(c1_b), 1, pad_d, d_i32)?;

            let act2 = Activation1d::from_weights(
                weights,
                &format!("{prefix}.activations.{}", 2 * idx + 1),
                channels,
            )?;
            let c2_w = weights.tensor_array_shaped(
                &format!("{prefix}.convs2.{idx}.weight"),
                &[channels, kernel_size, channels],
            )?;
            let c2_b =
                weights.tensor_array_shaped(&format!("{prefix}.convs2.{idx}.bias"), &[channels])?;
            let conv2 = Conv1dF32::new(c2_w, Some(c2_b), 1, pad_1, 1)?;

            branches.push((act1, conv1, act2, conv2));
        }

        Ok(Self { branches })
    }

    fn forward(&self, x: &Array, resample: &KaiserResampleFilters) -> Result<Array> {
        let mut h = x.clone();
        for (act1, conv1, act2, conv2) in &self.branches {
            let xt = act1.forward(&h, resample)?;
            let xt = conv1.forward(&xt)?;
            let xt = act2.forward(&xt, resample)?;
            let xt = conv2.forward(&xt)?;
            h = h.add(&xt)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
struct UpsampleStage {
    up: ConvTranspose1d,
    resample: KaiserResampleFilters,
    resblocks: Vec<AmpBlock1>,
}

/// `BigVGANv2` neural vocoder (`raw/bigvgan.safetensors`).
#[derive(Debug, Clone)]
pub struct BigVgan {
    conv_pre: Conv1dF32,
    stages: Vec<UpsampleStage>,
    act_post: Activation1d,
    post_resample: KaiserResampleFilters,
    conv_post: Conv1dF32,
    inv_num_kernels: Array,
}

impl BigVgan {
    /// Constructs a [`BigVgan`] vocoder from pre-filtered weights.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if any required tensor or Kaiser-Sinc buffer is
    /// missing or malformed.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::{
    ///     BigVgan, ModelWeightKind, SafeTensorsLoader, resolve_model_dir,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let weights = SafeTensorsLoader::load_from_path(
    ///         ModelWeightKind::BigVgan,
    ///         dir.join("raw/bigvgan.safetensors"),
    ///     )?;
    ///     let vocoder = BigVgan::from_weights(&weights)?;
    ///     assert_eq!(vocoder.num_stages(), 6);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[allow(clippy::cast_precision_loss)]
    pub fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let filter_tensor = weights.buffers.get("_kaiser_sinc_filter").ok_or_else(|| {
            IndexTtsError::weight_load(
                "raw/bigvgan.safetensors",
                "missing deduplicated `_kaiser_sinc_filter` buffer",
            )
        })?;
        let filter_slice = filter_tensor.as_f32()?;
        if filter_slice.len() != BIGVGAN_FILTER_LEN {
            return Err(IndexTtsError::weight_load(
                "_kaiser_sinc_filter",
                format!(
                    "expected {} filter taps, got {}",
                    BIGVGAN_FILTER_LEN,
                    filter_slice.len()
                ),
            ));
        }

        let pre_w = weights.tensor_array_shaped(
            "conv_pre.weight",
            &[BIGVGAN_UP_INIT_CH, 7, BIGVGAN_NUM_MELS],
        )?;
        let pre_b = weights.tensor_array_shaped("conv_pre.bias", &[BIGVGAN_UP_INIT_CH])?;
        let conv_pre = Conv1dF32::new(pre_w, Some(pre_b), 1, 3, 1)?;

        let mut stages = Vec::with_capacity(BIGVGAN_UP_RATES.len());
        let mut ch = BIGVGAN_UP_INIT_CH;
        for (stage_idx, (&rate, &up_k)) in BIGVGAN_UP_RATES
            .iter()
            .zip(BIGVGAN_UP_KERNELS.iter())
            .enumerate()
        {
            let ch_out = ch / 2;
            let up_w = weights
                .tensor_array_shaped(&format!("ups.{stage_idx}.weight"), &[ch_out, up_k, ch])?;
            let up_b = weights.tensor_array_shaped(&format!("ups.{stage_idx}.bias"), &[ch_out])?;
            let up = ConvTranspose1d::new(&up_w, up_b, up_k, rate)?;
            let resample = KaiserResampleFilters::new(filter_slice, ch_out)?;

            let mut resblocks = Vec::with_capacity(BIGVGAN_RES_KERNELS.len());
            for (k_idx, &res_k) in BIGVGAN_RES_KERNELS.iter().enumerate() {
                let rb_idx = stage_idx * BIGVGAN_RES_KERNELS.len() + k_idx;
                let block = AmpBlock1::from_weights(
                    weights,
                    &format!("resblocks.{rb_idx}"),
                    ch_out,
                    res_k,
                )?;
                resblocks.push(block);
            }

            stages.push(UpsampleStage {
                up,
                resample,
                resblocks,
            });
            ch = ch_out;
        }

        let act_post = Activation1d::from_weights(weights, "activation_post", ch)?;
        let post_resample = KaiserResampleFilters::new(filter_slice, ch)?;
        let post_w = weights.tensor_array_shaped("conv_post.weight", &[1, 7, ch])?;
        let post_b = if let Some(b) = weights.tensors.get("conv_post.bias") {
            Some(b.to_mlx_array()?)
        } else {
            None
        };
        let conv_post = Conv1dF32::new(post_w, post_b, 1, 3, 1)?;

        let inv_num_kernels = array!(1.0_f32 / (BIGVGAN_RES_KERNELS.len() as f32));

        Ok(Self {
            conv_pre,
            stages,
            act_post,
            post_resample,
            conv_post,
            inv_num_kernels,
        })
    }

    /// Loads [`BigVgan`] weights from `raw/bigvgan.safetensors` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if the file cannot be read or tensors fail validation.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let weights = SafeTensorsLoader::load_from_path(ModelWeightKind::BigVgan, path)?;
        Self::from_weights(&weights)
    }

    /// Returns the number of upsampling stages (`6`).
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Synthesizes a `[B, 1, 256 * T_mel]` audio waveform tensor from a `[B, 80, T_mel]` Mel
    /// spectrogram.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `mel` is not 3D `[B, 80, T_mel]` with `T_mel > 0`,
    /// or [`IndexTtsError::Mlx`] if GPU execution fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_rs::Array;
    /// use index_tts_rs_core::models::{BigVgan, resolve_model_dir};
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let vocoder = BigVgan::load_from_path(dir.join("raw/bigvgan.safetensors"))?;
    ///     let mel = Array::zeros::<f32>(&[1, 80, 4])?;
    ///     let wav = vocoder.forward(&mel)?;
    ///     assert_eq!(wav.shape(), &[1, 1, 1024]);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn forward(&self, mel: &Array) -> Result<Array> {
        if mel.ndim() != 3 || mel.dim(1) != BIGVGAN_NUM_MELS as i32 || mel.dim(2) <= 0 {
            return Err(IndexTtsError::validation(
                "mel",
                format!("expected [B, 80, T_mel], got shape {:?}", mel.shape()),
            ));
        }

        let mut h = mel.transpose_axes(&[0, 2, 1])?;
        h = self.conv_pre.forward(&h)?;

        for stage in &self.stages {
            h = stage.up.forward(&h)?;
            let mut acc: Option<Array> = None;
            for rb in &stage.resblocks {
                let y = rb.forward(&h, &stage.resample)?;
                acc = Some(match acc {
                    Some(prev) => prev.add(&y)?,
                    None => y,
                });
            }
            if let Some(summed) = acc {
                h = summed.multiply(&self.inv_num_kernels)?;
            }
            h.eval()?;
        }

        h = self.act_post.forward(&h, &self.post_resample)?;
        h = self.conv_post.forward(&h)?;
        h = clip(&h, (-1.0_f32, 1.0_f32))?;
        let out = h.transpose_axes(&[0, 2, 1])?.contiguous()?;
        out.eval()?;
        Ok(out)
    }

    /// Synthesizes a 1D `Vec<f32>` waveform (`256 * T_mel` samples) from a `[1, 80, T_mel]` Mel
    /// spectrogram.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if `mel.dim(0) != 1` or vocoder inference fails.
    pub fn synthesize_1d(&self, mel: &Array) -> Result<Vec<f32>> {
        if mel.dim(0) != 1 {
            return Err(IndexTtsError::validation(
                "mel",
                format!("synthesize_1d expects batch size 1, got {}", mel.dim(0)),
            ));
        }
        let wav_3d = self.forward(mel)?;
        let flat = wav_3d.reshape(&[-1])?;
        flat.eval()?;
        Ok(flat.try_as_slice::<f32>()?.to_vec())
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
    fn test_should_match_python_reference_cp14_bigvgan_waveform() {
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

        let vocoder_res = BigVgan::load_from_path(model_dir.join("raw/bigvgan.safetensors"));
        assert!(vocoder_res.is_ok());
        let Ok(vocoder) = vocoder_res else {
            return;
        };

        let (mel_shape, mel_data) = extract_f32(&st, "generated_mel");
        let (_, expected_wav) = extract_f32(&st, "wav_f32");
        let mel_shape_i32: Vec<i32> = mel_shape
            .iter()
            .map(|&d| i32::try_from(d).unwrap_or(0))
            .collect();
        let mel_arr = Array::from_slice(&mel_data, &mel_shape_i32);

        let wav_res = vocoder.synthesize_1d(&mel_arr);
        assert!(wav_res.is_ok());
        let Ok(rust_wav) = wav_res else {
            return;
        };

        assert_eq!(rust_wav.len(), expected_wav.len());
        let (cos, max_diff) = cosine_similarity_and_max_diff(&rust_wav, &expected_wav);
        let snr_db = compute_snr_db(&rust_wav, &expected_wav);
        assert!(
            cos >= 0.999_5,
            "CP-14 wav_f32 cosine similarity {cos} < 0.9995"
        );
        assert!(
            max_diff < 2e-3,
            "CP-14 wav_f32 max_abs_diff {max_diff} >= 2e-3"
        );
        assert!(snr_db > 50.0, "CP-14 wav_f32 SNR {snr_db} dB <= 50.0 dB");
    }

    fn compute_snr_db(test: &[f32], reference: &[f32]) -> f64 {
        let mut sig_pow = 0.0_f64;
        let mut err_pow = 0.0_f64;
        for (&t, &r) in test.iter().zip(reference.iter()) {
            let rf = f64::from(r);
            let d = f64::from(t) - rf;
            sig_pow += rf * rf;
            err_pow += d * d;
        }
        10.0 * (sig_pow / err_pow.max(1e-20)).log10()
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

//! 16 kHz Kaldi Povey Filterbank (`campplus_fbank` & `SeamlessM4T` `w2v_features`).
//!
//! Implements Kaldi `ExtractWindow` (DC offset removal, `0.97` preemphasis,
//! symmetric Povey window, 512-point power spectrum, and Kaldi Mel scale
//! $1127 \ln(1 + f/700)$) in pure Rust.

use std::{f64::consts::PI, sync::LazyLock};

use realfft::RealFftPlanner;

use crate::error::{IndexTtsError, Result};

/// Input sample rate for Kaldi filterbank extraction (`16000` Hz).
pub const FBANK_SAMPLE_RATE: u32 = 16_000;
/// Frame window length in samples (`25 ms @ 16 kHz = 400`).
pub const FBANK_FRAME_LEN: usize = 400;
/// Frame shift (hop) in samples (`10 ms @ 16 kHz = 160`).
pub const FBANK_FRAME_SHIFT: usize = 160;
/// Next power-of-two FFT size (`512`).
pub const FBANK_FFT_LEN: usize = 512;
/// One-sided RFFT bin count (`512 / 2 + 1 = 257`).
pub const FBANK_FREQ_BINS: usize = FBANK_FFT_LEN / 2 + 1;
/// Number of Kaldi Mel filterbank bins (`80`).
pub const FBANK_NUM_MELS: usize = 80;
/// Kaldi preemphasis coefficient (`0.97`).
pub const FBANK_PREEMPH: f64 = 0.97;
/// Kaldi Mel energy floor (`f32::EPSILON = 1.1920929e-7`).
pub const FBANK_MEL_FLOOR: f64 = 1.192_092_955_078_125e-7;
/// Temporal stacking stride for `w2v-bert-2.0` (`2`).
pub const W2V_STRIDE: usize = 2;

static POVEY_WINDOW_400: LazyLock<Vec<f64>> = LazyLock::new(|| povey_window(FBANK_FRAME_LEN));

static KALDI_MEL_FILTERS_80: LazyLock<Vec<f64>> = LazyLock::new(|| {
    kaldi_mel_filter_bank(
        FBANK_FREQ_BINS,
        FBANK_NUM_MELS,
        20.0,
        f64::from(FBANK_SAMPLE_RATE) / 2.0,
        FBANK_SAMPLE_RATE,
    )
});

/// Raw Kaldi log-Mel filterbank matrix of shape `[num_frames, 80]`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KaldiFbank {
    /// Row-major `f32` values of shape `[num_frames, 80]`.
    pub data: Vec<f32>,
    /// Number of extracted time frames (`T`).
    pub num_frames: usize,
    /// Number of Mel channels (`80`).
    pub num_mels: usize,
}

/// Utterance-mean-normalized filterbank features for `CAMPPlus` of shape `[1, T, 80]`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CampPlusFbank {
    /// Row-major `f32` values of shape `[1, num_frames, 80]`.
    pub data: Vec<f32>,
    /// Number of time frames (`T`).
    pub num_frames: usize,
    /// Number of Mel channels (`80`).
    pub num_mels: usize,
}

/// Extracted `SeamlessM4T` features and attention mask for `w2v-bert-2.0`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct W2vFeaturesOutput {
    /// Stacked filterbank features of shape `[1, num_frames, 160]`.
    pub input_features: Vec<f32>,
    /// Attention mask of shape `[1, num_frames]` (`1` = valid, `0` = padded).
    pub attention_mask: Vec<i32>,
    /// Number of strided frames (`T_even / 2`).
    pub num_frames: usize,
    /// Feature dimension per strided frame (`160`).
    pub feature_dim: usize,
}

/// Computes the 80-dimensional Kaldi Povey log-Mel filterbank from a `16 kHz` waveform.
///
/// When `scale_int16` is `true`, the input waveform is scaled by $2^{15} = 32768.0$
/// prior to window extraction (matching `SeamlessM4TFeatureExtractor`). When `false`,
/// unscaled `[-1.0, 1.0]` samples are used (matching `core/dsp.py::kaldi_fbank`).
///
/// # Errors
///
/// Returns [`IndexTtsError::Dsp`] if `audio_16k` has fewer than `400` samples (25 ms).
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::fbank::kaldi_fbank;
///
/// let audio = vec![0.05_f32; 16_000];
/// let fbank = kaldi_fbank(&audio, false)?;
/// assert_eq!(fbank.num_frames, 98);
/// assert_eq!(fbank.num_mels, 80);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn kaldi_fbank(audio_16k: &[f32], scale_int16: bool) -> Result<KaldiFbank> {
    if audio_16k.len() < FBANK_FRAME_LEN {
        return Err(IndexTtsError::Dsp {
            message: format!(
                "input 16kHz audio length {} is shorter than 1 frame ({FBANK_FRAME_LEN} samples)",
                audio_16k.len()
            ),
        });
    }

    let num_frames = 1 + (audio_16k.len() - FBANK_FRAME_LEN) / FBANK_FRAME_SHIFT;
    let scale = if scale_int16 { 32_768.0_f64 } else { 1.0_f64 };

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(FBANK_FFT_LEN);
    let mut fft_in = r2c.make_input_vec();
    let mut fft_out = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();

    let window = &*POVEY_WINDOW_400;
    let mel_filters = &*KALDI_MEL_FILTERS_80;
    let mut frame_buf = [0.0_f64; FBANK_FRAME_LEN];
    let mut power_spec = [0.0_f64; FBANK_FREQ_BINS];
    let mut out = Vec::with_capacity(num_frames.saturating_mul(FBANK_NUM_MELS));

    for frame_idx in 0..num_frames {
        let start = frame_idx.saturating_mul(FBANK_FRAME_SHIFT);
        let raw_slice = audio_16k
            .get(start..start.saturating_add(FBANK_FRAME_LEN))
            .ok_or_else(|| IndexTtsError::Dsp {
                message: "fbank frame slice out of bounds".to_owned(),
            })?;

        // 1. Scale (if int16 mode) and compute frame mean for DC offset removal
        let mut sum = 0.0_f64;
        for (dst, &s) in frame_buf.iter_mut().zip(raw_slice.iter()) {
            let val = f64::from(s) * scale;
            *dst = val;
            sum += val;
        }
        let mean = sum / (FBANK_FRAME_LEN as f64);
        for sample in &mut frame_buf {
            *sample -= mean;
        }

        // 2. Preemphasis (0.97) + Povey window into zero-padded 512-pt FFT buffer
        fft_in.fill(0.0);
        let first_val = frame_buf.first().copied().unwrap_or(0.0);
        if let (Some(dst0), Some(&w0)) = (fft_in.first_mut(), window.first()) {
            *dst0 = first_val * (1.0 - FBANK_PREEMPH) * w0;
        }
        for n in 1..FBANK_FRAME_LEN {
            if let (Some(&curr), Some(&prev), Some(&win), Some(dst)) = (
                frame_buf.get(n),
                frame_buf.get(n - 1),
                window.get(n),
                fft_in.get_mut(n),
            ) {
                *dst = (curr - FBANK_PREEMPH * prev) * win;
            }
        }

        // 3. 512-pt RFFT -> power spectrum (|X[k]|^2)
        r2c.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
            .map_err(|err| IndexTtsError::Dsp {
                message: format!("512-pt RFFT failed: {err}"),
            })?;
        for (p_slot, c) in power_spec.iter_mut().zip(fft_out.iter()) {
            *p_slot = c.re.mul_add(c.re, c.im * c.im);
        }

        // 4. Project onto 80 Kaldi Mel filters and take log
        for mel_row in mel_filters.as_chunks::<FBANK_FREQ_BINS>().0 {
            let mut energy = 0.0_f64;
            for (&w, &p) in mel_row.iter().zip(power_spec.iter()) {
                energy = w.mul_add(p, energy);
            }
            let log_e = energy.max(FBANK_MEL_FLOOR).ln() as f32;
            out.push(log_e);
        }
    }

    Ok(KaldiFbank {
        data: out,
        num_frames,
        num_mels: FBANK_NUM_MELS,
    })
}

/// Computes `campplus_fbank`: 80-dim Kaldi filterbank followed by utterance-level
/// mean normalization (`feat - mean(feat, axis=0)`), returning shape `[1, T, 80]`.
///
/// # Errors
///
/// Returns [`IndexTtsError::Dsp`] if `audio_16k` has fewer than `400` samples.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::fbank::campplus_fbank;
///
/// let audio = vec![0.02_f32; 16_000];
/// let feat = campplus_fbank(&audio)?;
/// assert_eq!(feat.num_frames, 98);
/// assert_eq!(feat.data.len(), 98 * 80);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
#[allow(clippy::cast_precision_loss)]
pub fn campplus_fbank(audio_16k: &[f32]) -> Result<CampPlusFbank> {
    let mut fbank = kaldi_fbank(audio_16k, false)?;
    let t = fbank.num_frames;
    let mels = fbank.num_mels;

    let inv_t = 1.0_f32 / (t as f32);
    let mut means = [0.0_f32; FBANK_NUM_MELS];
    for row in fbank.data.chunks_exact(mels) {
        for (m, &val) in means.iter_mut().zip(row.iter()) {
            *m += val;
        }
    }
    for m in &mut means {
        *m *= inv_t;
    }
    for row in fbank.data.chunks_exact_mut(mels) {
        for (val, &m) in row.iter_mut().zip(means.iter()) {
            *val -= m;
        }
    }

    Ok(CampPlusFbank {
        data: fbank.data,
        num_frames: t,
        num_mels: mels,
    })
}

/// Computes `w2v_features` matching `transformers.SeamlessM4TFeatureExtractor`:
///
/// 1. Scales `audio_16k` by $2^{15}$ and extracts 80-bin Kaldi log-Mel features `[T, 80]`.
/// 2. Applies per-channel zero-mean, unit-variance normalization using unbiased variance (`ddof =
///    1`, $\epsilon = 10^{-7}$).
/// 3. Pads `T` to a multiple of `2` with `padding_value = 1.0` (`mask = 0`), then stacks adjacent
///    frame pairs (`stride = 2`) into shape `[1, T_even / 2, 160]`.
///
/// # Errors
///
/// Returns [`IndexTtsError::Dsp`] if `audio_16k` has fewer than `560` samples (2 frames).
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::fbank::w2v_features;
///
/// let audio = vec![0.02_f32; 32_000];
/// let out = w2v_features(&audio)?;
/// assert_eq!(out.num_frames, 99);
/// assert_eq!(out.feature_dim, 160);
/// assert_eq!(out.input_features.len(), 99 * 160);
/// assert_eq!(out.attention_mask.len(), 99);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
#[allow(clippy::cast_precision_loss)]
pub fn w2v_features(audio_16k: &[f32]) -> Result<W2vFeaturesOutput> {
    let mut fbank = kaldi_fbank(audio_16k, true)?;
    let t = fbank.num_frames;
    if t < 2 {
        return Err(IndexTtsError::Dsp {
            message: format!("w2v_features requires at least 2 frames, got {t}"),
        });
    }
    let mels = fbank.num_mels;

    // Per-channel mean and unbiased variance (ddof = 1)
    let t_f32 = t as f32;
    let t_minus_1 = (t - 1) as f32;
    let mut means = [0.0_f32; FBANK_NUM_MELS];
    for row in fbank.data.chunks_exact(mels) {
        for (m, &val) in means.iter_mut().zip(row.iter()) {
            *m += val;
        }
    }
    for m in &mut means {
        *m /= t_f32;
    }

    let mut vars = [0.0_f32; FBANK_NUM_MELS];
    for row in fbank.data.chunks_exact(mels) {
        for ((v, &val), &m) in vars.iter_mut().zip(row.iter()).zip(means.iter()) {
            let diff = val - m;
            *v = diff.mul_add(diff, *v);
        }
    }
    let mut inv_stds = [0.0_f32; FBANK_NUM_MELS];
    for (inv_std, &var_sum) in inv_stds.iter_mut().zip(vars.iter()) {
        let var_ddof1 = var_sum / t_minus_1;
        *inv_std = 1.0_f32 / (var_ddof1 + 1e-7_f32).sqrt();
    }

    for row in fbank.data.chunks_exact_mut(mels) {
        for ((val, &m), &inv_s) in row.iter_mut().zip(means.iter()).zip(inv_stds.iter()) {
            *val = (*val - m) * inv_s;
        }
    }

    // Pad to multiple of stride=2 with padding_value = 1.0
    let mut attention_mask_raw = vec![1_i32; t];
    if t % W2V_STRIDE != 0 {
        fbank.data.resize(
            (t + 1).saturating_mul(mels),
            1.0_f32, // preprocessor_config.json: "padding_value": 1
        );
        attention_mask_raw.push(0);
    }

    let t_even = attention_mask_raw.len();
    let out_frames = t_even / W2V_STRIDE;
    let mut attention_mask = Vec::with_capacity(out_frames);
    for pair in attention_mask_raw.as_chunks::<W2V_STRIDE>().0 {
        // SeamlessM4TFeatureExtractor: attention_mask[:, indices % stride == 1]
        attention_mask.push(pair[1]);
    }

    Ok(W2vFeaturesOutput {
        input_features: fbank.data,
        attention_mask,
        num_frames: out_frames,
        feature_dim: mels * W2V_STRIDE,
    })
}

#[allow(clippy::cast_precision_loss)]
fn povey_window(window_length: usize) -> Vec<f64> {
    let denom = (window_length.saturating_sub(1)) as f64;
    let mut win = Vec::with_capacity(window_length);
    for n in 0..window_length {
        let hann = 0.5 - 0.5 * (2.0 * PI * (n as f64) / denom).cos();
        win.push(hann.powf(0.85));
    }
    win
}

#[allow(clippy::cast_precision_loss)]
fn kaldi_mel_filter_bank(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    min_frequency: f64,
    max_frequency: f64,
    sampling_rate: u32,
) -> Vec<f64> {
    let mel_min = hertz_to_mel_kaldi(min_frequency);
    let mel_max = hertz_to_mel_kaldi(max_frequency);
    let n_pts = num_mel_filters + 2;
    let mut mel_freqs = Vec::with_capacity(n_pts);
    for i in 0..n_pts {
        let ratio = (i as f64) / ((n_pts - 1) as f64);
        mel_freqs.push(mel_min + ratio * (mel_max - mel_min));
    }

    let fft_bin_width = f64::from(sampling_rate) / (((num_frequency_bins - 1) * 2) as f64);
    let mut fft_mel_freqs = Vec::with_capacity(num_frequency_bins);
    for k in 0..num_frequency_bins {
        fft_mel_freqs.push(hertz_to_mel_kaldi(fft_bin_width * (k as f64)));
    }

    // Return row-major [num_mel_filters, num_frequency_bins]
    let mut weights = Vec::with_capacity(num_mel_filters.saturating_mul(num_frequency_bins));
    for m in 0..num_mel_filters {
        let f_left = mel_freqs.get(m).copied().unwrap_or(0.0);
        let f_center = mel_freqs.get(m + 1).copied().unwrap_or(0.0);
        let f_right = mel_freqs.get(m + 2).copied().unwrap_or(0.0);
        let down_diff = f_center - f_left;
        let up_diff = f_right - f_center;

        for &fft_mel in &fft_mel_freqs {
            let down_slope = (fft_mel - f_left) / down_diff;
            let up_slope = (f_right - fft_mel) / up_diff;
            let w = 0.0_f64.max(down_slope.min(up_slope));
            weights.push(w);
        }
    }
    weights
}

fn hertz_to_mel_kaldi(freq: f64) -> f64 {
    1_127.0 * (1.0 + freq / 700.0).ln()
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use safetensors::SafeTensors;

    use super::*;
    use crate::dsp::cosine_similarity_and_max_diff;

    #[test]
    fn test_should_match_python_reference_cp03_campplus_fbank() {
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("golden_ref.safetensors");
        #[allow(clippy::disallowed_methods)]
        let bytes = fs::read(&fixture_path).unwrap_or_default();
        let Ok(st) = SafeTensors::deserialize(&bytes) else {
            return;
        };

        let audio_16k = extract_f32(&st, "audio_16k");
        let ref_camp = extract_f32(&st, "campplus_fbank");

        let rust_camp = campplus_fbank(&audio_16k);
        assert!(rust_camp.is_ok());
        if let Ok(rust_camp) = rust_camp {
            assert_eq!(rust_camp.num_frames, 198);
            assert_eq!(rust_camp.num_mels, 80);
            assert_eq!(rust_camp.data.len(), ref_camp.len());
            let (cos, max_diff) = cosine_similarity_and_max_diff(&rust_camp.data, &ref_camp);
            assert!(
                cos >= 0.999_95,
                "CP-03 campplus_fbank cosine similarity {cos} < 0.99995"
            );
            assert!(
                max_diff < 5e-4,
                "CP-03 campplus_fbank max abs diff {max_diff} >= 5e-4"
            );
        }
    }

    #[test]
    fn test_should_match_python_reference_cp04_w2v_features() {
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("golden_ref.safetensors");
        #[allow(clippy::disallowed_methods)]
        let bytes = fs::read(&fixture_path).unwrap_or_default();
        let Ok(st) = SafeTensors::deserialize(&bytes) else {
            return;
        };

        let audio_16k = extract_f32(&st, "audio_16k");
        let ref_w2v = extract_f32(&st, "w2v_input_features");
        let ref_mask = extract_i32(&st, "w2v_attention_mask");

        let rust_w2v = w2v_features(&audio_16k);
        assert!(rust_w2v.is_ok());
        if let Ok(rust_w2v) = rust_w2v {
            assert_eq!(rust_w2v.num_frames, 99);
            assert_eq!(rust_w2v.feature_dim, 160);
            assert_eq!(rust_w2v.attention_mask, ref_mask);
            let (cos, max_diff) =
                cosine_similarity_and_max_diff(&rust_w2v.input_features, &ref_w2v);
            assert!(
                cos >= 0.999_95,
                "CP-04 w2v_input_features cosine similarity {cos} < 0.99995"
            );
            assert!(
                max_diff < 5e-4,
                "CP-04 w2v_input_features max abs diff {max_diff} >= 5e-4"
            );
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_should_pad_odd_frame_count_with_ones_and_zero_mask() {
        // 32160 samples -> 1 + (32160 - 400)/160 = 199 frames (odd) -> padded to 200 -> 100 strided
        // frames
        let audio_16k: Vec<f32> = (0_usize..32_160_usize)
            .map(|i| ((i as f32) * 0.05).sin() * 0.2)
            .collect();
        let res = w2v_features(&audio_16k);
        assert!(res.is_ok());
        if let Ok(out) = res {
            assert_eq!(out.num_frames, 100);
            assert_eq!(out.attention_mask.last().copied(), Some(0));
            // Last 80 elements of the 100th strided frame must be 1.0 (padding_value = 1)
            let tail = out
                .input_features
                .get(out.input_features.len() - 80..)
                .unwrap_or(&[]);
            assert!(tail.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
        }
    }

    fn extract_f32(st: &SafeTensors<'_>, name: &str) -> Vec<f32> {
        let Ok(view) = st.tensor(name) else {
            return Vec::new();
        };
        view.data()
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect()
    }

    fn extract_i32(st: &SafeTensors<'_>, name: &str) -> Vec<i32> {
        let Ok(view) = st.tensor(name) else {
            return Vec::new();
        };
        view.data()
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(i32::from_le_bytes)
            .collect()
    }
}

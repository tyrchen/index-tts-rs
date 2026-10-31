//! Sinc-Hann resampling matching `torchaudio.functional.resample` (`sinc_interp_hann`).
//!
//! Replicates `core/dsp.py::resample_sinc_hann` bit-accurately using `f64` kernel
//! construction cast to `f32` and strided dot-product convolution.

use std::f64::consts::PI;

use crate::error::{IndexTtsError, Result};

/// Default low-pass filter zero-crossing width (`6`).
pub const DEFAULT_LOWPASS_FILTER_WIDTH: usize = 6;

/// Default Kaiser/Hann filter rolloff factor (`0.99`).
pub const DEFAULT_ROLLOFF: f64 = 0.99;

/// Resamples a mono `f32` waveform from `orig_freq` to `new_freq` using a Hann-windowed
/// sinc interpolation filter (`lowpass_filter_width = 6`, `rolloff = 0.99`).
///
/// # Errors
///
/// Returns [`IndexTtsError::Dsp`] if `orig_freq == 0` or `new_freq == 0`.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::resample::resample_sinc_hann;
///
/// let input = vec![0.0_f32; 24_000];
/// let resampled = resample_sinc_hann(&input, 24_000, 16_000)?;
/// assert_eq!(resampled.len(), 16_000);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
pub fn resample_sinc_hann(y: &[f32], orig_freq: u32, new_freq: u32) -> Result<Vec<f32>> {
    resample_sinc_hann_with_params(
        y,
        orig_freq,
        new_freq,
        DEFAULT_LOWPASS_FILTER_WIDTH,
        DEFAULT_ROLLOFF,
    )
}

/// Resamples a mono `f32` waveform with explicit `lowpass_filter_width` and `rolloff`.
///
/// # Errors
///
/// Returns [`IndexTtsError::Dsp`] if `orig_freq == 0`, `new_freq == 0`,
/// `lowpass_filter_width == 0`, or `rolloff` is not in `(0.0, 1.0]`.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::resample::resample_sinc_hann_with_params;
///
/// let input = vec![0.1_f32; 22_050];
/// let out = resample_sinc_hann_with_params(&input, 22_050, 16_000, 6, 0.99)?;
/// assert_eq!(out.len(), 16_000);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn resample_sinc_hann_with_params(
    y: &[f32],
    orig_freq: u32,
    new_freq: u32,
    lowpass_filter_width: usize,
    rolloff: f64,
) -> Result<Vec<f32>> {
    if orig_freq == 0 || new_freq == 0 {
        return Err(IndexTtsError::Dsp {
            message: format!(
                "sample rates must be non-zero (orig_freq={orig_freq}, new_freq={new_freq})"
            ),
        });
    }
    if lowpass_filter_width == 0 || !rolloff.is_finite() || rolloff <= 0.0 || rolloff > 1.0 {
        return Err(IndexTtsError::Dsp {
            message: format!(
                "invalid filter params: width={lowpass_filter_width}, rolloff={rolloff}"
            ),
        });
    }
    if y.is_empty() || orig_freq == new_freq {
        return Ok(y.to_vec());
    }

    let g = gcd_u32(orig_freq, new_freq);
    let orig = (orig_freq / g) as usize;
    let new = (new_freq / g) as usize;

    let orig_f64 = orig as f64;
    let new_f32 = new as f32;
    let base_freq = (orig.min(new) as f64) * rolloff;
    let lpf_f64 = lowpass_filter_width as f64;
    let width = (lpf_f64 * orig_f64 / base_freq).ceil() as usize;
    let kernel_len = width.saturating_mul(2).saturating_add(orig);

    // Precompute idx = np.arange(-width, width + orig, dtype=np.float64) / orig
    let mut idx = Vec::with_capacity(kernel_len);
    for col in 0..kernel_len {
        let signed = col.cast_signed() - width.cast_signed();
        idx.push((signed as f64) / orig_f64);
    }

    // Build kernels of shape [new, kernel_len] in f32
    let scale = base_freq / orig_f64;
    let mut kernels = Vec::with_capacity(new.saturating_mul(kernel_len));
    for j in 0..new {
        // np.arange(0, -new, -1, dtype=np.float32) / new
        let phase_f32 = -(j as f32) / new_f32;
        let phase_f64 = f64::from(phase_f32);
        for &idx_val in &idx {
            let mut t = (phase_f64 + idx_val) * base_freq;
            t = t.clamp(-lpf_f64, lpf_f64);
            let window = (t * PI / lpf_f64 / 2.0).cos().powi(2);
            let t_pi = t * PI;
            let sinc = if t_pi == 0.0 { 1.0 } else { t_pi.sin() / t_pi };
            let k_val = (sinc * window * scale) as f32;
            kernels.push(k_val);
        }
    }

    // Zero-pad input: (width, width + orig)
    let pad_right = width.saturating_add(orig);
    let padded_len = width.saturating_add(y.len()).saturating_add(pad_right);
    let mut yp = vec![0.0_f32; padded_len];
    if let Some(dst) = yp.get_mut(width..width.saturating_add(y.len())) {
        dst.copy_from_slice(y);
    }

    let n_pos = (padded_len.saturating_sub(kernel_len)) / orig + 1;
    let target_length = (((new as u128) * (y.len() as u128)).div_ceil(orig as u128)) as usize;

    let mut out = Vec::with_capacity(n_pos.saturating_mul(new));
    for p in 0..n_pos {
        let start = p.saturating_mul(orig);
        let Some(frame) = yp.get(start..start.saturating_add(kernel_len)) else {
            break;
        };
        for kernel_row in kernels.chunks_exact(kernel_len) {
            let dot: f32 = frame
                .iter()
                .zip(kernel_row.iter())
                .map(|(&x, &w)| x * w)
                .sum();
            out.push(dot);
            if out.len() >= target_length {
                return Ok(out);
            }
        }
    }

    out.truncate(target_length);
    Ok(out)
}

const fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use safetensors::SafeTensors;

    use super::*;
    use crate::dsp::cosine_similarity_and_max_diff;

    #[test]
    fn test_should_match_python_reference_cp01_resample_24k_to_22k_and_16k() {
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("golden_ref.safetensors");
        #[allow(clippy::disallowed_methods)]
        let bytes = fs::read(&fixture_path).unwrap_or_default();
        let st = SafeTensors::deserialize(&bytes);
        assert!(st.is_ok());
        let Ok(st) = st else { return };

        let raw_24k = extract_f32(&st, "raw_audio_24k");
        let ref_22k = extract_f32(&st, "audio_22k");
        let ref_16k = extract_f32(&st, "audio_16k");

        let rust_22k = resample_sinc_hann(&raw_24k, 24_000, 22_050).unwrap_or_default();
        assert_eq!(rust_22k.len(), ref_22k.len());
        let (cos_22k, max_diff_22k) = cosine_similarity_and_max_diff(&rust_22k, &ref_22k);
        assert!(
            cos_22k >= 0.999_99,
            "CP-01 22k cosine similarity {cos_22k} < 0.99999"
        );
        assert!(
            max_diff_22k < 5e-5,
            "CP-01 22k max abs diff {max_diff_22k} >= 5e-5"
        );

        let rust_16k = resample_sinc_hann(&rust_22k, 22_050, 16_000).unwrap_or_default();
        assert_eq!(rust_16k.len(), ref_16k.len());
        let (cos_16k, max_diff_16k) = cosine_similarity_and_max_diff(&rust_16k, &ref_16k);
        assert!(
            cos_16k >= 0.999_99,
            "CP-01 16k cosine similarity {cos_16k} < 0.99999"
        );
        assert!(
            max_diff_16k < 5e-5,
            "CP-01 16k max abs diff {max_diff_16k} >= 5e-5"
        );
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
}

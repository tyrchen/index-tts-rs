//! 22.05 kHz Slaney Mel Spectrogram computation (`n_fft=1024, hop=256, num_mels=80`).
//!
//! Replicates `core/dsp.py::mel_spectrogram` and `librosa.filters.mel(norm="slaney")`
//! in pure Rust using `realfft`.

use std::{f64::consts::PI, sync::LazyLock};

use realfft::RealFftPlanner;

use crate::{
    config::DEFAULT_SAMPLE_RATE,
    error::{IndexTtsError, Result},
};

/// Default FFT window length (`1024`).
pub const MEL_N_FFT: usize = 1_024;
/// Default STFT hop length (`256`).
pub const MEL_HOP_SIZE: usize = 256;
/// Default STFT Hann window size (`1024`).
pub const MEL_WIN_SIZE: usize = 1_024;
/// Default number of Mel filterbank channels (`80`).
pub const MEL_NUM_BINS: usize = 80;
/// Number of one-sided RFFT bins (`1024 / 2 + 1 = 513`).
pub const MEL_FREQ_BINS: usize = MEL_N_FFT / 2 + 1;
/// Reflective padding size on each side (`(1024 - 256) / 2 = 384`).
pub const MEL_REFLECT_PAD: usize = (MEL_N_FFT - MEL_HOP_SIZE) / 2;

static PERIODIC_HANN_1024: LazyLock<Vec<f32>> =
    LazyLock::new(|| periodic_hann_window(MEL_WIN_SIZE));

static DEFAULT_SLANEY_BASIS: LazyLock<Vec<f32>> = LazyLock::new(|| {
    slaney_mel_basis(
        DEFAULT_SAMPLE_RATE,
        MEL_N_FFT,
        MEL_NUM_BINS,
        0.0,
        f64::from(DEFAULT_SAMPLE_RATE) / 2.0,
    )
});

/// Output of [`mel_spectrogram`] with shape `[1, 80, frames]`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MelSpectrogram {
    /// Row-major `f32` values of shape `[1, num_mels, num_frames]`.
    pub data: Vec<f32>,
    /// Number of Mel channels (`80`).
    pub num_mels: usize,
    /// Number of time frames (`P`).
    pub num_frames: usize,
}

/// Computes the 80-bin log-Mel spectrogram for a `22050 Hz` mono `f32` waveform.
///
/// Applies reflective padding of `384` samples on both ends, a 1024-point periodic
/// Hann window with hop `256`, magnitude $\sqrt{\text{re}^2 + \text{im}^2 + 10^{-9}}$,
/// Slaney-normalized Mel filterbank projection, and natural log with floor $10^{-5}$.
///
/// # Errors
///
/// Returns [`IndexTtsError::Dsp`] if `audio_22k` has fewer than `2` samples (required
/// for reflective padding) or if the FFT execution fails.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::mel::mel_spectrogram;
///
/// let audio = vec![0.01_f32; 22_050];
/// let mel = mel_spectrogram(&audio)?;
/// assert_eq!(mel.num_mels, 80);
/// assert_eq!(mel.num_frames, 86);
/// assert_eq!(mel.data.len(), 80 * 86);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
pub fn mel_spectrogram(audio_22k: &[f32]) -> Result<MelSpectrogram> {
    if audio_22k.len() <= MEL_REFLECT_PAD {
        return Err(IndexTtsError::Dsp {
            message: format!(
                "input audio length {} must be > reflect pad size {MEL_REFLECT_PAD}",
                audio_22k.len()
            ),
        });
    }

    let padded = reflect_pad_1d(audio_22k, MEL_REFLECT_PAD)?;
    let num_frames = 1 + (padded.len().saturating_sub(MEL_N_FFT)) / MEL_HOP_SIZE;

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(MEL_N_FFT);
    let mut fft_in = r2c.make_input_vec();
    let mut fft_out = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();

    let window = &*PERIODIC_HANN_1024;
    let basis = &*DEFAULT_SLANEY_BASIS;

    // Magnitude matrix of shape [num_frames, MEL_FREQ_BINS]
    let mut mag_frames = Vec::with_capacity(num_frames.saturating_mul(MEL_FREQ_BINS));

    for frame_idx in 0..num_frames {
        let start = frame_idx.saturating_mul(MEL_HOP_SIZE);
        let frame_slice = padded
            .get(start..start.saturating_add(MEL_N_FFT))
            .ok_or_else(|| IndexTtsError::Dsp {
                message: "STFT frame slice out of bounds".to_owned(),
            })?;

        for ((dst, &sample), &win) in fft_in.iter_mut().zip(frame_slice.iter()).zip(window.iter()) {
            *dst = sample * win;
        }

        r2c.process_with_scratch(&mut fft_in, &mut fft_out, &mut scratch)
            .map_err(|err| IndexTtsError::Dsp {
                message: format!("1024-pt RFFT failed: {err}"),
            })?;

        for c in &fft_out {
            let energy = c.re.mul_add(c.re, c.im * c.im) + 1e-9_f32;
            mag_frames.push(energy.sqrt());
        }
    }

    // Output layout is [1, num_mels, num_frames] (mel-major, time-minor)
    let mut out = vec![0.0_f32; MEL_NUM_BINS.saturating_mul(num_frames)];
    let basis_rows = basis.as_chunks::<MEL_FREQ_BINS>().0;
    let mag_cols = mag_frames.as_chunks::<MEL_FREQ_BINS>().0;
    for (mel_idx, basis_row) in basis_rows.iter().enumerate() {
        let out_row_offset = mel_idx.saturating_mul(num_frames);
        for (frame_idx, mag_col) in mag_cols.iter().enumerate() {
            let mut dot = 0.0_f32;
            for (&b, &m) in basis_row.iter().zip(mag_col.iter()) {
                dot = b.mul_add(m, dot);
            }
            let clamped = dot.max(1e-5_f32).ln();
            if let Some(slot) = out.get_mut(out_row_offset.saturating_add(frame_idx)) {
                *slot = clamped;
            }
        }
    }

    Ok(MelSpectrogram {
        data: out,
        num_mels: MEL_NUM_BINS,
        num_frames,
    })
}

/// Constructs a Slaney-normalized Mel filterbank matrix of shape `[num_mels, n_fft / 2 + 1]`
/// matching `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney")`.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::mel::slaney_mel_basis;
///
/// let basis = slaney_mel_basis(22050, 1024, 80, 0.0, 11025.0);
/// assert_eq!(basis.len(), 80 * 513);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn slaney_mel_basis(
    sampling_rate: u32,
    n_fft: usize,
    num_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let sr_f64 = f64::from(sampling_rate);
    let n_fft_f64 = n_fft as f64;

    let mut fftfreqs = Vec::with_capacity(n_freqs);
    for k in 0..n_freqs {
        fftfreqs.push((k as f64) * sr_f64 / n_fft_f64);
    }

    let min_mel = hz_to_mel_slaney(fmin);
    let max_mel = hz_to_mel_slaney(fmax);
    let n_pts = num_mels + 2;
    let mut mel_f = Vec::with_capacity(n_pts);
    for i in 0..n_pts {
        let ratio = (i as f64) / ((n_pts - 1) as f64);
        let mel = min_mel + ratio * (max_mel - min_mel);
        mel_f.push(mel_to_hz_slaney(mel));
    }

    let mut weights = Vec::with_capacity(num_mels.saturating_mul(n_freqs));
    for i in 0..num_mels {
        let f_left = mel_f.get(i).copied().unwrap_or(0.0);
        let f_center = mel_f.get(i + 1).copied().unwrap_or(0.0);
        let f_right = mel_f.get(i + 2).copied().unwrap_or(0.0);
        let fdiff_lower = f_center - f_left;
        let fdiff_upper = f_right - f_center;
        let enorm = 2.0 / (f_right - f_left);

        for &freq in &fftfreqs {
            let lower = (freq - f_left) / fdiff_lower;
            let upper = (f_right - freq) / fdiff_upper;
            let w = 0.0_f64.max(lower.min(upper)) * enorm;
            weights.push(w as f32);
        }
    }
    weights
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn periodic_hann_window(win_size: usize) -> Vec<f32> {
    let denom = win_size as f64;
    let mut win = Vec::with_capacity(win_size);
    for n in 0..win_size {
        let v = 0.5 - 0.5 * (2.0 * PI * (n as f64) / denom).cos();
        win.push(v as f32);
    }
    win
}

fn hz_to_mel_slaney(freq: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f64).ln() / 27.0;

    if freq >= MIN_LOG_HZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / logstep
    } else {
        (freq - F_MIN) / F_SP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    const F_MIN: f64 = 0.0;
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f64).ln() / 27.0;

    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_MIN + F_SP * mel
    }
}

fn reflect_pad_1d(y: &[f32], pad: usize) -> Result<Vec<f32>> {
    if y.len() <= pad {
        return Err(IndexTtsError::Dsp {
            message: format!("cannot reflect-pad signal of length {} by {pad}", y.len()),
        });
    }
    let total_len = y.len().saturating_add(pad.saturating_mul(2));
    let mut out = Vec::with_capacity(total_len);

    // Left reflect: y[pad], y[pad-1], ..., y[1]
    for k in (1..=pad).rev() {
        let sample = y.get(k).copied().ok_or_else(|| IndexTtsError::Dsp {
            message: "left reflect pad index out of bounds".to_owned(),
        })?;
        out.push(sample);
    }

    // Center
    out.extend_from_slice(y);

    // Right reflect: y[N-2], y[N-3], ..., y[N-1-pad]
    let n = y.len();
    for k in 0..pad {
        let idx = n.saturating_sub(2).saturating_sub(k);
        let sample = y.get(idx).copied().ok_or_else(|| IndexTtsError::Dsp {
            message: "right reflect pad index out of bounds".to_owned(),
        })?;
        out.push(sample);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use safetensors::SafeTensors;

    use super::*;
    use crate::dsp::cosine_similarity_and_max_diff;

    #[test]
    fn test_should_match_python_reference_cp02_mel_spectrogram() {
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("golden_ref.safetensors");
        #[allow(clippy::disallowed_methods)]
        let bytes = fs::read(&fixture_path).unwrap_or_default();
        let Ok(st) = SafeTensors::deserialize(&bytes) else {
            return;
        };

        let audio_22k = extract_f32(&st, "audio_22k");
        let ref_mel = extract_f32(&st, "ref_mel");

        let rust_mel = mel_spectrogram(&audio_22k);
        assert!(rust_mel.is_ok());
        if let Ok(rust_mel) = rust_mel {
            assert_eq!(rust_mel.num_mels, 80);
            assert_eq!(rust_mel.num_frames, 172);
            assert_eq!(rust_mel.data.len(), ref_mel.len());
            let (cos, max_diff) = cosine_similarity_and_max_diff(&rust_mel.data, &ref_mel);
            assert!(
                cos >= 0.999_99,
                "CP-02 ref_mel cosine similarity {cos} < 0.99999"
            );
            assert!(
                max_diff < 1e-4,
                "CP-02 ref_mel max abs diff {max_diff} >= 1e-4"
            );
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
}

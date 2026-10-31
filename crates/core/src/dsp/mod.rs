//! Pure-Rust Digital Signal Processing (DSP) pipeline for IndexTTS-2.5.
//!
//! Provides audio container decoding/encoding, Hann-sinc resampling,
//! 22.05 kHz Slaney Mel spectrograms, and 16 kHz Kaldi Povey filterbanks.

pub mod audio;
pub mod fbank;
pub mod mel;
pub mod resample;

pub use audio::{
    DecodedAudio, decode_audio_bytes, decode_audio_file, encode_wav_i16_bytes, f32_to_pcm_i16,
    truncate_ref_audio, write_wav_i16_file,
};
pub use fbank::{
    CampPlusFbank, KaldiFbank, W2vFeaturesOutput, campplus_fbank, kaldi_fbank, w2v_features,
};
pub use mel::{MelSpectrogram, mel_spectrogram, slaney_mel_basis};
pub use resample::{resample_sinc_hann, resample_sinc_hann_with_params};

/// Computes the cosine similarity and maximum absolute element-wise difference
/// between two `f32` slices of equal length.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::cosine_similarity_and_max_diff;
///
/// let a = [1.0_f32, 2.0, 3.0];
/// let b = [1.0_f32, 2.0, 3.0];
/// let (cos, max_diff) = cosine_similarity_and_max_diff(&a, &b);
/// assert!((cos - 1.0).abs() < 1e-6);
/// assert!(max_diff < 1e-6);
/// ```
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn cosine_similarity_and_max_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    if a.is_empty() || a.len() != b.len() {
        return (0.0, f32::INFINITY);
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    let mut max_diff = 0.0_f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = f64::from(x);
        let yf = f64::from(y);
        dot = xf.mul_add(yf, dot);
        norm_a = xf.mul_add(xf, norm_a);
        norm_b = yf.mul_add(yf, norm_b);
        let diff = (x - y).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-12);
    ((dot / denom) as f32, max_diff)
}

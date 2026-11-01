//! `LavaSR` v2 bandwidth extension: native 22.05 kHz speech to 48 kHz.
//!
//! The vocoder's output stops at 11.025 kHz. The extender (a 14 M-parameter Vocos GAN,
//! Apache-2.0, `YatharthS/LavaSR` `enhancer_v2`) re-synthesizes the full band from a log-Mel
//! view of the upsampled signal, then a spectral crossover around [`CROSSOVER_HZ`] keeps the
//! original audio below it and takes only the generated band above it, so timbre and timing are
//! untouched.
//!
//! ```text
//! 22.05k PCM ─► sinc resample 48k ─► |STFT| (Hann 2048, hop 512, reflect "same" pad)
//!   ─► Mel (80 bands, filterbank from the checkpoint) ─► log ─► Conv1d(k7) ─► LN
//!   ─► 8× ConvNeXt(512/1536) ─► LN ─► Linear(512→2050) ─► exp|·|, phase ─► iSTFT (OLA)
//!   ─► crossover with the upsampled input ─► 48k PCM
//! ```
//!
//! Spectral transforms run on the CPU (`realfft`); the network runs on the GPU.

use std::{fmt, path::Path, sync::Arc};

use mlx_rs::{
    Array,
    fast::layer_norm,
    ops::{addmm, conv1d, exp, maximum, minimum, split_equal},
    transforms::eval,
};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex, num_complex::Complex32};

use crate::{
    config::{DEFAULT_SAMPLE_RATE, HIFI_SAMPLE_RATE},
    dsp::resample_sinc_hann,
    error::{IndexTtsError, Result},
    models::{FilteredWeights, ModelWeightKind, SafeTensorsLoader, convnext::ConvNeXtBlock},
};

/// STFT size (`2048`).
pub const BWE_N_FFT: usize = 2_048;
/// STFT hop (`512`).
pub const BWE_HOP: usize = 512;
/// Mel bands (`80`).
pub const BWE_MELS: usize = 80;
/// Backbone width (`512`).
pub const BWE_DIM: usize = 512;
/// `ConvNeXt` intermediate width (`1536`).
pub const BWE_INTERMEDIATE: usize = 1_536;
/// `ConvNeXt` blocks (`8`).
pub const BWE_LAYERS: usize = 8;
/// Frequency bins of the one-sided spectrum (`1025`).
pub const BWE_BINS: usize = BWE_N_FFT / 2 + 1;
/// Centre of the crossover between the original and the generated band. The native signal
/// already rolls off above ~10 kHz (vocoder and resampler band edge), so the blend sits below
/// that edge: the output is the original signal up to 9.5 kHz and the generated band from
/// 10.5 kHz, which keeps the spectrum continuous across the 10-11 kHz edge.
pub const CROSSOVER_HZ: f32 = 10_000.0;
/// Width of the smoothstep crossover band centred on [`CROSSOVER_HZ`].
pub const CROSSOVER_WIDTH_HZ: f32 = 1_000.0;

/// "Same" padding on each side: `(n_fft - hop) / 2`.
const PAD: usize = (BWE_N_FFT - BWE_HOP) / 2;
const NORM_EPS: f32 = 1e-6;
/// Floor applied before the log (`safe_log` in Vocos).
const LOG_FLOOR: f32 = 1e-5;
/// Magnitude ceiling of the predicted spectrum.
const MAGNITUDE_CEIL: f32 = 1e3;

/// The `LavaSR` v2 bandwidth extender.
#[derive(Clone)]
pub struct BandwidthExtender {
    mel_basis: Array,
    embed_weight: Array,
    embed_bias: Array,
    norm_weight: Array,
    norm_bias: Array,
    blocks: Vec<ConvNeXtBlock>,
    final_norm_weight: Array,
    final_norm_bias: Array,
    out_weight: Array,
    out_bias: Array,
    analysis_window: Vec<f32>,
    synthesis_window: Vec<f32>,
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
}

impl fmt::Debug for BandwidthExtender {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BandwidthExtender")
            .field("layers", &self.blocks.len())
            .finish_non_exhaustive()
    }
}

impl BandwidthExtender {
    /// Builds the extender from filtered `enhancer_v2` weights.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if a tensor is missing or mis-shaped.
    pub fn from_weights(weights: &FilteredWeights) -> Result<Self> {
        let tensor = |name: &str, shape: &[usize]| weights.tensor_array_shaped(name, shape);
        let blocks = (0..BWE_LAYERS)
            .map(|i| {
                ConvNeXtBlock::from_weights(
                    weights,
                    &format!("backbone.convnext.{i}"),
                    BWE_DIM,
                    BWE_INTERMEDIATE,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let window = |name: &str| -> Result<Vec<f32>> {
            let array = tensor(name, &[BWE_N_FFT])?;
            Ok(array.try_as_slice::<f32>()?.to_vec())
        };
        let mut planner = RealFftPlanner::<f32>::new();
        Ok(Self {
            mel_basis: tensor(
                "feature_extractor.mel_spec.mel_scale.fb",
                &[BWE_BINS, BWE_MELS],
            )?,
            embed_weight: tensor("backbone.embed.weight", &[BWE_DIM, 7, BWE_MELS])?,
            embed_bias: tensor("backbone.embed.bias", &[BWE_DIM])?,
            norm_weight: tensor("backbone.norm.weight", &[BWE_DIM])?,
            norm_bias: tensor("backbone.norm.bias", &[BWE_DIM])?,
            blocks,
            final_norm_weight: tensor("backbone.final_layer_norm.weight", &[BWE_DIM])?,
            final_norm_bias: tensor("backbone.final_layer_norm.bias", &[BWE_DIM])?,
            out_weight: tensor("head.out.weight", &[2 * BWE_BINS, BWE_DIM])?,
            out_bias: tensor("head.out.bias", &[2 * BWE_BINS])?,
            analysis_window: window("feature_extractor.mel_spec.spectrogram.window")?,
            synthesis_window: window("head.istft.window")?,
            forward_fft: planner.plan_fft_forward(BWE_N_FFT),
            inverse_fft: planner.plan_fft_inverse(BWE_N_FFT),
        })
    }

    /// Loads the extender from an `enhancer_v2/model.safetensors` file.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if the file cannot be read or fails validation.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let weights = SafeTensorsLoader::load_from_path(ModelWeightKind::BandwidthExtender, path)?;
        Self::from_weights(&weights)
    }

    /// Extends native 22.05 kHz audio (`[-1, 1]` floats) to 48 kHz.
    ///
    /// The output has exactly the length of a plain 22.05 → 48 kHz resample, so durations and
    /// timings are preserved.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Dsp`] for transform failures or [`IndexTtsError::Mlx`] if GPU
    /// execution fails.
    pub fn extend(&self, audio_22k: &[f32]) -> Result<Vec<f32>> {
        let upsampled = resample_sinc_hann(audio_22k, DEFAULT_SAMPLE_RATE, HIFI_SAMPLE_RATE)?;
        // Reflect padding needs more samples than the pad; such clips are a few milliseconds.
        if upsampled.len() <= PAD + BWE_HOP {
            return Ok(upsampled);
        }
        let (frames, magnitudes) = self.magnitudes(&upsampled)?;
        let (real, imag) = self.predict_spectrum(frames, &magnitudes)?;
        let generated = self.overlap_add(frames, &real, &imag)?;
        crossover(&generated, &upsampled)
    }

    /// `|STFT|` with "same" reflect padding: `[frames, 1025]` row-major.
    fn magnitudes(&self, audio: &[f32]) -> Result<(usize, Vec<f32>)> {
        let padded = reflect_pad(audio, PAD);
        let frames = (padded.len() - BWE_N_FFT) / BWE_HOP + 1;
        let mut input = self.forward_fft.make_input_vec();
        let mut spectrum = self.forward_fft.make_output_vec();
        let mut magnitudes = Vec::with_capacity(frames * BWE_BINS);
        for t in 0..frames {
            let frame = padded
                .get(t * BWE_HOP..t * BWE_HOP + BWE_N_FFT)
                .unwrap_or_default();
            for ((slot, &sample), &w) in input.iter_mut().zip(frame).zip(&self.analysis_window) {
                *slot = sample * w;
            }
            self.forward_fft
                .process(&mut input, &mut spectrum)
                .map_err(|err| dsp_error(&err))?;
            magnitudes.extend(spectrum.iter().map(|bin| bin.norm()));
        }
        Ok((frames, magnitudes))
    }

    /// Runs the network: returns the predicted spectrum's real and imaginary parts.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn predict_spectrum(&self, frames: usize, magnitudes: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let t = frames as i32;
        let magnitude = Array::from_slice(magnitudes, &[t, BWE_BINS as i32]);
        let mel = magnitude.matmul(&self.mel_basis)?;
        let features = maximum(&mel, Array::from_f32(LOG_FLOOR))?
            .log()?
            .reshape(&[1, t, BWE_MELS as i32])?;
        let mut h = conv1d(&features, &self.embed_weight, 1, 3, 1, 1)?.add(&self.embed_bias)?;
        h = layer_norm(&h, Some(&self.norm_weight), Some(&self.norm_bias), NORM_EPS)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        h = layer_norm(
            &h,
            Some(&self.final_norm_weight),
            Some(&self.final_norm_bias),
            NORM_EPS,
        )?;
        let out = addmm(&self.out_bias, &h, self.out_weight.t(), None, None)?;
        let parts = split_equal(&out, 2, -1)?;
        let (Some(log_magnitude), Some(phase)) = (parts.first(), parts.get(1)) else {
            return Err(IndexTtsError::mlx("failed to split the predicted spectrum"));
        };
        let magnitude = minimum(&exp(log_magnitude)?, Array::from_f32(MAGNITUDE_CEIL))?;
        let real = magnitude.multiply(phase.cos()?)?;
        let imag = magnitude.multiply(phase.sin()?)?;
        eval([&real, &imag])?;
        Ok((
            real.try_as_slice::<f32>()?.to_vec(),
            imag.try_as_slice::<f32>()?.to_vec(),
        ))
    }

    /// Inverse STFT with "same" padding: windowed overlap-add normalized by the squared-window
    /// envelope, trimmed by [`PAD`] on both sides (`frames * hop` samples).
    #[allow(clippy::cast_precision_loss)]
    fn overlap_add(&self, frames: usize, real: &[f32], imag: &[f32]) -> Result<Vec<f32>> {
        let total = (frames - 1) * BWE_HOP + BWE_N_FFT;
        let mut signal = vec![0.0_f32; total];
        let mut envelope = vec![0.0_f32; total];
        let mut spectrum = self.inverse_fft.make_input_vec();
        let mut frame = self.inverse_fft.make_output_vec();
        let scale = 1.0 / BWE_N_FFT as f32;
        for t in 0..frames {
            let row = t * BWE_BINS..(t + 1) * BWE_BINS;
            let (re, im) = (
                real.get(row.clone()).unwrap_or_default(),
                imag.get(row).unwrap_or_default(),
            );
            for (bin, (&r, &i)) in spectrum.iter_mut().zip(re.iter().zip(im)) {
                *bin = Complex32::new(r, i);
            }
            // A real signal's DC and Nyquist bins are real; `irfft` ignores their imaginary part.
            if let Some(dc) = spectrum.first_mut() {
                dc.im = 0.0;
            }
            if let Some(nyquist) = spectrum.last_mut() {
                nyquist.im = 0.0;
            }
            self.inverse_fft
                .process(&mut spectrum, &mut frame)
                .map_err(|err| dsp_error(&err))?;
            let at = t * BWE_HOP;
            let out = signal.get_mut(at..at + BWE_N_FFT).unwrap_or_default();
            let env = envelope.get_mut(at..at + BWE_N_FFT).unwrap_or_default();
            for (((o, e), &x), &w) in out
                .iter_mut()
                .zip(env)
                .zip(&frame)
                .zip(&self.synthesis_window)
            {
                *o += x * scale * w;
                *e += w * w;
            }
        }
        Ok(signal
            .get(PAD..total - PAD)
            .unwrap_or_default()
            .iter()
            .zip(envelope.get(PAD..total - PAD).unwrap_or_default())
            .map(|(&x, &e)| if e > 1e-11 { x / e } else { 0.0 })
            .collect())
    }
}

/// Keeps `original` below [`CROSSOVER_HZ`] and `generated` above it, with a smoothstep blend
/// across [`CROSSOVER_WIDTH_HZ`]. Samples past the end of `generated` come from `original`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn crossover(generated: &[f32], original: &[f32]) -> Result<Vec<f32>> {
    let n = generated.len().min(original.len());
    if n < 2 {
        return Ok(original.to_vec());
    }
    let mut planner = RealFftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(n);
    let inverse = planner.plan_fft_inverse(n);
    let spectrum_of = |signal: &[f32]| -> Result<Vec<Complex32>> {
        let mut input = signal.get(..n).unwrap_or_default().to_vec();
        let mut spectrum = forward.make_output_vec();
        forward
            .process(&mut input, &mut spectrum)
            .map_err(|err| dsp_error(&err))?;
        Ok(spectrum)
    };
    let high = spectrum_of(generated)?;
    let mut merged = spectrum_of(original)?;
    let bins = merged.len();
    let hz_per_bin = HIFI_SAMPLE_RATE as f32 / n as f32;
    let lo = (CROSSOVER_HZ - CROSSOVER_WIDTH_HZ / 2.0) / hz_per_bin;
    let hi = (CROSSOVER_HZ + CROSSOVER_WIDTH_HZ / 2.0) / hz_per_bin;
    for (k, (low, &gen_bin)) in merged.iter_mut().zip(&high).enumerate().take(bins) {
        let x = ((k as f32 - lo) / (hi - lo)).clamp(0.0, 1.0);
        let weight = x * x * (3.0 - 2.0 * x);
        *low += (gen_bin - *low) * weight;
    }
    if let Some(dc) = merged.first_mut() {
        dc.im = 0.0;
    }
    if n.is_multiple_of(2)
        && let Some(nyquist) = merged.last_mut()
    {
        nyquist.im = 0.0;
    }
    let mut output = inverse.make_output_vec();
    inverse
        .process(&mut merged, &mut output)
        .map_err(|err| dsp_error(&err))?;
    let scale = 1.0 / n as f32;
    let mut out: Vec<f32> = output.into_iter().map(|x| x * scale).collect();
    out.extend_from_slice(original.get(n..).unwrap_or_default());
    Ok(out)
}

/// `numpy.pad(mode="reflect")`: mirrors around the edge samples without repeating them.
fn reflect_pad(audio: &[f32], pad: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(audio.len() + 2 * pad);
    out.extend((1..=pad).rev().filter_map(|i| audio.get(i)));
    out.extend_from_slice(audio);
    let last = audio.len().saturating_sub(1);
    out.extend((1..=pad).filter_map(|i| last.checked_sub(i).and_then(|j| audio.get(j))));
    out
}

fn dsp_error(err: &impl fmt::Display) -> IndexTtsError {
    IndexTtsError::Dsp {
        message: format!("bandwidth extension FFT failed: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use realfft::{RealFftPlanner, num_complex::Complex32};

    use super::{
        BWE_HOP, BWE_N_FFT, BandwidthExtender, CROSSOVER_HZ, CROSSOVER_WIDTH_HZ, PAD, crossover,
        reflect_pad,
    };
    use crate::{
        config::{DEFAULT_SAMPLE_RATE, HIFI_SAMPLE_RATE},
        dsp::resample_sinc_hann,
        models::resolve_hifi_weights,
    };

    fn spectrum(signal: Option<&[f32]>) -> Option<Vec<Complex32>> {
        let mut input = signal?.to_vec();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(input.len());
        let mut out = fft.make_output_vec();
        fft.process(&mut input, &mut out).ok()?;
        Some(out)
    }

    /// Energy per band `[lo, hi)` Hz of a 48 kHz signal.
    #[allow(clippy::cast_precision_loss)]
    fn band_energy(signal: &[f32], lo: f32, hi: f32) -> f32 {
        let mut input = signal.to_vec();
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(input.len());
        let mut spectrum = fft.make_output_vec();
        if fft.process(&mut input, &mut spectrum).is_err() {
            return f32::NAN;
        }
        let hz = HIFI_SAMPLE_RATE as f32 / signal.len() as f32;
        spectrum
            .iter()
            .enumerate()
            .filter(|(k, _)| (lo..hi).contains(&(*k as f32 * hz)))
            .map(|(_, bin)| bin.norm_sqr())
            .sum()
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn test_should_extend_bandwidth_preserving_length_and_low_band() {
        let Ok(path) = resolve_hifi_weights() else {
            return;
        };
        let Ok(extender) = BandwidthExtender::load_from_path(path) else {
            panic!("bandwidth extender failed to load");
        };
        // One second of a voiced, speech-like signal: a 140 Hz harmonic series below 9 kHz.
        let audio: Vec<f32> = (0..DEFAULT_SAMPLE_RATE)
            .map(|i| {
                let t = i as f32 / DEFAULT_SAMPLE_RATE as f32;
                (1..60)
                    .map(|h| (2.0 * PI * 140.0 * h as f32 * t).sin() * 0.3 / h as f32)
                    .sum()
            })
            .collect();
        let Ok(extended) = extender.extend(&audio) else {
            panic!("extension failed");
        };
        let Ok(upsampled) = resample_sinc_hann(&audio, DEFAULT_SAMPLE_RATE, HIFI_SAMPLE_RATE)
        else {
            panic!("resample failed");
        };
        assert_eq!(extended.len(), upsampled.len());
        // Within the generated span (whole STFT frames), every bin below the crossover band is
        // the original's.
        let frames = (upsampled.len() + 2 * PAD - BWE_N_FFT) / BWE_HOP + 1;
        let span = frames * BWE_HOP;
        let (Some(ours), Some(original)) = (
            spectrum(extended.get(..span)),
            spectrum(upsampled.get(..span)),
        ) else {
            panic!("spectrum failed");
        };
        let below = ((CROSSOVER_HZ - CROSSOVER_WIDTH_HZ / 2.0) / HIFI_SAMPLE_RATE as f32
            * span as f32) as usize;
        let changed: f32 = ours
            .iter()
            .zip(&original)
            .take(below)
            .map(|(a, b)| (a - b).norm_sqr())
            .sum();
        let total: f32 = original.iter().take(below).map(|b| b.norm_sqr()).sum();
        assert!(
            changed / total < 1e-9,
            "low band changed by {:e} (relative energy)",
            changed / total
        );
        let generated = band_energy(&extended, 11_000.0, 24_000.0);
        assert!(generated > band_energy(&upsampled, 11_000.0, 24_000.0) * 10.0);
        assert!(extended.iter().all(|x| x.is_finite() && x.abs() <= 1.5));
    }

    #[test]
    fn test_should_reflect_without_repeating_edges() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }

    #[test]
    fn test_should_keep_low_band_of_original_and_high_band_of_generated() {
        let n = 48_000;
        #[allow(clippy::cast_precision_loss)]
        let tone = |hz: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (2.0 * PI * hz * i as f32 / 48_000.0).sin())
                .collect()
        };
        let low = tone(1_000.0);
        let high = tone(15_000.0);
        // The original has only the low tone; the generated signal has both tones plus a
        // corrupted copy of the low band, which must be discarded.
        let generated: Vec<f32> = low.iter().zip(&high).map(|(l, h)| -0.5 * l + h).collect();
        let merged = crossover(&generated, &low).unwrap_or_default();
        let expected: Vec<f32> = low.iter().zip(&high).map(|(l, h)| l + h).collect();
        let max_err = merged
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // A wrong crossover leaves errors of order 0.5; f32 FFT round-off is a few 1e-3.
        assert!(
            max_err < 1e-2,
            "max error {max_err} (crossover {CROSSOVER_HZ} Hz)"
        );
    }
}

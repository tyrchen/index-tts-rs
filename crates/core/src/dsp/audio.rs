//! Pure-Rust audio decoding (`symphonia`) and WAV encoding (`hound`).
//!
//! Decodes arbitrary mono or multi-channel audio files (`.wav`, `.mp3`, `.flac`, `.m4a`)
//! into normalized `f32` mono waveforms, enforces reference duration caps, and
//! encodes synthesized `i16` PCM waveforms to 22050 Hz 16-bit mono WAV files.

use std::{fs, io::Cursor, path::Path};

use hound::{SampleFormat, WavSpec, WavWriter};
use symphonia::core::{
    codecs::{
        CodecParameters,
        audio::{AudioDecoder, AudioDecoderOptions},
    },
    errors::Error as SymphoniaError,
    formats::{FormatOptions, FormatReader, TrackType, probe::Hint},
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
};

use crate::{
    config::{DEFAULT_SAMPLE_RATE, MAX_REF_SECONDS, validate_safe_file_path},
    error::{IndexTtsError, Result},
};

/// Hard cap on decoded audio samples (60 seconds at 96 kHz) to prevent decompression bombs.
pub const MAX_DECODED_AUDIO_SAMPLES: usize = 60 * 96_000;

/// Maximum input audio file size in bytes (50 MiB).
pub const MAX_AUDIO_FILE_BYTES: usize = 50 * 1024 * 1024;

/// Decoded mono `f32` waveform with its native sample rate.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DecodedAudio {
    /// Mono audio samples in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Native sample rate in Hz.
    pub sample_rate: u32,
}

/// Decodes an in-memory audio container (`WAV`, `MP3`, `FLAC`, `M4A`) into mono `f32` samples.
///
/// Multi-channel streams are downmixed to mono by averaging all channels per frame,
/// matching `soundfile.read(..., always_2d=True).mean(axis=1)`.
///
/// # Errors
///
/// Returns [`IndexTtsError::Validation`] if `bytes` is empty or exceeds [`MAX_AUDIO_FILE_BYTES`],
/// or [`IndexTtsError::AudioDecode`] if the container or codec stream is invalid.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::audio::{decode_audio_bytes, encode_wav_i16_bytes};
///
/// let wav = encode_wav_i16_bytes(&[0, 16384, -16384, 0], 22050)?;
/// let decoded = decode_audio_bytes(&wav, Some("wav"))?;
/// assert_eq!(decoded.sample_rate, 22050);
/// assert_eq!(decoded.samples.len(), 4);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
pub fn decode_audio_bytes(bytes: &[u8], extension_hint: Option<&str>) -> Result<DecodedAudio> {
    if bytes.is_empty() || bytes.len() > MAX_AUDIO_FILE_BYTES {
        return Err(IndexTtsError::validation(
            "audio_bytes",
            format!(
                "audio byte length {} must be in 1..={MAX_AUDIO_FILE_BYTES}",
                bytes.len()
            ),
        ));
    }

    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = extension_hint {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|err| IndexTtsError::AudioDecode {
            message: format!("failed to probe audio container format: {err}"),
        })?;

    let track =
        format
            .default_track(TrackType::Audio)
            .ok_or_else(|| IndexTtsError::AudioDecode {
                message: "no default audio track found in container".to_owned(),
            })?;

    let track_id = track.id;
    let audio_params = match &track.codec_params {
        Some(CodecParameters::Audio(params)) => params.clone(),
        _ => {
            return Err(IndexTtsError::AudioDecode {
                message: "default track does not contain audio codec parameters".to_owned(),
            });
        }
    };
    let sample_rate = audio_params
        .sample_rate
        .ok_or_else(|| IndexTtsError::AudioDecode {
            message: "audio track is missing sample_rate metadata".to_owned(),
        })?;
    if sample_rate == 0 {
        return Err(IndexTtsError::AudioDecode {
            message: "audio track has invalid sample_rate 0".to_owned(),
        });
    }

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())
        .map_err(|err| IndexTtsError::AudioDecode {
            message: format!("failed to initialize audio codec decoder: {err}"),
        })?;

    let mono_samples = decode_track_packets(format.as_mut(), decoder.as_mut(), track_id)?;
    if mono_samples.is_empty() {
        return Err(IndexTtsError::AudioDecode {
            message: "decoded audio contains 0 samples".to_owned(),
        });
    }

    Ok(DecodedAudio {
        samples: mono_samples,
        sample_rate,
    })
}

fn decode_track_packets(
    format: &mut dyn FormatReader,
    decoder: &mut dyn AudioDecoder,
    track_id: u32,
) -> Result<Vec<f32>> {
    let mut mono_samples = Vec::new();
    let mut interleaved_buf = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::IoError(ref io_err))
                if io_err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(IndexTtsError::AudioDecode {
                    message: "stream reset required mid-file (unsupported)".to_owned(),
                });
            }
            Err(err) => {
                return Err(IndexTtsError::AudioDecode {
                    message: format!("error reading audio packet: {err}"),
                });
            }
        };

        if packet.track_id != track_id {
            continue;
        }

        let audio_buf = decoder
            .decode(&packet)
            .map_err(|err| IndexTtsError::AudioDecode {
                message: format!("error decoding audio packet: {err}"),
            })?;

        let channels = audio_buf.num_planes().max(1);
        let frames = audio_buf.frames();
        if frames == 0 {
            continue;
        }

        let new_len = mono_samples.len().saturating_add(frames);
        if new_len > MAX_DECODED_AUDIO_SAMPLES {
            return Err(IndexTtsError::Validation {
                field: "audio_duration",
                reason: format!(
                    "decoded sample count exceeds hard cap of {MAX_DECODED_AUDIO_SAMPLES} samples"
                ),
            });
        }

        interleaved_buf.clear();
        audio_buf.copy_to_vec_interleaved::<f32>(&mut interleaved_buf);

        if channels == 1 {
            mono_samples.extend_from_slice(&interleaved_buf);
        } else {
            #[allow(clippy::cast_precision_loss)]
            let inv_ch = 1.0_f32 / (channels as f32);
            for frame in interleaved_buf.chunks_exact(channels) {
                let sum: f32 = frame.iter().copied().sum();
                mono_samples.push(sum * inv_ch);
            }
        }
    }

    Ok(mono_samples)
}

/// Reads and decodes an audio file from disk into mono `f32` samples.
///
/// # Errors
///
/// Returns [`IndexTtsError::Validation`] if the path is unsafe, [`IndexTtsError::Io`] on
/// file read failure, or [`IndexTtsError::AudioDecode`] if decoding fails.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::audio::{decode_audio_file, write_wav_i16_file};
///
/// let dir = tempfile::tempdir().unwrap();
/// let wav_path = dir.path().join("tone.wav");
/// write_wav_i16_file(&wav_path, &[0, 8192, -8192, 0], 22050)?;
/// let decoded = decode_audio_file(&wav_path)?;
/// assert_eq!(decoded.sample_rate, 22050);
/// assert_eq!(decoded.samples.len(), 4);
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
#[allow(clippy::disallowed_methods)]
pub fn decode_audio_file(path: impl AsRef<Path>) -> Result<DecodedAudio> {
    let path_ref = path.as_ref();
    validate_safe_file_path(path_ref, "audio_path")?;
    let bytes = fs::read(path_ref).map_err(|err| IndexTtsError::io(path_ref, err))?;
    let ext = path_ref.extension().and_then(|e| e.to_str());
    decode_audio_bytes(&bytes, ext)
}

/// Truncates a reference audio waveform to at most `max_seconds * sample_rate` samples.
///
/// Mirrors `audio[:MAX_REF_SECONDS * sr]` in `core/pipeline.py`.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::audio::truncate_ref_audio;
///
/// let samples = vec![0.1_f32; 40_000];
/// let truncated = truncate_ref_audio(&samples, 16_000, 2);
/// assert_eq!(truncated.len(), 32_000);
/// ```
#[must_use]
pub fn truncate_ref_audio(samples: &[f32], sample_rate: u32, max_seconds: u32) -> &[f32] {
    let effective_secs = if max_seconds == 0 {
        MAX_REF_SECONDS
    } else {
        max_seconds
    };
    let max_samples = (sample_rate as usize).saturating_mul(effective_secs as usize);
    samples.get(..max_samples).unwrap_or(samples)
}

/// Converts normalized `f32` audio samples (`[-1.0, 1.0]`) into 16-bit signed PCM (`i16`),
/// matching `np.clip(32767.0 * wav, -32767.0, 32767.0).astype(np.int16)`.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::audio::f32_to_pcm_i16;
///
/// let pcm = f32_to_pcm_i16(&[0.0, 1.0, -1.0, 1.5]);
/// assert_eq!(pcm, vec![0, 32767, -32767, 32767]);
/// ```
#[must_use]
pub fn f32_to_pcm_i16(samples: &[f32]) -> Vec<i16> {
    let mut out = Vec::with_capacity(samples.len());
    for &s in samples {
        if !s.is_finite() {
            out.push(0);
            continue;
        }
        let scaled = (s * 32_767.0).clamp(-32_767.0, 32_767.0);
        #[allow(clippy::cast_possible_truncation)]
        let sample_i16 = scaled as i16;
        out.push(sample_i16);
    }
    out
}

/// Encodes 16-bit mono PCM samples into an in-memory `.wav` byte buffer.
///
/// # Errors
///
/// Returns [`IndexTtsError::AudioEncode`] if `sample_rate == 0` or WAV writing fails.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::audio::encode_wav_i16_bytes;
///
/// let wav = encode_wav_i16_bytes(&[0, 1000, -1000], 22050)?;
/// assert!(wav.starts_with(b"RIFF"));
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
pub fn encode_wav_i16_bytes(pcm_i16: &[i16], sample_rate: u32) -> Result<Vec<u8>> {
    let effective_sr = if sample_rate == 0 {
        DEFAULT_SAMPLE_RATE
    } else {
        sample_rate
    };
    let spec = WavSpec {
        channels: 1,
        sample_rate: effective_sr,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::with_capacity(pcm_i16.len().saturating_mul(2) + 44));
    {
        let mut writer =
            WavWriter::new(&mut cursor, spec).map_err(|err| IndexTtsError::AudioEncode {
                message: format!("failed to initialize WAV writer: {err}"),
            })?;
        for &sample in pcm_i16 {
            writer
                .write_sample(sample)
                .map_err(|err| IndexTtsError::AudioEncode {
                    message: format!("failed writing WAV sample: {err}"),
                })?;
        }
        writer
            .finalize()
            .map_err(|err| IndexTtsError::AudioEncode {
                message: format!("failed finalizing WAV header: {err}"),
            })?;
    }
    Ok(cursor.into_inner())
}

/// Writes 16-bit mono PCM samples to a `.wav` file on disk.
///
/// # Errors
///
/// Returns [`IndexTtsError::Validation`] if the path is unsafe,
/// [`IndexTtsError::AudioEncode`] on WAV encoding errors, or [`IndexTtsError::Io`] on write
/// failure.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::dsp::audio::write_wav_i16_file;
///
/// let dir = tempfile::tempdir().unwrap();
/// let out = dir.path().join("out.wav");
/// write_wav_i16_file(&out, &[0, 500, -500], 22050)?;
/// assert!(out.exists());
/// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
/// ```
#[allow(clippy::disallowed_methods)]
pub fn write_wav_i16_file(path: impl AsRef<Path>, pcm_i16: &[i16], sample_rate: u32) -> Result<()> {
    let path_ref = path.as_ref();
    validate_safe_file_path(path_ref, "wav_output_path")?;
    if let Some(parent) = path_ref.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| IndexTtsError::io(parent, err))?;
    }
    let wav_bytes = encode_wav_i16_bytes(pcm_i16, sample_rate)?;
    fs::write(path_ref, wav_bytes).map_err(|err| IndexTtsError::io(path_ref, err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_roundtrip_wav_encode_and_decode() {
        let pcm: Vec<i16> = vec![0, 4096, 8192, -4096, -8192, 0];
        let wav_bytes = encode_wav_i16_bytes(&pcm, 22_050);
        assert!(wav_bytes.is_ok());
        if let Ok(bytes) = wav_bytes {
            let decoded = decode_audio_bytes(&bytes, Some("wav"));
            assert!(decoded.is_ok());
            if let Ok(decoded) = decoded {
                assert_eq!(decoded.sample_rate, 22_050);
                assert_eq!(decoded.samples.len(), pcm.len());
                let re_pcm = f32_to_pcm_i16(&decoded.samples);
                for (a, b) in pcm.iter().zip(re_pcm.iter()) {
                    assert!((a - b).abs() <= 1);
                }
            }
        }
    }

    #[test]
    fn test_should_truncate_reference_audio_to_max_seconds() {
        let samples = vec![0.25_f32; 22_050 * 20];
        let truncated = truncate_ref_audio(&samples, 22_050, 15);
        assert_eq!(truncated.len(), 22_050 * 15);
    }
}

//! Worker-side forced alignment of external recordings (`EngineCommand::Align`).
//!
//! The recording is turned into the GPT's own semantic codes (`w2v-BERT` features quantized by
//! the codec encoder), replayed through the decoder with teacher forcing next to the text, and
//! decoded with the same attention aligner that times synthesized speech.

use std::{borrow::Cow, time::Instant};

use mlx_rs::Array;
use tracing::info;

use crate::{
    align::{
        AlignmentOutput, AlignmentRequest, AlignmentTimings, CODE_FRAME_SECS, Conditioning,
        MAX_ALIGN_CODES, MIN_ALIGN_SECS, Recording, TextMap, align_recording,
        phones::{SpokenText, split_phones},
    },
    config::EmoAlpha,
    dsp::{FBANK_SAMPLE_RATE, audio::decode_audio_file, f32_to_pcm_i16, resample_sinc_hann},
    engine::actor::MlxWorkerState,
    error::{IndexTtsError, Result},
    frontend::PreparedText,
    models::{DEFAULT_KV_MAX_SEQ, SemanticEncoder},
    voice::{CONDS_DIM, CONDS_FRAMES},
};

/// Milliseconds elapsed since `since`.
fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

impl MlxWorkerState {
    /// Aligns one recording to its text.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    pub(super) fn handle_align(&mut self, req: &AlignmentRequest) -> Result<AlignmentOutput> {
        let started = Instant::now();
        let prepared = self.prepare_alignment_text(req)?;
        let (samples, sample_rate) = match &req.recording {
            Recording::File(path) => {
                let decoded = decode_audio_file(path)?;
                (Cow::Owned(decoded.samples), decoded.sample_rate)
            }
            Recording::Pcm {
                samples,
                sample_rate,
            } => (Cow::Borrowed(samples.as_slice()), *sample_rate),
        };
        let duration_secs = check_duration(samples.len(), sample_rate)?;
        let audio_ms = ms(started);

        let t_features = Instant::now();
        let audio_16k = resample_sinc_hann(&samples, sample_rate, FBANK_SAMPLE_RATE)?;
        drop(samples);
        self.ensure_voice_extractor()?;
        self.ensure_semantic_encoder()?;
        let (Some(extractor), Some(encoder)) = (&self.voice_extractor, &self.semantic_encoder)
        else {
            return Err(IndexTtsError::actor("alignment encoders unavailable"));
        };
        let (features, frames) = extractor.semantic_features(&audio_16k)?;
        features.eval()?;
        let features_ms = ms(t_features);

        let t_encode = Instant::now();
        let codes = encoder.quantize(&features)?;
        let encode_ms = ms(t_encode);
        let segment = prepared
            .segments
            .first()
            .ok_or_else(|| IndexTtsError::validation("text", "nothing to align"))?;
        check_sequence(segment.tokens.len(), codes.len())?;

        let t_condition = Instant::now();
        let conds = match &req.conditioning {
            Conditioning::Recording => extractor.recording_conds(&features, frames, &audio_16k)?,
            Conditioning::Speaker(source) => self
                .speaker_context(source, None, EmoAlpha::default())?
                .conds
                .clone(),
        };
        drop(features);
        let conds = Array::from_slice(&conds, &[1, CONDS_FRAMES as i32, CONDS_DIM as i32]);
        let condition_ms = ms(t_condition);

        let t_attention = Instant::now();
        let attention = self.gpt.alignment_attention(
            &segment.token_ids(),
            &conds,
            req.language.token_id(),
            &[],
            &codes,
            &mut self.kv_cache,
        )?;
        let _ = mlx_rs::memory::clear_cache();
        let pcm = f32_to_pcm_i16(&audio_16k);
        let map = TextMap::new(&prepared.normalized, req.text.as_str());
        let aligned = align_recording(
            segment,
            &map,
            &attention,
            codes.len(),
            &pcm,
            FBANK_SAMPLE_RATE,
        );
        let phones = if req.phones {
            let text = SpokenText {
                normalized: &prepared.normalized,
                body: segment.normalized_range(),
                map: &map,
                source: req.text.as_str(),
            };
            split_phones(&text, &aligned.tokens, &pcm, FBANK_SAMPLE_RATE)
        } else {
            Vec::new()
        };
        let attention_ms = ms(t_attention);

        Ok(AlignmentOutput {
            spans: aligned.spans,
            quality: aligned.quality,
            phones,
            duration_secs,
            codes: codes.len(),
            timings: AlignmentTimings {
                audio_ms,
                features_ms,
                encode_ms,
                condition_ms,
                attention_ms,
                total_ms: ms(started),
            },
        })
    }

    /// Normalizes and tokenizes the text as one decoder pass.
    fn prepare_alignment_text(&self, req: &AlignmentRequest) -> Result<PreparedText> {
        let prepared =
            self.frontend
                .prepare_text(req.text.as_str(), req.language.as_code(), usize::MAX);
        match prepared.segments.as_slice() {
            [segment] if segment.tokens.iter().any(|token| segment.is_spoken(token)) => {
                Ok(prepared)
            }
            [] | [_] => Err(IndexTtsError::validation(
                "text",
                "the text has nothing to pronounce",
            )),
            _ => Err(IndexTtsError::too_long(
                "text",
                "the text exceeds one decoder pass (600 tokens); align shorter pieces",
            )),
        }
    }

    /// Loads the codec encoder on first use (about 100 MB of weights).
    fn ensure_semantic_encoder(&mut self) -> Result<&SemanticEncoder> {
        if self.semantic_encoder.is_none() {
            let t0 = Instant::now();
            let encoder = SemanticEncoder::load_from_path(
                self.model_dir.join("raw/semantic_codec.safetensors"),
            )?;
            info!(
                elapsed_ms = ms(t0),
                "Lazily loaded the semantic codec encoder"
            );
            self.semantic_encoder = Some(encoder);
        }
        self.semantic_encoder
            .as_ref()
            .ok_or_else(|| IndexTtsError::actor("failed to initialize SemanticEncoder"))
    }
}

/// Recording duration in seconds, rejecting recordings outside one decoder pass.
#[allow(clippy::cast_precision_loss)]
fn check_duration(samples: usize, sample_rate: u32) -> Result<f32> {
    if sample_rate == 0 {
        return Err(IndexTtsError::validation(
            "audio",
            "sample rate must be positive",
        ));
    }
    let duration = samples as f32 / sample_rate as f32;
    let longest = MAX_ALIGN_CODES as f32 * CODE_FRAME_SECS;
    if duration < MIN_ALIGN_SECS {
        return Err(IndexTtsError::validation(
            "audio",
            format!("the recording lasts {duration:.3} s; at least {MIN_ALIGN_SECS} s is needed"),
        ));
    }
    if duration > longest {
        return Err(IndexTtsError::too_long(
            "audio",
            format!(
                "the recording lasts {duration:.1} s; one alignment covers at most {longest:.1} s"
            ),
        ));
    }
    Ok(duration)
}

/// Rejects a replay longer than the decoder's KV cache: `conds`, `<BOS>`, text, `<EOS>`,
/// `<START_MEL>`, and all but the last code.
fn check_sequence(tokens: usize, codes: usize) -> Result<()> {
    let length = CONDS_FRAMES + tokens + 2 + codes;
    if codes > MAX_ALIGN_CODES || length > DEFAULT_KV_MAX_SEQ {
        return Err(IndexTtsError::too_long(
            "audio",
            format!(
                "{tokens} text tokens and {codes} codes exceed one decoder pass \
                 ({DEFAULT_KV_MAX_SEQ} positions); align shorter pieces"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_duration, check_sequence};
    use crate::error::IndexTtsError;

    #[test]
    fn test_should_reject_replays_beyond_the_kv_cache() {
        assert!(check_sequence(100, 1_000).is_ok());
        assert!(matches!(
            check_sequence(600, 1_500),
            Err(IndexTtsError::TooLong { .. })
        ));
        assert!(matches!(
            check_sequence(10, 1_816),
            Err(IndexTtsError::TooLong { .. })
        ));
    }

    #[test]
    fn test_should_bound_the_recording_duration() {
        assert!(matches!(
            check_duration(1_000, 16_000),
            Err(IndexTtsError::Validation { .. })
        ));
        assert!(check_duration(16_000, 16_000).is_ok());
        assert!(matches!(
            check_duration(80 * 16_000, 16_000),
            Err(IndexTtsError::TooLong { .. })
        ));
        assert!(check_duration(16_000, 0).is_err());
    }
}

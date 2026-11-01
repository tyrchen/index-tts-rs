//! The job event stream shared by `itts synth --output jsonl`, the daemon socket protocol, and
//! MCP progress reporting: `started` → `chunk`* → `done` | `error`.

use std::{collections::BTreeMap, path::PathBuf};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use index_tts_rs_core::engine::{AudioChunk, StageTimings};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    error::ServiceError,
    request::PreparedJob,
    run::JobChunk,
    transcript::{Alignment, SentenceTiming},
};

/// Where a job executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum Backend {
    /// The warm-model daemon.
    Daemon,
    /// A model loaded inside the calling process.
    InProcess,
}

/// Emitted once before any audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct JobStarted {
    /// Speaker of the first utterance.
    pub speaker: String,
    /// Language code of the first utterance.
    pub language: String,
    /// Number of planned utterances (paragraphs and control boundaries).
    pub utterances: usize,
    /// Execution backend.
    pub backend: Backend,
    /// Seed driving this job; pass it as `advanced.seed` to reproduce the audio.
    pub seed: u64,
}

impl JobStarted {
    /// Describes a prepared job about to run on `backend`.
    #[must_use]
    pub fn new(job: &PreparedJob, backend: Backend) -> Self {
        Self {
            speaker: job.speaker.clone(),
            language: job.language.as_code().to_owned(),
            utterances: job.plan.len(),
            backend,
            seed: job.seed,
        }
    }
}

/// One synthesized segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct ChunkInfo {
    /// Zero-based utterance index.
    pub utterance_index: usize,
    /// Number of utterances.
    pub total_utterances: usize,
    /// Zero-based sentence segment within the utterance.
    pub segment_index: usize,
    /// Number of segments in the utterance.
    pub total_segments: usize,
    /// Sample count, including prepended silence.
    pub samples: usize,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Duration of this chunk in seconds.
    pub duration_secs: f32,
    /// Generated Mel codes.
    pub mel_tokens: usize,
    /// Per-stage timings in milliseconds.
    #[schemars(with = "BTreeMap<String, f64>")]
    pub timings: StageTimings,
    /// Where this chunk (including its leading silence) starts in the job audio, in seconds.
    pub start_secs: f32,
    /// The spoken text this chunk reads, including its closing punctuation.
    pub text: String,
    /// Character index of `text` within the job's spoken text (the script without controls).
    pub text_offset: usize,
    /// Character timings of `text` on the job timeline, when `timestamps` was requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<Alignment>,
    /// Base64 little-endian `i16` mono PCM, present only when the client asked for audio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pcm_base64: Option<String>,
}

impl ChunkInfo {
    pub(crate) fn new(chunk: &JobChunk, include_pcm: bool) -> Self {
        let audio = &chunk.audio;
        Self {
            utterance_index: chunk.utterance_index,
            total_utterances: chunk.total_utterances,
            segment_index: audio.segment_index,
            total_segments: audio.total_segments,
            samples: audio.pcm_i16.len(),
            sample_rate: audio.sample_rate,
            duration_secs: audio.duration_secs(),
            mel_tokens: audio.mel_tokens,
            timings: audio.timings,
            start_secs: chunk.start_secs,
            text: chunk.text.clone(),
            text_offset: chunk.text_offset,
            alignment: chunk.alignment.clone(),
            pcm_base64: include_pcm.then(|| encode_pcm(&audio.pcm_i16)),
        }
    }

    /// Reconstructs the [`JobChunk`] carried by this event.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ErrorCode::Internal`] if the PCM payload is absent or malformed.
    pub fn into_job_chunk(self) -> Result<JobChunk, ServiceError> {
        let encoded = self
            .pcm_base64
            .ok_or_else(|| ServiceError::internal("chunk event carries no PCM"))?;
        let bytes = BASE64
            .decode(encoded)
            .map_err(|err| ServiceError::internal(format!("invalid PCM payload: {err}")))?;
        let pcm = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| i16::from_le_bytes(*pair))
            .collect();
        Ok(JobChunk {
            utterance_index: self.utterance_index,
            total_utterances: self.total_utterances,
            start_secs: self.start_secs,
            text: self.text,
            text_offset: self.text_offset,
            alignment: self.alignment,
            audio: AudioChunk::new(
                self.segment_index,
                self.total_segments,
                pcm,
                self.sample_rate,
                self.mel_tokens,
                self.timings,
            ),
        })
    }
}

/// Final job statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct JobSummary {
    /// Absolute path of the written WAV, when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_path: Option<PathBuf>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Total samples.
    pub samples: usize,
    /// Audio duration in seconds.
    pub duration_secs: f32,
    /// GPU synthesis time in seconds.
    pub synthesis_secs: f32,
    /// Wall-clock time in seconds, including queueing.
    pub wall_secs: f32,
    /// Real-time factor (`synthesisSecs / durationSecs`; below 1.0 is faster than real time).
    pub rtf: f32,
    /// Utterances synthesized.
    pub utterances: usize,
    /// Sentence segments synthesized.
    pub segments: usize,
    /// Generated Mel codes.
    pub mel_tokens: usize,
    /// Aggregated per-stage timings in milliseconds.
    #[schemars(with = "BTreeMap<String, f64>")]
    pub timings: StageTimings,
    /// Seed that produced this audio; pass it as `advanced.seed` to reproduce it.
    pub seed: u64,
    /// Character timings over the spoken text, when `timestamps` was requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<Alignment>,
    /// Sentence timings derived from `alignment`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sentences: Vec<SentenceTiming>,
}

/// One event of a job stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event", rename_all = "camelCase")]
#[non_exhaustive]
pub enum JobEvent {
    /// The job was accepted and planned.
    Started(JobStarted),
    /// A segment finished.
    Chunk(ChunkInfo),
    /// The job completed.
    Done(JobSummary),
    /// The job failed; no further events follow.
    Error {
        /// Failure details.
        error: ServiceError,
    },
}

fn encode_pcm(samples: &[i16]) -> String {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    BASE64.encode(bytes)
}

#[cfg(test)]
mod tests {
    use index_tts_rs_core::engine::{AudioChunk, StageTimings};
    use serde_json::{from_value, json, to_value};

    use super::{ChunkInfo, JobEvent};
    use crate::{ErrorCode, ServiceError, run::JobChunk, transcript::Alignment};

    #[test]
    fn test_should_round_trip_pcm_text_and_alignment_through_chunk_events()
    -> Result<(), ServiceError> {
        let alignment = Alignment {
            text: "你好。".to_owned(),
            start: vec![1.0, 1.2, 1.4],
            end: vec![1.2, 1.4, 1.5],
        };
        let chunk = JobChunk {
            utterance_index: 1,
            total_utterances: 2,
            audio: AudioChunk::new(0, 3, vec![-2, 0, 32767], 22050, 7, StageTimings::default()),
            start_secs: 0.8,
            text: "你好。".to_owned(),
            text_offset: 4,
            alignment: Some(alignment.clone()),
        };
        let info: ChunkInfo = chunk.info(true);
        let restored = info.into_job_chunk()?;
        assert_eq!(restored.audio.pcm_i16, vec![-2, 0, 32767]);
        assert_eq!(restored.utterance_index, 1);
        assert_eq!(restored.text, "你好。");
        assert_eq!(restored.text_offset, 4);
        assert_eq!(restored.alignment, Some(alignment));
        Ok(())
    }

    #[test]
    fn test_should_tag_events_by_name() -> serde_json::Result<()> {
        let event = JobEvent::Error {
            error: ServiceError::new(ErrorCode::Busy, "full"),
        };
        let value = to_value(&event)?;
        assert_eq!(
            value,
            json!({"event":"error","error":{"code":"BUSY","message":"full"}})
        );
        assert_eq!(from_value::<JobEvent>(value)?, event);
        Ok(())
    }
}

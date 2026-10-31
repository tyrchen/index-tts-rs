//! Executes a [`PreparedJob`] against an [`IndexTtsEngine`] and accounts for its output.
//!
//! [`JobRun`] is a pull-based cursor over every utterance's audio chunks with paragraph pauses
//! and sentence silences already applied, so frontends only decide how to deliver PCM (socket,
//! speaker, HTTP stream, file). Dropping a [`JobRun`] cancels the job: the engine actor stops at
//! the next segment boundary once its chunk receiver is gone.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
    vec::IntoIter,
};

use index_tts_rs_core::{
    dsp::write_wav_i16_file,
    engine::{AudioChunk, IndexTtsEngine, StageTimings, SynthesisStream},
};
use index_tts_rs_director::{PlannedUtterance, prepend_silence};

use crate::{
    error::ServiceError,
    events::{ChunkInfo, JobSummary},
    request::PreparedJob,
};

/// One synthesized segment, positioned within the job.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct JobChunk {
    /// Zero-based utterance (paragraph) index.
    pub utterance_index: usize,
    /// Number of utterances in the job.
    pub total_utterances: usize,
    /// Segment audio with its leading pause or sentence silence prepended.
    pub audio: AudioChunk,
}

impl JobChunk {
    /// Describes this chunk for an event stream, optionally embedding base64 PCM.
    #[must_use]
    pub fn info(&self, include_pcm: bool) -> ChunkInfo {
        ChunkInfo::new(self, include_pcm)
    }
}

#[derive(Debug)]
struct ActiveUtterance {
    stream: SynthesisStream,
    pause_before_ms: u32,
    interval_ms: u32,
}

/// Pull-based cursor yielding a job's [`JobChunk`]s in playback order.
#[derive(Debug)]
pub struct JobRun {
    engine: IndexTtsEngine,
    utterances: IntoIter<PlannedUtterance>,
    total_utterances: usize,
    next_index: usize,
    active: Option<ActiveUtterance>,
}

impl JobRun {
    /// Creates a cursor; no work is submitted until the first [`JobRun::next_chunk`].
    #[must_use]
    pub fn new(engine: IndexTtsEngine, plan: Vec<PlannedUtterance>) -> Self {
        let total_utterances = plan.len();
        Self {
            engine,
            utterances: plan.into_iter(),
            total_utterances,
            next_index: 0,
            active: None,
        }
    }

    /// Number of utterances in this job.
    #[must_use]
    pub const fn total_utterances(&self) -> usize {
        self.total_utterances
    }

    /// Awaits the next chunk, or `None` once every utterance is complete.
    ///
    /// # Errors
    ///
    /// Returns the engine or silence-insertion failure of the current utterance.
    pub async fn next_chunk(&mut self) -> Result<Option<JobChunk>, ServiceError> {
        loop {
            if let Some(active) = &self.active {
                match active.stream.next_chunk().await {
                    Some(chunk) => {
                        let mut audio = chunk?;
                        let silence = if audio.segment_index == 0 {
                            active.pause_before_ms
                        } else {
                            active.interval_ms
                        };
                        prepend_silence(&mut audio.pcm_i16, audio.sample_rate, silence)?;
                        return Ok(Some(JobChunk {
                            utterance_index: self.next_index.saturating_sub(1),
                            total_utterances: self.total_utterances,
                            audio,
                        }));
                    }
                    None => self.active = None,
                }
            } else if let Some(part) = self.utterances.next() {
                self.next_index = self.next_index.saturating_add(1);
                self.active = Some(ActiveUtterance {
                    pause_before_ms: part.pause_before_ms,
                    interval_ms: part.request.interval_silence_ms,
                    stream: self.engine.synthesize_stream(part.request)?,
                });
            } else {
                return Ok(None);
            }
        }
    }
}

/// Accumulates chunk statistics (and optionally PCM) into a [`JobSummary`].
#[derive(Debug)]
pub struct JobAccumulator {
    sample_rate: u32,
    pcm: Option<Vec<i16>>,
    samples: usize,
    segments: usize,
    utterances: usize,
    mel_tokens: usize,
    timings: StageTimings,
}

impl JobAccumulator {
    /// Creates an accumulator; `keep_pcm` retains audio for WAV output.
    #[must_use]
    pub fn new(sample_rate: u32, keep_pcm: bool) -> Self {
        Self {
            sample_rate,
            pcm: if keep_pcm { Some(Vec::new()) } else { None },
            samples: 0,
            segments: 0,
            utterances: 0,
            mel_tokens: 0,
            timings: StageTimings::default(),
        }
    }

    /// Records one chunk.
    pub fn push(&mut self, chunk: &JobChunk) {
        let audio = &chunk.audio;
        self.sample_rate = audio.sample_rate;
        self.samples = self.samples.saturating_add(audio.pcm_i16.len());
        self.segments = self.segments.saturating_add(1);
        self.utterances = chunk.total_utterances;
        self.mel_tokens = self.mel_tokens.saturating_add(audio.mel_tokens);
        self.timings.accumulate_segment(&audio.timings);
        if let Some(pcm) = self.pcm.as_mut() {
            pcm.extend_from_slice(&audio.pcm_i16);
        }
    }

    /// Total samples recorded so far.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.samples
    }

    /// Retained PCM, if `keep_pcm` was requested.
    #[must_use]
    pub fn pcm(&self) -> Option<&[i16]> {
        self.pcm.as_deref()
    }

    /// Finishes the job, writing a WAV to `output_path` when given.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ErrorCode::InvalidArgument`] if no audio was produced, or the WAV write
    /// failure.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "audio lengths and stage timings are far below f32 precision limits"
    )]
    pub fn finish(
        mut self,
        wall: Duration,
        output_path: Option<&Path>,
    ) -> Result<JobSummary, ServiceError> {
        if self.samples == 0 {
            return Err(ServiceError::invalid(
                "no audio was produced; check that the text contains speakable content",
            ));
        }
        if let Some(path) = output_path {
            let pcm = self
                .pcm
                .as_deref()
                .ok_or_else(|| ServiceError::internal("PCM was not retained for WAV output"))?;
            write_wav_i16_file(path, pcm, self.sample_rate)?;
        }
        let t = &self.timings;
        let stage_ms =
            t.frontend_ms + t.gpt_ms + t.codec_ms + t.regulator_ms + t.cfm_ms + t.bigvgan_ms;
        self.timings.total_ms = t.voice_extract_ms + stage_ms;
        let duration_secs = self.samples as f32 / self.sample_rate.max(1) as f32;
        let synthesis_secs = if stage_ms > 0.0 {
            (self.timings.total_ms / 1000.0) as f32
        } else {
            wall.as_secs_f32()
        };
        Ok(JobSummary {
            output_path: output_path.map(Path::to_path_buf),
            sample_rate: self.sample_rate,
            samples: self.samples,
            duration_secs,
            synthesis_secs,
            wall_secs: wall.as_secs_f32(),
            rtf: synthesis_secs / duration_secs.max(f32::EPSILON),
            utterances: self.utterances,
            segments: self.segments,
            mel_tokens: self.mel_tokens,
            timings: self.timings,
        })
    }
}

/// Runs a whole job on `engine`, invoking `on_chunk` for every chunk, then writes the WAV if
/// the job requested `outputPath`.
///
/// # Errors
///
/// Returns the first engine, callback, or output failure.
pub async fn run_job(
    engine: IndexTtsEngine,
    job: PreparedJob,
    mut on_chunk: impl FnMut(&JobChunk) -> Result<(), ServiceError>,
) -> Result<JobSummary, ServiceError> {
    let started = Instant::now();
    let output_path: Option<PathBuf> = job.output_path;
    let mut acc = JobAccumulator::new(engine.config().sample_rate, output_path.is_some());
    let mut run = JobRun::new(engine, job.plan);
    while let Some(chunk) = run.next_chunk().await? {
        on_chunk(&chunk)?;
        acc.push(&chunk);
    }
    acc.finish(started.elapsed(), output_path.as_deref())
}

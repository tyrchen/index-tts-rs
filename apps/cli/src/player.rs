//! Real-time macOS `CoreAudio` streaming player (`cpal`).
//!
//! Accepts `22,050 Hz` mono `i16` PCM chunks sentence-by-sentence, automatically resamples to the
//! active hardware output device's native sample rate and channel count, and plays gaplessly.

use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use cpal::{
    Device, FromSample, OutputCallbackInfo, Sample, SampleFormat, SizedSample, Stream,
    StreamConfig, default_host,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use flume::Sender;
use index_tts_rs_core::dsp::resample_sinc_hann;
use index_tts_rs_service::{ErrorCode, ServiceError};
use tracing::warn;

/// Streaming audio player backed by the system default output audio device (`CoreAudio` on macOS).
pub struct AudioPlayer {
    device_sample_rate: u32,
    device_channels: usize,
    queue: Arc<Mutex<VecDeque<f32>>>,
    finished_enqueue: Arc<AtomicBool>,
    stream: Stream,
}

impl std::fmt::Debug for AudioPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioPlayer")
            .field("device_sample_rate", &self.device_sample_rate)
            .field("device_channels", &self.device_channels)
            .finish_non_exhaustive()
    }
}

impl AudioPlayer {
    /// Opens the default system audio output device and starts a streaming playback queue.
    ///
    /// # Errors
    ///
    /// Returns an error if no default audio output device is available or if building/starting
    /// the output stream fails.
    pub fn try_new() -> Result<Self> {
        let host = default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default audio output device found"))?;
        let supported_config = device
            .default_output_config()
            .context("failed to query default output stream configuration")?;

        let sample_format = supported_config.sample_format();
        let config: StreamConfig = supported_config.into();
        let device_sample_rate = config.sample_rate;
        let device_channels = usize::from(config.channels.max(1));

        let queue = Arc::new(Mutex::new(VecDeque::with_capacity(
            (device_sample_rate as usize) * device_channels * 2,
        )));
        let finished_enqueue = Arc::new(AtomicBool::new(false));

        let stream = match sample_format {
            SampleFormat::F32 => build_output_stream::<f32>(&device, &config, Arc::clone(&queue))?,
            SampleFormat::I16 => build_output_stream::<i16>(&device, &config, Arc::clone(&queue))?,
            SampleFormat::U16 => build_output_stream::<u16>(&device, &config, Arc::clone(&queue))?,
            other => {
                return Err(anyhow!("unsupported audio output sample format: {other:?}"));
            }
        };

        stream
            .play()
            .context("failed to start audio output stream playback")?;

        Ok(Self {
            device_sample_rate,
            device_channels,
            queue,
            finished_enqueue,
            stream,
        })
    }

    /// Converts a `mono` `i16` PCM chunk at `src_sample_rate` to the output device's sample rate
    /// and channel layout, then appends it to the playback buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if resampling fails or the playback buffer lock is poisoned.
    #[allow(clippy::cast_precision_loss)]
    pub fn enqueue_pcm_i16(&self, pcm_i16: &[i16], src_sample_rate: u32) -> Result<()> {
        if pcm_i16.is_empty() {
            return Ok(());
        }
        let mono_f32: Vec<f32> = pcm_i16
            .iter()
            .map(|&s| (f32::from(s) / 32_767.0).clamp(-1.0, 1.0))
            .collect();

        let resampled = if src_sample_rate == self.device_sample_rate {
            mono_f32
        } else {
            resample_sinc_hann(&mono_f32, src_sample_rate, self.device_sample_rate)
                .context("failed to resample audio chunk for output device")?
        };

        let mut guard = self
            .queue
            .lock()
            .map_err(|_| anyhow!("audio playback queue mutex poisoned"))?;
        if self.device_channels <= 1 {
            guard.extend(resampled);
        } else {
            for sample in resampled {
                for _ in 0..self.device_channels {
                    guard.push_back(sample);
                }
            }
        }
        Ok(())
    }

    /// Blocks until all queued audio samples have been consumed by the hardware output callback.
    pub fn drain(self) {
        self.finished_enqueue.store(true, Ordering::Release);
        loop {
            let remaining = self.queue.lock().map_or(0, |q| q.len());
            if remaining == 0 {
                break;
            }
            thread::sleep(Duration::from_millis(15));
        }
        // Allow one final hardware buffer period (30ms) to flush to speakers
        thread::sleep(Duration::from_millis(30));
        drop(self.stream);
    }
}

fn build_output_stream<T>(
    device: &Device,
    config: &StreamConfig,
    queue: Arc<Mutex<VecDeque<f32>>>,
) -> Result<Stream>
where
    T: SizedSample + Sample + FromSample<f32> + Send + 'static,
{
    let err_fn = |err| {
        warn!(error = %err, "Audio output stream error");
    };
    let stream = device
        .build_output_stream(
            *config,
            move |data: &mut [T], _: &OutputCallbackInfo| {
                if let Ok(mut guard) = queue.lock() {
                    for slot in data.iter_mut() {
                        let val = guard.pop_front().unwrap_or(0.0_f32);
                        *slot = T::from_sample(val);
                    }
                } else {
                    for slot in data.iter_mut() {
                        *slot = T::from_sample(0.0_f32);
                    }
                }
            },
            err_fn,
            None,
        )
        .context("failed to build CoreAudio output stream")?;
    Ok(stream)
}

/// `Send` handle that plays PCM on a dedicated thread owning the non-`Send` `CoreAudio` stream,
/// so async tasks (MCP tool calls, daemon clients) can feed audio across `.await` points.
#[derive(Debug)]
pub struct Playback {
    tx: Sender<(Vec<i16>, u32)>,
    handle: JoinHandle<Result<()>>,
}

impl Playback {
    /// Opens the default output device on a playback thread.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::AudioOutputUnavailable`] if no output device can be opened.
    pub fn start() -> Result<Self, ServiceError> {
        let (tx, rx) = flume::unbounded::<(Vec<i16>, u32)>();
        let (ready_tx, ready_rx) = flume::bounded::<Result<(), String>>(1);
        let handle = thread::Builder::new()
            .name("itts-playback".to_owned())
            .spawn(move || {
                let player = match AudioPlayer::try_new() {
                    Ok(player) => {
                        let _ = ready_tx.send(Ok(()));
                        player
                    }
                    Err(err) => {
                        let _ = ready_tx.send(Err(format!("{err:#}")));
                        return Err(err);
                    }
                };
                for (pcm, sample_rate) in rx {
                    player.enqueue_pcm_i16(&pcm, sample_rate)?;
                }
                player.drain();
                Ok(())
            })
            .map_err(|err| unavailable(&err.to_string()))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self { tx, handle }),
            Ok(Err(message)) => Err(unavailable(&message)),
            Err(_) => Err(unavailable("playback thread exited during startup")),
        }
    }

    /// Queues mono PCM for playback.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::AudioOutputUnavailable`] if the playback thread has stopped.
    pub fn enqueue(&self, pcm: Vec<i16>, sample_rate: u32) -> Result<(), ServiceError> {
        self.tx
            .send((pcm, sample_rate))
            .map_err(|_| unavailable("audio playback stopped unexpectedly"))
    }

    /// Blocks until every queued sample has been played.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::AudioOutputUnavailable`] if playback failed.
    pub fn finish(self) -> Result<(), ServiceError> {
        drop(self.tx);
        self.handle
            .join()
            .map_err(|_| unavailable("audio playback thread panicked"))?
            .map_err(|err| unavailable(&format!("{err:#}")))
    }
}

/// Name of the default output device, if any.
pub fn default_output_device_name() -> Option<String> {
    default_host()
        .default_output_device()
        .and_then(|device| device.description().ok().map(|d| d.name().to_owned()))
}

fn unavailable(message: &str) -> ServiceError {
    ServiceError::new(
        ErrorCode::AudioOutputUnavailable,
        format!("audio output unavailable: {message}"),
    )
    .with_hint("connect an output device, or write to a file with --out <file.wav>")
}

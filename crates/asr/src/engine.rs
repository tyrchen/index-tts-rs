//! The recognizer actor: ONNX Runtime (via sherpa-onnx) on its own worker thread.
//!
//! The worker owns every loaded `OfflineRecognizer` and creates each one the first time its
//! model is used; models stay resident until [`AsrEngine::evict`] or shutdown. Model files are
//! resolved (and downloaded on first use) on the caller's async task, so a download never
//! blocks recognition with another model.

use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use arc_swap::ArcSwap;
use flume::{Receiver, Sender};
use sherpa_onnx::{
    OfflineParaformerModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig, OfflineTransducerModelConfig,
};
use tokio::sync::Semaphore;
use tracing::{debug, info};

use crate::{
    ASR_SAMPLE_RATE,
    error::{AsrError, Result},
    models::{AsrModel, ModelFiles, ensure, resolve},
    tokens::{HeardToken, RawToken, heard_tokens},
};

/// ONNX Runtime intra-op threads per recognizer (4 is 1.6× faster than 2; more does not help).
pub const ASR_THREADS: i32 = 4;
/// Queued commands before senders wait.
const CHANNEL_CAPACITY: usize = 32;

/// One recognition.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Transcript {
    /// Model that recognized the audio.
    pub model: AsrModel,
    /// Recognized units in order.
    pub tokens: Vec<HeardToken>,
    /// Whether the model files were downloaded for this request.
    pub downloaded: bool,
    /// Milliseconds spent loading the model (0 when resident).
    pub load_ms: f64,
    /// Milliseconds spent recognizing.
    pub recognize_ms: f64,
}

enum Command {
    Transcribe {
        files: ModelFiles,
        samples: Vec<f32>,
        reply: Sender<Result<Transcript>>,
    },
    Evict {
        reply: Sender<usize>,
    },
    Shutdown {
        reply: Sender<()>,
    },
}

/// Handle to the recognizer actor. Cloning shares the same worker.
#[derive(Debug, Clone)]
pub struct AsrEngine {
    cmd_tx: Sender<Command>,
    running: Arc<AtomicBool>,
    loaded: Arc<ArcSwap<Vec<AsrModel>>>,
    downloads: Arc<Semaphore>,
    worker: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl AsrEngine {
    /// Starts the worker thread. No model is loaded until it is first used.
    ///
    /// # Errors
    ///
    /// Returns [`AsrError::Actor`] when the thread cannot be spawned.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_asr::AsrEngine;
    ///
    /// let engine = AsrEngine::start()?;
    /// assert!(engine.is_running());
    /// assert!(engine.loaded().is_empty());
    /// engine.stop()?;
    /// # Ok::<(), index_tts_rs_asr::AsrError>(())
    /// ```
    pub fn start() -> Result<Self> {
        let loaded = Arc::new(ArcSwap::from_pointee(Vec::new()));
        let (cmd_tx, running, handle) = spawn_worker(Arc::clone(&loaded))?;
        Ok(Self {
            cmd_tx,
            running,
            loaded,
            downloads: Arc::new(Semaphore::new(1)),
            worker: Arc::new(Mutex::new(Some(handle))),
        })
    }

    /// Whether the worker accepts commands.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire) && !self.cmd_tx.is_disconnected()
    }

    /// Models currently resident in the worker.
    #[must_use]
    pub fn loaded(&self) -> Vec<AsrModel> {
        self.loaded.load().as_ref().clone()
    }

    /// Recognizes 16 kHz mono `samples` with `model`, downloading its files first when they
    /// are not cached.
    ///
    /// # Errors
    ///
    /// Returns [`AsrError::ModelUnavailable`] when the model cannot be downloaded or loaded,
    /// or [`AsrError::Actor`] when the worker has stopped.
    pub async fn transcribe(&self, model: AsrModel, samples: Vec<f32>) -> Result<Transcript> {
        let (files, downloaded) = self.files(model).await?;
        let (reply, reply_rx) = flume::bounded(1);
        self.cmd_tx
            .send_async(Command::Transcribe {
                files,
                samples,
                reply,
            })
            .await
            .map_err(|_| AsrError::actor("the recognizer worker has stopped"))?;
        let mut transcript = reply_rx
            .recv_async()
            .await
            .map_err(|_| AsrError::actor("the recognizer worker dropped the request"))??;
        transcript.downloaded = downloaded;
        Ok(transcript)
    }

    /// Cached files of `model`; one download at a time, rechecking the cache after waiting.
    async fn files(&self, model: AsrModel) -> Result<(ModelFiles, bool)> {
        if let Ok(files) = resolve(model) {
            return Ok((files, false));
        }
        let _permit = self
            .downloads
            .acquire()
            .await
            .map_err(|_| AsrError::actor("model downloads are closed"))?;
        ensure(model).await
    }

    /// Unloads every resident model; returns how many were loaded.
    ///
    /// # Errors
    ///
    /// Returns [`AsrError::Actor`] when the worker has stopped.
    pub async fn evict(&self) -> Result<usize> {
        let (reply, reply_rx) = flume::bounded(1);
        self.cmd_tx
            .send_async(Command::Evict { reply })
            .await
            .map_err(|_| AsrError::actor("the recognizer worker has stopped"))?;
        reply_rx
            .recv_async()
            .await
            .map_err(|_| AsrError::actor("the recognizer worker dropped the request"))
    }

    /// Stops the worker after the request in progress and joins its thread.
    ///
    /// # Errors
    ///
    /// Returns [`AsrError::Actor`] when the worker thread panicked.
    pub fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::Release);
        let (reply, reply_rx) = flume::bounded(1);
        if self.cmd_tx.send(Command::Shutdown { reply }).is_ok() {
            let _ = reply_rx.recv();
        }
        if let Ok(mut guard) = self.worker.lock()
            && let Some(handle) = guard.take()
        {
            handle
                .join()
                .map_err(|_| AsrError::actor("the recognizer worker panicked"))?;
        }
        Ok(())
    }

    /// Stops the worker (if running) and starts a fresh one with no models loaded.
    ///
    /// # Errors
    ///
    /// Returns [`AsrError::Actor`] when the new thread cannot be spawned.
    pub fn restart(&mut self) -> Result<()> {
        let _ = self.stop();
        let (cmd_tx, running, handle) = spawn_worker(Arc::clone(&self.loaded))?;
        self.cmd_tx = cmd_tx;
        self.running = running;
        if let Ok(mut guard) = self.worker.lock() {
            *guard = Some(handle);
        }
        Ok(())
    }
}

impl Drop for AsrEngine {
    fn drop(&mut self) {
        if Arc::strong_count(&self.worker) == 1 {
            let _ = self.stop();
        }
    }
}

#[allow(clippy::type_complexity)]
fn spawn_worker(
    loaded: Arc<ArcSwap<Vec<AsrModel>>>,
) -> Result<(Sender<Command>, Arc<AtomicBool>, JoinHandle<()>)> {
    let (cmd_tx, cmd_rx) = flume::bounded(CHANNEL_CAPACITY);
    let running = Arc::new(AtomicBool::new(true));
    let flag = Arc::clone(&running);
    let handle = thread::Builder::new()
        .name("index-tts-asr-worker".to_owned())
        .spawn(move || {
            Worker {
                recognizers: BTreeMap::new(),
                loaded,
            }
            .run(&cmd_rx);
            flag.store(false, Ordering::Release);
        })
        .map_err(|err| AsrError::actor(format!("cannot spawn the recognizer worker: {err}")))?;
    Ok((cmd_tx, running, handle))
}

struct Worker {
    recognizers: BTreeMap<AsrModel, OfflineRecognizer>,
    loaded: Arc<ArcSwap<Vec<AsrModel>>>,
}

impl Worker {
    fn run(mut self, cmd_rx: &Receiver<Command>) {
        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                Command::Transcribe {
                    files,
                    samples,
                    reply,
                } => {
                    let _ = reply.send(self.transcribe(&files, &samples));
                }
                Command::Evict { reply } => {
                    let count = self.recognizers.len();
                    self.recognizers.clear();
                    self.publish();
                    let _ = reply.send(count);
                }
                Command::Shutdown { reply } => {
                    let _ = reply.send(());
                    break;
                }
            }
        }
    }

    fn publish(&self) {
        self.loaded
            .store(Arc::new(self.recognizers.keys().copied().collect()));
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "sample counts stay far below f32 precision limits"
    )]
    fn transcribe(&mut self, files: &ModelFiles, samples: &[f32]) -> Result<Transcript> {
        let model = files.model;
        let started = Instant::now();
        let load_ms = match self.recognizers.entry(model) {
            Entry::Occupied(_) => 0.0,
            Entry::Vacant(slot) => {
                let recognizer =
                    OfflineRecognizer::create(&recognizer_config(files)?).ok_or_else(|| {
                        AsrError::model(model.as_str(), "sherpa-onnx failed to load it")
                    })?;
                slot.insert(recognizer);
                self.publish();
                let ms = started.elapsed().as_secs_f64() * 1000.0;
                info!(
                    model = model.as_str(),
                    load_ms = ms,
                    "Loaded speech recognizer"
                );
                ms
            }
        };
        let recognizer = self
            .recognizers
            .get(&model)
            .ok_or_else(|| AsrError::actor("recognizer missing after load"))?;
        let t_recognize = Instant::now();
        let stream = recognizer.create_stream();
        stream.accept_waveform(i32::try_from(ASR_SAMPLE_RATE).unwrap_or(i32::MAX), samples);
        recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| AsrError::model(model.as_str(), "the recognizer returned no result"))?;
        let recognize_ms = t_recognize.elapsed().as_secs_f64() * 1000.0;
        let timestamps = result.timestamps.unwrap_or_default();
        let durations = result.durations.unwrap_or_default();
        let raw: Vec<RawToken> = result
            .tokens
            .into_iter()
            .enumerate()
            .map(|(i, text)| RawToken {
                text,
                start: timestamps.get(i).copied().unwrap_or_default(),
                duration: durations.get(i).copied(),
            })
            .collect();
        let duration = samples.len() as f32 / ASR_SAMPLE_RATE as f32;
        debug!(
            model = model.as_str(),
            tokens = raw.len(),
            recognize_ms,
            "Recognized audio"
        );
        Ok(Transcript {
            model,
            tokens: heard_tokens(model, &raw, duration),
            downloaded: false,
            load_ms,
            recognize_ms,
        })
    }
}

/// sherpa-onnx configuration of a resolved model: no inverse text normalization (numbers stay
/// as spoken), automatic language for `SenseVoice`.
fn recognizer_config(files: &ModelFiles) -> Result<OfflineRecognizerConfig> {
    let mut config = OfflineRecognizerConfig::default();
    let model = &mut config.model_config;
    model.num_threads = ASR_THREADS;
    model.provider = Some("cpu".to_owned());
    model.tokens = Some(files.path("tokens.txt")?);
    match files.model {
        AsrModel::SenseVoice => {
            model.sense_voice = OfflineSenseVoiceModelConfig {
                model: Some(files.path("model.int8.onnx")?),
                language: Some("auto".to_owned()),
                use_itn: false,
            };
        }
        AsrModel::Paraformer => {
            model.paraformer = OfflineParaformerModelConfig {
                model: Some(files.path("model.int8.onnx")?),
            };
        }
        AsrModel::Parakeet => {
            model.transducer = OfflineTransducerModelConfig {
                encoder: Some(files.path("encoder.int8.onnx")?),
                decoder: Some(files.path("decoder.int8.onnx")?),
                joiner: Some(files.path("joiner.int8.onnx")?),
            };
            model.model_type = Some("nemo_transducer".to_owned());
        }
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use super::AsrEngine;
    use crate::{
        ASR_SAMPLE_RATE,
        models::{AsrModel, resolve},
    };

    #[test]
    fn test_should_start_and_restart_without_loading_models() -> crate::Result<()> {
        let mut engine = AsrEngine::start()?;
        assert!(engine.is_running());
        engine.restart()?;
        assert!(engine.is_running() && engine.loaded().is_empty());
        engine.stop()?;
        assert!(!engine.is_running());
        Ok(())
    }

    /// Runs when `SenseVoice` is cached: a tone yields monotonic tokens inside the recording.
    #[tokio::test]
    #[allow(clippy::cast_precision_loss)]
    async fn test_should_recognize_with_a_cached_model() -> crate::Result<()> {
        if resolve(AsrModel::SenseVoice).is_err() {
            return Ok(());
        }
        let engine = AsrEngine::start()?;
        let samples: Vec<f32> = (0..ASR_SAMPLE_RATE)
            .map(|i| 0.1 * (TAU * 220.0 * i as f32 / ASR_SAMPLE_RATE as f32).sin())
            .collect();
        let transcript = engine.transcribe(AsrModel::SenseVoice, samples).await?;
        assert!(transcript.load_ms > 0.0);
        assert!(
            transcript
                .tokens
                .windows(2)
                .all(|w| w[0].start <= w[1].start)
        );
        assert!(transcript.tokens.iter().all(|t| t.end <= 1.0));
        assert_eq!(engine.loaded(), [AsrModel::SenseVoice]);
        assert_eq!(engine.evict().await?, 1);
        assert_eq!(engine.loaded(), []);
        Ok(())
    }
}

//! Dedicated OS-thread `MlxWorkerActor` and thread-safe [`IndexTtsEngine`] handle.
//!
//! Because Apple Silicon Metal command buffers and `mlx_rs::Array` operations are bound to thread
//! stream contexts and do not implement `Send`/`Sync`, the worker actor owns all MLX model
//! weights on a dedicated worker thread and communicates with callers over bounded `flume`
//! channels.

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use flume::{Receiver, Sender};
use mlx_rs::{
    Array,
    ops::concatenate,
    random::{key, normal},
};
use tracing::{debug, info};

use crate::{
    config::{EmoAlpha, EngineConfig, SpeakerSource, SynthesisRequest, VoiceExtractRequest},
    dsp::f32_to_pcm_i16,
    engine::{
        stream::{AudioChunk, StageTimings, SynthesisResult, SynthesisStream},
        telemetry::{EngineTelemetry, StartupEvent, StartupObserver, TelemetrySnapshot},
    },
    error::{IndexTtsError, Result},
    frontend::Frontend,
    models::{
        BigVgan, CFM_IN_CHANNELS, CfmDit, DEFAULT_KV_MAX_SEQ, Gpt, KvCache, LengthRegulator,
        SamplerRng, SemanticCodec, SpkProjWeights, generate_codes, resolve_model_dir,
    },
    voice::{
        CONDS_DIM, CONDS_FRAMES, EmoMatrix, PROMPT_CONDITION_DIM, REF_MEL_BINS, STYLE_DIM,
        SpeakerContext, VoiceEntry, VoiceExtractor, VoiceStore, default_voices_dir,
    },
};

/// Bounded capacity of the actor command channel.
pub const ENGINE_CHANNEL_CAPACITY: usize = 32;
/// Bounded capacity of per-request streaming chunk channels.
pub const STREAM_CHANNEL_CAPACITY: usize = 8;
/// Ratio of Mel frames to semantic codec frames (`22050 / (256 * 50) = 1.72265625`, rounded to
/// `1.72` in `pipeline.py:120`).
pub const MEL_TO_CODEC_RATIO: f64 = 1.72;

enum EngineCommand {
    SynthesizeStream {
        req: Box<SynthesisRequest>,
        chunk_tx: Sender<Result<AudioChunk>>,
    },
    ExtractVoice {
        req: VoiceExtractRequest,
        reply_tx: Sender<Result<VoiceEntry>>,
    },
    EvictVoiceExtractor {
        reply_tx: Sender<Result<bool>>,
    },
    Shutdown {
        reply_tx: Sender<()>,
    },
}

struct CachedRefAudio {
    ref_path: PathBuf,
    emo_path: Option<PathBuf>,
    emo_alpha: EmoAlpha,
    context: Arc<SpeakerContext>,
}

struct MlxWorkerState {
    config: EngineConfig,
    model_dir: PathBuf,
    voice_store: VoiceStore,
    frontend: Frontend,
    gpt: Gpt,
    kv_cache: KvCache,
    codec: SemanticCodec,
    regulator: LengthRegulator,
    cfm: CfmDit,
    bigvgan: BigVgan,
    spk_proj: SpkProjWeights,
    emo_matrix: EmoMatrix,
    voice_extractor: Option<VoiceExtractor>,
    last_ref_cache: Option<CachedRefAudio>,
    shutdown_flag: Arc<AtomicBool>,
    telemetry: Arc<EngineTelemetry>,
}

/// Loads one component, reporting its load time to the startup observer.
fn load_component<T>(
    observer: &dyn Fn(&StartupEvent),
    component: &'static str,
    load: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let started = Instant::now();
    let loaded = load()?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    debug!(component, elapsed_ms, "Loaded synthesis component");
    observer(&StartupEvent::ComponentLoaded {
        component,
        elapsed_ms,
    });
    Ok(loaded)
}

impl MlxWorkerState {
    fn initialize(
        config: EngineConfig,
        model_dir: PathBuf,
        voice_store: VoiceStore,
        shutdown_flag: Arc<AtomicBool>,
        telemetry: Arc<EngineTelemetry>,
        observer: &dyn Fn(&StartupEvent),
    ) -> Result<Self> {
        let t0 = Instant::now();
        let _ = mlx_rs::memory::set_cache_limit(128 * 1024 * 1024);
        let frontend = load_component(observer, "frontend", || {
            Frontend::from_model_dir(
                &model_dir,
                config.text_pos_capacity,
                config.use_normalization,
                config.glossary_path.as_deref(),
            )
        })?;
        let gpt = load_component(observer, "gpt", || {
            Gpt::load_from_path(model_dir.join("gpt_int8.safetensors"))
        })?;
        let kv_cache = KvCache::new(DEFAULT_KV_MAX_SEQ)?;
        let codec = load_component(observer, "semantic_codec", || {
            SemanticCodec::load_from_path(model_dir.join("raw/semantic_codec.safetensors"))
        })?;
        let regulator = load_component(observer, "length_regulator", || {
            LengthRegulator::load_from_path(model_dir.join("raw/length_regulator.safetensors"))
        })?;
        let cfm = load_component(observer, "cfm", || {
            CfmDit::load_from_path(model_dir.join("cfm_estimator.safetensors"))
        })?;
        let bigvgan = load_component(observer, "bigvgan", || {
            BigVgan::load_from_path(model_dir.join("raw/bigvgan.safetensors"))
        })?;
        let (spk_proj, emo_matrix) = load_component(observer, "emotion_projection", || {
            Ok((
                SpkProjWeights::load_from_path(model_dir.join("spk_proj.npz"))?,
                EmoMatrix::load_or_embedded(Some(&model_dir))?,
            ))
        })?;
        let _ = mlx_rs::memory::clear_cache();
        telemetry.publish_memory();

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        info!(
            elapsed_ms,
            model_dir = %model_dir.display(),
            "Initialized IndexTTS-2.5 Stage 1..3 synthesis models on Metal GPU"
        );
        observer(&StartupEvent::EngineReady { elapsed_ms });

        Ok(Self {
            config,
            model_dir,
            voice_store,
            frontend,
            gpt,
            kv_cache,
            codec,
            regulator,
            cfm,
            bigvgan,
            spk_proj,
            emo_matrix,
            voice_extractor: None,
            last_ref_cache: None,
            shutdown_flag,
            telemetry,
        })
    }

    fn ensure_voice_extractor(&mut self) -> Result<&VoiceExtractor> {
        if self.voice_extractor.is_none() {
            let t0 = Instant::now();
            let extractor = VoiceExtractor::load_from_model_dir(&self.model_dir)?;
            info!(
                elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0,
                "Lazily loaded Stage 0 VoiceExtractor encoders"
            );
            self.voice_extractor = Some(extractor);
            self.telemetry.set_voice_extractor_loaded(true);
        }
        self.voice_extractor
            .as_ref()
            .ok_or_else(|| IndexTtsError::actor("failed to initialize VoiceExtractor"))
    }

    fn resolve_speaker(&mut self, req: &SynthesisRequest) -> Result<(Arc<SpeakerContext>, f64)> {
        let t0 = Instant::now();
        match &req.speaker {
            SpeakerSource::VoiceId(voice_id) => {
                let ctx = self.voice_store.get(voice_id)?;
                Ok((ctx, t0.elapsed().as_secs_f64() * 1000.0))
            }
            SpeakerSource::VoiceFile(path) => {
                let ctx = Arc::new(SpeakerContext::load_from_path(path)?);
                Ok((ctx, t0.elapsed().as_secs_f64() * 1000.0))
            }
            SpeakerSource::ReferenceAudio(ref_path) => {
                if let Some(cached) = &self.last_ref_cache
                    && cached.ref_path == *ref_path
                    && cached.emo_path == req.emo_audio_path
                    && (cached.emo_alpha.get() - req.emo_alpha.get()).abs() < 1e-6
                {
                    return Ok((
                        Arc::clone(&cached.context),
                        t0.elapsed().as_secs_f64() * 1000.0,
                    ));
                }
                let extracted = if let Some(extractor) = self.voice_extractor.as_ref() {
                    Arc::new(extractor.extract_from_path(
                        ref_path,
                        req.emo_audio_path.as_deref(),
                        req.emo_alpha,
                        &self.regulator,
                    )?)
                } else {
                    Arc::new(VoiceExtractor::extract_from_model_dir(
                        &self.model_dir,
                        ref_path,
                        req.emo_audio_path.as_deref(),
                        req.emo_alpha,
                        &self.regulator,
                    )?)
                };
                self.last_ref_cache = Some(CachedRefAudio {
                    ref_path: ref_path.clone(),
                    emo_path: req.emo_audio_path.clone(),
                    emo_alpha: req.emo_alpha,
                    context: Arc::clone(&extracted),
                });
                Ok((extracted, t0.elapsed().as_secs_f64() * 1000.0))
            }
        }
    }

    fn handle_extract_voice(&mut self, req: &VoiceExtractRequest) -> Result<VoiceEntry> {
        if self.voice_extractor.is_none() {
            let _ = self.ensure_voice_extractor()?;
        }
        let Some(extractor) = self.voice_extractor.as_ref() else {
            return Err(IndexTtsError::actor("VoiceExtractor unavailable"));
        };
        let mut ctx = extractor.extract_from_path(
            &req.ref_audio_path,
            req.emo_audio_path.as_deref(),
            req.emo_alpha,
            &self.regulator,
        )?;
        if let Some(ref emo_vec) = req.emo_vector
            && !emo_vec.is_zero()
        {
            ctx.conds = ctx.resolve_conds_with_emotion(
                &self.spk_proj,
                &self.emo_matrix,
                emo_vec,
                req.emo_alpha,
            )?;
        }
        let _ = self.voice_store.register(req.voice_id.clone(), ctx)?;
        self.voice_store.inspect(&req.voice_id)
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn handle_synthesize_stream(
        &mut self,
        req: &SynthesisRequest,
        chunk_tx: &Sender<Result<AudioChunk>>,
    ) {
        let (spk, voice_extract_ms) = match self.resolve_speaker(req) {
            Ok(pair) => pair,
            Err(err) => {
                let _ = chunk_tx.send(Err(err));
                return;
            }
        };

        let t_fe = Instant::now();
        let max_tok = req.max_text_tokens_per_segment.get() as usize;
        let segments = self
            .frontend
            .prepare(req.text.as_str(), req.language, max_tok);
        let frontend_ms = t_fe.elapsed().as_secs_f64() * 1000.0;

        if segments.is_empty() {
            return;
        }

        let resolved_conds = match req.emo_vector.as_ref() {
            Some(emo_vec) if !emo_vec.is_zero() => match spk.resolve_conds_with_emotion(
                &self.spk_proj,
                &self.emo_matrix,
                emo_vec,
                req.emo_alpha,
            ) {
                Ok(c) => Cow::Owned(c),
                Err(err) => {
                    let _ = chunk_tx.send(Err(err));
                    return;
                }
            },
            _ => Cow::Borrowed(spk.conds.as_slice()),
        };

        let p_i32 = spk.prompt_frames.get() as i32;
        let conds_arr =
            Array::from_slice(&resolved_conds, &[1, CONDS_FRAMES as i32, CONDS_DIM as i32]);
        let prompt_cond_arr = Array::from_slice(
            &spk.prompt_condition,
            &[1, p_i32, PROMPT_CONDITION_DIM as i32],
        );
        let ref_mel_arr = Array::from_slice(&spk.ref_mel, &[1, REF_MEL_BINS as i32, p_i32]);
        let style_arr = Array::from_slice(&spk.style, &[1, STYLE_DIM as i32]);

        let lang_id = req.language.token_id();
        let mut rng = SamplerRng::new(req.sampling.seed.unwrap_or(0x5EED_CAFE_2026));
        let total_segments = segments.len();

        for (seg_idx, seg_ids) in segments.into_iter().enumerate() {
            if self.shutdown_flag.load(Ordering::Relaxed) || chunk_tx.is_disconnected() {
                debug!(
                    seg_idx,
                    "Synthesis stream cancelled or engine shutting down"
                );
                break;
            }

            let seg_ids_i32: Vec<i32> = seg_ids.into_iter().map(|id| id as i32).collect();
            let seg_res = self.synthesize_single_segment(
                seg_idx,
                total_segments,
                &seg_ids_i32,
                lang_id,
                &conds_arr,
                &prompt_cond_arr,
                &ref_mel_arr,
                &style_arr,
                req,
                &mut rng,
                if seg_idx == 0 { voice_extract_ms } else { 0.0 },
                if seg_idx == 0 { frontend_ms } else { 0.0 },
            );

            match seg_res {
                Ok(Some(chunk)) => {
                    if chunk_tx.send(Ok(chunk)).is_err() {
                        break;
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    let _ = chunk_tx.send(Err(err));
                    break;
                }
            }
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::too_many_arguments
    )]
    fn synthesize_single_segment(
        &mut self,
        seg_idx: usize,
        total_segments: usize,
        token_ids: &[i32],
        lang_id: i32,
        conds_arr: &Array,
        prompt_cond_arr: &Array,
        ref_mel_arr: &Array,
        style_arr: &Array,
        req: &SynthesisRequest,
        rng: &mut SamplerRng,
        voice_extract_ms: f64,
        frontend_ms: f64,
    ) -> Result<Option<AudioChunk>> {
        let seg_start = Instant::now();

        // Stage 1: Int8 GPT Autoregressive code generation
        let t_gpt = Instant::now();
        let codes = generate_codes(
            &self.gpt,
            token_ids,
            conds_arr,
            lang_id,
            &mut self.kv_cache,
            &req.sampling,
            rng,
        )?;
        let gpt_ms = t_gpt.elapsed().as_secs_f64() * 1000.0;
        if codes.is_empty() {
            return Ok(None);
        }

        // Stage 2a: SemanticCodec VQ decode + ConvNeXt + 2x temporal repeat
        let t_codec = Instant::now();
        let codes_arr = Array::from_slice(&codes, &[1, codes.len() as i32]);
        let s_infer = self.codec.decode(&codes_arr)?;
        let codec_ms = t_codec.elapsed().as_secs_f64() * 1000.0;

        // Stage 2b: LengthRegulator + prompt concatenation
        let t_reg = Instant::now();
        let s_len = usize::try_from(s_infer.dim(1)).unwrap_or(1);
        let duration_factor = f64::from(req.cfm.duration_factor.get());
        let target_len = ((s_len as f64) * MEL_TO_CODEC_RATIO * duration_factor).floor() as usize;
        let target_len = target_len.max(1);
        let cond = self.regulator.forward(&s_infer, target_len)?;
        let cat_condition = concatenate(&[prompt_cond_arr, &cond], 1)?;
        cat_condition.eval()?;
        let regulator_ms = t_reg.elapsed().as_secs_f64() * 1000.0;

        // Stage 2c: CFM 25-step all-GPU Euler ODE solver
        let t_cfm = Instant::now();
        let t_total_i32 = cat_condition.dim(1);
        let cfm_key = req
            .sampling
            .seed
            .map(|s| key(s.wrapping_add(seg_idx as u64)));
        let resolved_key = match cfm_key {
            Some(k) => Some(k?),
            None => None,
        };
        let noise_z = normal::<f32>(
            &[1, CFM_IN_CHANNELS as i32, t_total_i32],
            None,
            None,
            resolved_key.as_ref(),
        )?;
        let mel = self.cfm.solve_euler(
            &cat_condition,
            ref_mel_arr,
            style_arr,
            &noise_z,
            req.cfm.n_timesteps as usize,
            req.cfm.cfg_rate.get(),
        )?;
        let cfm_ms = t_cfm.elapsed().as_secs_f64() * 1000.0;

        // Stage 3: BigVGAN neural vocoder
        let t_voc = Instant::now();
        let wav_arr = self.bigvgan.forward(&mel)?;
        let wav_slice = wav_arr.try_as_slice::<f32>()?;
        let pcm_i16 = f32_to_pcm_i16(wav_slice);
        drop(wav_arr);
        drop(mel);
        let _ = mlx_rs::memory::clear_cache();
        let bigvgan_ms = t_voc.elapsed().as_secs_f64() * 1000.0;

        let total_ms = voice_extract_ms + frontend_ms + seg_start.elapsed().as_secs_f64() * 1000.0;
        let timings = StageTimings {
            voice_extract_ms,
            frontend_ms,
            gpt_ms,
            codec_ms,
            regulator_ms,
            cfm_ms,
            bigvgan_ms,
            total_ms,
        };

        Ok(Some(AudioChunk::new(
            seg_idx,
            total_segments,
            pcm_i16,
            self.config.sample_rate,
            codes.len(),
            timings,
        )))
    }
}

fn run_actor_loop(mut state: MlxWorkerState, cmd_rx: &Receiver<EngineCommand>) {
    while let Ok(cmd) = cmd_rx.recv() {
        let shutdown = matches!(cmd, EngineCommand::Shutdown { .. });
        state.handle_command(cmd);
        if shutdown {
            break;
        }
        state.telemetry.publish_memory();
    }
}

impl MlxWorkerState {
    fn handle_command(&mut self, cmd: EngineCommand) {
        match cmd {
            EngineCommand::SynthesizeStream { req, chunk_tx } => {
                self.handle_synthesize_stream(&req, &chunk_tx);
            }
            EngineCommand::ExtractVoice { req, reply_tx } => {
                let res = self.handle_extract_voice(&req);
                let _ = reply_tx.send(res);
            }
            EngineCommand::EvictVoiceExtractor { reply_tx } => {
                let had_extractor = self.voice_extractor.take().is_some();
                self.telemetry.set_voice_extractor_loaded(false);
                let _ = mlx_rs::memory::clear_cache();
                let _ = reply_tx.send(Ok(had_extractor));
            }
            EngineCommand::Shutdown { reply_tx } => {
                self.shutdown_flag.store(true, Ordering::Relaxed);
                let _ = reply_tx.send(());
            }
        }
    }
}

/// Thread-safe handle to the `IndexTTS-2.5` inference engine Actor.
///
/// Owns a dedicated background OS thread hosting the MLX Metal GPU models and provides
/// both streaming ([`IndexTtsEngine::synthesize_stream`]) and full-utterance
/// ([`IndexTtsEngine::synthesize`], [`IndexTtsEngine::synthesize_blocking`]) synthesis APIs,
/// as well as lifecycle management (`start`, `stop`, `restart`, `evict_voice_extractor`).
#[derive(Debug, Clone)]
pub struct IndexTtsEngine {
    config: EngineConfig,
    model_dir: PathBuf,
    voice_store: VoiceStore,
    cmd_tx: Sender<EngineCommand>,
    shutdown_flag: Arc<AtomicBool>,
    telemetry: Arc<EngineTelemetry>,
    worker_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl IndexTtsEngine {
    /// Starts the [`IndexTtsEngine`] actor thread and initializes Stage 1..3 synthesis models on
    /// the Metal GPU.
    ///
    /// Stage 0 voice extraction encoders (`w2v-bert-2.0`, `EmoVec`, `CAMPPlus`) are NOT loaded at
    /// startup unless needed, keeping baseline steady-state memory at ~1.68 GB.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if configuration validation fails, the local model snapshot
    /// cannot be resolved, or weight loading fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{config::EngineConfig, engine::IndexTtsEngine, models::resolve_model_dir};
    ///
    /// if resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx").is_ok() {
    ///     let engine = IndexTtsEngine::start(EngineConfig::default())?;
    ///     assert!(engine.is_running());
    ///     engine.stop()?;
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn start(config: EngineConfig) -> Result<Self> {
        Self::start_with_observer(config, Box::new(|_| {}))
    }

    /// Starts the engine like [`IndexTtsEngine::start`], reporting per-component load progress
    /// to `observer` from the worker thread.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if configuration validation, model resolution, or weight
    /// loading fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{config::EngineConfig, engine::IndexTtsEngine, models::resolve_model_dir};
    ///
    /// if resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx").is_ok() {
    ///     let engine = IndexTtsEngine::start_with_observer(
    ///         EngineConfig::default(),
    ///         Box::new(|event| eprintln!("{event:?}")),
    ///     )?;
    ///     engine.stop()?;
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn start_with_observer(config: EngineConfig, observer: StartupObserver) -> Result<Self> {
        config.validate_config()?;
        let model_dir = resolve_model_dir(config.model_dir.as_deref(), &config.model_repo)?;
        observer(&StartupEvent::ModelResolved {
            model_dir: model_dir.clone(),
        });
        let voices_dir = config.voices_dir.clone().unwrap_or_else(default_voices_dir);
        let voice_store = VoiceStore::new(voices_dir)?;
        let telemetry = Arc::new(EngineTelemetry::default());
        let (cmd_tx, shutdown_flag, handle) = spawn_worker_thread(
            config.clone(),
            model_dir.clone(),
            voice_store.clone(),
            Arc::clone(&telemetry),
            observer,
        )?;

        Ok(Self {
            config,
            model_dir,
            voice_store,
            cmd_tx,
            shutdown_flag,
            telemetry,
            worker_handle: Arc::new(Mutex::new(Some(handle))),
        })
    }

    /// Returns a lock-free snapshot of worker runtime state (memory, resident encoders).
    #[must_use]
    pub fn telemetry(&self) -> TelemetrySnapshot {
        self.telemetry.snapshot()
    }

    /// Alias for [`IndexTtsEngine::start`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if startup fails.
    pub fn new(config: EngineConfig) -> Result<Self> {
        Self::start(config)
    }

    /// Returns whether the background `MlxWorkerActor` thread is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.shutdown_flag.load(Ordering::Relaxed) && !self.cmd_tx.is_disconnected()
    }

    /// Returns a reference to the active [`EngineConfig`].
    #[must_use]
    pub const fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Returns the resolved model snapshot directory path.
    #[must_use]
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Returns a reference to the [`VoiceStore`].
    #[must_use]
    pub const fn voice_store(&self) -> &VoiceStore {
        &self.voice_store
    }

    /// Submits a streaming synthesis request and returns a [`SynthesisStream`] yielding
    /// [`AudioChunk`]s sentence-by-sentence.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Actor`] if the background worker has stopped.
    pub fn synthesize_stream(&self, req: SynthesisRequest) -> Result<SynthesisStream> {
        if !self.is_running() {
            return Err(IndexTtsError::actor("IndexTtsEngine actor is not running"));
        }
        let (chunk_tx, chunk_rx) = flume::bounded(STREAM_CHANNEL_CAPACITY);
        self.cmd_tx
            .send(EngineCommand::SynthesizeStream {
                req: Box::new(req),
                chunk_tx,
            })
            .map_err(|_| {
                IndexTtsError::actor("failed to send SynthesizeStream command to actor")
            })?;
        Ok(SynthesisStream::new(chunk_rx))
    }

    /// Asynchronously synthesizes an entire [`SynthesisRequest`] and concatenates all sentence
    /// segments (with `interval_silence_ms` between segments) into a [`SynthesisResult`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if speaker resolution, tokenization, or GPU synthesis fails.
    #[allow(clippy::cast_precision_loss)]
    pub async fn synthesize(&self, req: SynthesisRequest) -> Result<SynthesisResult> {
        let start = Instant::now();
        let interval_ms = req.interval_silence_ms;
        let sample_rate = self.config.sample_rate;
        let stream = self.synthesize_stream(req)?;

        let mut chunks = Vec::new();
        while let Some(chunk_res) = stream.next_chunk().await {
            chunks.push(chunk_res?);
        }
        Ok(assemble_synthesis_result(
            chunks,
            sample_rate,
            interval_ms,
            start.elapsed().as_secs_f32(),
        ))
    }

    /// Synchronously synthesizes an entire [`SynthesisRequest`] into a [`SynthesisResult`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if speaker resolution, tokenization, or GPU synthesis fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn synthesize_blocking(&self, req: SynthesisRequest) -> Result<SynthesisResult> {
        let start = Instant::now();
        let interval_ms = req.interval_silence_ms;
        let sample_rate = self.config.sample_rate;
        let stream = self.synthesize_stream(req)?;

        let mut chunks = Vec::new();
        for chunk_res in stream {
            chunks.push(chunk_res?);
        }
        Ok(assemble_synthesis_result(
            chunks,
            sample_rate,
            interval_ms,
            start.elapsed().as_secs_f32(),
        ))
    }

    /// Asynchronously extracts a [`SpeakerContext`] from a reference audio file and registers it
    /// in the [`VoiceStore`] as `<voice_id>.itts_voice`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if extraction or persistence fails.
    pub async fn extract_voice(&self, req: VoiceExtractRequest) -> Result<VoiceEntry> {
        let (reply_tx, reply_rx) = flume::bounded(1);
        self.cmd_tx
            .send(EngineCommand::ExtractVoice { req, reply_tx })
            .map_err(|_| IndexTtsError::actor("failed to send ExtractVoice command to actor"))?;
        reply_rx
            .recv_async()
            .await
            .map_err(|_| IndexTtsError::actor("actor dropped ExtractVoice reply channel"))?
    }

    /// Synchronously extracts a [`SpeakerContext`] from a reference audio file and registers it
    /// in the [`VoiceStore`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if extraction or persistence fails.
    pub fn extract_voice_blocking(&self, req: VoiceExtractRequest) -> Result<VoiceEntry> {
        let (reply_tx, reply_rx) = flume::bounded(1);
        self.cmd_tx
            .send(EngineCommand::ExtractVoice { req, reply_tx })
            .map_err(|_| IndexTtsError::actor("failed to send ExtractVoice command to actor"))?;
        reply_rx
            .recv()
            .map_err(|_| IndexTtsError::actor("actor dropped ExtractVoice reply channel"))?
    }

    /// Asynchronously evicts the Stage 0 [`VoiceExtractor`] models (`w2v-bert-2.0`, `EmoVec`,
    /// `CAMPPlus`) from memory if loaded, returning `true` if models were evicted.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Actor`] if the worker thread has stopped.
    pub async fn evict_voice_extractor(&self) -> Result<bool> {
        let (reply_tx, reply_rx) = flume::bounded(1);
        self.cmd_tx
            .send(EngineCommand::EvictVoiceExtractor { reply_tx })
            .map_err(|_| IndexTtsError::actor("failed to send EvictVoiceExtractor command"))?;
        reply_rx
            .recv_async()
            .await
            .map_err(|_| IndexTtsError::actor("actor dropped EvictVoiceExtractor reply channel"))?
    }

    /// Synchronously evicts the Stage 0 [`VoiceExtractor`] models from memory if loaded.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Actor`] if the worker thread has stopped.
    pub fn evict_voice_extractor_blocking(&self) -> Result<bool> {
        let (reply_tx, reply_rx) = flume::bounded(1);
        self.cmd_tx
            .send(EngineCommand::EvictVoiceExtractor { reply_tx })
            .map_err(|_| IndexTtsError::actor("failed to send EvictVoiceExtractor command"))?;
        reply_rx
            .recv()
            .map_err(|_| IndexTtsError::actor("actor dropped EvictVoiceExtractor reply channel"))?
    }

    /// Signals the background `MlxWorkerActor` to stop after completing any in-flight segment and
    /// joins the OS thread.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Actor`] if joining the worker thread fails.
    pub fn stop(&self) -> Result<()> {
        self.shutdown_flag.store(true, Ordering::Relaxed);
        let (reply_tx, reply_rx) = flume::bounded(1);
        if self
            .cmd_tx
            .send(EngineCommand::Shutdown { reply_tx })
            .is_ok()
        {
            let _ = reply_rx.recv();
        }
        if let Ok(mut guard) = self.worker_handle.lock()
            && let Some(handle) = guard.take()
        {
            handle
                .join()
                .map_err(|_| IndexTtsError::actor("MlxWorkerActor thread panicked during join"))?;
        }
        Ok(())
    }

    /// Stops the current worker thread (if running) and spawns a fresh `MlxWorkerActor` thread.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if stopping or re-initializing the worker fails.
    pub fn restart(&mut self) -> Result<()> {
        let _ = self.stop();
        let (cmd_tx, shutdown_flag, handle) = spawn_worker_thread(
            self.config.clone(),
            self.model_dir.clone(),
            self.voice_store.clone(),
            Arc::clone(&self.telemetry),
            Box::new(|_| {}),
        )?;
        self.cmd_tx = cmd_tx;
        self.shutdown_flag = shutdown_flag;
        if let Ok(mut guard) = self.worker_handle.lock() {
            *guard = Some(handle);
        }
        Ok(())
    }
}

impl Drop for IndexTtsEngine {
    fn drop(&mut self) {
        if Arc::strong_count(&self.worker_handle) == 1 {
            let _ = self.stop();
        }
    }
}

#[allow(clippy::type_complexity)]
fn spawn_worker_thread(
    config: EngineConfig,
    model_dir: PathBuf,
    voice_store: VoiceStore,
    telemetry: Arc<EngineTelemetry>,
    observer: StartupObserver,
) -> Result<(Sender<EngineCommand>, Arc<AtomicBool>, JoinHandle<()>)> {
    let (cmd_tx, cmd_rx) = flume::bounded(ENGINE_CHANNEL_CAPACITY);
    let (init_tx, init_rx) = flume::bounded::<Result<()>>(1);
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let flag_clone = Arc::clone(&shutdown_flag);

    let handle = thread::Builder::new()
        .name("index-tts-mlx-worker".to_owned())
        .spawn(move || {
            match MlxWorkerState::initialize(
                config,
                model_dir,
                voice_store,
                flag_clone,
                telemetry,
                observer.as_ref(),
            ) {
                Ok(state) => {
                    let _ = init_tx.send(Ok(()));
                    run_actor_loop(state, &cmd_rx);
                }
                Err(err) => {
                    let _ = init_tx.send(Err(err));
                }
            }
        })
        .map_err(|err| {
            IndexTtsError::actor(format!("failed to spawn MlxWorkerActor thread: {err}"))
        })?;

    match init_rx.recv() {
        Ok(Ok(())) => Ok((cmd_tx, shutdown_flag, handle)),
        Ok(Err(err)) => {
            let _ = handle.join();
            Err(err)
        }
        Err(_) => {
            let _ = handle.join();
            Err(IndexTtsError::actor(
                "MlxWorkerActor thread terminated unexpectedly during initialization",
            ))
        }
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn assemble_synthesis_result(
    chunks: Vec<AudioChunk>,
    sample_rate: u32,
    interval_silence_ms: u32,
    synthesis_secs: f32,
) -> SynthesisResult {
    let segments_count = chunks.len();
    let silence_samples = if interval_silence_ms > 0 && segments_count > 1 {
        ((f64::from(sample_rate) * f64::from(interval_silence_ms)) / 1000.0) as usize
    } else {
        0
    };

    let total_samples: usize = chunks.iter().map(|c| c.pcm_i16.len()).sum::<usize>()
        + silence_samples.saturating_mul(segments_count.saturating_sub(1));

    let mut pcm_i16 = Vec::with_capacity(total_samples);
    let mut total_mel_tokens = 0_usize;
    let mut timings = StageTimings::default();

    for (idx, chunk) in chunks.into_iter().enumerate() {
        if idx > 0 && silence_samples > 0 {
            pcm_i16.resize(pcm_i16.len() + silence_samples, 0_i16);
        }
        pcm_i16.extend_from_slice(&chunk.pcm_i16);
        total_mel_tokens += chunk.mel_tokens;
        timings.accumulate_segment(&chunk.timings);
    }
    timings.total_ms = f64::from(synthesis_secs) * 1000.0;

    SynthesisResult::new(
        pcm_i16,
        sample_rate,
        synthesis_secs,
        segments_count,
        total_mel_tokens,
        timings,
    )
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::config::{DEFAULT_MODEL_REPO, Language, SamplingParams, SynthesisText};

    #[test]
    fn test_should_synthesize_with_golden_speaker_and_handle_stream_cancellation() {
        if resolve_model_dir(None, DEFAULT_MODEL_REPO).is_err() {
            return;
        }
        let Ok(temp_voices) = tempfile::tempdir() else {
            return;
        };
        let cfg = EngineConfig::builder()
            .voices_dir(Some(temp_voices.path().to_path_buf()))
            .build();

        let engine_res = IndexTtsEngine::start(cfg);
        assert!(
            engine_res.is_ok(),
            "IndexTtsEngine::start failed: {engine_res:?}"
        );
        let Ok(engine) = engine_res else {
            return;
        };
        assert!(engine.is_running());

        let golden_voice = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("golden_speaker.itts_voice");

        let Ok(text) = SynthesisText::new("你好。") else {
            return;
        };
        let sampling = SamplingParams::builder()
            .greedy(true)
            .max_mel_tokens(32)
            .build();
        let req = SynthesisRequest::builder()
            .text(text)
            .speaker(SpeakerSource::VoiceFile(golden_voice.clone()))
            .language(Language::Zh)
            .sampling(sampling.clone())
            .build();

        let synth_res = engine.synthesize_blocking(req);
        assert!(
            synth_res.is_ok(),
            "synthesize_blocking failed: {synth_res:?}"
        );
        if let Ok(res) = synth_res {
            assert_ne!(res.pcm_i16, [] as [i16; 0]);
            assert_eq!(res.sample_rate, 22_050);
            assert!(res.duration_secs > 0.0);
            assert_eq!(res.segments_count, 1);
            assert!(res.timings.gpt_ms > 0.0);
            assert!(res.timings.cfm_ms > 0.0);
            assert!(res.timings.bigvgan_ms > 0.0);
            assert!(res.to_wav_bytes().is_ok());
        }

        // Test streaming cancellation after first segment
        let Ok(multi_text) = SynthesisText::new("第一句测试。第二句测试。第三句测试。")
        else {
            return;
        };
        let stream_req = SynthesisRequest::builder()
            .text(multi_text)
            .speaker(SpeakerSource::VoiceFile(golden_voice))
            .language(Language::Zh)
            .sampling(sampling)
            .max_text_tokens_per_segment(NonZeroU32::new(6).unwrap_or(NonZeroU32::MIN))
            .build();

        if let Ok(mut stream) = engine.synthesize_stream(stream_req) {
            let first = stream.next();
            assert!(first.is_some());
            if let Some(Ok(chunk)) = first {
                assert_eq!(chunk.segment_index, 0);
                assert!(chunk.total_segments >= 2);
                assert_ne!(chunk.pcm_i16, [] as [i16; 0]);
            }
            drop(stream);
        }

        let evicted = engine.evict_voice_extractor_blocking();
        assert_eq!(evicted.ok(), Some(false));
        assert!(engine.stop().is_ok());
    }
}

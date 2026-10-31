//! Stage 0 [`VoiceExtractor`] and [`VoiceStore`] (`.itts_voice` cache & persistence).
//!
//! - [`VoiceExtractor`] encapsulates the heavy Stage 0 reference-audio encoders ([`SemanticModel`],
//!   [`EmoVec`], [`CampPlus`], [`SpkProjWeights`]) so they can be lazily loaded on demand and
//!   evicted immediately after speaker extraction.
//! - [`VoiceStore`] manages in-memory (`DashMap`) and on-disk
//!   (`~/.cache/index-tts-rs/voices/*.itts_voice`) voice snapshots.

use std::{
    env, fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::Arc,
};

use dashmap::DashMap;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

use crate::{
    config::{
        DEFAULT_MAX_REF_SECONDS, DEFAULT_SAMPLE_RATE, EmoAlpha, VoiceId, validate_safe_file_path,
    },
    dsp::{
        FBANK_SAMPLE_RATE, campplus_fbank, decode_audio_bytes, mel_spectrogram, resample_sinc_hann,
        truncate_ref_audio, w2v_features,
    },
    error::{IndexTtsError, Result},
    models::{CampPlus, EmoVec, LengthRegulator, SemanticModel, SpkProjWeights},
    voice::{SpeakerContext, VoiceMetadata},
};

/// Default relative directory under `$HOME` for `.itts_voice` snapshots.
pub const DEFAULT_VOICES_SUBDIR: &str = ".cache/index-tts-rs/voices";

/// Maximum number of 22,050 Hz samples retained from reference audio (`15 * 22050 = 330,750`).
pub const MAX_REF_SAMPLES_22K: usize =
    (DEFAULT_MAX_REF_SECONDS as usize) * (DEFAULT_SAMPLE_RATE as usize);
/// Maximum number of 16,000 Hz samples retained from reference audio (`15 * 16000 = 240,000`).
pub const MAX_REF_SAMPLES_16K: usize =
    (DEFAULT_MAX_REF_SECONDS as usize) * (FBANK_SAMPLE_RATE as usize);

/// Summary descriptor for a registered `.itts_voice` snapshot in [`VoiceStore`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct VoiceEntry {
    /// Registered voice identifier.
    pub voice_id: String,
    /// Absolute or relative path to the `.itts_voice` file on disk.
    pub path: PathBuf,
    /// Number of reference Mel frames `P`.
    pub prompt_frames: usize,
    /// Approximate duration of the reference prompt in seconds (`P * 256 / 22050`).
    pub duration_secs: f32,
    /// File size on disk in bytes.
    pub size_bytes: u64,
    /// Provenance metadata stored in the `.itts_voice` header.
    pub metadata: VoiceMetadata,
}

/// Stage 0 Reference Audio Feature Extractor (`w2v-bert-2.0` + `EmoVec` + `CAMPPlus` + `spk_proj`).
#[derive(Debug, Clone)]
pub struct VoiceExtractor {
    semantic_model: SemanticModel,
    emo_vec: EmoVec,
    campplus: CampPlus,
    spk_proj: SpkProjWeights,
}

impl VoiceExtractor {
    /// Loads all Stage 0 encoder models from a verified model snapshot directory.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if any weight file is missing or malformed.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{models::resolve_model_dir, voice::VoiceExtractor};
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let extractor = VoiceExtractor::load_from_model_dir(&dir)?;
    ///     assert_eq!(extractor.semantic_model().num_layers(), 17);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn load_from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = model_dir.as_ref();
        let semantic_model =
            SemanticModel::load_from_path(dir.join("raw/semantic_model.safetensors"))?;
        let emo_vec = EmoVec::load_from_path(dir.join("raw/emo.safetensors"))?;
        let campplus = CampPlus::load_from_path(dir.join("raw/campplus.safetensors"))?;
        let spk_proj = SpkProjWeights::load_from_path(dir.join("spk_proj.npz"))?;
        Ok(Self {
            semantic_model,
            emo_vec,
            campplus,
            spk_proj,
        })
    }

    /// Returns a reference to the loaded [`SemanticModel`].
    #[must_use]
    pub const fn semantic_model(&self) -> &SemanticModel {
        &self.semantic_model
    }

    /// Returns a reference to the loaded [`EmoVec`] model.
    #[must_use]
    pub const fn emo_vec(&self) -> &EmoVec {
        &self.emo_vec
    }

    /// Returns a reference to the loaded [`CampPlus`] model.
    #[must_use]
    pub const fn campplus(&self) -> &CampPlus {
        &self.campplus
    }

    /// Extracts a [`SpeakerContext`] from a reference audio file on disk (and an optional separate
    /// emotion reference audio file).
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if audio decoding, DSP feature extraction, or GPU model
    /// execution fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{
    ///     config::EmoAlpha,
    ///     models::{LengthRegulator, resolve_model_dir},
    ///     voice::VoiceExtractor,
    /// };
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let extractor = VoiceExtractor::load_from_model_dir(&dir)?;
    ///     let regulator = LengthRegulator::load_from_path(dir.join("raw/length_regulator.safetensors"))?;
    ///     let wav_22k = vec![0.01_f32; 22050];
    ///     let wav_16k = vec![0.01_f32; 16000];
    ///     let ctx = extractor.extract_from_pcm(
    ///         &wav_22k,
    ///         &wav_16k,
    ///         None,
    ///         "0".repeat(64),
    ///         EmoAlpha::default(),
    ///         &regulator,
    ///     )?;
    ///     assert_eq!(ctx.style.len(), 192);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[allow(clippy::disallowed_methods)]
    pub fn extract_from_path(
        &self,
        ref_audio_path: impl AsRef<Path>,
        emo_audio_path: Option<&Path>,
        emo_alpha: EmoAlpha,
        regulator: &LengthRegulator,
    ) -> Result<SpeakerContext> {
        let ref_path = ref_audio_path.as_ref();
        validate_safe_file_path(ref_path, "ref_audio_path")?;
        let raw_bytes = fs::read(ref_path).map_err(|err| IndexTtsError::io(ref_path, err))?;
        let source_blake3 = blake3::hash(&raw_bytes).to_hex().to_string();
        let ext = ref_path.extension().and_then(|e| e.to_str());
        let decoded = decode_audio_bytes(&raw_bytes, ext)?;
        let audio_22k =
            resample_sinc_hann(&decoded.samples, decoded.sample_rate, DEFAULT_SAMPLE_RATE)?;
        let audio_16k =
            resample_sinc_hann(&decoded.samples, decoded.sample_rate, FBANK_SAMPLE_RATE)?;

        let emo_16k = match emo_audio_path {
            Some(emo_path) if emo_path != ref_path => {
                validate_safe_file_path(emo_path, "emo_audio_path")?;
                let emo_bytes =
                    fs::read(emo_path).map_err(|err| IndexTtsError::io(emo_path, err))?;
                let emo_ext = emo_path.extension().and_then(|e| e.to_str());
                let emo_dec = decode_audio_bytes(&emo_bytes, emo_ext)?;
                Some(resample_sinc_hann(
                    &emo_dec.samples,
                    emo_dec.sample_rate,
                    FBANK_SAMPLE_RATE,
                )?)
            }
            _ => None,
        };

        self.extract_from_pcm(
            &audio_22k,
            &audio_16k,
            emo_16k.as_deref(),
            source_blake3,
            emo_alpha,
            regulator,
        )
    }

    /// Extracts a [`SpeakerContext`] from pre-resampled `22,050 Hz` and `16,000 Hz` PCM buffers.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if feature extraction or GPU inference fails.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn extract_from_pcm(
        &self,
        audio_22k: &[f32],
        audio_16k: &[f32],
        emo_audio_16k: Option<&[f32]>,
        source_audio_blake3: String,
        emo_alpha: EmoAlpha,
        regulator: &LengthRegulator,
    ) -> Result<SpeakerContext> {
        let wav_22k = truncate_ref_audio(audio_22k, DEFAULT_SAMPLE_RATE, DEFAULT_MAX_REF_SECONDS);
        let wav_16k = truncate_ref_audio(audio_16k, FBANK_SAMPLE_RATE, DEFAULT_MAX_REF_SECONDS);

        // 1. W2V-BERT semantic conditioning features [1, Te, 1024]
        let (spk_cond_emb, spk_frames) = self.encode_semantic(wav_16k)?;

        // 2. Emotion vector [1, 1280]
        let emovec_arr = if let Some(emo_16k) = emo_audio_16k {
            let emo_trimmed =
                truncate_ref_audio(emo_16k, FBANK_SAMPLE_RATE, DEFAULT_MAX_REF_SECONDS);
            let (emo_cond_emb, emo_frames) = self.encode_semantic(emo_trimmed)?;
            self.emo_vec.forward(&emo_cond_emb, &[emo_frames])?
        } else {
            self.emo_vec.forward(&spk_cond_emb, &[spk_frames])?
        };
        let emovec_slice = emovec_arr.try_as_slice::<f32>()?;

        // 3. CAMPPlus speaker style vector [1, 192] -> spk_proj [1280]
        let fbank = campplus_fbank(wav_16k)?;
        let fbank_arr = Array::from_slice(&fbank.data, &[1, fbank.num_frames as i32, 80]);
        let style_arr = self.campplus.forward(&fbank_arr)?;
        let style_slice = style_arr.try_as_slice::<f32>()?;

        let spk_cond = self.spk_proj.project(style_slice)?;
        let conds = SpeakerContext::build_conds(&spk_cond, emovec_slice, emo_alpha)?;

        // 4. Reference Mel spectrogram [1, 80, P] and LengthRegulator prompt_condition [1, P, 512]
        let mel = mel_spectrogram(wav_22k)?;
        let prompt_frames = NonZeroUsize::new(mel.num_frames).ok_or_else(|| {
            IndexTtsError::validation("audio_22k", "reference audio produced 0 Mel frames")
        })?;
        let prompt_cond_arr = regulator.forward(&spk_cond_emb, mel.num_frames)?;
        let prompt_cond_slice = prompt_cond_arr.try_as_slice::<f32>()?;
        let prompt_cond_vec = prompt_cond_slice.to_vec();
        let style_vec = style_slice.to_vec();
        let _ = mlx_rs::memory::clear_cache();

        SpeakerContext::new(
            conds,
            prompt_cond_vec,
            mel.data,
            style_vec,
            prompt_frames,
            VoiceMetadata::new(source_audio_blake3, emo_alpha),
        )
    }

    /// Performs one-shot sequential speaker extraction from a model directory, loading and
    /// immediately dropping each Stage 0 encoder (`SemanticModel` -> `EmoVec` -> `CampPlus`) so
    /// only one Stage 0 model resides in memory at any point in time.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if model loading, audio decoding, DSP feature extraction, or GPU
    /// execution fails.
    #[allow(
        clippy::disallowed_methods,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    pub fn extract_from_model_dir(
        model_dir: impl AsRef<Path>,
        ref_audio_path: impl AsRef<Path>,
        emo_audio_path: Option<&Path>,
        emo_alpha: EmoAlpha,
        regulator: &LengthRegulator,
    ) -> Result<SpeakerContext> {
        let dir = model_dir.as_ref();
        let ref_path = ref_audio_path.as_ref();
        validate_safe_file_path(ref_path, "ref_audio_path")?;
        let raw_bytes = fs::read(ref_path).map_err(|err| IndexTtsError::io(ref_path, err))?;
        let source_blake3 = blake3::hash(&raw_bytes).to_hex().to_string();
        let ext = ref_path.extension().and_then(|e| e.to_str());
        let decoded = decode_audio_bytes(&raw_bytes, ext)?;
        let audio_22k =
            resample_sinc_hann(&decoded.samples, decoded.sample_rate, DEFAULT_SAMPLE_RATE)?;
        let audio_16k =
            resample_sinc_hann(&decoded.samples, decoded.sample_rate, FBANK_SAMPLE_RATE)?;

        let emo_16k = match emo_audio_path {
            Some(emo_path) if emo_path != ref_path => {
                validate_safe_file_path(emo_path, "emo_audio_path")?;
                let emo_bytes =
                    fs::read(emo_path).map_err(|err| IndexTtsError::io(emo_path, err))?;
                let emo_ext = emo_path.extension().and_then(|e| e.to_str());
                let emo_dec = decode_audio_bytes(&emo_bytes, emo_ext)?;
                Some(resample_sinc_hann(
                    &emo_dec.samples,
                    emo_dec.sample_rate,
                    FBANK_SAMPLE_RATE,
                )?)
            }
            _ => None,
        };

        let wav_22k = truncate_ref_audio(&audio_22k, DEFAULT_SAMPLE_RATE, DEFAULT_MAX_REF_SECONDS);
        let wav_16k = truncate_ref_audio(&audio_16k, FBANK_SAMPLE_RATE, DEFAULT_MAX_REF_SECONDS);

        // 1. Load & run SemanticModel, then drop it immediately (~859 MB freed)
        let (spk_cond_emb, spk_frames, emo_cond_opt) = {
            let semantic_model =
                SemanticModel::load_from_path(dir.join("raw/semantic_model.safetensors"))?;
            let (spk_emb, spk_f) = Self::encode_semantic_with(&semantic_model, wav_16k)?;
            spk_emb.eval()?;
            let emo_opt = if let Some(emo_pcm) = emo_16k.as_deref() {
                let emo_trimmed =
                    truncate_ref_audio(emo_pcm, FBANK_SAMPLE_RATE, DEFAULT_MAX_REF_SECONDS);
                let (emo_emb, emo_f) = Self::encode_semantic_with(&semantic_model, emo_trimmed)?;
                emo_emb.eval()?;
                Some((emo_emb, emo_f))
            } else {
                None
            };
            (spk_emb, spk_f, emo_opt)
        };
        let _ = mlx_rs::memory::clear_cache();

        // 2. Load & run EmoVec, then drop it immediately (~621 MB freed)
        let emovec_vec = {
            let emo_vec = EmoVec::load_from_path(dir.join("raw/emo.safetensors"))?;
            let emovec_arr = if let Some((emo_cond_emb, emo_frames)) = emo_cond_opt {
                emo_vec.forward(&emo_cond_emb, &[emo_frames])?
            } else {
                emo_vec.forward(&spk_cond_emb, &[spk_frames])?
            };
            emovec_arr.try_as_slice::<f32>()?.to_vec()
        };
        let _ = mlx_rs::memory::clear_cache();

        // 3. Load & run CampPlus + SpkProjWeights, then drop them immediately
        let (style_vec, conds) = {
            let campplus = CampPlus::load_from_path(dir.join("raw/campplus.safetensors"))?;
            let spk_proj = SpkProjWeights::load_from_path(dir.join("spk_proj.npz"))?;
            let fbank = campplus_fbank(wav_16k)?;
            let fbank_arr = Array::from_slice(&fbank.data, &[1, fbank.num_frames as i32, 80]);
            let style_arr = campplus.forward(&fbank_arr)?;
            let style_slice = style_arr.try_as_slice::<f32>()?;
            let spk_cond = spk_proj.project(style_slice)?;
            let conds = SpeakerContext::build_conds(&spk_cond, &emovec_vec, emo_alpha)?;
            (style_slice.to_vec(), conds)
        };
        let _ = mlx_rs::memory::clear_cache();

        // 4. Reference Mel spectrogram [1, 80, P] and LengthRegulator prompt_condition [1, P, 512]
        let mel = mel_spectrogram(wav_22k)?;
        let prompt_frames = NonZeroUsize::new(mel.num_frames).ok_or_else(|| {
            IndexTtsError::validation("audio_22k", "reference audio produced 0 Mel frames")
        })?;
        let prompt_cond_arr = regulator.forward(&spk_cond_emb, mel.num_frames)?;
        let prompt_cond_vec = prompt_cond_arr.try_as_slice::<f32>()?.to_vec();
        drop(spk_cond_emb);
        drop(prompt_cond_arr);
        let _ = mlx_rs::memory::clear_cache();

        SpeakerContext::new(
            conds,
            prompt_cond_vec,
            mel.data,
            style_vec,
            prompt_frames,
            VoiceMetadata::new(source_blake3, emo_alpha),
        )
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn encode_semantic(&self, wav_16k: &[f32]) -> Result<(Array, usize)> {
        Self::encode_semantic_with(&self.semantic_model, wav_16k)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn encode_semantic_with(
        semantic_model: &SemanticModel,
        wav_16k: &[f32],
    ) -> Result<(Array, usize)> {
        let w2v = w2v_features(wav_16k)?;
        let frames_i32 = w2v.num_frames as i32;
        let feat_arr = Array::from_slice(&w2v.input_features, &[1, frames_i32, 160]);
        let mask_arr = Array::from_slice(&w2v.attention_mask, &[1, frames_i32]);
        let out = semantic_model.forward(&feat_arr, Some(&mask_arr))?;
        Ok((out, w2v.num_frames))
    }
}

/// Concurrent in-memory (`DashMap`) and on-disk (`.itts_voice`) voice snapshot store.
#[derive(Debug, Clone)]
pub struct VoiceStore {
    voices_dir: PathBuf,
    cache: Arc<DashMap<VoiceId, Arc<SpeakerContext>>>,
}

impl VoiceStore {
    /// Creates a [`VoiceStore`] rooted at `voices_dir`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `voices_dir` is unsafe.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::voice::VoiceStore;
    ///
    /// let dir = tempfile::tempdir().unwrap();
    /// let store = VoiceStore::new(dir.path())?;
    /// assert!(store.list()?.is_empty());
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(voices_dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = voices_dir.into();
        validate_safe_file_path(&dir, "voices_dir")?;
        Ok(Self {
            voices_dir: dir,
            cache: Arc::new(DashMap::new()),
        })
    }

    /// Creates a [`VoiceStore`] using the default user cache directory
    /// (`~/.cache/index-tts-rs/voices`).
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if the path fails validation.
    pub fn with_default_dir() -> Result<Self> {
        Self::new(default_voices_dir())
    }

    /// Returns the root directory where `.itts_voice` files are stored.
    #[must_use]
    pub fn voices_dir(&self) -> &Path {
        &self.voices_dir
    }

    /// Returns the filesystem path for `voice_id` (`<voices_dir>/<voice_id>.itts_voice`).
    #[must_use]
    pub fn voice_path(&self, voice_id: &VoiceId) -> PathBuf {
        self.voices_dir
            .join(format!("{}.itts_voice", voice_id.as_str()))
    }

    /// Retrieves a [`SpeakerContext`] by [`VoiceId`], checking the in-memory `DashMap` first and
    /// falling back to `<voices_dir>/<voice_id>.itts_voice` on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::VoiceNotFound`] if the voice is neither cached nor on disk, or
    /// [`IndexTtsError::WeightLoad`] if the `.itts_voice` file is corrupted.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use index_tts_rs_core::{
    ///     config::VoiceId,
    ///     voice::{SpeakerContext, VoiceMetadata, VoiceStore},
    /// };
    ///
    /// let dir = tempfile::tempdir().unwrap();
    /// let store = VoiceStore::new(dir.path())?;
    /// let id = VoiceId::new("narrator")?;
    /// let ctx = SpeakerContext::new(
    ///     vec![0.0; 3 * 1280],
    ///     vec![0.0; 4 * 512],
    ///     vec![0.0; 80 * 4],
    ///     vec![0.0; 192],
    ///     NonZeroUsize::new(4).unwrap(),
    ///     VoiceMetadata::default(),
    /// )?;
    /// store.register(id.clone(), ctx)?;
    /// let loaded = store.get(&id)?;
    /// assert_eq!(loaded.prompt_frames.get(), 4);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn get(&self, voice_id: &VoiceId) -> Result<Arc<SpeakerContext>> {
        if let Some(entry) = self.cache.get(voice_id) {
            return Ok(Arc::clone(entry.value()));
        }

        let path = self.voice_path(voice_id);
        if !path.is_file() {
            return Err(IndexTtsError::VoiceNotFound {
                voice_id: voice_id.as_str().to_owned(),
            });
        }

        let ctx = Arc::new(SpeakerContext::load_from_path(&path)?);
        self.cache.insert(voice_id.clone(), Arc::clone(&ctx));
        Ok(ctx)
    }

    /// Persists a [`SpeakerContext`] to `<voices_dir>/<voice_id>.itts_voice` and caches it in
    /// memory.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Io`] if writing to disk fails.
    pub fn register(&self, voice_id: VoiceId, context: SpeakerContext) -> Result<PathBuf> {
        let path = self.voice_path(&voice_id);
        context.save_to_path(&path)?;
        self.cache.insert(voice_id, Arc::new(context));
        Ok(path)
    }

    /// Removes `voice_id` from both the in-memory cache and disk
    /// (`<voices_dir>/<voice_id>.itts_voice`). Returns `true` if the voice existed in memory or
    /// on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Io`] if removing the on-disk file fails for a reason other than
    /// `NotFound`.
    #[allow(clippy::disallowed_methods)]
    pub fn remove(&self, voice_id: &VoiceId) -> Result<bool> {
        let removed_mem = self.cache.remove(voice_id).is_some();
        let path = self.voice_path(voice_id);
        if path.is_file() {
            fs::remove_file(&path).map_err(|err| IndexTtsError::io(&path, err))?;
            return Ok(true);
        }
        Ok(removed_mem)
    }

    /// Inspects a single registered voice and returns its [`VoiceEntry`] descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::VoiceNotFound`] if the voice does not exist.
    #[allow(clippy::cast_precision_loss, clippy::disallowed_methods)]
    pub fn inspect(&self, voice_id: &VoiceId) -> Result<VoiceEntry> {
        let ctx = self.get(voice_id)?;
        let path = self.voice_path(voice_id);
        let size_bytes = fs::metadata(&path).map_or(0, |m| m.len());
        let p = ctx.prompt_frames.get();
        let duration_secs = (p as f32 * 256.0) / (DEFAULT_SAMPLE_RATE as f32);
        Ok(VoiceEntry {
            voice_id: voice_id.as_str().to_owned(),
            path,
            prompt_frames: p,
            duration_secs,
            size_bytes,
            metadata: ctx.metadata.clone(),
        })
    }

    /// Lists all `.itts_voice` snapshots stored in `voices_dir`, sorted by `voice_id`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Io`] if reading the directory fails.
    #[allow(clippy::disallowed_methods)]
    pub fn list(&self) -> Result<Vec<VoiceEntry>> {
        if !self.voices_dir.exists() {
            return Ok(Vec::new());
        }
        let entries = fs::read_dir(&self.voices_dir)
            .map_err(|err| IndexTtsError::io(&self.voices_dir, err))?;
        let mut out = Vec::new();
        for entry_res in entries {
            let entry = entry_res.map_err(|err| IndexTtsError::io(&self.voices_dir, err))?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("itts_voice") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Ok(voice_id) = VoiceId::new(stem) else {
                continue;
            };
            if let Ok(info) = self.inspect(&voice_id) {
                out.push(info);
            }
        }
        out.sort_by(|a, b| a.voice_id.cmp(&b.voice_id));
        Ok(out)
    }
}

/// Resolves the default user voices cache directory (`~/.cache/index-tts-rs/voices`).
#[must_use]
pub fn default_voices_dir() -> PathBuf {
    if let Ok(home) = env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home).join(DEFAULT_VOICES_SUBDIR);
    }
    PathBuf::from(DEFAULT_VOICES_SUBDIR)
}

#[cfg(test)]
mod tests {
    use safetensors::SafeTensors;

    use super::*;
    use crate::{
        config::DEFAULT_MODEL_REPO, dsp::cosine_similarity_and_max_diff, models::resolve_model_dir,
    };

    #[test]
    fn test_should_match_python_reference_end_to_end_voice_extractor() {
        let Ok(model_dir) = resolve_model_dir(None, DEFAULT_MODEL_REPO) else {
            return;
        };
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let golden_ref_path = fixture_dir.join("golden_ref.safetensors");
        let golden_voice_path = fixture_dir.join("golden_speaker.itts_voice");

        #[allow(clippy::disallowed_methods)]
        let Ok(ref_bytes) = fs::read(&golden_ref_path) else {
            return;
        };
        let Ok(st) = SafeTensors::deserialize(&ref_bytes) else {
            return;
        };
        let Ok(expected_ctx) = SpeakerContext::load_from_path(&golden_voice_path) else {
            return;
        };

        let (_, audio_22k) = extract_f32(&st, "audio_22k");
        let (_, audio_16k) = extract_f32(&st, "audio_16k");

        let extractor_res = VoiceExtractor::load_from_model_dir(&model_dir);
        assert!(
            extractor_res.is_ok(),
            "VoiceExtractor load failed: {extractor_res:?}"
        );
        let Ok(extractor) = extractor_res else {
            return;
        };

        let reg_res =
            LengthRegulator::load_from_path(model_dir.join("raw/length_regulator.safetensors"));
        assert!(reg_res.is_ok(), "LengthRegulator load failed: {reg_res:?}");
        let Ok(regulator) = reg_res else {
            return;
        };

        let ctx_res = extractor.extract_from_pcm(
            &audio_22k,
            &audio_16k,
            None,
            expected_ctx.metadata.source_audio_blake3.clone(),
            EmoAlpha::default(),
            &regulator,
        );
        assert!(
            ctx_res.is_ok(),
            "VoiceExtractor::extract_from_pcm failed: {ctx_res:?}"
        );
        let Ok(ctx) = ctx_res else {
            return;
        };

        assert_eq!(ctx.prompt_frames, expected_ctx.prompt_frames);

        let (cos_style, diff_style) =
            cosine_similarity_and_max_diff(&ctx.style, &expected_ctx.style);
        assert!(
            cos_style >= 0.999_9 && diff_style < 1e-3,
            "VoiceExtractor style mismatch: cos={cos_style}, max_diff={diff_style}"
        );

        let (cos_conds, diff_conds) =
            cosine_similarity_and_max_diff(&ctx.conds, &expected_ctx.conds);
        assert!(
            cos_conds >= 0.999_9 && diff_conds < 1e-3,
            "VoiceExtractor conds mismatch: cos={cos_conds}, max_diff={diff_conds}"
        );

        let (cos_prompt, _) =
            cosine_similarity_and_max_diff(&ctx.prompt_condition, &expected_ctx.prompt_condition);
        assert!(
            cos_prompt >= 0.999_9,
            "VoiceExtractor prompt_condition cosine similarity {cos_prompt} < 0.9999"
        );

        let (cos_mel, diff_mel) =
            cosine_similarity_and_max_diff(&ctx.ref_mel, &expected_ctx.ref_mel);
        assert!(
            cos_mel >= 0.999_9 && diff_mel < 1e-4,
            "VoiceExtractor ref_mel mismatch: cos={cos_mel}, max_diff={diff_mel}"
        );
    }

    #[test]
    fn test_should_register_list_inspect_and_remove_voices_in_store() {
        let Ok(temp) = tempfile::tempdir() else {
            return;
        };
        let Ok(store) = VoiceStore::new(temp.path()) else {
            return;
        };
        let Ok(id) = VoiceId::new("alice_en") else {
            return;
        };
        let p = NonZeroUsize::new(10).unwrap_or(NonZeroUsize::MIN);
        let Ok(ctx) = SpeakerContext::new(
            vec![0.1; 3 * 1280],
            vec![0.2; 10 * 512],
            vec![-3.5; 80 * 10],
            vec![0.05; 192],
            p,
            VoiceMetadata::default(),
        ) else {
            return;
        };

        let path_res = store.register(id.clone(), ctx);
        assert!(path_res.is_ok());

        let list = store.list().unwrap_or_default();
        assert_eq!(list.len(), 1);
        assert_eq!(list.first().map(|e| e.voice_id.as_str()), Some("alice_en"));

        let removed = store.remove(&id).unwrap_or(false);
        assert!(removed);
        assert!(matches!(
            store.get(&id),
            Err(IndexTtsError::VoiceNotFound { .. })
        ));
    }

    fn extract_f32(st: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<f32>) {
        let Ok(view) = st.tensor(name) else {
            return (Vec::new(), Vec::new());
        };
        let mut out = Vec::with_capacity(view.data().len() / 4);
        for &b in view.data().as_chunks::<4>().0 {
            out.push(f32::from_le_bytes(b));
        }
        (view.shape().to_vec(), out)
    }
}

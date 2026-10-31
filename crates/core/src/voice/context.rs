//! [`SpeakerContext`] and `.itts_voice` `SafeTensors` serialization.
//!
//! `.itts_voice` files store the four pre-extracted conditioning tensors
//! (`conds`, `prompt_condition`, `ref_mel`, `style`) in standard `SafeTensors`
//! format along with header metadata (`format_version`, `model_id`,
//! `source_audio_blake3`, `emo_alpha`).

use std::{borrow::Cow, collections::HashMap, fs, num::NonZeroUsize, path::Path};

use safetensors::{
    Dtype, SafeTensors, View,
    tensor::{SafeTensorError, TensorView},
};
use serde::{Deserialize, Serialize};

use crate::{
    config::{DEFAULT_MODEL_REPO, EmoAlpha, EmotionVector, validate_safe_file_path},
    error::{IndexTtsError, Result},
    models::SpkProjWeights,
};

/// Current `.itts_voice` binary format version.
pub const ITTS_VOICE_FORMAT_VERSION: &str = "1";

/// Expected feature dimension of `conds` (`[1, 3, 1280]`).
pub const CONDS_FRAMES: usize = 3;
/// Expected channel dimension of `conds` (`1280`).
pub const CONDS_DIM: usize = 1_280;
/// Expected channel dimension of `prompt_condition` (`512`).
pub const PROMPT_CONDITION_DIM: usize = 512;
/// Expected mel bin count of `ref_mel` (`80`).
pub const REF_MEL_BINS: usize = 80;
/// Expected speaker style embedding dimension (`192`).
pub const STYLE_DIM: usize = 192;

/// Number of prototype rows per emotion category in `emo_matrix.safetensors` (`sum = 73`).
pub const EMO_PROTOTYPE_COUNTS: [usize; 8] = [3, 17, 2, 8, 4, 5, 10, 24];
/// Total number of prototype rows across all 8 emotion categories (`73`).
pub const EMO_PROTOTYPE_ROWS: usize = 73;

const EMBEDDED_EMO_MATRIX_BYTES: &[u8] = include_bytes!("../../assets/emo_matrix.safetensors");

/// Pre-extracted 8-class emotion prototype matrices (`spk_matrix [73, 192]` and
/// `emo_matrix [73, 1280]`) from upstream `IndexTeam/IndexTTS-2.5`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmoMatrix {
    spk_matrix: Vec<f32>,
    emo_matrix: Vec<f32>,
}

impl EmoMatrix {
    /// Loads the compiled-in [`EmoMatrix`] from `crates/core/assets/emo_matrix.safetensors`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if the embedded asset is corrupted.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::voice::EmoMatrix;
    ///
    /// let matrix = EmoMatrix::embedded()?;
    /// assert_eq!(matrix.num_rows(), 73);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn embedded() -> Result<Self> {
        Self::from_safetensors_bytes(EMBEDDED_EMO_MATRIX_BYTES)
    }

    /// Loads `emo_matrix.safetensors` from `model_dir` if present on disk, otherwise falls back
    /// to [`Self::embedded`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if an override file exists on disk but is invalid.
    #[allow(clippy::disallowed_methods)]
    pub fn load_or_embedded(model_dir: Option<&Path>) -> Result<Self> {
        if let Some(dir) = model_dir {
            let candidate = dir.join("emo_matrix.safetensors");
            if candidate.is_file() {
                validate_safe_file_path(&candidate, "emo_matrix_path")?;
                let bytes =
                    fs::read(&candidate).map_err(|err| IndexTtsError::io(&candidate, err))?;
                return Self::from_safetensors_bytes(&bytes);
            }
        }
        Self::embedded()
    }

    /// Deserializes and validates an [`EmoMatrix`] from `SafeTensors` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] or [`IndexTtsError::Validation`] if `spk_matrix`
    /// (`[73, 192]`) or `emo_matrix` (`[73, 1280]`) is missing, has an unexpected shape, or
    /// contains non-finite values.
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self> {
        let st = SafeTensors::deserialize(bytes).map_err(|err| {
            IndexTtsError::weight_load(
                "emo_matrix.safetensors",
                format!("failed to parse safetensors: {err}"),
            )
        })?;
        let spk_view = st.tensor("spk_matrix").map_err(|err| {
            IndexTtsError::weight_load(
                "emo_matrix.safetensors",
                format!("missing `spk_matrix` tensor: {err}"),
            )
        })?;
        let emo_view = st.tensor("emo_matrix").map_err(|err| {
            IndexTtsError::weight_load(
                "emo_matrix.safetensors",
                format!("missing `emo_matrix` tensor: {err}"),
            )
        })?;

        if spk_view.shape() != [EMO_PROTOTYPE_ROWS, STYLE_DIM] {
            return Err(IndexTtsError::weight_load(
                "emo_matrix.safetensors",
                format!(
                    "invalid `spk_matrix` shape {:?}, expected [{EMO_PROTOTYPE_ROWS}, {STYLE_DIM}]",
                    spk_view.shape()
                ),
            ));
        }
        if emo_view.shape() != [EMO_PROTOTYPE_ROWS, CONDS_DIM] {
            return Err(IndexTtsError::weight_load(
                "emo_matrix.safetensors",
                format!(
                    "invalid `emo_matrix` shape {:?}, expected [{EMO_PROTOTYPE_ROWS}, {CONDS_DIM}]",
                    emo_view.shape()
                ),
            ));
        }

        let spk_matrix = decode_f32_tensor("spk_matrix", &spk_view)?;
        let emo_matrix = decode_f32_tensor("emo_matrix", &emo_view)?;

        if spk_matrix.iter().any(|v| !v.is_finite()) {
            return Err(IndexTtsError::validation(
                "spk_matrix",
                "tensor contains non-finite values",
            ));
        }
        if emo_matrix.iter().any(|v| !v.is_finite()) {
            return Err(IndexTtsError::validation(
                "emo_matrix",
                "tensor contains non-finite values",
            ));
        }

        Ok(Self {
            spk_matrix,
            emo_matrix,
        })
    }

    /// Returns the total number of emotion prototype rows (`73`).
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        EMO_PROTOTYPE_ROWS
    }

    /// Selects the speaker-matched emotion prototype for each of the 8 emotion categories by
    /// maximizing cosine similarity against `style` (`[192]`), and returns the weighted sum
    /// $\sum_{c=0}^7 w_c \cdot \text{emo\_matrix}[c, r^*_c]$ (`[1280]`).
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `style.len() != 192`.
    pub fn select_emotion_vector(&self, style: &[f32], weights: &[f32; 8]) -> Result<Vec<f32>> {
        if style.len() != STYLE_DIM {
            return Err(IndexTtsError::validation(
                "style",
                format!("expected {STYLE_DIM} elements, got {}", style.len()),
            ));
        }

        let style_norm = style.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
        let mut combined = vec![0.0_f32; CONDS_DIM];
        let mut row_start = 0_usize;

        for (&count, &weight) in EMO_PROTOTYPE_COUNTS.iter().zip(weights.iter()) {
            if weight > 0.0 {
                let mut best_row = row_start;
                let mut best_cos = f32::NEG_INFINITY;
                for row in row_start..(row_start + count) {
                    let spk_slice = &self.spk_matrix[row * STYLE_DIM..(row + 1) * STYLE_DIM];
                    let mut dot = 0.0_f32;
                    let mut row_norm_sq = 0.0_f32;
                    for (&s, &r) in style.iter().zip(spk_slice.iter()) {
                        dot = s.mul_add(r, dot);
                        row_norm_sq = r.mul_add(r, row_norm_sq);
                    }
                    let cos = dot / (style_norm * row_norm_sq.sqrt().max(1e-12));
                    if cos > best_cos {
                        best_cos = cos;
                        best_row = row;
                    }
                }
                let emo_slice = &self.emo_matrix[best_row * CONDS_DIM..(best_row + 1) * CONDS_DIM];
                for (dst, &val) in combined.iter_mut().zip(emo_slice.iter()) {
                    *dst = weight.mul_add(val, *dst);
                }
            }
            row_start += count;
        }

        Ok(combined)
    }

    /// Blends `emo_vector` with the speaker's `base_emovec` (`[1280]`) following upstream
    /// `IndexTTS-2.5` `merge_emovec`:
    ///
    /// $$\mathbf{e}_{\text{final}} = \sum_{c=0}^7 \hat{w}_c \mathbf{E}_{c, r^*_c} + \left(1 -
    /// \sum_{c=0}^7 \hat{w}_c\right) \mathbf{e}_{\text{base}}$$
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `style.len() != 192` or `base_emovec.len() !=
    /// 1280`.
    pub fn blend_emovec(
        &self,
        style: &[f32],
        base_emovec: &[f32],
        emo_vector: &EmotionVector,
        emo_alpha: EmoAlpha,
    ) -> Result<Vec<f32>> {
        if base_emovec.len() != CONDS_DIM {
            return Err(IndexTtsError::validation(
                "base_emovec",
                format!("expected {CONDS_DIM} elements, got {}", base_emovec.len()),
            ));
        }
        emo_vector.validate_weights()?;
        let weights = emo_vector.normalize(true, emo_alpha);
        let mut emo_mat = self.select_emotion_vector(style, &weights)?;
        let weight_sum: f32 = weights.iter().sum();
        let base_scale = (1.0 - weight_sum).max(0.0);
        for (dst, &base) in emo_mat.iter_mut().zip(base_emovec.iter()) {
            *dst = base_scale.mul_add(base, *dst);
        }
        Ok(emo_mat)
    }
}

/// Metadata stored in the `SafeTensors` `__metadata__` header of an `.itts_voice` snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct VoiceMetadata {
    /// Format version string (must be `"1"`).
    pub format_version: String,
    /// Model repository identifier (`"yunfengwang/IndexTTS-2.5-mlx"`).
    pub model_id: String,
    /// BLAKE3 hex digest of the source reference audio.
    pub source_audio_blake3: String,
    /// Emotion blending coefficient used when computing `conds`.
    pub emo_alpha: EmoAlpha,
}

impl Default for VoiceMetadata {
    fn default() -> Self {
        Self {
            format_version: ITTS_VOICE_FORMAT_VERSION.to_owned(),
            model_id: DEFAULT_MODEL_REPO.to_owned(),
            source_audio_blake3: "0".repeat(64),
            emo_alpha: EmoAlpha::default(),
        }
    }
}

impl VoiceMetadata {
    /// Constructs new [`VoiceMetadata`] with the given source audio BLAKE3 digest and `emo_alpha`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{config::EmoAlpha, voice::VoiceMetadata};
    ///
    /// let meta = VoiceMetadata::new("abcd", EmoAlpha::default());
    /// assert_eq!(meta.format_version, "1");
    /// ```
    #[must_use]
    pub fn new(source_audio_blake3: impl Into<String>, emo_alpha: EmoAlpha) -> Self {
        Self {
            format_version: ITTS_VOICE_FORMAT_VERSION.to_owned(),
            model_id: DEFAULT_MODEL_REPO.to_owned(),
            source_audio_blake3: source_audio_blake3.into(),
            emo_alpha,
        }
    }
}

/// Pre-computed speaker conditioning tensors bridging Stage 0 and Stages 1..3.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpeakerContext {
    /// GPT conditioning prefix tensor of shape `[1, 3, 1280]` (`f32`).
    /// Frame 0 is `spk_cond + emo_alpha * emovec`; frames 1..2 are zero-filled.
    pub conds: Vec<f32>,
    /// CFM prompt condition tensor of shape `[1, P, 512]` (`f32`).
    pub prompt_condition: Vec<f32>,
    /// Reference audio Mel spectrogram tensor of shape `[1, 80, P]` (`f32`).
    pub ref_mel: Vec<f32>,
    /// `CAMPPlus` speaker style vector of shape `[1, 192]` (`f32`).
    pub style: Vec<f32>,
    /// Number of Mel frames `P` in the reference prompt.
    pub prompt_frames: NonZeroUsize,
    /// Snapshot provenance and hyperparameter metadata.
    pub metadata: VoiceMetadata,
}

struct F32TensorPayload<'a> {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Cow<'a, [u8]>,
}

impl View for F32TensorPayload<'_> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

impl SpeakerContext {
    /// Validates shapes and constructs a [`SpeakerContext`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if any tensor length does not match
    /// its expected dimensions or contains non-finite values (`NaN` / `Inf`).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use index_tts_rs_core::voice::{SpeakerContext, VoiceMetadata};
    ///
    /// let p = NonZeroUsize::new(10).unwrap();
    /// let ctx = SpeakerContext::new(
    ///     vec![0.0; 3 * 1280],
    ///     vec![0.0; 10 * 512],
    ///     vec![0.0; 80 * 10],
    ///     vec![0.0; 192],
    ///     p,
    ///     VoiceMetadata::default(),
    /// )?;
    /// assert_eq!(ctx.prompt_frames.get(), 10);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn new(
        conds: Vec<f32>,
        prompt_condition: Vec<f32>,
        ref_mel: Vec<f32>,
        style: Vec<f32>,
        prompt_frames: NonZeroUsize,
        metadata: VoiceMetadata,
    ) -> Result<Self> {
        let p = prompt_frames.get();
        let expected_conds = CONDS_FRAMES * CONDS_DIM;
        let expected_prompt = p.checked_mul(PROMPT_CONDITION_DIM).ok_or_else(|| {
            IndexTtsError::validation("prompt_condition", "prompt_frames * 512 overflowed")
        })?;
        let expected_mel = p
            .checked_mul(REF_MEL_BINS)
            .ok_or_else(|| IndexTtsError::validation("ref_mel", "prompt_frames * 80 overflowed"))?;

        if conds.len() != expected_conds {
            return Err(IndexTtsError::validation(
                "conds",
                format!(
                    "expected {expected_conds} elements ([1, 3, 1280]), got {}",
                    conds.len()
                ),
            ));
        }
        if prompt_condition.len() != expected_prompt {
            return Err(IndexTtsError::validation(
                "prompt_condition",
                format!(
                    "expected {expected_prompt} elements ([1, {p}, 512]), got {}",
                    prompt_condition.len()
                ),
            ));
        }
        if ref_mel.len() != expected_mel {
            return Err(IndexTtsError::validation(
                "ref_mel",
                format!(
                    "expected {expected_mel} elements ([1, 80, {p}]), got {}",
                    ref_mel.len()
                ),
            ));
        }
        if style.len() != STYLE_DIM {
            return Err(IndexTtsError::validation(
                "style",
                format!(
                    "expected {STYLE_DIM} elements ([1, 192]), got {}",
                    style.len()
                ),
            ));
        }

        for (name, slice) in [
            ("conds", conds.as_slice()),
            ("prompt_condition", prompt_condition.as_slice()),
            ("ref_mel", ref_mel.as_slice()),
            ("style", style.as_slice()),
        ] {
            if slice.iter().any(|v| !v.is_finite()) {
                return Err(IndexTtsError::validation(
                    name,
                    "tensor contains non-finite (NaN or Inf) values",
                ));
            }
        }

        Ok(Self {
            conds,
            prompt_condition,
            ref_mel,
            style,
            prompt_frames,
            metadata,
        })
    }

    /// Computes the `[1, 3, 1280]` `conds` tensor by blending `spk_cond` (`[1280]`)
    /// and `emovec` (`[1280]`) with `emo_alpha`, padding frames 1 and 2 with zeros.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `spk_cond` or `emovec` is not of length `1280`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::{config::EmoAlpha, voice::SpeakerContext};
    ///
    /// let spk = vec![1.0_f32; 1280];
    /// let emo = vec![0.5_f32; 1280];
    /// let conds = SpeakerContext::build_conds(&spk, &emo, EmoAlpha::new(0.5)?)?;
    /// assert_eq!(conds.len(), 3 * 1280);
    /// assert!((conds[0] - 1.25).abs() < 1e-6);
    /// assert!((conds[1280]).abs() < 1e-6);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn build_conds(spk_cond: &[f32], emovec: &[f32], emo_alpha: EmoAlpha) -> Result<Vec<f32>> {
        if spk_cond.len() != CONDS_DIM {
            return Err(IndexTtsError::validation(
                "spk_cond",
                format!("expected {CONDS_DIM} elements, got {}", spk_cond.len()),
            ));
        }
        if emovec.len() != CONDS_DIM {
            return Err(IndexTtsError::validation(
                "emovec",
                format!("expected {CONDS_DIM} elements, got {}", emovec.len()),
            ));
        }
        let alpha = emo_alpha.get();
        let mut conds = Vec::with_capacity(CONDS_FRAMES * CONDS_DIM);
        for (&s, &e) in spk_cond.iter().zip(emovec.iter()) {
            conds.push(s.mul_add(1.0, alpha * e));
        }
        conds.resize(CONDS_FRAMES * CONDS_DIM, 0.0);
        Ok(conds)
    }

    /// Dynamically synthesizes a new `[1, 3, 1280]` `conds` tensor for this speaker snapshot by
    /// recovering $\mathbf{s}_{\text{spk}} = \text{spk\_proj}(\text{style})$ and
    /// $\mathbf{e}_{\text{base}} = (\text{conds}[0] - \mathbf{s}_{\text{spk}}) / \alpha_0$,
    /// blending `emo_vector` via [`EmoMatrix::blend_emovec`], and adding $\mathbf{s}_{\text{spk}}$.
    ///
    /// This runs in microseconds on CPU without loading any Stage 0 neural encoders.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if `emo_vector` is invalid or tensor dimensions mismatch.
    pub fn resolve_conds_with_emotion(
        &self,
        spk_proj: &SpkProjWeights,
        emo_matrix: &EmoMatrix,
        emo_vector: &EmotionVector,
        emo_alpha: EmoAlpha,
    ) -> Result<Vec<f32>> {
        if emo_vector.is_zero() {
            return Ok(self.conds.clone());
        }
        let spk_cond = spk_proj.project(&self.style)?;
        let alpha_0 = self.metadata.emo_alpha.get();
        let mut base_emovec = vec![0.0_f32; CONDS_DIM];
        if alpha_0 >= 1e-4 {
            for (i, dst) in base_emovec.iter_mut().enumerate() {
                *dst = (self.conds[i] - spk_cond[i]) / alpha_0;
            }
        }
        let blended = emo_matrix.blend_emovec(&self.style, &base_emovec, emo_vector, emo_alpha)?;
        Self::build_conds(&spk_cond, &blended, EmoAlpha::default())
    }

    /// Updates `self.conds` in place by loading `spk_proj.npz` and [`EmoMatrix`] from `model_dir`
    /// and blending `emo_vector`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError`] if loading `spk_proj.npz` or blending fails.
    pub fn apply_emotion_from_model_dir(
        &mut self,
        model_dir: impl AsRef<Path>,
        emo_vector: &EmotionVector,
        emo_alpha: EmoAlpha,
    ) -> Result<()> {
        if emo_vector.is_zero() {
            return Ok(());
        }
        let dir = model_dir.as_ref();
        let spk_proj = SpkProjWeights::load_from_path(dir.join("spk_proj.npz"))?;
        let emo_matrix = EmoMatrix::load_or_embedded(Some(dir))?;
        self.conds =
            self.resolve_conds_with_emotion(&spk_proj, &emo_matrix, emo_vector, emo_alpha)?;
        Ok(())
    }

    /// Serializes this [`SpeakerContext`] into standard `.itts_voice` `SafeTensors` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if `SafeTensors` serialization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use index_tts_rs_core::voice::{SpeakerContext, VoiceMetadata};
    ///
    /// let p = NonZeroUsize::new(4).unwrap();
    /// let ctx = SpeakerContext::new(
    ///     vec![0.1; 3 * 1280],
    ///     vec![0.2; 4 * 512],
    ///     vec![-5.0; 80 * 4],
    ///     vec![0.05; 192],
    ///     p,
    ///     VoiceMetadata::default(),
    /// )?;
    /// let bytes = ctx.to_safetensors_bytes()?;
    /// let decoded = SpeakerContext::from_safetensors_bytes(&bytes)?;
    /// assert_eq!(ctx, decoded);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn to_safetensors_bytes(&self) -> Result<Vec<u8>> {
        let p = self.prompt_frames.get();
        let tensors = [
            (
                "conds",
                F32TensorPayload {
                    dtype: Dtype::F32,
                    shape: vec![1, CONDS_FRAMES, CONDS_DIM],
                    bytes: Cow::Owned(f32_slice_to_le_bytes(&self.conds)),
                },
            ),
            (
                "prompt_condition",
                F32TensorPayload {
                    dtype: Dtype::F32,
                    shape: vec![1, p, PROMPT_CONDITION_DIM],
                    bytes: Cow::Owned(f32_slice_to_le_bytes(&self.prompt_condition)),
                },
            ),
            (
                "ref_mel",
                F32TensorPayload {
                    dtype: Dtype::F32,
                    shape: vec![1, REF_MEL_BINS, p],
                    bytes: Cow::Owned(f32_slice_to_le_bytes(&self.ref_mel)),
                },
            ),
            (
                "style",
                F32TensorPayload {
                    dtype: Dtype::F32,
                    shape: vec![1, STYLE_DIM],
                    bytes: Cow::Owned(f32_slice_to_le_bytes(&self.style)),
                },
            ),
        ];

        let mut meta = HashMap::with_capacity(4);
        meta.insert(
            "format_version".to_owned(),
            self.metadata.format_version.clone(),
        );
        meta.insert("model_id".to_owned(), self.metadata.model_id.clone());
        meta.insert(
            "source_audio_blake3".to_owned(),
            self.metadata.source_audio_blake3.clone(),
        );
        meta.insert(
            "emo_alpha".to_owned(),
            format!("{:.4}", self.metadata.emo_alpha.get()),
        );

        safetensors::serialize(tensors, Some(meta)).map_err(|err: SafeTensorError| {
            IndexTtsError::weight_load(
                ".itts_voice",
                format!("safetensors serialize failed: {err}"),
            )
        })
    }

    /// Deserializes and validates a [`SpeakerContext`] from `.itts_voice` `SafeTensors` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] or [`IndexTtsError::Validation`] if the binary
    /// header, metadata, tensor shapes, or dtypes are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    /// use index_tts_rs_core::voice::SpeakerContext;
    ///
    /// let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/golden_speaker.itts_voice");
    /// let ctx = SpeakerContext::load_from_path(&path)?;
    /// assert_eq!(ctx.prompt_frames.get(), 172);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self> {
        let (_, header_meta) = SafeTensors::read_metadata(bytes).map_err(|err| {
            IndexTtsError::weight_load(".itts_voice", format!("invalid safetensors header: {err}"))
        })?;

        let metadata = parse_voice_metadata(header_meta.metadata().as_ref())?;
        let st = SafeTensors::deserialize(bytes).map_err(|err| {
            IndexTtsError::weight_load(".itts_voice", format!("failed to parse safetensors: {err}"))
        })?;

        let conds_view = st.tensor("conds").map_err(|err| {
            IndexTtsError::weight_load(".itts_voice", format!("missing `conds` tensor: {err}"))
        })?;
        let prompt_view = st.tensor("prompt_condition").map_err(|err| {
            IndexTtsError::weight_load(
                ".itts_voice",
                format!("missing `prompt_condition` tensor: {err}"),
            )
        })?;
        let mel_view = st.tensor("ref_mel").map_err(|err| {
            IndexTtsError::weight_load(".itts_voice", format!("missing `ref_mel` tensor: {err}"))
        })?;
        let style_view = st.tensor("style").map_err(|err| {
            IndexTtsError::weight_load(".itts_voice", format!("missing `style` tensor: {err}"))
        })?;

        if conds_view.shape() != [1, CONDS_FRAMES, CONDS_DIM] {
            return Err(IndexTtsError::weight_load(
                ".itts_voice",
                format!(
                    "invalid `conds` shape {:?}, expected [1, 3, 1280]",
                    conds_view.shape()
                ),
            ));
        }
        if style_view.shape() != [1, STYLE_DIM] {
            return Err(IndexTtsError::weight_load(
                ".itts_voice",
                format!(
                    "invalid `style` shape {:?}, expected [1, 192]",
                    style_view.shape()
                ),
            ));
        }

        let prompt_shape = prompt_view.shape();
        if prompt_shape.len() != 3
            || prompt_shape.first().copied() != Some(1)
            || prompt_shape.get(2).copied() != Some(PROMPT_CONDITION_DIM)
        {
            return Err(IndexTtsError::weight_load(
                ".itts_voice",
                format!("invalid `prompt_condition` shape {prompt_shape:?}, expected [1, P, 512]"),
            ));
        }
        let p_raw = prompt_shape.get(1).copied().unwrap_or(0);
        let prompt_frames = NonZeroUsize::new(p_raw).ok_or_else(|| {
            IndexTtsError::weight_load(".itts_voice", "`prompt_condition` has 0 frames")
        })?;

        if mel_view.shape() != [1, REF_MEL_BINS, p_raw] {
            return Err(IndexTtsError::weight_load(
                ".itts_voice",
                format!(
                    "`ref_mel` shape {:?} does not match [1, 80, {p_raw}]",
                    mel_view.shape()
                ),
            ));
        }

        let conds = decode_f32_tensor("conds", &conds_view)?;
        let prompt_condition = decode_f32_tensor("prompt_condition", &prompt_view)?;
        let ref_mel = decode_f32_tensor("ref_mel", &mel_view)?;
        let style = decode_f32_tensor("style", &style_view)?;

        Self::new(
            conds,
            prompt_condition,
            ref_mel,
            style,
            prompt_frames,
            metadata,
        )
    }

    /// Saves this [`SpeakerContext`] to disk at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if the path is unsafe, or
    /// [`IndexTtsError::Io`] if writing fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use index_tts_rs_core::voice::{SpeakerContext, VoiceMetadata};
    ///
    /// let dir = tempfile::tempdir().unwrap();
    /// let file_path = dir.path().join("demo.itts_voice");
    /// let ctx = SpeakerContext::new(
    ///     vec![0.0; 3 * 1280],
    ///     vec![0.0; 5 * 512],
    ///     vec![0.0; 80 * 5],
    ///     vec![0.0; 192],
    ///     NonZeroUsize::new(5).unwrap(),
    ///     VoiceMetadata::default(),
    /// )?;
    /// ctx.save_to_path(&file_path)?;
    /// let loaded = SpeakerContext::load_from_path(&file_path)?;
    /// assert_eq!(ctx, loaded);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[allow(clippy::disallowed_methods)]
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let path_ref = path.as_ref();
        validate_safe_file_path(path_ref, "voice_path")?;
        if let Some(parent) = path_ref.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|err| IndexTtsError::io(parent, err))?;
        }
        let bytes = self.to_safetensors_bytes()?;
        fs::write(path_ref, bytes).map_err(|err| IndexTtsError::io(path_ref, err))
    }

    /// Loads a [`SpeakerContext`] from an `.itts_voice` file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if the path is unsafe,
    /// [`IndexTtsError::Io`] on read errors, or [`IndexTtsError::WeightLoad`] if malformed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    /// use index_tts_rs_core::voice::SpeakerContext;
    ///
    /// let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/golden_speaker.itts_voice");
    /// let ctx = SpeakerContext::load_from_path(&path)?;
    /// assert_eq!(ctx.style.len(), 192);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    #[allow(clippy::disallowed_methods)]
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        validate_safe_file_path(path_ref, "voice_path")?;
        let bytes = fs::read(path_ref).map_err(|err| IndexTtsError::io(path_ref, err))?;
        Self::from_safetensors_bytes(&bytes)
    }
}

fn parse_voice_metadata(raw_meta: Option<&HashMap<String, String>>) -> Result<VoiceMetadata> {
    let Some(map) = raw_meta else {
        return Ok(VoiceMetadata::default());
    };
    let format_version = map
        .get("format_version")
        .cloned()
        .unwrap_or_else(|| ITTS_VOICE_FORMAT_VERSION.to_owned());
    if format_version != ITTS_VOICE_FORMAT_VERSION {
        return Err(IndexTtsError::weight_load(
            ".itts_voice",
            format!(
                "unsupported format_version `{format_version}`, expected \
                 `{ITTS_VOICE_FORMAT_VERSION}`"
            ),
        ));
    }
    let model_id = map
        .get("model_id")
        .cloned()
        .unwrap_or_else(|| DEFAULT_MODEL_REPO.to_owned());
    let source_audio_blake3 = map
        .get("source_audio_blake3")
        .cloned()
        .unwrap_or_else(|| "0".repeat(64));
    let emo_alpha = match map.get("emo_alpha") {
        Some(raw) => {
            let parsed: f32 = raw.parse().map_err(|_| {
                IndexTtsError::weight_load(".itts_voice", format!("invalid emo_alpha `{raw}`"))
            })?;
            EmoAlpha::new(parsed)?
        }
        None => EmoAlpha::default(),
    };

    Ok(VoiceMetadata {
        format_version,
        model_id,
        source_audio_blake3,
        emo_alpha,
    })
}

fn decode_f32_tensor(name: &str, view: &TensorView<'_>) -> Result<Vec<f32>> {
    if view.dtype() != Dtype::F32 {
        return Err(IndexTtsError::weight_load(
            ".itts_voice",
            format!("tensor `{name}` has dtype {:?}, expected F32", view.dtype()),
        ));
    }
    let data = view.data();
    if data.len() % 4 != 0 {
        return Err(IndexTtsError::weight_load(
            ".itts_voice",
            format!(
                "tensor `{name}` byte length {} is not divisible by 4",
                data.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for &arr in data.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

fn f32_slice_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_should_load_golden_speaker_itts_voice_fixture() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("golden_speaker.itts_voice");
        let ctx = SpeakerContext::load_from_path(&fixture);
        assert!(
            ctx.is_ok(),
            "failed to load golden_speaker.itts_voice: {ctx:?}"
        );
        if let Ok(ctx) = ctx {
            assert_eq!(ctx.prompt_frames.get(), 172);
            assert_eq!(ctx.conds.len(), 3 * 1280);
            assert_eq!(ctx.prompt_condition.len(), 172 * 512);
            assert_eq!(ctx.ref_mel.len(), 80 * 172);
            assert_eq!(ctx.style.len(), 192);
            assert_eq!(ctx.metadata.format_version, "1");
            assert_eq!(ctx.metadata.model_id, DEFAULT_MODEL_REPO);
        }
    }

    #[test]
    fn test_should_roundtrip_speaker_context_via_safetensors() {
        let p = NonZeroUsize::new(12).unwrap_or(NonZeroUsize::MIN);
        let ctx = SpeakerContext::new(
            vec![0.25; 3 * 1280],
            vec![-0.1; 12 * 512],
            vec![-4.5; 80 * 12],
            vec![0.03; 192],
            p,
            VoiceMetadata::default(),
        );
        assert!(ctx.is_ok());
        if let Ok(ctx) = ctx {
            let bytes = ctx.to_safetensors_bytes();
            assert!(bytes.is_ok());
            if let Ok(bytes) = bytes {
                let restored = SpeakerContext::from_safetensors_bytes(&bytes);
                assert_eq!(restored.ok().as_ref(), Some(&ctx));
            }
        }
    }

    #[test]
    fn test_should_reject_nan_in_speaker_context() {
        let p = NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN);
        let mut style = vec![0.0; 192];
        if let Some(first) = style.first_mut() {
            *first = f32::NAN;
        }
        let res = SpeakerContext::new(
            vec![0.0; 3 * 1280],
            vec![0.0; 2 * 512],
            vec![0.0; 80 * 2],
            style,
            p,
            VoiceMetadata::default(),
        );
        assert!(matches!(
            res,
            Err(IndexTtsError::Validation { field: "style", .. })
        ));
    }

    #[test]
    fn test_should_load_embedded_emo_matrix_and_blend_conds() {
        let matrix = EmoMatrix::embedded();
        assert!(
            matrix.is_ok(),
            "failed to load embedded EmoMatrix: {matrix:?}"
        );
        let Ok(matrix) = matrix else {
            return;
        };
        assert_eq!(matrix.num_rows(), 73);

        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("golden_speaker.itts_voice");
        let Ok(ctx) = SpeakerContext::load_from_path(&fixture) else {
            return;
        };

        let emo: EmotionVector = "melancholic:0.45,calm:0.25".parse().unwrap_or_default();
        let base_emovec = vec![0.1_f32; CONDS_DIM];
        let blended = matrix.blend_emovec(&ctx.style, &base_emovec, &emo, EmoAlpha::default());
        assert!(blended.is_ok());
        if let Ok(blended) = blended {
            assert_eq!(blended.len(), CONDS_DIM);
            assert!(blended.iter().all(|v| v.is_finite()));
            assert!(
                blended
                    .iter()
                    .zip(base_emovec.iter())
                    .any(|(a, b)| (a - b).abs() > 1e-4)
            );
        }
    }
}

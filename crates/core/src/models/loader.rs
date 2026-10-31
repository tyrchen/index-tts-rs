//! Model directory resolution, `spk_proj.npz` parser, and dead-weight filtering
//! `SafeTensors` loader with automatic PyTorch-to-MLX convolution layout transposition.

use std::{
    collections::HashMap,
    env, fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use mlx_rs::Array;
use safetensors::{
    Dtype, SafeTensors,
    tensor::{Metadata, TensorView},
};

use crate::{
    config::{DEFAULT_MODEL_REPO, validate_safe_file_path},
    error::{IndexTtsError, Result},
};

/// Expected output dimension of `spk_proj.npz` (`1280`).
pub const SPK_PROJ_OUT_DIM: usize = 1_280;
/// Expected input dimension of `spk_proj.npz` (`192`).
pub const SPK_PROJ_IN_DIM: usize = 192;
/// Number of active `w2v-bert-2.0` Conformer layers (`0..17`).
pub const SEMANTIC_ACTIVE_LAYERS: usize = 17;

const REQUIRED_MODEL_FILES: [&str; 4] = [
    "gpt_int8.safetensors",
    "cfm_estimator.safetensors",
    "spk_proj.npz",
    "multilingual_zh_ja_yue_char_del.tiktoken",
];

/// Identifies the specific IndexTTS-2.5 sub-model weight file and its filtering/transposition
/// rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelWeightKind {
    /// `gpt_int8.safetensors` (pre-quantized MLX layout).
    GptInt8,
    /// `cfm_estimator.safetensors` (pre-converted MLX layout).
    CfmEstimator,
    /// `raw/semantic_model.safetensors` (skips layers `17..=23` and `masked_spec_embed`).
    SemanticModel,
    /// `raw/semantic_codec.safetensors` (skips `encoder.*`, `down.*`, `*.in_project.*`).
    SemanticCodec,
    /// `raw/length_regulator.safetensors` (skips `embedding.weight`, `mask_token`).
    LengthRegulator,
    /// `raw/bigvgan.safetensors` (deduplicates 218 Kaiser-Sinc filters, transposes
    /// `ConvTranspose1d`).
    BigVgan,
    /// `raw/emo.safetensors` (extracts `pe`, remaps `embed_conv`/`embed_out`, transposes
    /// Conv1d/2d).
    EmoVec,
    /// `raw/campplus.safetensors` (skips `*.num_batches_tracked`, transposes Conv1d/2d).
    CampPlus,
}

impl ModelWeightKind {
    /// All 8 `SafeTensors` weight artifacts in `IndexTTS-2.5-mlx`.
    pub const ALL: [Self; 8] = [
        Self::GptInt8,
        Self::CfmEstimator,
        Self::SemanticModel,
        Self::SemanticCodec,
        Self::LengthRegulator,
        Self::BigVgan,
        Self::EmoVec,
        Self::CampPlus,
    ];
    /// Returns the relative path of this weight file inside the model snapshot directory.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::ModelWeightKind;
    ///
    /// assert_eq!(ModelWeightKind::GptInt8.relative_path(), "gpt_int8.safetensors");
    /// assert_eq!(
    ///     ModelWeightKind::SemanticCodec.relative_path(),
    ///     "raw/semantic_codec.safetensors"
    /// );
    /// ```
    #[must_use]
    pub const fn relative_path(self) -> &'static str {
        match self {
            Self::GptInt8 => "gpt_int8.safetensors",
            Self::CfmEstimator => "cfm_estimator.safetensors",
            Self::SemanticModel => "raw/semantic_model.safetensors",
            Self::SemanticCodec => "raw/semantic_codec.safetensors",
            Self::LengthRegulator => "raw/length_regulator.safetensors",
            Self::BigVgan => "raw/bigvgan.safetensors",
            Self::EmoVec => "raw/emo.safetensors",
            Self::CampPlus => "raw/campplus.safetensors",
        }
    }
}

/// Decoded tensor data ready for MLX or CPU consumption.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TensorStorage {
    /// 32-bit floating-point elements (`F32`).
    F32(Vec<f32>),
    /// 32-bit unsigned integer elements (`U32`, used by packed Int8 quantized weights).
    U32(Vec<u32>),
    /// 32-bit signed integer elements (`I32`).
    I32(Vec<i32>),
    /// Pre-materialized MLX unified-memory [`Array`] (avoids duplicate CPU heap copies).
    Mlx(Array),
}

impl PartialEq for TensorStorage {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F32(a), Self::F32(b)) => a == b,
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::Mlx(a), Self::Mlx(b)) => {
                a.shape() == b.shape()
                    && a.dtype() == b.dtype()
                    && a.try_as_slice::<u8>().ok() == b.try_as_slice::<u8>().ok()
            }
            _ => false,
        }
    }
}

/// A single loaded and layout-normalized model tensor.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LoadedTensor {
    /// Remapped MLX parameter key name.
    pub name: String,
    /// Tensor dimensions (after `PyTorch` -> MLX transposition if applicable).
    pub shape: Vec<usize>,
    /// Underlying typed element storage.
    pub storage: TensorStorage,
}

impl LoadedTensor {
    /// Returns the `f32` slice if this tensor has `F32` storage.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if the tensor is not `F32`.
    pub fn as_f32(&self) -> Result<&[f32]> {
        match &self.storage {
            TensorStorage::F32(v) => Ok(v),
            TensorStorage::Mlx(arr) => arr.try_as_slice::<f32>().map_err(|err| {
                IndexTtsError::weight_load(
                    &self.name,
                    format!("expected F32 MLX array slice: {err}"),
                )
            }),
            _ => Err(IndexTtsError::weight_load(
                &self.name,
                "expected F32 tensor storage",
            )),
        }
    }

    /// Returns the `u32` slice if this tensor has `U32` storage.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if the tensor is not `U32`.
    pub fn as_u32(&self) -> Result<&[u32]> {
        match &self.storage {
            TensorStorage::U32(v) => Ok(v),
            TensorStorage::Mlx(arr) => arr.try_as_slice::<u32>().map_err(|err| {
                IndexTtsError::weight_load(
                    &self.name,
                    format!("expected U32 MLX array slice: {err}"),
                )
            }),
            _ => Err(IndexTtsError::weight_load(
                &self.name,
                "expected U32 tensor storage",
            )),
        }
    }

    /// Converts this loaded tensor into an [`mlx_rs::Array`], validating element count and
    /// dimension bounds.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if any dimension exceeds `i32::MAX` or the element
    /// count does not match the product of `self.shape`.
    pub fn to_mlx_array(&self) -> Result<Array> {
        let mut shape_i32 = Vec::with_capacity(self.shape.len());
        let mut expected_len = 1_usize;
        for &dim in &self.shape {
            let dim_i32 = i32::try_from(dim).map_err(|_| {
                IndexTtsError::weight_load(&self.name, format!("dimension {dim} exceeds i32::MAX"))
            })?;
            shape_i32.push(dim_i32);
            expected_len = expected_len.checked_mul(dim).ok_or_else(|| {
                IndexTtsError::weight_load(&self.name, "shape product overflowed usize")
            })?;
        }

        match &self.storage {
            TensorStorage::F32(v) => {
                if v.len() != expected_len {
                    return Err(IndexTtsError::weight_load(
                        &self.name,
                        format!(
                            "F32 length {} does not match shape {:?} (expected {expected_len})",
                            v.len(),
                            self.shape
                        ),
                    ));
                }
                Ok(Array::from_slice(v, &shape_i32))
            }
            TensorStorage::U32(v) => {
                if v.len() != expected_len {
                    return Err(IndexTtsError::weight_load(
                        &self.name,
                        format!(
                            "U32 length {} does not match shape {:?} (expected {expected_len})",
                            v.len(),
                            self.shape
                        ),
                    ));
                }
                Ok(Array::from_slice(v, &shape_i32))
            }
            TensorStorage::I32(v) => {
                if v.len() != expected_len {
                    return Err(IndexTtsError::weight_load(
                        &self.name,
                        format!(
                            "I32 length {} does not match shape {:?} (expected {expected_len})",
                            v.len(),
                            self.shape
                        ),
                    ));
                }
                Ok(Array::from_slice(v, &shape_i32))
            }
            TensorStorage::Mlx(arr) => {
                if arr.shape() != shape_i32 {
                    return Err(IndexTtsError::weight_load(
                        &self.name,
                        format!(
                            "MLX array shape {:?} does not match expected {:?}",
                            arr.shape(),
                            shape_i32
                        ),
                    ));
                }
                Ok(arr.clone())
            }
        }
    }
}

/// Result of loading a `.safetensors` file through [`SafeTensorsLoader`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FilteredWeights {
    /// Active parameter tensors keyed by their remapped MLX parameter name.
    pub tensors: HashMap<String, LoadedTensor>,
    /// Auxiliary non-parameter buffers (e.g., `_mean`, `_std`, `enc.embed.pos_enc.pe`,
    /// `_kaiser_sinc_filter`).
    pub buffers: HashMap<String, LoadedTensor>,
    /// Number of dead-weight tensors skipped during header filtering.
    pub skipped_count: usize,
    /// Total bytes of dead weights avoided.
    pub skipped_bytes: usize,
}

impl FilteredWeights {
    /// Looks up a parameter tensor by name and converts it into an [`mlx_rs::Array`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if the tensor is missing or invalid.
    pub fn tensor_array(&self, name: &str) -> Result<Array> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| IndexTtsError::weight_load(name, "missing required parameter tensor"))?;
        tensor.to_mlx_array()
    }

    /// Looks up a parameter tensor by name, verifies its exact shape, and converts it into an
    /// [`mlx_rs::Array`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if the tensor is missing, has a mismatched shape,
    /// or fails conversion.
    pub fn tensor_array_shaped(&self, name: &str, expected_shape: &[usize]) -> Result<Array> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| IndexTtsError::weight_load(name, "missing required parameter tensor"))?;
        if tensor.shape != expected_shape {
            return Err(IndexTtsError::weight_load(
                name,
                format!(
                    "expected shape {expected_shape:?}, found {:?}",
                    tensor.shape
                ),
            ));
        }
        tensor.to_mlx_array()
    }

    /// Looks up an auxiliary buffer tensor by name and converts it into an [`mlx_rs::Array`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if the buffer is missing or invalid.
    pub fn buffer_array(&self, name: &str) -> Result<Array> {
        let tensor = self
            .buffers
            .get(name)
            .ok_or_else(|| IndexTtsError::weight_load(name, "missing required buffer tensor"))?;
        tensor.to_mlx_array()
    }
}

/// Speaker projection layer weights loaded from `spk_proj.npz` (`[1280, 192]` weight + `[1280]`
/// bias).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpkProjWeights {
    /// Row-major weight matrix of shape `[1280, 192]`.
    pub weight: Vec<f32>,
    /// Bias vector of shape `[1280]`.
    pub bias: Vec<f32>,
}

impl SpkProjWeights {
    /// Parses `spk_proj.npz` from raw bytes (uncompressed ZIP archive containing `weight.npy` and
    /// `bias.npy`).
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if the ZIP or NPY headers are malformed or shapes
    /// mismatch.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::{SpkProjWeights, resolve_model_dir};
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let proj = SpkProjWeights::load_from_path(dir.join("spk_proj.npz"))?;
    ///     assert_eq!(proj.weight.len(), 1280 * 192);
    ///     assert_eq!(proj.bias.len(), 1280);
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn from_npz_bytes(bytes: &[u8]) -> Result<Self> {
        let entries = parse_stored_zip_entries(bytes)?;
        let weight_npy = entries.get("weight.npy").ok_or_else(|| {
            IndexTtsError::weight_load("spk_proj.npz", "missing `weight.npy` entry in archive")
        })?;
        let bias_npy = entries.get("bias.npy").ok_or_else(|| {
            IndexTtsError::weight_load("spk_proj.npz", "missing `bias.npy` entry in archive")
        })?;

        let (w_shape, weight) = parse_npy_f32("spk_proj.npz:weight.npy", weight_npy)?;
        let (b_shape, bias) = parse_npy_f32("spk_proj.npz:bias.npy", bias_npy)?;

        if w_shape != [SPK_PROJ_OUT_DIM, SPK_PROJ_IN_DIM] {
            return Err(IndexTtsError::weight_load(
                "spk_proj.npz",
                format!(
                    "unexpected weight shape {w_shape:?}, expected [{SPK_PROJ_OUT_DIM}, \
                     {SPK_PROJ_IN_DIM}]"
                ),
            ));
        }
        if b_shape != [SPK_PROJ_OUT_DIM] {
            return Err(IndexTtsError::weight_load(
                "spk_proj.npz",
                format!("unexpected bias shape {b_shape:?}, expected [{SPK_PROJ_OUT_DIM}]"),
            ));
        }

        Ok(Self { weight, bias })
    }

    /// Loads `spk_proj.npz` from disk.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if the path is unsafe, [`IndexTtsError::Io`] on
    /// read failure, or [`IndexTtsError::WeightLoad`] if malformed.
    #[allow(clippy::disallowed_methods)]
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        validate_safe_file_path(path_ref, "spk_proj_path")?;
        let bytes = fs::read(path_ref).map_err(|err| IndexTtsError::io(path_ref, err))?;
        Self::from_npz_bytes(&bytes)
    }

    /// Creates a new [`SpkProjWeights`] after validating shapes `[1280, 192]` and `[1280]`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `weight` or `bias` lengths do not match.
    pub fn new(weight: Vec<f32>, bias: Vec<f32>) -> Result<Self> {
        if weight.len() != SPK_PROJ_OUT_DIM * SPK_PROJ_IN_DIM {
            return Err(IndexTtsError::validation(
                "weight",
                format!(
                    "expected {} elements, got {}",
                    SPK_PROJ_OUT_DIM * SPK_PROJ_IN_DIM,
                    weight.len()
                ),
            ));
        }
        if bias.len() != SPK_PROJ_OUT_DIM {
            return Err(IndexTtsError::validation(
                "bias",
                format!("expected {SPK_PROJ_OUT_DIM} elements, got {}", bias.len()),
            ));
        }
        Ok(Self { weight, bias })
    }

    /// Projects a `[192]` speaker style vector into a `[1280]` speaker conditioning vector:
    /// `spk_cond = style @ weight.T + bias`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if `style.len() != 192`.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::SpkProjWeights;
    ///
    /// let proj = SpkProjWeights::new(vec![0.0; 1280 * 192], vec![0.5; 1280])?;
    /// let out = proj.project(&vec![1.0; 192])?;
    /// assert_eq!(out.len(), 1280);
    /// assert!((out[0] - 0.5).abs() < 1e-6);
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn project(&self, style: &[f32]) -> Result<Vec<f32>> {
        if style.len() != SPK_PROJ_IN_DIM {
            return Err(IndexTtsError::validation(
                "style",
                format!("expected {SPK_PROJ_IN_DIM} elements, got {}", style.len()),
            ));
        }
        let mut out = Vec::with_capacity(SPK_PROJ_OUT_DIM);
        for (row, &b) in self
            .weight
            .as_chunks::<SPK_PROJ_IN_DIM>()
            .0
            .iter()
            .zip(self.bias.iter())
        {
            let dot: f32 = row.iter().zip(style.iter()).map(|(&w, &s)| w * s).sum();
            out.push(dot + b);
        }
        Ok(out)
    }
}

/// Resolves the local directory containing `IndexTTS-2.5-mlx` model weights.
///
/// Checks `explicit_dir` first, then `$INDEXTTS_MODEL_DIR`, and finally searches
/// the local Hugging Face Hub cache (`$HF_HUB_CACHE`, `$HF_HOME/hub`, or
/// `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev>`).
///
/// # Errors
///
/// Returns [`IndexTtsError::Validation`] if `explicit_dir` is unsafe or
/// [`IndexTtsError::WeightLoad`] if no valid model snapshot directory is found.
///
/// # Examples
///
/// ```
/// use index_tts_rs_core::models::resolve_model_dir;
///
/// // Resolves from local Hugging Face cache if available:
/// let resolved = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx");
/// if let Ok(path) = resolved {
///     assert!(path.join("spk_proj.npz").exists());
/// }
/// ```
#[allow(clippy::disallowed_methods)]
pub fn resolve_model_dir(explicit_dir: Option<&Path>, repo_id: &str) -> Result<PathBuf> {
    if let Some(dir) = explicit_dir {
        validate_safe_file_path(dir, "model_dir")?;
        verify_model_snapshot_dir(dir)?;
        return Ok(dir.to_path_buf());
    }

    if let Ok(env_dir) = env::var("INDEXTTS_MODEL_DIR")
        && !env_dir.trim().is_empty()
    {
        let dir = PathBuf::from(env_dir);
        validate_safe_file_path(&dir, "INDEXTTS_MODEL_DIR")?;
        verify_model_snapshot_dir(&dir)?;
        return Ok(dir);
    }

    let effective_repo = if repo_id.trim().is_empty() {
        DEFAULT_MODEL_REPO
    } else {
        repo_id.trim()
    };
    let folder_name = format!("models--{}", effective_repo.replace('/', "--"));

    for hub_root in candidate_hf_hub_roots() {
        let repo_root = hub_root.join(&folder_name);
        let snapshots_dir = repo_root.join("snapshots");
        if !snapshots_dir.is_dir() {
            continue;
        }

        // 1. Check `refs/main` first
        let refs_main = repo_root.join("refs").join("main");
        if let Ok(rev) = fs::read_to_string(&refs_main) {
            let candidate = snapshots_dir.join(rev.trim());
            if verify_model_snapshot_dir(&candidate).is_ok() {
                return Ok(candidate);
            }
        }

        // 2. Fallback: scan subdirectories in `snapshots/`
        if let Ok(entries) = fs::read_dir(&snapshots_dir) {
            let mut subdirs: Vec<PathBuf> = entries
                .filter_map(std::result::Result::ok)
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect();
            subdirs.sort();
            for candidate in subdirs.into_iter().rev() {
                if verify_model_snapshot_dir(&candidate).is_ok() {
                    return Ok(candidate);
                }
            }
        }
    }

    Err(IndexTtsError::weight_load(
        effective_repo,
        "could not locate valid local model snapshot in Hugging Face cache; run `itts models \
         pull` or pass `--model-dir`",
    ))
}

/// `SafeTensors` loader that strips dead weights at header-inspection time and applies
/// PyTorch-to-MLX layout transpositions.
#[derive(Debug, Default, Clone, Copy)]
pub struct SafeTensorsLoader;

impl SafeTensorsLoader {
    /// Determines whether a raw tensor key in `kind` should be loaded (`true`) or
    /// discarded as a dead weight (`false`).
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::{ModelWeightKind, SafeTensorsLoader};
    ///
    /// assert!(SafeTensorsLoader::should_keep_tensor(
    ///     ModelWeightKind::SemanticModel,
    ///     "encoder.layers.16.final_layer_norm.weight"
    /// ));
    /// assert!(!SafeTensorsLoader::should_keep_tensor(
    ///     ModelWeightKind::SemanticModel,
    ///     "encoder.layers.17.final_layer_norm.weight"
    /// ));
    /// assert!(!SafeTensorsLoader::should_keep_tensor(
    ///     ModelWeightKind::SemanticCodec,
    ///     "encoder.blocks.0.weight"
    /// ));
    /// ```
    #[must_use]
    pub fn should_keep_tensor(kind: ModelWeightKind, raw_key: &str) -> bool {
        match kind {
            ModelWeightKind::SemanticModel => {
                if raw_key == "masked_spec_embed" {
                    return false;
                }
                if let Some(rest) = raw_key.strip_prefix("encoder.layers.")
                    && let Some(layer_str) = rest.split('.').next()
                    && let Ok(layer_idx) = layer_str.parse::<usize>()
                {
                    return layer_idx < SEMANTIC_ACTIVE_LAYERS;
                }
                true
            }
            ModelWeightKind::SemanticCodec => {
                !raw_key.starts_with("encoder.")
                    && !raw_key.starts_with("down.")
                    && !raw_key.contains(".in_project.")
            }
            ModelWeightKind::LengthRegulator => {
                raw_key != "embedding.weight" && raw_key != "mask_token"
            }
            ModelWeightKind::CampPlus => !raw_key.ends_with("num_batches_tracked"),
            ModelWeightKind::BigVgan
            | ModelWeightKind::EmoVec
            | ModelWeightKind::GptInt8
            | ModelWeightKind::CfmEstimator => true,
        }
    }

    /// Loads and transforms a `SafeTensors` byte buffer according to `kind`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::WeightLoad`] if `SafeTensors` deserialization or tensor
    /// transposition fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use index_tts_rs_core::models::{ModelWeightKind, SafeTensorsLoader, resolve_model_dir};
    ///
    /// if let Ok(dir) = resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx") {
    ///     let loaded = SafeTensorsLoader::load_from_path(
    ///         ModelWeightKind::LengthRegulator,
    ///         dir.join("raw/length_regulator.safetensors"),
    ///     )?;
    ///     assert_eq!(loaded.skipped_count, 2);
    ///     assert!(loaded.tensors.contains_key("model.layers.0.weight"));
    /// }
    /// # Ok::<(), index_tts_rs_core::error::IndexTtsError>(())
    /// ```
    pub fn load_from_bytes(kind: ModelWeightKind, bytes: &[u8]) -> Result<FilteredWeights> {
        let st = SafeTensors::deserialize(bytes).map_err(|err| {
            IndexTtsError::weight_load(
                kind.relative_path(),
                format!("safetensors parse failed: {err}"),
            )
        })?;

        let mut tensors = HashMap::new();
        let mut buffers = HashMap::new();
        let mut skipped_count = 0_usize;
        let mut skipped_bytes = 0_usize;
        let mut extracted_bigvgan_filter = false;

        for (raw_key, view) in st.tensors() {
            if !Self::should_keep_tensor(kind, &raw_key) {
                skipped_count = skipped_count.saturating_add(1);
                skipped_bytes = skipped_bytes.saturating_add(view.data().len());
                continue;
            }

            // Handle BigVGAN's 218 duplicate Kaiser-Sinc filter buffers
            if kind == ModelWeightKind::BigVgan
                && (raw_key.ends_with(".upsample.filter") || raw_key.ends_with(".lowpass.filter"))
            {
                if extracted_bigvgan_filter {
                    skipped_count = skipped_count.saturating_add(1);
                    skipped_bytes = skipped_bytes.saturating_add(view.data().len());
                } else {
                    let vals = decode_f32_view(kind.relative_path(), &raw_key, &view)?;
                    let len = vals.len();
                    buffers.insert(
                        "_kaiser_sinc_filter".to_owned(),
                        LoadedTensor {
                            name: "_kaiser_sinc_filter".to_owned(),
                            shape: vec![len],
                            storage: TensorStorage::F32(vals),
                        },
                    );
                    extracted_bigvgan_filter = true;
                }
                continue;
            }

            // Handle SemanticModel `_mean` / `_std` and EmoVec `enc.embed.pos_enc.pe` buffers
            if (kind == ModelWeightKind::SemanticModel && (raw_key == "_mean" || raw_key == "_std"))
                || (kind == ModelWeightKind::EmoVec && raw_key == "enc.embed.pos_enc.pe")
            {
                let loaded = decode_and_transform_tensor(kind, &raw_key, &view)?;
                buffers.insert(loaded.name.clone(), loaded);
                continue;
            }

            let loaded = decode_and_transform_tensor(kind, &raw_key, &view)?;
            tensors.insert(loaded.name.clone(), loaded);
        }

        Ok(FilteredWeights {
            tensors,
            buffers,
            skipped_count,
            skipped_bytes,
        })
    }

    /// Loads and transforms a `SafeTensors` file from disk according to `kind`, streaming only
    /// active tensors directly into MLX unified memory without reading skipped dead weights or
    /// allocating a full-file buffer.
    ///
    /// # Errors
    ///
    /// Returns [`IndexTtsError::Validation`] if the path is unsafe, [`IndexTtsError::Io`] on
    /// read failure, or [`IndexTtsError::WeightLoad`] if parsing fails.
    #[allow(
        clippy::disallowed_methods,
        clippy::disallowed_types,
        clippy::too_many_lines
    )]
    pub fn load_from_path(
        kind: ModelWeightKind,
        path: impl AsRef<Path>,
    ) -> Result<FilteredWeights> {
        const MAX_SAFETENSORS_HEADER_BYTES: usize = 100_000_000;

        let path_ref = path.as_ref();
        validate_safe_file_path(path_ref, "weight_path")?;
        let artifact = kind.relative_path();

        let mut file = fs::File::open(path_ref).map_err(|err| IndexTtsError::io(path_ref, err))?;
        let file_len = usize::try_from(
            file.metadata()
                .map_err(|err| IndexTtsError::io(path_ref, err))?
                .len(),
        )
        .map_err(|_| IndexTtsError::weight_load(artifact, "file size exceeds usize::MAX"))?;

        if file_len < 8 {
            return Err(IndexTtsError::weight_load(
                artifact,
                "safetensors file smaller than 8-byte header prefix",
            ));
        }

        let mut len_bytes = [0_u8; 8];
        file.read_exact(&mut len_bytes)
            .map_err(|err| IndexTtsError::io(path_ref, err))?;
        let header_len = usize::try_from(u64::from_le_bytes(len_bytes)).map_err(|_| {
            IndexTtsError::weight_load(artifact, "safetensors header length overflow")
        })?;
        if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
            return Err(IndexTtsError::weight_load(
                artifact,
                format!("invalid safetensors header length {header_len}"),
            ));
        }
        let data_base = 8_usize.checked_add(header_len).ok_or_else(|| {
            IndexTtsError::weight_load(artifact, "safetensors header offset overflow")
        })?;
        if data_base > file_len {
            return Err(IndexTtsError::weight_load(
                artifact,
                "safetensors header length exceeds file size",
            ));
        }

        let mut header_buf = vec![0_u8; header_len];
        file.read_exact(&mut header_buf)
            .map_err(|err| IndexTtsError::io(path_ref, err))?;
        let metadata: Metadata = serde_json::from_slice(&header_buf).map_err(|err| {
            IndexTtsError::weight_load(artifact, format!("safetensors header parse failed: {err}"))
        })?;
        drop(header_buf);

        if data_base.saturating_add(metadata.data_len()) != file_len {
            return Err(IndexTtsError::weight_load(
                artifact,
                "safetensors data section length does not match file size",
            ));
        }

        let mut tensors = HashMap::new();
        let mut buffers = HashMap::new();
        let mut skipped_count = 0_usize;
        let mut skipped_bytes = 0_usize;
        let mut extracted_bigvgan_filter = false;

        for raw_key in metadata.offset_keys() {
            let Some(info) = metadata.info(&raw_key) else {
                continue;
            };
            let (start, end) = info.data_offsets;
            let byte_len = end.saturating_sub(start);

            if !Self::should_keep_tensor(kind, &raw_key) {
                skipped_count = skipped_count.saturating_add(1);
                skipped_bytes = skipped_bytes.saturating_add(byte_len);
                continue;
            }

            if kind == ModelWeightKind::BigVgan
                && (raw_key.ends_with(".upsample.filter") || raw_key.ends_with(".lowpass.filter"))
            {
                if extracted_bigvgan_filter {
                    skipped_count = skipped_count.saturating_add(1);
                    skipped_bytes = skipped_bytes.saturating_add(byte_len);
                    continue;
                }
                let file_offset = (data_base + start) as u64;
                file.seek(SeekFrom::Start(file_offset))
                    .map_err(|err| IndexTtsError::io(path_ref, err))?;
                let mut raw_bytes = vec![0_u8; byte_len];
                file.read_exact(&mut raw_bytes)
                    .map_err(|err| IndexTtsError::io(path_ref, err))?;
                let view =
                    TensorView::new(info.dtype, info.shape.clone(), &raw_bytes).map_err(|err| {
                        IndexTtsError::weight_load(artifact, format!("invalid tensor view: {err}"))
                    })?;
                let vals = decode_f32_view(artifact, &raw_key, &view)?;
                let len = vals.len();
                buffers.insert(
                    "_kaiser_sinc_filter".to_owned(),
                    LoadedTensor {
                        name: "_kaiser_sinc_filter".to_owned(),
                        shape: vec![len],
                        storage: TensorStorage::F32(vals),
                    },
                );
                extracted_bigvgan_filter = true;
                continue;
            }

            let file_offset = (data_base + start) as u64;
            file.seek(SeekFrom::Start(file_offset))
                .map_err(|err| IndexTtsError::io(path_ref, err))?;
            let mut raw_bytes = vec![0_u8; byte_len];
            file.read_exact(&mut raw_bytes)
                .map_err(|err| IndexTtsError::io(path_ref, err))?;
            let view =
                TensorView::new(info.dtype, info.shape.clone(), &raw_bytes).map_err(|err| {
                    IndexTtsError::weight_load(artifact, format!("invalid tensor view: {err}"))
                })?;

            let loaded = decode_and_transform_tensor(kind, &raw_key, &view)?;
            drop(raw_bytes);

            if (kind == ModelWeightKind::SemanticModel && (raw_key == "_mean" || raw_key == "_std"))
                || (kind == ModelWeightKind::EmoVec && raw_key == "enc.embed.pos_enc.pe")
            {
                buffers.insert(loaded.name.clone(), loaded);
            } else {
                tensors.insert(loaded.name.clone(), loaded);
            }
        }

        Ok(FilteredWeights {
            tensors,
            buffers,
            skipped_count,
            skipped_bytes,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransposeRule {
    None,
    Axes021,
    Axes120,
    Axes0231,
}

fn decode_and_transform_tensor(
    kind: ModelWeightKind,
    raw_key: &str,
    view: &TensorView<'_>,
) -> Result<LoadedTensor> {
    let artifact = kind.relative_path();
    let shape = view.shape().to_vec();

    match view.dtype() {
        Dtype::F32 => {
            let (new_key, rule) = remap_key_and_transpose_rule(kind, raw_key, &shape);
            if rule == TransposeRule::None {
                let arr = Array::try_from(view.clone()).map_err(|err| {
                    IndexTtsError::weight_load(
                        artifact,
                        format!("failed to convert F32 tensor `{raw_key}` to MLX Array: {err}"),
                    )
                })?;
                return Ok(LoadedTensor {
                    name: new_key,
                    shape,
                    storage: TensorStorage::Mlx(arr),
                });
            }

            let values = decode_f32_view(artifact, raw_key, view)?;
            let (new_shape, transposed) = match rule {
                TransposeRule::Axes021 => transpose_3d_021(&shape, &values)?,
                TransposeRule::Axes120 => transpose_3d_120(&shape, &values)?,
                TransposeRule::Axes0231 => transpose_4d_0231(&shape, &values)?,
                TransposeRule::None => (shape, values),
            };
            let shape_i32: Vec<i32> = new_shape
                .iter()
                .map(|&d| {
                    i32::try_from(d).map_err(|_| {
                        IndexTtsError::weight_load(&new_key, format!("dim {d} exceeds i32::MAX"))
                    })
                })
                .collect::<Result<Vec<i32>>>()?;
            let arr = Array::from_slice(&transposed, &shape_i32);
            Ok(LoadedTensor {
                name: new_key,
                shape: new_shape,
                storage: TensorStorage::Mlx(arr),
            })
        }
        Dtype::U32 | Dtype::I32 => {
            let arr = Array::try_from(view.clone()).map_err(|err| {
                IndexTtsError::weight_load(
                    artifact,
                    format!("failed to convert integer tensor `{raw_key}` to MLX Array: {err}"),
                )
            })?;
            Ok(LoadedTensor {
                name: raw_key.to_owned(),
                shape,
                storage: TensorStorage::Mlx(arr),
            })
        }
        other => Err(IndexTtsError::weight_load(
            artifact,
            format!("unsupported dtype {other:?} for tensor `{raw_key}`"),
        )),
    }
}

fn remap_key_and_transpose_rule(
    kind: ModelWeightKind,
    raw_key: &str,
    shape: &[usize],
) -> (String, TransposeRule) {
    match kind {
        ModelWeightKind::GptInt8 | ModelWeightKind::CfmEstimator => {
            (raw_key.to_owned(), TransposeRule::None)
        }
        ModelWeightKind::SemanticModel | ModelWeightKind::SemanticCodec => {
            let rule = if shape.len() == 3 {
                TransposeRule::Axes021
            } else {
                TransposeRule::None
            };
            (raw_key.to_owned(), rule)
        }
        ModelWeightKind::LengthRegulator => {
            let parts: Vec<&str> = raw_key.split('.').collect();
            if parts.first().copied() == Some("model")
                && parts.len() == 3
                && let Some(pos_str) = parts.get(1)
                && let Ok(pos) = pos_str.parse::<usize>()
                && let Some(param) = parts.get(2)
            {
                let new_key = format!("model.layers.{pos}.{param}");
                let rule = if matches!(pos, 0 | 3 | 6 | 9 | 12)
                    && *param == "weight"
                    && shape.len() == 3
                {
                    TransposeRule::Axes021
                } else {
                    TransposeRule::None
                };
                return (new_key, rule);
            }
            (raw_key.to_owned(), TransposeRule::None)
        }
        ModelWeightKind::BigVgan => {
            if raw_key.starts_with("ups.") {
                let parts: Vec<&str> = raw_key.split('.').collect();
                if parts.len() == 4
                    && let (Some(idx), Some(param)) = (parts.get(1), parts.get(3))
                {
                    let new_key = format!("ups.{idx}.{param}");
                    let rule = if shape.len() == 3 {
                        TransposeRule::Axes120
                    } else {
                        TransposeRule::None
                    };
                    return (new_key, rule);
                }
            }
            let rule = if shape.len() == 3 {
                TransposeRule::Axes021
            } else {
                TransposeRule::None
            };
            (raw_key.to_owned(), rule)
        }
        ModelWeightKind::EmoVec => {
            if let Some(suffix) = raw_key.strip_prefix("enc.embed.conv.0.") {
                let new_key = format!("enc.embed_conv.{suffix}");
                let rule = if suffix == "weight" && shape.len() == 4 {
                    TransposeRule::Axes0231
                } else {
                    TransposeRule::None
                };
                return (new_key, rule);
            }
            if let Some(suffix) = raw_key.strip_prefix("enc.embed.out.0.") {
                return (format!("enc.embed_out.{suffix}"), TransposeRule::None);
            }
            if (raw_key.ends_with(".pointwise_conv1.weight")
                || raw_key.ends_with(".depthwise_conv.weight")
                || raw_key.ends_with(".pointwise_conv2.weight"))
                && shape.len() == 3
            {
                return (raw_key.to_owned(), TransposeRule::Axes021);
            }
            (raw_key.to_owned(), TransposeRule::None)
        }
        ModelWeightKind::CampPlus => {
            if raw_key.ends_with(".weight") {
                if shape.len() == 4 {
                    return (raw_key.to_owned(), TransposeRule::Axes0231);
                }
                if shape.len() == 3 {
                    return (raw_key.to_owned(), TransposeRule::Axes021);
                }
            }
            (raw_key.to_owned(), TransposeRule::None)
        }
    }
}

/// Transposes a 3D tensor from `PyTorch` `Conv1d` layout `[D0, D1, D2]` (`[Out, In, K]`)
/// to MLX `Conv1d` layout `[D0, D2, D1]` (`[Out, K, In]`).
pub(crate) fn transpose_3d_021(shape: &[usize], data: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
    let [d0, d1, d2] = <[usize; 3]>::try_from(shape).map_err(|_| {
        IndexTtsError::weight_load(
            "transpose_3d_021",
            format!("expected 3D shape, got {shape:?}"),
        )
    })?;
    let expected = d0.saturating_mul(d1).saturating_mul(d2);
    if data.len() != expected {
        return Err(IndexTtsError::weight_load(
            "transpose_3d_021",
            format!("data length {} does not match shape {shape:?}", data.len()),
        ));
    }
    let mut out = vec![0.0_f32; expected];
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                let src_idx = (i0 * d1 + i1) * d2 + i2;
                let dst_idx = (i0 * d2 + i2) * d1 + i1;
                if let (Some(&val), Some(slot)) = (data.get(src_idx), out.get_mut(dst_idx)) {
                    *slot = val;
                }
            }
        }
    }
    Ok((vec![d0, d2, d1], out))
}

/// Transposes a 3D tensor from `PyTorch` `ConvTranspose1d` layout `[D0, D1, D2]` (`[In, Out, K]`)
/// to MLX layout `[D1, D2, D0]` (`[Out, K, In]`).
pub(crate) fn transpose_3d_120(shape: &[usize], data: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
    let [d0, d1, d2] = <[usize; 3]>::try_from(shape).map_err(|_| {
        IndexTtsError::weight_load(
            "transpose_3d_120",
            format!("expected 3D shape, got {shape:?}"),
        )
    })?;
    let expected = d0.saturating_mul(d1).saturating_mul(d2);
    if data.len() != expected {
        return Err(IndexTtsError::weight_load(
            "transpose_3d_120",
            format!("data length {} does not match shape {shape:?}", data.len()),
        ));
    }
    let mut out = vec![0.0_f32; expected];
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                let src_idx = (i0 * d1 + i1) * d2 + i2;
                let dst_idx = (i1 * d2 + i2) * d0 + i0;
                if let (Some(&val), Some(slot)) = (data.get(src_idx), out.get_mut(dst_idx)) {
                    *slot = val;
                }
            }
        }
    }
    Ok((vec![d1, d2, d0], out))
}

/// Transposes a 4D tensor from `PyTorch` `Conv2d` layout `[D0, D1, D2, D3]` (`[Out, In, H, W]`)
/// to MLX `Conv2d` layout `[D0, D2, D3, D1]` (`[Out, H, W, In]`).
pub(crate) fn transpose_4d_0231(shape: &[usize], data: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
    let [d0, d1, d2, d3] = <[usize; 4]>::try_from(shape).map_err(|_| {
        IndexTtsError::weight_load(
            "transpose_4d_0231",
            format!("expected 4D shape, got {shape:?}"),
        )
    })?;
    let expected = d0.saturating_mul(d1).saturating_mul(d2).saturating_mul(d3);
    if data.len() != expected {
        return Err(IndexTtsError::weight_load(
            "transpose_4d_0231",
            format!("data length {} does not match shape {shape:?}", data.len()),
        ));
    }
    let mut out = vec![0.0_f32; expected];
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                for i3 in 0..d3 {
                    let src_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
                    let dst_idx = ((i0 * d2 + i2) * d3 + i3) * d1 + i1;
                    if let (Some(&val), Some(slot)) = (data.get(src_idx), out.get_mut(dst_idx)) {
                        *slot = val;
                    }
                }
            }
        }
    }
    Ok((vec![d0, d2, d3, d1], out))
}

fn decode_f32_view(artifact: &str, key: &str, view: &TensorView<'_>) -> Result<Vec<f32>> {
    let raw = view.data();
    if raw.len() % 4 != 0 {
        return Err(IndexTtsError::weight_load(
            artifact,
            format!(
                "tensor `{key}` byte length {} is not a multiple of 4",
                raw.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(raw.len() / 4);
    for &bytes in raw.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

fn parse_stored_zip_entries(bytes: &[u8]) -> Result<HashMap<String, &[u8]>> {
    const LOCAL_FILE_HEADER_SIG: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];
    const CENTRAL_DIR_SIG: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];

    let mut entries = HashMap::new();
    let mut offset = 0_usize;

    while let Some(sig) = bytes.get(offset..offset.saturating_add(4)) {
        if sig == CENTRAL_DIR_SIG || sig == [0x50, 0x4b, 0x05, 0x06] {
            break;
        }
        if sig != LOCAL_FILE_HEADER_SIG {
            return Err(IndexTtsError::weight_load(
                "spk_proj.npz",
                format!("invalid ZIP local file header signature at byte offset {offset}"),
            ));
        }
        let header = bytes
            .get(offset..offset.saturating_add(30))
            .ok_or_else(|| {
                IndexTtsError::weight_load("spk_proj.npz", "truncated ZIP local file header")
            })?;
        let compression = u16::from_le_bytes([
            header.get(8).copied().unwrap_or(0),
            header.get(9).copied().unwrap_or(0),
        ]);
        if compression != 0 {
            return Err(IndexTtsError::weight_load(
                "spk_proj.npz",
                format!("unsupported ZIP compression method {compression}, expected STORED (0)"),
            ));
        }
        let raw_comp_size = u32::from_le_bytes([
            header.get(18).copied().unwrap_or(0),
            header.get(19).copied().unwrap_or(0),
            header.get(20).copied().unwrap_or(0),
            header.get(21).copied().unwrap_or(0),
        ]);
        let name_len = u16::from_le_bytes([
            header.get(26).copied().unwrap_or(0),
            header.get(27).copied().unwrap_or(0),
        ]) as usize;
        let extra_len = u16::from_le_bytes([
            header.get(28).copied().unwrap_or(0),
            header.get(29).copied().unwrap_or(0),
        ]) as usize;

        let name_start = offset.saturating_add(30);
        let name_end = name_start.saturating_add(name_len);
        let name_bytes = bytes.get(name_start..name_end).ok_or_else(|| {
            IndexTtsError::weight_load("spk_proj.npz", "truncated ZIP entry filename")
        })?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|err| {
                IndexTtsError::weight_load(
                    "spk_proj.npz",
                    format!("invalid UTF-8 entry name: {err}"),
                )
            })?
            .to_owned();

        let data_start = name_end.saturating_add(extra_len);
        let extra_bytes = bytes.get(name_end..data_start).ok_or_else(|| {
            IndexTtsError::weight_load("spk_proj.npz", "truncated ZIP extra field")
        })?;
        let compressed_size = if raw_comp_size == u32::MAX {
            parse_zip64_compressed_size(extra_bytes)?
        } else {
            raw_comp_size as usize
        };
        let data_end = data_start.saturating_add(compressed_size);
        let payload = bytes.get(data_start..data_end).ok_or_else(|| {
            IndexTtsError::weight_load("spk_proj.npz", format!("truncated payload for `{name}`"))
        })?;
        entries.insert(name, payload);
        offset = data_end;
    }

    if entries.is_empty() {
        return Err(IndexTtsError::weight_load(
            "spk_proj.npz",
            "no ZIP entries found in `.npz` archive",
        ));
    }
    Ok(entries)
}

fn parse_zip64_compressed_size(extra: &[u8]) -> Result<usize> {
    let mut pos = 0_usize;
    while pos.saturating_add(4) <= extra.len() {
        let header_id = u16::from_le_bytes([
            extra.get(pos).copied().unwrap_or(0),
            extra.get(pos + 1).copied().unwrap_or(0),
        ]);
        let data_size = u16::from_le_bytes([
            extra.get(pos + 2).copied().unwrap_or(0),
            extra.get(pos + 3).copied().unwrap_or(0),
        ]) as usize;
        let field_start = pos.saturating_add(4);
        let field_end = field_start.saturating_add(data_size);
        if header_id == 0x0001 {
            let field = extra.get(field_start..field_end).ok_or_else(|| {
                IndexTtsError::weight_load("spk_proj.npz", "truncated ZIP64 extra block")
            })?;
            let size_bytes = if field.len() >= 16 {
                field.get(8..16)
            } else {
                field.get(0..8)
            };
            let arr: [u8; 8] = size_bytes.and_then(|s| s.try_into().ok()).ok_or_else(|| {
                IndexTtsError::weight_load("spk_proj.npz", "invalid ZIP64 size field")
            })?;
            #[allow(clippy::cast_possible_truncation)]
            return Ok(u64::from_le_bytes(arr) as usize);
        }
        pos = field_end;
    }
    Err(IndexTtsError::weight_load(
        "spk_proj.npz",
        "ZIP64 sentinel size without 0x0001 ZIP64 extra header",
    ))
}

fn parse_npy_f32(artifact: &str, npy_bytes: &[u8]) -> Result<(Vec<usize>, Vec<f32>)> {
    const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";
    if npy_bytes.get(0..6) != Some(NPY_MAGIC) {
        return Err(IndexTtsError::weight_load(
            artifact,
            "missing `\\x93NUMPY` magic bytes",
        ));
    }
    let major = npy_bytes.get(6).copied().unwrap_or(0);
    let (header_len, data_offset) = match major {
        1 => {
            let len = u16::from_le_bytes([
                npy_bytes.get(8).copied().unwrap_or(0),
                npy_bytes.get(9).copied().unwrap_or(0),
            ]) as usize;
            (len, 10_usize.saturating_add(len))
        }
        2 => {
            let len = u32::from_le_bytes([
                npy_bytes.get(8).copied().unwrap_or(0),
                npy_bytes.get(9).copied().unwrap_or(0),
                npy_bytes.get(10).copied().unwrap_or(0),
                npy_bytes.get(11).copied().unwrap_or(0),
            ]) as usize;
            (len, 12_usize.saturating_add(len))
        }
        other => {
            return Err(IndexTtsError::weight_load(
                artifact,
                format!("unsupported NPY major version {other}"),
            ));
        }
    };

    let header_start = data_offset.saturating_sub(header_len);
    let header_slice = npy_bytes
        .get(header_start..data_offset)
        .ok_or_else(|| IndexTtsError::weight_load(artifact, "truncated NPY header"))?;
    let header_str = std::str::from_utf8(header_slice).map_err(|err| {
        IndexTtsError::weight_load(artifact, format!("invalid UTF-8 in NPY header: {err}"))
    })?;

    if !header_str.contains("'<f4'") && !header_str.contains("\"<f4\"") {
        return Err(IndexTtsError::weight_load(
            artifact,
            format!("expected little-endian float32 (`<f4`) NPY descriptor, got {header_str}"),
        ));
    }
    if header_str.contains("'fortran_order': True") {
        return Err(IndexTtsError::weight_load(
            artifact,
            "Fortran-ordered NPY arrays are not supported",
        ));
    }

    let shape = parse_npy_shape(artifact, header_str)?;
    let expected_elems: usize = shape.iter().copied().product();
    let payload = npy_bytes
        .get(data_offset..)
        .ok_or_else(|| IndexTtsError::weight_load(artifact, "missing NPY data section"))?;
    if payload.len() != expected_elems.saturating_mul(4) {
        return Err(IndexTtsError::weight_load(
            artifact,
            format!(
                "NPY byte payload length {} does not match shape {shape:?}",
                payload.len()
            ),
        ));
    }

    let mut values = Vec::with_capacity(expected_elems);
    for &arr in payload.as_chunks::<4>().0 {
        values.push(f32::from_le_bytes(arr));
    }

    Ok((shape, values))
}

fn parse_npy_shape(artifact: &str, header_str: &str) -> Result<Vec<usize>> {
    let shape_pos = header_str.find("'shape'").ok_or_else(|| {
        IndexTtsError::weight_load(artifact, "missing `'shape'` field in NPY header")
    })?;
    let after_shape = header_str.get(shape_pos..).unwrap_or("");
    let open_paren = after_shape
        .find('(')
        .ok_or_else(|| IndexTtsError::weight_load(artifact, "missing `(` in NPY shape"))?;
    let close_paren = after_shape
        .find(')')
        .ok_or_else(|| IndexTtsError::weight_load(artifact, "missing `)` in NPY shape"))?;
    let inside = after_shape
        .get(open_paren.saturating_add(1)..close_paren)
        .ok_or_else(|| IndexTtsError::weight_load(artifact, "malformed NPY shape tuple"))?;

    let mut dims = Vec::new();
    for part in inside.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let dim = trimmed.parse::<usize>().map_err(|_| {
            IndexTtsError::weight_load(artifact, format!("invalid dimension `{trimmed}`"))
        })?;
        dims.push(dim);
    }
    Ok(dims)
}

fn candidate_hf_hub_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(hub_cache) = env::var("HF_HUB_CACHE")
        && !hub_cache.trim().is_empty()
    {
        roots.push(PathBuf::from(hub_cache));
    }
    if let Ok(hf_home) = env::var("HF_HOME")
        && !hf_home.trim().is_empty()
    {
        roots.push(PathBuf::from(hf_home).join("hub"));
    }
    if let Ok(home) = env::var("HOME")
        && !home.trim().is_empty()
    {
        roots.push(
            PathBuf::from(home)
                .join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }
    roots
}

fn verify_model_snapshot_dir(dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        return Err(IndexTtsError::weight_load(
            dir.display().to_string(),
            "model directory does not exist",
        ));
    }
    for req in REQUIRED_MODEL_FILES {
        let file_path = dir.join(req);
        if !file_path.exists() {
            return Err(IndexTtsError::weight_load(
                dir.display().to_string(),
                format!("missing required model file `{req}`"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use safetensors::{View, serialize};

    use super::*;

    struct DummyTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl View for DummyTensor {
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

    #[test]
    #[ignore = "requires local IndexTTS model weights"]
    fn test_should_resolve_local_model_dir_and_load_spk_proj_npz() {
        let model_dir = resolve_model_dir(None, DEFAULT_MODEL_REPO);
        assert!(
            model_dir.is_ok(),
            "expected local HF cache snapshot: {model_dir:?}"
        );
        if let Ok(dir) = model_dir {
            let proj = SpkProjWeights::load_from_path(dir.join("spk_proj.npz"));
            assert!(proj.is_ok(), "failed loading spk_proj.npz: {proj:?}");
            if let Ok(proj) = proj {
                assert_eq!(proj.weight.len(), SPK_PROJ_OUT_DIM * SPK_PROJ_IN_DIM);
                assert_eq!(proj.bias.len(), SPK_PROJ_OUT_DIM);
                // Verify exact first weight & bias values against Python np.load reference:
                let first_w = proj.weight.first().copied().unwrap_or(0.0);
                let first_b = proj.bias.first().copied().unwrap_or(0.0);
                assert!((first_w - 0.129_760_73).abs() < 1e-6);
                assert!((first_b - (-0.179_650_49)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_should_skip_unused_w2v_layers_17_to_23() {
        let tensors = vec![
            (
                "_mean",
                DummyTensor {
                    dtype: Dtype::F32,
                    shape: vec![1, 1, 1024],
                    bytes: vec![0_u8; 1024 * 4],
                },
            ),
            (
                "_std",
                DummyTensor {
                    dtype: Dtype::F32,
                    shape: vec![1, 1, 1024],
                    bytes: vec![0_u8; 1024 * 4],
                },
            ),
            (
                "masked_spec_embed",
                DummyTensor {
                    dtype: Dtype::F32,
                    shape: vec![1024],
                    bytes: vec![0_u8; 1024 * 4],
                },
            ),
            (
                "encoder.layers.16.conv_module.depthwise_conv.weight",
                DummyTensor {
                    dtype: Dtype::F32,
                    shape: vec![4, 1, 3],
                    bytes: vec![0_u8; 12 * 4],
                },
            ),
            (
                "encoder.layers.17.final_layer_norm.weight",
                DummyTensor {
                    dtype: Dtype::F32,
                    shape: vec![1024],
                    bytes: vec![0_u8; 1024 * 4],
                },
            ),
            (
                "encoder.layers.23.final_layer_norm.weight",
                DummyTensor {
                    dtype: Dtype::F32,
                    shape: vec![1024],
                    bytes: vec![0_u8; 1024 * 4],
                },
            ),
        ];

        let raw_bytes = serialize(tensors, None).unwrap_or_default();
        let filtered =
            SafeTensorsLoader::load_from_bytes(ModelWeightKind::SemanticModel, &raw_bytes);
        assert!(filtered.is_ok());
        if let Ok(filtered) = filtered {
            assert_eq!(filtered.skipped_count, 3);
            assert!(filtered.buffers.contains_key("_mean"));
            assert!(filtered.buffers.contains_key("_std"));
            let conv = filtered
                .tensors
                .get("encoder.layers.16.conv_module.depthwise_conv.weight");
            assert!(conv.is_some());
            if let Some(conv) = conv {
                // Transposed from [4, 1, 3] to [4, 3, 1]
                assert_eq!(conv.shape, vec![4, 3, 1]);
            }
            assert!(
                !filtered
                    .tensors
                    .contains_key("encoder.layers.17.final_layer_norm.weight")
            );
            assert!(
                !filtered
                    .tensors
                    .contains_key("encoder.layers.23.final_layer_norm.weight")
            );
        }
    }

    #[test]
    fn test_should_skip_codec_encoder_and_down_layers() {
        let tensors = vec![
            (
                "encoder.0.weight",
                DummyTensor {
                    dtype: Dtype::F32,
                    shape: vec![8, 4, 3],
                    bytes: vec![0_u8; 96 * 4],
                },
            ),
            (
                "down.weight",
                DummyTensor {
                    dtype: Dtype::F32,
                    shape: vec![8, 4, 3],
                    bytes: vec![0_u8; 96 * 4],
                },
            ),
            (
                "quantizer.quantizers.0.in_project.weight",
                DummyTensor {
                    dtype: Dtype::F32,
                    shape: vec![8, 1024, 1],
                    bytes: vec![0_u8; 8192 * 4],
                },
            ),
            (
                "quantizer.quantizers.0.out_project.weight",
                DummyTensor {
                    dtype: Dtype::F32,
                    shape: vec![1024, 8, 1],
                    bytes: vec![0_u8; 8192 * 4],
                },
            ),
        ];

        let raw_bytes = serialize(tensors, None).unwrap_or_default();
        let filtered =
            SafeTensorsLoader::load_from_bytes(ModelWeightKind::SemanticCodec, &raw_bytes);
        assert!(filtered.is_ok());
        if let Ok(filtered) = filtered {
            assert_eq!(filtered.skipped_count, 3);
            assert_eq!(filtered.tensors.len(), 1);
            let out_proj = filtered
                .tensors
                .get("quantizer.quantizers.0.out_project.weight");
            assert!(out_proj.is_some());
            if let Some(out_proj) = out_proj {
                assert_eq!(out_proj.shape, vec![1024, 1, 8]);
            }
        }
    }

    #[test]
    fn test_should_filter_real_semantic_codec_and_length_regulator_checkpoints() {
        let Ok(dir) = resolve_model_dir(None, DEFAULT_MODEL_REPO) else {
            return;
        };
        let lr = SafeTensorsLoader::load_from_path(
            ModelWeightKind::LengthRegulator,
            dir.join("raw/length_regulator.safetensors"),
        );
        assert!(lr.is_ok());
        if let Ok(lr) = lr {
            assert_eq!(lr.skipped_count, 2);
            assert!(lr.tensors.contains_key("content_in_proj.weight"));
            assert!(lr.tensors.contains_key("model.layers.0.weight"));
        }

        let codec = SafeTensorsLoader::load_from_path(
            ModelWeightKind::SemanticCodec,
            dir.join("raw/semantic_codec.safetensors"),
        );
        assert!(codec.is_ok());
        if let Ok(codec) = codec {
            assert!(codec.skipped_bytes > 100_000_000);
            assert!(
                codec
                    .tensors
                    .contains_key("quantizer.quantizers.0.out_project.weight")
            );
            assert!(!codec.tensors.keys().any(|k| k.starts_with("encoder.")));
        }
    }
}

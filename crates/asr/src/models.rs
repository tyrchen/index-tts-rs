//! The recognizer models itts can run, where their files live, and first-use downloads.
//!
//! Every model is a pinned revision of a sherpa-onnx export on Hugging Face Hub. Files are
//! looked up in the Hub cache (`HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`) and
//! must have their recorded sizes; a fresh download is also checked against its BLAKE3 digest.

use std::{
    env, fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::error::{AsrError, Result};

/// The language a recording is spoken in, as far as recognizer choice is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SpokenLanguage {
    /// Mandarin Chinese.
    Zh,
    /// Mandarin mixed with English words.
    ZhEn,
    /// Cantonese.
    Yue,
    /// Japanese.
    Ja,
    /// English.
    En,
}

/// A local speech recognizer model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum AsrModel {
    /// SenseVoice-Small (non-autoregressive CTC; zh, yue, ja, en, mixed zh/en).
    SenseVoice,
    /// Paraformer-large (non-autoregressive CIF; zh and mixed zh/en).
    Paraformer,
    /// Parakeet-TDT 0.6B v2 (transducer; English).
    Parakeet,
}

/// One file of a model snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFile {
    /// Path inside the repository.
    pub name: &'static str,
    /// Size in bytes.
    pub size: u64,
    /// BLAKE3 digest (hex) checked after a download.
    pub blake3: &'static str,
}

const SENSE_VOICE_FILES: &[ModelFile] = &[
    ModelFile {
        name: "model.int8.onnx",
        size: 239_233_841,
        blake3: "3171a61e94132c6270584b5a0f2f100289d5958c28795695a4d43ef23660ab8e",
    },
    ModelFile {
        name: "tokens.txt",
        size: 315_894,
        blake3: "dc65b6ac2b8bab079278769054812b6972114f2f515983491dfd7750e20987e1",
    },
];

const PARAFORMER_FILES: &[ModelFile] = &[
    ModelFile {
        name: "model.int8.onnx",
        size: 243_371_218,
        blake3: "fd0662a68b9b38c3c8771f74ccc8beb0a8fad117a62fb7eb0bb7972c2f2609a0",
    },
    ModelFile {
        name: "tokens.txt",
        size: 75_756,
        blake3: "9c19ad99b316b3a07d1cf08d4baea01a29fa70549b6845d94966d7222bc14d7d",
    },
];

const PARAKEET_FILES: &[ModelFile] = &[
    ModelFile {
        name: "encoder.int8.onnx",
        size: 652_184_296,
        blake3: "6c6819389f60b03c7ece6d7252482872d5a7cb3a56912c55f04da1fe48a42f98",
    },
    ModelFile {
        name: "decoder.int8.onnx",
        size: 7_257_753,
        blake3: "a6918fd115a447954740266f9574893a70d689f65ee46e0c36f41839f9000163",
    },
    ModelFile {
        name: "joiner.int8.onnx",
        size: 1_739_080,
        blake3: "3d82238adbbeac2d7d1e1f2cfc384914fb0815b36d2f6664b990c397e2545092",
    },
    ModelFile {
        name: "tokens.txt",
        size: 9_384,
        blake3: "2f9d32dbe3d0cd0e299a5c4764c56d841f5dc8625b6dfe65c3d4940938beac8d",
    },
];

impl AsrModel {
    /// Every model, in documentation order.
    pub const ALL: [Self; 3] = [Self::SenseVoice, Self::Paraformer, Self::Parakeet];

    /// Wire name (`sense-voice`, `paraformer`, `parakeet`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SenseVoice => "sense-voice",
            Self::Paraformer => "paraformer",
            Self::Parakeet => "parakeet",
        }
    }

    /// Hugging Face repository of the sherpa-onnx export.
    #[must_use]
    pub const fn repo(self) -> &'static str {
        match self {
            Self::SenseVoice => "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            Self::Paraformer => "csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14",
            Self::Parakeet => "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        }
    }

    /// Pinned repository revision.
    #[must_use]
    pub const fn revision(self) -> &'static str {
        match self {
            Self::SenseVoice => "2365baeacb507f821a0c8120fcee3d484dba7a07",
            Self::Paraformer => "def027084691107096b5ebba69785756d63de6c5",
            Self::Parakeet => "1ab9323565ddb038682214b292f588070a538ce2",
        }
    }

    /// Files the recognizer loads.
    #[must_use]
    pub const fn files(self) -> &'static [ModelFile] {
        match self {
            Self::SenseVoice => SENSE_VOICE_FILES,
            Self::Paraformer => PARAFORMER_FILES,
            Self::Parakeet => PARAKEET_FILES,
        }
    }

    /// Download size in bytes.
    ///
    /// ```
    /// use index_tts_rs_asr::AsrModel;
    /// assert!(AsrModel::SenseVoice.total_bytes() > 200 * 1024 * 1024);
    /// ```
    #[must_use]
    pub fn total_bytes(self) -> u64 {
        self.files().iter().map(|file| file.size).sum()
    }

    /// Whether this model recognizes `language`.
    #[must_use]
    pub const fn supports(self, language: SpokenLanguage) -> bool {
        match self {
            Self::SenseVoice => true,
            Self::Paraformer => matches!(language, SpokenLanguage::Zh | SpokenLanguage::ZhEn),
            Self::Parakeet => matches!(language, SpokenLanguage::En),
        }
    }

    /// The model used when a request does not name one: `Paraformer` for Mandarin (as accurate
    /// as `SenseVoice` on the evaluation, with fewer one-character mishearings), `SenseVoice` for
    /// Cantonese and Japanese, `Parakeet` for English.
    ///
    /// ```
    /// use index_tts_rs_asr::{AsrModel, SpokenLanguage};
    /// assert_eq!(AsrModel::default_for(SpokenLanguage::Zh), AsrModel::Paraformer);
    /// assert_eq!(AsrModel::default_for(SpokenLanguage::Yue), AsrModel::SenseVoice);
    /// assert_eq!(AsrModel::default_for(SpokenLanguage::En), AsrModel::Parakeet);
    /// ```
    #[must_use]
    pub const fn default_for(language: SpokenLanguage) -> Self {
        match language {
            SpokenLanguage::Zh | SpokenLanguage::ZhEn => Self::Paraformer,
            SpokenLanguage::En => Self::Parakeet,
            _ => Self::SenseVoice,
        }
    }

    /// `model` if given (it must support `language`), else [`Self::default_for`].
    ///
    /// # Errors
    ///
    /// Returns [`AsrError::InvalidArgument`] when the model does not recognize the language.
    pub fn choose(model: Option<Self>, language: SpokenLanguage) -> Result<Self> {
        match model {
            None => Ok(Self::default_for(language)),
            Some(model) if model.supports(language) => Ok(model),
            Some(model) => Err(AsrError::invalid(format!(
                "model `{model}` does not recognize {language:?} speech"
            ))),
        }
    }
}

impl fmt::Display for AsrModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AsrModel {
    type Err = AsrError;

    fn from_str(value: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|model| model.as_str() == value)
            .ok_or_else(|| {
                AsrError::invalid(format!(
                    "unknown speech recognition model `{value}`; expected one of sense-voice, \
                     paraformer, parakeet"
                ))
            })
    }
}

/// A resolved model snapshot whose files all have their recorded sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFiles {
    /// The model.
    pub model: AsrModel,
    /// Snapshot directory.
    pub dir: PathBuf,
}

impl ModelFiles {
    /// Path of `name` inside the snapshot, as UTF-8 (sherpa-onnx takes strings).
    ///
    /// # Errors
    ///
    /// Returns [`AsrError::ModelUnavailable`] for a path that is not valid UTF-8.
    pub fn path(&self, name: &str) -> Result<String> {
        let path = self.dir.join(name);
        path.to_str().map(str::to_owned).ok_or_else(|| {
            AsrError::model(
                self.model.as_str(),
                format!("path `{}` is not valid UTF-8", path.display()),
            )
        })
    }
}

/// Hub cache directories searched for snapshots, in priority order.
#[must_use]
pub fn hub_roots() -> Vec<PathBuf> {
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

/// Snapshot directory of `model` under one Hub cache root.
#[must_use]
pub fn snapshot_dir(root: &Path, model: AsrModel) -> PathBuf {
    root.join(format!("models--{}", model.repo().replace('/', "--")))
        .join("snapshots")
        .join(model.revision())
}

/// Whether every file of `model` exists in `dir` with its recorded size.
#[must_use]
pub fn is_complete(dir: &Path, model: AsrModel) -> bool {
    model.files().iter().all(|file| {
        dir.join(file.name)
            .metadata()
            .is_ok_and(|meta| meta.len() == file.size)
    })
}

/// Finds a complete cached snapshot of `model`.
///
/// # Errors
///
/// Returns [`AsrError::ModelUnavailable`] when no Hub cache root holds every file.
pub fn resolve(model: AsrModel) -> Result<ModelFiles> {
    hub_roots()
        .iter()
        .map(|root| snapshot_dir(root, model))
        .find(|dir| is_complete(dir, model))
        .map(|dir| ModelFiles { model, dir })
        .ok_or_else(|| AsrError::model(model.as_str(), "not downloaded"))
}

#[cfg(feature = "recognizer")]
pub use download::ensure;

#[cfg(feature = "recognizer")]
mod download {
    use std::{io, path::Path};

    use blake3::Hasher;
    use hf_hub::{HFClient, split_id};
    use tokio::{fs::File, io::AsyncReadExt};
    use tracing::info;

    use super::{AsrModel, ModelFile, ModelFiles, resolve};
    use crate::error::{AsrError, Result};

    /// Bytes hashed per read when checking a download.
    const DIGEST_CHUNK_BYTES: usize = 4 << 20;

    /// Resolves `model` from the cache, downloading its pinned revision first when missing.
    /// Returns the files and whether they were downloaded now.
    ///
    /// # Errors
    ///
    /// Returns [`AsrError::ModelUnavailable`] when the download fails or a downloaded file does
    /// not match its recorded size and digest.
    pub async fn ensure(model: AsrModel) -> Result<(ModelFiles, bool)> {
        if let Ok(files) = resolve(model) {
            return Ok((files, false));
        }
        let name = model.as_str();
        let client = HFClient::new()
            .map_err(|err| AsrError::model(name, format!("Hugging Face client: {err}")))?;
        let (owner, repo) = split_id(model.repo());
        let hub = client.model(owner, repo);
        info!(
            model = name,
            bytes = model.total_bytes(),
            "Downloading speech recognition model"
        );
        for file in model.files() {
            let path = hub
                .download_file()
                .filename(file.name)
                .revision(model.revision())
                .send()
                .await
                .map_err(|err| {
                    AsrError::model(
                        name,
                        format!("download of `{}` from `{}`: {err}", file.name, model.repo()),
                    )
                })?;
            verify(model, *file, &path).await?;
        }
        Ok((resolve(model)?, true))
    }

    async fn verify(model: AsrModel, file: ModelFile, path: &Path) -> Result<()> {
        let shown = path.display().to_string();
        let unreadable = |err: io::Error| {
            AsrError::model(model.as_str(), format!("cannot read `{shown}`: {err}"))
        };
        let mut reader = File::open(path).await.map_err(unreadable)?;
        let mut hasher = Hasher::new();
        let mut buf = vec![0u8; DIGEST_CHUNK_BYTES];
        loop {
            let read = reader.read(&mut buf).await.map_err(unreadable)?;
            let Some(chunk) = buf.get(..read).filter(|chunk| !chunk.is_empty()) else {
                break;
            };
            hasher.update(chunk);
        }
        let digest = hasher.finalize().to_hex();
        if digest.as_str() == file.blake3 {
            return Ok(());
        }
        Err(AsrError::model(
            model.as_str(),
            format!(
                "`{shown}` does not match its recorded digest; delete the model's directory in \
                 the Hugging Face cache and retry"
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, io};

    use rstest::rstest;
    use tempfile::tempdir;

    use super::{AsrModel, SpokenLanguage, is_complete, snapshot_dir};

    #[rstest]
    #[case(SpokenLanguage::Zh, AsrModel::Paraformer)]
    #[case(SpokenLanguage::ZhEn, AsrModel::Paraformer)]
    #[case(SpokenLanguage::Yue, AsrModel::SenseVoice)]
    #[case(SpokenLanguage::Ja, AsrModel::SenseVoice)]
    #[case(SpokenLanguage::En, AsrModel::Parakeet)]
    fn test_should_default_to_a_model_supporting_the_language(
        #[case] language: SpokenLanguage,
        #[case] model: AsrModel,
    ) {
        assert_eq!(AsrModel::default_for(language), model);
        assert!(model.supports(language));
    }

    #[test]
    fn test_should_reject_a_model_for_another_language() {
        assert!(AsrModel::choose(Some(AsrModel::Parakeet), SpokenLanguage::Zh).is_err());
        assert!(AsrModel::choose(Some(AsrModel::Paraformer), SpokenLanguage::Ja).is_err());
        assert!(
            AsrModel::choose(Some(AsrModel::Paraformer), SpokenLanguage::ZhEn)
                .is_ok_and(|m| m == AsrModel::Paraformer)
        );
    }

    #[test]
    fn test_should_parse_wire_names() {
        for model in AsrModel::ALL {
            assert_eq!(model.as_str().parse::<AsrModel>().ok(), Some(model));
        }
        assert!("whisper".parse::<AsrModel>().is_err());
    }

    #[test]
    #[allow(clippy::disallowed_methods)]
    fn test_should_treat_a_size_mismatch_as_missing() -> io::Result<()> {
        let root = tempdir()?;
        let dir = snapshot_dir(root.path(), AsrModel::Parakeet);
        fs::create_dir_all(&dir)?;
        for file in AsrModel::Parakeet.files() {
            fs::write(dir.join(file.name), b"short")?;
        }
        assert!(!is_complete(&dir, AsrModel::Parakeet));
        Ok(())
    }
}

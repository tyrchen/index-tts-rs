//! `itts models {pull,info}`.

use std::{
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, Result, anyhow};
use clap::Subcommand;
use hf_hub::{
    HFClient,
    progress::{DownloadEvent, ProgressEvent, ProgressHandler},
    split_id,
};
use index_tts_rs_core::{
    config::{DEFAULT_MODEL_REPO, EngineConfig},
    models::{ModelWeightKind, resolve_model_dir},
};
use index_tts_rs_service::ServiceError;
use serde::Serialize;
use serde_json::json;
use tokio::fs;

use crate::{daemon::lifecycle::display_path, output::OutputMode};

/// Every file the engine needs from the model repository.
const REQUIRED_REPO_FILES: &[&str] = &[
    "config.yaml",
    "glossary.yaml",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "spk_proj.npz",
    "gpt_int8.safetensors",
    "cfm_estimator.safetensors",
    "raw/semantic_model.safetensors",
    "raw/emo.safetensors",
    "raw/campplus.safetensors",
    "raw/length_regulator.safetensors",
    "raw/semantic_codec.safetensors",
    "raw/bigvgan.safetensors",
];
const MIB: f64 = 1024.0 * 1024.0;

/// Model subcommands.
#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    /// Download the model weights from Hugging Face Hub (no-op when already complete).
    Pull {
        /// Hugging Face repository ID.
        #[arg(long, default_value = DEFAULT_MODEL_REPO)]
        repo: String,
    },
    /// Show the resolved model snapshot and its files.
    #[command(visible_alias = "inspect")]
    Info,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactInfo {
    relative_path: &'static str,
    size_bytes: u64,
}

/// Runs a models subcommand.
///
/// # Errors
///
/// Returns the classified failure.
pub async fn run(config: &EngineConfig, command: ModelsCommand, output: OutputMode) -> Result<()> {
    match command {
        ModelsCommand::Pull { repo } => {
            let (model_dir, downloaded) =
                match resolve_model_dir(config.model_dir.as_deref(), &repo) {
                    Ok(dir) => (dir, false),
                    Err(_) => (download(&repo, output).await?, true),
                };
            let total_bytes = artifacts(&model_dir)
                .await
                .iter()
                .map(|a| a.size_bytes)
                .sum::<u64>();
            output.emit(
                &json!({ "status": "ok", "repo": repo, "modelDir": model_dir,
                    "downloaded": downloaded, "totalWeightBytes": total_bytes }),
                |out| {
                    let verb = if downloaded {
                        "downloaded"
                    } else {
                        "already present"
                    };
                    writeln!(
                        out,
                        "✔ model {verb}: {repo} ({:.2} GiB)",
                        to_mib(total_bytes) / 1024.0
                    )?;
                    writeln!(out, "  → {}", display_path(&model_dir))
                },
            )
        }
        ModelsCommand::Info => {
            let model_dir = resolve_model_dir(config.model_dir.as_deref(), &config.model_repo)?;
            let artifacts = artifacts(&model_dir).await;
            let total_bytes = artifacts.iter().map(|a| a.size_bytes).sum::<u64>();
            output.emit(
                &json!({ "modelRepo": config.model_repo, "modelDir": model_dir,
                    "totalWeightBytes": total_bytes, "artifacts": artifacts }),
                |out| {
                    writeln!(out, "repo   {}", config.model_repo)?;
                    writeln!(out, "dir    {}", display_path(&model_dir))?;
                    for a in &artifacts {
                        writeln!(
                            out,
                            "  {:<36} {:>9.2} MiB",
                            a.relative_path,
                            to_mib(a.size_bytes)
                        )?;
                    }
                    writeln!(out, "total  {:.2} GiB", to_mib(total_bytes) / 1024.0)
                },
            )
        }
    }
}

async fn artifacts(model_dir: &Path) -> Vec<ArtifactInfo> {
    let mut out = Vec::with_capacity(ModelWeightKind::ALL.len());
    for kind in ModelWeightKind::ALL {
        let relative_path = kind.relative_path();
        let size_bytes = fs::metadata(model_dir.join(relative_path))
            .await
            .map_or(0, |m| m.len());
        out.push(ArtifactInfo {
            relative_path,
            size_bytes,
        });
    }
    out
}

async fn download(repo: &str, output: OutputMode) -> Result<PathBuf> {
    let client = HFClient::new().context("failed to initialize Hugging Face Hub client")?;
    let (owner, name) = split_id(repo);
    let hf_repo = client.model(owner, name);
    let total = REQUIRED_REPO_FILES.len();
    let mut snapshot_root = None;
    for (index, &rel) in REQUIRED_REPO_FILES.iter().enumerate() {
        let reporter = FileReporter {
            output,
            label: format!("[{}/{total}] {rel}", index + 1),
            last_bucket: AtomicU64::new(u64::MAX),
        };
        let downloaded = hf_repo
            .download_file()
            .filename(rel)
            .progress(reporter)
            .send()
            .await
            .with_context(|| format!("failed to download `{rel}` from `{repo}`"))?;
        if rel == "config.yaml" {
            snapshot_root = downloaded.parent().map(Path::to_path_buf);
        }
    }
    let root =
        snapshot_root.ok_or_else(|| anyhow!("failed to resolve downloaded snapshot root"))?;
    resolve_model_dir(Some(&root), repo).map_err(|err| {
        anyhow!(
            ServiceError::from(err)
                .with_hint("the download is incomplete; run `itts models pull` again")
        )
    })
}

/// Reports one file's download progress in 10% steps (JSON lines in `jsonl` mode).
struct FileReporter {
    output: OutputMode,
    label: String,
    last_bucket: AtomicU64,
}

impl ProgressHandler for FileReporter {
    fn on_progress(&self, event: &ProgressEvent) {
        let (done, total) = match event {
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                files.iter().fold((0, 0), |(d, t), f| {
                    (d + f.bytes_completed, t + f.total_bytes)
                })
            }
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            }) => (*bytes_completed, *total_bytes),
            ProgressEvent::Download(DownloadEvent::Complete) => (1, 1),
            _ => return,
        };
        let bucket = done
            .saturating_mul(10)
            .checked_div(total)
            .unwrap_or(10)
            .min(10);
        if self.last_bucket.swap(bucket, Ordering::Relaxed) == bucket {
            return;
        }
        let percent = bucket * 10;
        match self.output {
            OutputMode::Jsonl => {
                let _ = OutputMode::emit_line(&json!({
                    "event": "progress", "file": self.label, "percent": percent,
                    "bytesCompleted": done, "totalBytes": total,
                }));
            }
            OutputMode::Text => {
                let mut err = io::stderr().lock();
                let (lead, tail) = if err.is_terminal() {
                    ("\r", "")
                } else {
                    ("", "\n")
                };
                let _ = write!(
                    err,
                    "{lead}  ▸ {:<52} {percent:>3}% of {:>9.1} MiB{tail}",
                    self.label,
                    to_mib(total)
                );
                if percent >= 100 && err.is_terminal() {
                    let _ = writeln!(err);
                }
            }
            OutputMode::Json => {}
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "sizes are displayed approximately"
)]
fn to_mib(bytes: u64) -> f64 {
    bytes as f64 / MIB
}

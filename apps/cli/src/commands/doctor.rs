//! `itts doctor`: one-shot environment report with actionable fixes.

use std::{env, path::Path};

use anyhow::Result;
use index_tts_rs_core::{
    config::EngineConfig,
    models::{HIFI_MODEL_REPO, ModelWeightKind, resolve_hifi_weights, resolve_model_dir},
    voice::{VoiceStore, default_voices_dir},
};
use index_tts_rs_service::ServiceError;
use schemars::JsonSchema;
use serde::Serialize;
use tokio::{fs, task::spawn_blocking};

use crate::{
    daemon::{
        client::DaemonClient, default_socket_path, lifecycle::display_path, protocol::DaemonState,
    },
    output::OutputMode,
    player::default_output_device_name,
};

/// Health of one check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CheckStatus {
    /// Working.
    Ok,
    /// Optional capability missing.
    Warn,
    /// Synthesis cannot work until fixed.
    Error,
}

/// One diagnostic.
#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Check {
    /// Area (`platform`, `model`, `hifi`, `voices`, `daemon`, `director`, `audio`).
    pub name: &'static str,
    /// Health.
    pub status: CheckStatus,
    /// What was found.
    pub summary: String,
    /// How to fix it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

/// Full environment report.
#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DoctorReport {
    /// `itts` version.
    pub version: &'static str,
    /// Whether synthesis can run (no `error` checks).
    pub ready: bool,
    /// Individual checks.
    pub checks: Vec<Check>,
}

impl Check {
    fn new(name: &'static str, status: CheckStatus, summary: impl Into<String>) -> Self {
        Self {
            name,
            status,
            summary: summary.into(),
            hint: None,
        }
    }

    fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

/// Collects the report.
pub async fn collect(config: &EngineConfig, socket: &Path) -> DoctorReport {
    let mut checks = vec![
        platform_check(),
        model_check(config).await,
        hifi_check().await,
    ];
    checks.extend(voice_checks(config));
    checks.push(daemon_check(socket).await);
    checks.push(director_check());
    checks.push(audio_check());
    DoctorReport {
        version: env!("CARGO_PKG_VERSION"),
        ready: checks.iter().all(|c| c.status != CheckStatus::Error),
        checks,
    }
}

/// Runs `itts doctor`.
///
/// # Errors
///
/// Returns an output failure.
pub async fn run(config: &EngineConfig, output: OutputMode) -> Result<()> {
    let report = collect(config, &default_socket_path()).await;
    output.emit(&report, |out| {
        for check in &report.checks {
            let mark = match check.status {
                CheckStatus::Ok => "✔",
                CheckStatus::Warn => "⚠",
                CheckStatus::Error => "✖",
            };
            writeln!(out, "{mark} {:<9} {}", check.name, check.summary)?;
            if let Some(hint) = &check.hint {
                writeln!(out, "  {:<9} hint: {hint}", "")?;
            }
        }
        Ok(())
    })
}

fn platform_check() -> Check {
    let (os, arch) = (env::consts::OS, env::consts::ARCH);
    if os == "macos" && arch == "aarch64" {
        Check::new(
            "platform",
            CheckStatus::Ok,
            "macOS on Apple Silicon (Metal)",
        )
    } else {
        Check::new(
            "platform",
            CheckStatus::Error,
            format!("{os}/{arch} is not supported"),
        )
        .hint("the MLX Metal backend requires macOS on Apple Silicon")
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "sizes are displayed approximately"
)]
async fn model_check(config: &EngineConfig) -> Check {
    match resolve_model_dir(config.model_dir.as_deref(), &config.model_repo) {
        Ok(dir) => {
            let mut bytes = 0_u64;
            for kind in ModelWeightKind::ALL {
                let size = fs::metadata(dir.join(kind.relative_path()))
                    .await
                    .map_or(0, |m| m.len());
                bytes = bytes.saturating_add(size);
            }
            Check::new(
                "model",
                CheckStatus::Ok,
                format!(
                    "{} ({:.2} GiB) at {}",
                    config.model_repo,
                    bytes as f64 / f64::from(1_u32 << 30),
                    display_path(&dir)
                ),
            )
        }
        Err(err) => {
            let err = ServiceError::from(err);
            Check::new("model", CheckStatus::Error, err.message).hint(
                err.hint
                    .unwrap_or_else(|| "run `itts models pull`".to_owned()),
            )
        }
    }
}

async fn hifi_check() -> Check {
    match spawn_blocking(resolve_hifi_weights).await {
        Ok(Ok(path)) => Check::new(
            "hifi",
            CheckStatus::Ok,
            format!(
                "48 kHz bandwidth extender ({HIFI_MODEL_REPO}) at {}",
                display_path(&path)
            ),
        ),
        Ok(Err(err)) => Check::new(
            "hifi",
            CheckStatus::Warn,
            format!(
                "48 kHz output unavailable: {}",
                ServiceError::from(err).message
            ),
        )
        .hint("run `itts models pull` to fetch the 56 MB extender weights"),
        Err(err) => Check::new("hifi", CheckStatus::Warn, format!("check failed: {err}")),
    }
}

fn voice_checks(config: &EngineConfig) -> Vec<Check> {
    let dir = config.voices_dir.clone().unwrap_or_else(default_voices_dir);
    let listed = VoiceStore::new(dir.clone())
        .map_err(ServiceError::from)
        .and_then(|store| store.list().map_err(ServiceError::from));
    let entries = match listed {
        Ok(entries) => entries,
        Err(err) => {
            return vec![Check::new("voices", CheckStatus::Error, err.message)];
        }
    };
    let names: Vec<&str> = entries.iter().map(|e| e.voice_id.as_str()).collect();
    let mut checks = vec![if names.is_empty() {
        Check::new(
            "voices",
            CheckStatus::Warn,
            format!("none in {}", display_path(&dir)),
        )
        .hint("register one with `itts voice add <ID> --ref <AUDIO>`, or pass --ref to synth")
    } else {
        Check::new(
            "voices",
            CheckStatus::Ok,
            format!(
                "{} in {} — {}",
                names.len(),
                display_path(&dir),
                names.join(", ")
            ),
        )
    }];
    if let Some(default) = &config.default_voice
        && !names.contains(&default.as_str())
    {
        checks.push(
            Check::new(
                "voices",
                CheckStatus::Warn,
                format!("default_voice `{}` is not registered", default.as_str()),
            )
            .hint("register it or change `default_voice` in the config"),
        );
    }
    checks
}

async fn daemon_check(socket: &Path) -> Check {
    match DaemonClient::connect(socket).await {
        Ok(Some(mut client)) => match client.status().await {
            Ok(status) => {
                let state = match status.state {
                    DaemonState::Ready => "ready",
                    DaemonState::Loading => "loading models",
                };
                Check::new(
                    "daemon",
                    CheckStatus::Ok,
                    format!("{state} · pid {} · v{}", status.pid, status.version),
                )
            }
            Err(err) => Check::new("daemon", CheckStatus::Error, err.message)
                .hint(err.hint.unwrap_or_default()),
        },
        Ok(None) => Check::new(
            "daemon",
            CheckStatus::Warn,
            "not running (each synth loads models)",
        )
        .hint("run `itts daemon start` to keep models warm"),
        Err(err) => {
            Check::new("daemon", CheckStatus::Error, err.message).hint(err.hint.unwrap_or_default())
        }
    }
}

fn director_check() -> Check {
    if env::var("DEEPSEEK_API_KEY").is_ok_and(|key| !key.trim().is_empty()) {
        Check::new(
            "director",
            CheckStatus::Ok,
            "DEEPSEEK_API_KEY set — `itts annotate` available",
        )
    } else {
        Check::new(
            "director",
            CheckStatus::Warn,
            "DEEPSEEK_API_KEY not set — `itts annotate` unavailable",
        )
        .hint("export DEEPSEEK_API_KEY=<key> to enable AI pacing annotation")
    }
}

fn audio_check() -> Check {
    match default_output_device_name() {
        Some(name) => Check::new(
            "audio",
            CheckStatus::Ok,
            format!("output device \"{name}\""),
        ),
        None => Check::new(
            "audio",
            CheckStatus::Warn,
            "no default output device — --play unavailable",
        )
        .hint("write files with --out <file.wav> instead"),
    }
}

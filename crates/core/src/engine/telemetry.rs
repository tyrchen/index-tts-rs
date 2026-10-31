//! Engine startup progress events and lock-free runtime telemetry.
//!
//! The MLX worker thread owns every model, so callers cannot inspect it directly while a
//! synthesis is running. Instead the worker publishes a small set of atomics after each command,
//! and [`IndexTtsEngine::telemetry`](crate::engine::IndexTtsEngine::telemetry) reads them without
//! queueing behind in-flight GPU work.

use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use mlx_rs::memory::{active_memory, peak_memory};
use serde::{Deserialize, Serialize};

/// Progress reported while [`IndexTtsEngine`](crate::engine::IndexTtsEngine) loads its models.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "camelCase")]
#[non_exhaustive]
pub enum StartupEvent {
    /// The local model snapshot directory was resolved and verified.
    #[serde(rename_all = "camelCase")]
    ModelResolved {
        /// Resolved snapshot directory.
        model_dir: PathBuf,
    },
    /// One synthesis component finished loading onto the Metal GPU.
    #[serde(rename_all = "camelCase")]
    ComponentLoaded {
        /// Stable component name (`frontend`, `gpt`, `semantic_codec`, ...).
        component: &'static str,
        /// Load time for this component in milliseconds.
        elapsed_ms: f64,
    },
    /// All Stage 1..3 synthesis components are resident and the actor accepts commands.
    #[serde(rename_all = "camelCase")]
    EngineReady {
        /// Total model initialization time in milliseconds.
        elapsed_ms: f64,
    },
}

/// Callback receiving [`StartupEvent`]s from the worker thread during engine startup.
pub type StartupObserver = Box<dyn Fn(&StartupEvent) + Send>;

/// Point-in-time engine runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct TelemetrySnapshot {
    /// Whether the Stage 0 voice extraction encoders are currently resident.
    pub voice_extractor_loaded: bool,
    /// MLX allocator bytes in active use after the last completed command.
    pub active_memory_bytes: u64,
    /// MLX allocator peak bytes since engine start.
    pub peak_memory_bytes: u64,
}

/// Worker-published runtime counters shared with every engine handle.
#[derive(Debug, Default)]
pub(crate) struct EngineTelemetry {
    voice_extractor_loaded: AtomicBool,
    active_memory_bytes: AtomicU64,
    peak_memory_bytes: AtomicU64,
}

impl EngineTelemetry {
    pub(crate) fn set_voice_extractor_loaded(&self, loaded: bool) {
        self.voice_extractor_loaded.store(loaded, Ordering::Relaxed);
    }

    /// Samples the MLX allocator. Must be called from the worker thread between commands.
    pub(crate) fn publish_memory(&self) {
        if let Ok(active) = active_memory() {
            self.active_memory_bytes
                .store(u64::try_from(active).unwrap_or(u64::MAX), Ordering::Relaxed);
        }
        if let Ok(peak) = peak_memory() {
            self.peak_memory_bytes
                .store(u64::try_from(peak).unwrap_or(u64::MAX), Ordering::Relaxed);
        }
    }

    pub(crate) fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            voice_extractor_loaded: self.voice_extractor_loaded.load(Ordering::Relaxed),
            active_memory_bytes: self.active_memory_bytes.load(Ordering::Relaxed),
            peak_memory_bytes: self.peak_memory_bytes.load(Ordering::Relaxed),
        }
    }
}

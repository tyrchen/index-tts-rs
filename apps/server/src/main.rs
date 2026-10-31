//! HTTP server binary entrypoint for `index-tts-rs`.

#![forbid(unsafe_code)]
#![warn(rust_2024_compatibility, missing_docs, missing_debug_implementations)]

use anyhow::Result;
use index_tts_rs_core::EngineConfig;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cfg = EngineConfig::default();
    info!(
        model_repo = %cfg.model_repo,
        sample_rate = cfg.sample_rate,
        "index-tts-rs-server initialized"
    );
    Ok(())
}

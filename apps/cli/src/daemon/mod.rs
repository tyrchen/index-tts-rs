//! Warm-model daemon: one resident `IndexTtsEngine` per machine shared by the CLI and MCP.
//!
//! - [`protocol`]: versioned JSON Lines wire format.
//! - [`server`]: the daemon process (`itts daemon run`).
//! - [`client`]: connection with handshake, used by `synth`, `voice add`, `status`, and MCP.
//! - [`lifecycle`]: detached start with readiness reporting, stop, restart, ensure, logs.

pub mod client;
pub mod lifecycle;
pub mod protocol;
pub mod server;
mod transport;

use std::{
    env,
    path::{Path, PathBuf},
};

/// Default socket path relative to `$HOME`.
const DEFAULT_SOCKET_SUBPATH: &str = ".cache/index-tts-rs/daemon.sock";
/// Default idle auto-shutdown timeout (10 minutes).
pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 600;
/// Default maximum concurrent jobs before `BUSY`.
pub const DEFAULT_MAX_IN_FLIGHT: usize = 16;

/// Files owned by one daemon instance, all derived from its socket path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonPaths {
    /// Unix domain socket.
    pub socket: PathBuf,
    /// Exclusive lock held for the daemon's lifetime (single-instance guard).
    pub lock: PathBuf,
    /// Process id of the running daemon.
    pub pid: PathBuf,
    /// Daemon log (stderr of the detached process).
    pub log: PathBuf,
}

impl DaemonPaths {
    /// Derives sibling paths (`.lock`, `.pid`, `.log`) from `socket`.
    pub fn new(socket: &Path) -> Self {
        Self {
            socket: socket.to_path_buf(),
            lock: socket.with_extension("lock"),
            pid: socket.with_extension("pid"),
            log: socket.with_extension("log"),
        }
    }

    /// Paths for the default socket (`~/.cache/index-tts-rs/daemon.sock`).
    pub fn default_paths() -> Self {
        Self::new(&default_socket_path())
    }
}

/// Resolves the default daemon socket path.
pub fn default_socket_path() -> PathBuf {
    env::var("HOME")
        .ok()
        .filter(|home| !home.trim().is_empty())
        .map_or_else(
            || PathBuf::from(DEFAULT_SOCKET_SUBPATH),
            |home| PathBuf::from(home).join(DEFAULT_SOCKET_SUBPATH),
        )
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::DaemonPaths;

    #[test]
    fn test_should_derive_sibling_paths_from_socket() {
        let paths = DaemonPaths::new(Path::new("/tmp/x/daemon.sock"));
        assert_eq!(paths.lock, Path::new("/tmp/x/daemon.lock"));
        assert_eq!(paths.pid, Path::new("/tmp/x/daemon.pid"));
        assert_eq!(paths.log, Path::new("/tmp/x/daemon.log"));
    }
}

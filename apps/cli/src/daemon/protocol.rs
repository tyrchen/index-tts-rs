//! Daemon wire protocol (version [`PROTOCOL_VERSION`]): newline-delimited JSON over a `0600`
//! Unix domain socket.
//!
//! Every line is an envelope `{"id": N, "method": ...}` (client → daemon) or
//! `{"id": N, "event": ...}` (daemon → client). Clients send `hello` first; requests on one
//! connection are multiplexed by `id` and can be cancelled with `cancel`. Disconnecting
//! cancels every in-flight request of that connection.

use std::path::PathBuf;

use index_tts_rs_core::{engine::TelemetrySnapshot, voice::VoiceEntry};
use index_tts_rs_service::{
    AnnotateJob, AnnotateResult, ApiKey, ChunkInfo, JobEvent, JobStarted, JobSummary, ServiceError,
    SpeechJob,
};
use serde::{Deserialize, Serialize};

/// Protocol version; bumped on any incompatible wire change.
pub const PROTOCOL_VERSION: u32 = 3;
/// Maximum accepted request line length in bytes.
pub const MAX_LINE_BYTES: usize = 256 * 1024;

/// A client request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "camelCase")]
pub enum Request {
    /// Version handshake.
    #[serde(rename_all = "camelCase")]
    Hello {
        /// Client binary version.
        client_version: String,
        /// Client protocol version.
        protocol_version: u32,
    },
    /// Runtime status; does not reset the idle timer.
    Status,
    /// Graceful shutdown.
    Stop,
    /// Run a speech job.
    #[serde(rename_all = "camelCase")]
    Synth {
        /// Job with absolute paths.
        job: Box<SpeechJob>,
        /// Embed base64 PCM in `chunk` events.
        #[serde(default)]
        include_pcm: bool,
    },
    /// Cancel an in-flight request on this connection.
    Cancel {
        /// Request `id` to cancel.
        target: u64,
    },
    /// Choose a delivery per unit; `apiKey` overrides the daemon's `DEEPSEEK_API_KEY`.
    #[serde(rename_all = "camelCase")]
    Annotate {
        /// Units to direct.
        job: Box<AnnotateJob>,
        /// Provider key for this request only.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        api_key: Option<ApiKey>,
    },
    /// Extract and register a voice using the resident models.
    #[serde(rename_all = "camelCase")]
    VoiceAdd {
        /// Voice identifier.
        voice_id: String,
        /// Absolute reference audio path.
        ref_audio: PathBuf,
        /// Absolute emotion reference audio path.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emo_ref: Option<PathBuf>,
        /// Emotion blending weight.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emo_alpha: Option<f32>,
        /// Emotion vector to bake into the snapshot.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        emo: Option<String>,
    },
}

/// Client → daemon line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestEnvelope {
    /// Correlation id chosen by the client.
    pub id: u64,
    /// The request.
    #[serde(flatten)]
    pub request: Request,
}

/// Daemon lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DaemonState {
    /// Models are loading; jobs wait until ready.
    Loading,
    /// Models are resident.
    Ready,
}

/// Registered voice summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceSummary {
    /// Voice identifier.
    pub id: String,
    /// Reference prompt duration in seconds.
    pub duration_secs: f32,
}

impl From<VoiceEntry> for VoiceSummary {
    fn from(entry: VoiceEntry) -> Self {
        Self {
            id: entry.voice_id,
            duration_secs: entry.duration_secs,
        }
    }
}

/// Job counters since daemon start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobCounters {
    /// Jobs currently running or queued behind the GPU actor.
    pub in_flight: usize,
    /// Maximum concurrent jobs before `BUSY`.
    pub max_in_flight: usize,
    /// Jobs completed successfully.
    pub completed: u64,
    /// Jobs that failed.
    pub failed: u64,
    /// Jobs cancelled by clients.
    pub cancelled: u64,
}

/// The most recent job failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LastError {
    /// Failure details.
    pub error: ServiceError,
    /// Unix timestamp in seconds.
    pub at_unix_secs: u64,
}

/// `itts daemon status` payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DaemonStatus {
    /// Lifecycle state.
    pub state: DaemonState,
    /// Daemon binary version.
    pub version: String,
    /// Protocol version.
    pub protocol_version: u32,
    /// Process id.
    pub pid: u32,
    /// Socket path.
    pub socket_path: PathBuf,
    /// Log file path.
    pub log_path: PathBuf,
    /// Resolved model snapshot directory (once ready).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_dir: Option<PathBuf>,
    /// Voices directory.
    pub voices_dir: PathBuf,
    /// Seconds since start.
    pub uptime_secs: u64,
    /// Seconds since the last job.
    pub idle_secs: u64,
    /// Idle auto-shutdown timeout (`0` disables it).
    pub idle_timeout_secs: u64,
    /// Registered voices.
    pub voices: Vec<VoiceSummary>,
    /// Engine runtime telemetry (once ready).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<TelemetrySnapshot>,
    /// Job counters.
    pub jobs: JobCounters,
    /// Most recent failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<LastError>,
}

/// A daemon response or job event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "camelCase")]
pub enum Response {
    /// Handshake reply.
    #[serde(rename_all = "camelCase")]
    Welcome {
        /// Daemon binary version.
        server_version: String,
        /// Daemon protocol version.
        protocol_version: u32,
    },
    /// Status reply.
    Status(Box<DaemonStatus>),
    /// Shutdown acknowledged.
    Stopping,
    /// Cancel acknowledged.
    Cancelled {
        /// Whether the target was still running.
        found: bool,
    },
    /// Voice registered.
    Voice {
        /// Registered voice.
        voice: Box<VoiceEntry>,
    },
    /// Deliveries chosen for an `annotate` request.
    Annotated(Box<AnnotateResult>),
    /// Job planned.
    Started(JobStarted),
    /// Job chunk.
    Chunk(Box<ChunkInfo>),
    /// Job finished.
    Done(Box<JobSummary>),
    /// Request failed.
    Error {
        /// Failure details.
        error: ServiceError,
    },
}

impl From<JobEvent> for Response {
    fn from(event: JobEvent) -> Self {
        match event {
            JobEvent::Started(started) => Self::Started(started),
            JobEvent::Chunk(chunk) => Self::Chunk(Box::new(chunk)),
            JobEvent::Done(summary) => Self::Done(Box::new(summary)),
            JobEvent::Error { error } => Self::Error { error },
            _ => Self::Error {
                error: ServiceError::internal("unsupported job event"),
            },
        }
    }
}

/// Daemon → client line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseEnvelope {
    /// Correlation id of the originating request (`0` for unparseable lines).
    pub id: u64,
    /// The response.
    #[serde(flatten)]
    pub response: Response,
}

/// Facts reported once the daemon is ready.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadyInfo {
    /// Process id.
    pub pid: u32,
    /// Daemon version.
    pub version: String,
    /// Protocol version.
    pub protocol_version: u32,
    /// Socket path.
    pub socket_path: PathBuf,
    /// Log file path.
    pub log_path: PathBuf,
    /// Model snapshot directory.
    pub model_dir: PathBuf,
    /// Voices directory.
    pub voices_dir: PathBuf,
    /// Registered voice ids.
    pub voices: Vec<String>,
    /// Model load time in milliseconds.
    pub load_ms: f64,
    /// Idle auto-shutdown timeout in seconds.
    pub idle_timeout_secs: u64,
}

/// Startup progress written by `itts daemon run` (one JSON object per line).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "camelCase")]
pub enum StartupLine {
    /// A startup step finished.
    #[serde(rename_all = "camelCase")]
    Progress {
        /// Step name.
        step: String,
        /// Extra detail (path, voice list).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        /// Step duration in milliseconds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        elapsed_ms: Option<f64>,
    },
    /// The daemon accepts jobs.
    Ready(Box<ReadyInfo>),
    /// Another daemon already owns the socket.
    #[serde(rename_all = "camelCase")]
    AlreadyRunning {
        /// Its process id, when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pid: Option<u32>,
        /// Socket path.
        socket_path: PathBuf,
    },
    /// Startup failed.
    Failed {
        /// Failure details.
        error: ServiceError,
    },
}

#[cfg(test)]
mod tests {
    use serde_json::{from_str, json, to_value};

    use super::{Request, RequestEnvelope, Response, ResponseEnvelope};

    #[test]
    fn test_should_flatten_method_into_request_envelope() -> serde_json::Result<()> {
        let envelope: RequestEnvelope = from_str(r#"{"id":7,"method":"cancel","target":3}"#)?;
        assert_eq!(envelope.id, 7);
        assert!(matches!(envelope.request, Request::Cancel { target: 3 }));
        let synth: RequestEnvelope =
            from_str(r#"{"id":1,"method":"synth","job":{"input":"你好","voice":"a"}}"#)?;
        assert!(matches!(
            synth.request,
            Request::Synth {
                include_pcm: false,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn test_should_accept_annotate_requests_without_leaking_the_key() -> serde_json::Result<()> {
        let envelope: RequestEnvelope = from_str(
            r#"{"id":4,"method":"annotate","job":{"units":[{"id":"a","text":"你好。"}]},"apiKey":"sk-secret"}"#,
        )?;
        assert!(matches!(
            envelope.request,
            Request::Annotate {
                api_key: Some(_),
                ..
            }
        ));
        assert!(!format!("{envelope:?}").contains("sk-secret"));
        Ok(())
    }

    #[test]
    fn test_should_flatten_event_into_response_envelope() -> serde_json::Result<()> {
        let envelope = ResponseEnvelope {
            id: 2,
            response: Response::Cancelled { found: true },
        };
        assert_eq!(
            to_value(&envelope)?,
            json!({"id":2,"event":"cancelled","found":true})
        );
        Ok(())
    }
}

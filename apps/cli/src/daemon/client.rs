//! Daemon client with protocol handshake.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use index_tts_rs_core::voice::VoiceEntry;
use index_tts_rs_service::{ErrorCode, JobEvent, JobSummary, ServiceError, SpeechJob};
use serde_json::{Value, from_value};
use tokio::{
    io::BufReader,
    net::{
        UnixStream,
        unix::{OwnedReadHalf, OwnedWriteHalf},
    },
    time::timeout,
};

use crate::daemon::{
    protocol::{
        DaemonStatus, PROTOCOL_VERSION, Request, RequestEnvelope, Response, ResponseEnvelope,
    },
    server::version_mismatch,
    transport::{read_json, write_json},
};

/// Connect timeout; a live daemon accepts within microseconds.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(400);
/// Handshake and control-request timeout.
const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);

/// Parameters for registering a voice through the daemon.
#[derive(Debug, Clone)]
pub struct VoiceAddRequest {
    /// Voice identifier.
    pub voice_id: String,
    /// Absolute reference audio path.
    pub ref_audio: PathBuf,
    /// Absolute emotion reference path.
    pub emo_ref: Option<PathBuf>,
    /// Emotion blending weight.
    pub emo_alpha: Option<f32>,
    /// Emotion vector spec.
    pub emo: Option<String>,
}

/// An open, version-checked daemon connection.
#[derive(Debug)]
pub struct DaemonClient {
    reader: BufReader<OwnedReadHalf>,
    writer: OwnedWriteHalf,
    buf: Vec<u8>,
    next_id: u64,
}

impl DaemonClient {
    /// Connects and performs the `hello` handshake.
    ///
    /// Returns `Ok(None)` when no daemon is listening.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::DaemonVersionMismatch`] if the daemon speaks another protocol
    /// version (including pre-handshake daemons), or [`ErrorCode::DaemonUnavailable`] if it
    /// accepts but does not answer.
    pub async fn connect(socket: &Path) -> Result<Option<Self>, ServiceError> {
        if !socket.exists() {
            return Ok(None);
        }
        let Ok(Ok(stream)) = timeout(CONNECT_TIMEOUT, UnixStream::connect(socket)).await else {
            return Ok(None);
        };
        let (read_half, writer) = stream.into_split();
        let mut client = Self {
            reader: BufReader::new(read_half),
            writer,
            buf: Vec::new(),
            next_id: 1,
        };
        let id = client
            .send(Request::Hello {
                client_version: env!("CARGO_PKG_VERSION").to_owned(),
                protocol_version: PROTOCOL_VERSION,
            })
            .await?;
        match client.handshake_reply(id).await? {
            Response::Welcome {
                protocol_version, ..
            } if protocol_version == PROTOCOL_VERSION => Ok(Some(client)),
            Response::Welcome {
                protocol_version, ..
            } => Err(version_mismatch(protocol_version, PROTOCOL_VERSION)),
            Response::Error { error } if error.code == ErrorCode::DaemonVersionMismatch => {
                Err(error)
            }
            _ => Err(version_mismatch(1, PROTOCOL_VERSION)),
        }
    }

    /// Fetches runtime status.
    ///
    /// # Errors
    ///
    /// Returns a transport or protocol failure.
    pub async fn status(&mut self) -> Result<DaemonStatus, ServiceError> {
        let id = self.send(Request::Status).await?;
        match self.recv_control(id).await? {
            Response::Status(status) => Ok(*status),
            other => Err(unexpected(&other)),
        }
    }

    /// Requests graceful shutdown.
    ///
    /// # Errors
    ///
    /// Returns a transport or protocol failure.
    pub async fn stop(&mut self) -> Result<(), ServiceError> {
        let id = self.send(Request::Stop).await?;
        match self.recv_control(id).await? {
            Response::Stopping => Ok(()),
            other => Err(unexpected(&other)),
        }
    }

    /// Runs a job, forwarding every event to `on_event`. Dropping the returned future (or the
    /// client) disconnects, which cancels the job in the daemon.
    ///
    /// # Errors
    ///
    /// Returns the job's failure, or the first error returned by `on_event`.
    pub async fn synth(
        &mut self,
        job: &SpeechJob,
        include_pcm: bool,
        mut on_event: impl FnMut(JobEvent) -> Result<(), ServiceError>,
    ) -> Result<JobSummary, ServiceError> {
        let id = self
            .send(Request::Synth {
                job: Box::new(job.clone()),
                include_pcm,
            })
            .await?;
        loop {
            match self.recv(id).await? {
                Response::Started(started) => on_event(JobEvent::Started(started))?,
                Response::Chunk(chunk) => on_event(JobEvent::Chunk(*chunk))?,
                Response::Done(summary) => {
                    on_event(JobEvent::Done((*summary).clone()))?;
                    return Ok(*summary);
                }
                Response::Error { error } => return Err(error),
                other => return Err(unexpected(&other)),
            }
        }
    }

    /// Registers a voice using the daemon's resident models.
    ///
    /// # Errors
    ///
    /// Returns the extraction failure.
    pub async fn voice_add(
        &mut self,
        request: VoiceAddRequest,
    ) -> Result<VoiceEntry, ServiceError> {
        let id = self
            .send(Request::VoiceAdd {
                voice_id: request.voice_id,
                ref_audio: request.ref_audio,
                emo_ref: request.emo_ref,
                emo_alpha: request.emo_alpha,
                emo: request.emo,
            })
            .await?;
        match self.recv(id).await? {
            Response::Voice { voice } => Ok(*voice),
            Response::Error { error } => Err(error),
            other => Err(unexpected(&other)),
        }
    }

    async fn send(&mut self, request: Request) -> Result<u64, ServiceError> {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        write_json(&mut self.writer, &RequestEnvelope { id, request })
            .await
            .map_err(|err| unavailable(&err.to_string()))?;
        Ok(id)
    }

    /// Reads the `hello` reply. Daemons predating the handshake answer with a differently
    /// shaped error line, which is reported as a protocol mismatch.
    async fn handshake_reply(&mut self, id: u64) -> Result<Response, ServiceError> {
        let line: Option<Value> =
            timeout(CONTROL_TIMEOUT, read_json(&mut self.reader, &mut self.buf))
                .await
                .map_err(|_| unavailable("daemon did not answer the handshake within 5 s"))?
                .map_err(|err| unavailable(&err.to_string()))?;
        let line = line.ok_or_else(|| unavailable("daemon closed the connection"))?;
        match from_value::<ResponseEnvelope>(line) {
            Ok(envelope) if envelope.id == id || envelope.id == 0 => Ok(envelope.response),
            _ => Err(version_mismatch(1, PROTOCOL_VERSION)),
        }
    }

    async fn recv_control(&mut self, id: u64) -> Result<Response, ServiceError> {
        timeout(CONTROL_TIMEOUT, self.recv(id))
            .await
            .map_err(|_| unavailable("daemon did not answer within 5 s"))?
    }

    async fn recv(&mut self, id: u64) -> Result<Response, ServiceError> {
        loop {
            let envelope: Option<ResponseEnvelope> = read_json(&mut self.reader, &mut self.buf)
                .await
                .map_err(|err| unavailable(&err.to_string()))?;
            let envelope = envelope.ok_or_else(|| unavailable("daemon closed the connection"))?;
            if envelope.id == id || envelope.id == 0 {
                return Ok(envelope.response);
            }
        }
    }
}

fn unavailable(message: &str) -> ServiceError {
    ServiceError::new(
        ErrorCode::DaemonUnavailable,
        format!("daemon connection failed: {message}"),
    )
    .with_hint("check `itts daemon status` and `itts daemon logs`")
}

fn unexpected(response: &Response) -> ServiceError {
    ServiceError::internal(format!("unexpected daemon response: {response:?}"))
}

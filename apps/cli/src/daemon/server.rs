//! The daemon process (`itts daemon run` / `itts daemon start --foreground`).
//!
//! Startup order guarantees a single model copy per socket: take an exclusive lock, bind the
//! socket, then load models while already answering `status` (state `loading`). Jobs submitted
//! during loading wait for readiness instead of loading their own copy in-process.

use std::{
    collections::HashMap,
    fs::{Permissions, TryLockError},
    io::{self, Write},
    iter::once,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arc_swap::ArcSwapOption;
use index_tts_rs_core::{
    config::{EmoAlpha, EmotionVector, EngineConfig, VoiceExtractRequest, VoiceId},
    engine::{IndexTtsEngine, StartupEvent},
    voice::{VoiceEntry, VoiceStore, default_voices_dir},
};
use index_tts_rs_director::DirectorConfig;
use index_tts_rs_service::{
    Backend, ErrorCode, JobAccumulator, JobRun, JobStarted, JobSummary, PathPolicy, ServiceError,
    SpeechJob,
};
use tokio::{
    fs,
    io::BufReader,
    net::{UnixListener, UnixStream},
    signal::unix::{Signal, SignalKind, signal},
    sync::{mpsc, watch},
    task::{AbortHandle, JoinSet, spawn_blocking},
    time::{MissedTickBehavior, interval, sleep, timeout},
};
use tracing::{info, warn};

use crate::daemon::{
    DaemonPaths,
    protocol::{
        DaemonState, DaemonStatus, JobCounters, LastError, PROTOCOL_VERSION, ReadyInfo, Request,
        RequestEnvelope, Response, ResponseEnvelope, StartupLine, VoiceSummary,
    },
    transport::{encode_line, read_line, write_json},
};

/// How long a job submitted during model loading waits for readiness.
const LOAD_WAIT: Duration = Duration::from_secs(300);
/// How long shutdown waits for in-flight jobs.
const DRAIN_WAIT: Duration = Duration::from_secs(10);
/// Idle timer resolution.
const IDLE_TICK: Duration = Duration::from_secs(5);
/// Per-connection outbound queue depth.
const CONNECTION_QUEUE: usize = 64;

/// Daemon runtime options.
#[derive(Debug, Clone)]
pub struct DaemonOptions {
    /// Socket, lock, pid, and log paths.
    pub paths: DaemonPaths,
    /// Idle auto-shutdown timeout in seconds (`0` disables it).
    pub idle_timeout_secs: u64,
    /// Maximum concurrent jobs before `BUSY`.
    pub max_in_flight: usize,
}

/// Receives startup progress.
pub trait StartupReporter: Send {
    /// Handles one startup line.
    fn report(&mut self, line: &StartupLine);
}

/// Writes startup lines as JSON to stdout (read by the detaching parent process).
#[derive(Debug, Default)]
pub struct JsonLinesReporter;

impl StartupReporter for JsonLinesReporter {
    fn report(&mut self, line: &StartupLine) {
        if let Ok(encoded) = encode_line(line) {
            let mut out = io::stdout().lock();
            let _ = out.write_all(&encoded);
            let _ = out.flush();
        }
    }
}

#[derive(Debug, Clone)]
enum Phase {
    Loading,
    Ready(Box<IndexTtsEngine>),
    Failed(ServiceError),
}

#[derive(Debug)]
struct Shared {
    opts: DaemonOptions,
    voices: VoiceStore,
    started: Instant,
    last_active_ms: AtomicU64,
    phase: watch::Receiver<Phase>,
    shutdown: watch::Sender<bool>,
    in_flight: AtomicUsize,
    completed: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    last_error: ArcSwapOption<LastError>,
}

/// Decrements the in-flight count and refreshes the idle timer when a job ends.
#[derive(Debug)]
struct SlotGuard(Arc<Shared>);

impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::AcqRel);
        self.0.touch();
    }
}

impl Shared {
    fn touch(&self) {
        self.last_active_ms.store(now_unix_ms(), Ordering::Relaxed);
    }

    fn idle_secs(&self) -> u64 {
        if self.in_flight.load(Ordering::Acquire) > 0 {
            return 0;
        }
        now_unix_ms().saturating_sub(self.last_active_ms.load(Ordering::Relaxed)) / 1000
    }

    fn try_acquire_slot(self: &Arc<Self>) -> Option<SlotGuard> {
        let max = self.opts.max_in_flight;
        self.in_flight
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < max).then_some(n + 1)
            })
            .ok()
            .map(|_| {
                self.touch();
                SlotGuard(Arc::clone(self))
            })
    }

    fn request_shutdown(&self) {
        self.shutdown.send_replace(true);
    }

    fn record_failure(&self, error: &ServiceError) {
        if error.code == ErrorCode::Cancelled {
            self.cancelled.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.failed.fetch_add(1, Ordering::Relaxed);
        self.last_error.store(Some(Arc::new(LastError {
            error: error.clone(),
            at_unix_secs: now_unix_ms() / 1000,
        })));
    }

    async fn ready_engine(&self) -> Result<IndexTtsEngine, ServiceError> {
        let mut phase = self.phase.clone();
        let current = timeout(LOAD_WAIT, phase.wait_for(|p| !matches!(p, Phase::Loading)))
            .await
            .map_err(|_| {
                ServiceError::new(ErrorCode::Timeout, "daemon is still loading models")
                    .with_hint("check `itts daemon logs`")
            })?
            .map_err(|_| ServiceError::internal("daemon is shutting down"))?
            .clone();
        match current {
            Phase::Ready(engine) => Ok(*engine),
            Phase::Failed(error) => Err(error),
            Phase::Loading => Err(ServiceError::internal("daemon is still loading models")),
        }
    }

    fn status(&self) -> DaemonStatus {
        let phase = self.phase.borrow().clone();
        let (state, engine) = match &phase {
            Phase::Ready(engine) => (DaemonState::Ready, Some(engine.as_ref())),
            Phase::Loading | Phase::Failed(_) => (DaemonState::Loading, None),
        };
        DaemonStatus {
            state,
            version: env!("CARGO_PKG_VERSION").to_owned(),
            protocol_version: PROTOCOL_VERSION,
            pid: process::id(),
            socket_path: self.opts.paths.socket.clone(),
            log_path: self.opts.paths.log.clone(),
            model_dir: engine.map(|e| e.model_dir().to_path_buf()),
            voices_dir: self.voices.voices_dir().to_path_buf(),
            uptime_secs: self.started.elapsed().as_secs(),
            idle_secs: self.idle_secs(),
            idle_timeout_secs: self.opts.idle_timeout_secs,
            voices: self
                .voices
                .list()
                .unwrap_or_default()
                .into_iter()
                .map(VoiceSummary::from)
                .collect(),
            engine: engine.map(IndexTtsEngine::telemetry),
            jobs: JobCounters {
                in_flight: self.in_flight.load(Ordering::Acquire),
                max_in_flight: self.opts.max_in_flight,
                completed: self.completed.load(Ordering::Relaxed),
                failed: self.failed.load(Ordering::Relaxed),
                cancelled: self.cancelled.load(Ordering::Relaxed),
            },
            last_error: self.last_error.load_full().map(|e| (*e).clone()),
        }
    }
}

/// Runs the daemon until `stop`, `SIGTERM`/`SIGINT`, or idle timeout.
///
/// Returns `Ok(())` without doing anything if another daemon holds the lock (reported as
/// [`StartupLine::AlreadyRunning`]).
///
/// # Errors
///
/// Returns (and reports) the failure if the socket cannot be bound or models fail to load.
pub async fn run_daemon(
    config: EngineConfig,
    opts: DaemonOptions,
    reporter: &mut dyn StartupReporter,
) -> Result<(), ServiceError> {
    match serve(config, opts, reporter).await {
        Ok(()) => Ok(()),
        Err(error) => {
            reporter.report(&StartupLine::Failed {
                error: error.clone(),
            });
            Err(error)
        }
    }
}

async fn serve(
    config: EngineConfig,
    opts: DaemonOptions,
    reporter: &mut dyn StartupReporter,
) -> Result<(), ServiceError> {
    let paths = opts.paths.clone();
    let Some(_lock) = acquire_lock(&paths, reporter).await? else {
        return Ok(());
    };
    let listener = bind_socket(&paths).await?;
    fs::write(&paths.pid, process::id().to_string())
        .await
        .map_err(|err| io_error(&paths.pid, &err))?;
    reporter.report(&progress(
        "socket",
        Some(paths.socket.display().to_string()),
        None,
    ));

    let mut signals = Signals::install()?;
    let voices_dir = config.voices_dir.clone().unwrap_or_else(default_voices_dir);
    let (phase_tx, phase_rx) = watch::channel(Phase::Loading);
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let shared = Arc::new(Shared {
        voices: VoiceStore::new(voices_dir)?,
        opts,
        started: Instant::now(),
        last_active_ms: AtomicU64::new(now_unix_ms()),
        phase: phase_rx,
        shutdown: shutdown_tx,
        in_flight: AtomicUsize::new(0),
        completed: AtomicU64::new(0),
        failed: AtomicU64::new(0),
        cancelled: AtomicU64::new(0),
        last_error: ArcSwapOption::empty(),
    });
    let accept_task = tokio::spawn(accept_loop(listener, Arc::clone(&shared)));

    let loaded = load_engine(config, reporter, &mut signals).await;
    let engine = match loaded {
        Ok(engine) => engine,
        Err(error) => {
            phase_tx.send_replace(Phase::Failed(error.clone()));
            accept_task.abort();
            remove_runtime_files(&paths).await;
            return Err(error);
        }
    };
    phase_tx.send_replace(Phase::Ready(Box::new(engine.clone())));
    report_ready(&shared, &engine, reporter);

    if !signals.stop_requested {
        wait_for_shutdown(&shared, &mut shutdown_rx, &mut signals).await;
    }
    shared.request_shutdown();
    accept_task.abort();
    drain_in_flight(&shared).await;
    remove_runtime_files(&paths).await;
    spawn_blocking(move || engine.stop().map_err(ServiceError::from))
        .await
        .map_err(|err| ServiceError::internal(format!("engine shutdown task failed: {err}")))??;
    info!("Daemon stopped");
    Ok(())
}

/// Exclusive advisory lock on `<socket>.lock`, released when the daemon process exits.
///
/// `tokio::fs::File` exposes no locking API; the std handle is only held, never read or
/// written, and `try_lock` never blocks.
#[derive(Debug)]
#[allow(clippy::disallowed_types)]
struct InstanceLock {
    _file: std::fs::File,
}

async fn acquire_lock(
    paths: &DaemonPaths,
    reporter: &mut dyn StartupReporter,
) -> Result<Option<InstanceLock>, ServiceError> {
    if let Some(parent) = paths.socket.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .await
            .map_err(|err| io_error(parent, &err))?;
    }
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&paths.lock)
        .await
        .map_err(|err| io_error(&paths.lock, &err))?
        .into_std()
        .await;
    match file.try_lock() {
        Ok(()) => Ok(Some(InstanceLock { _file: file })),
        Err(TryLockError::WouldBlock) => {
            reporter.report(&StartupLine::AlreadyRunning {
                pid: read_pid(&paths.pid).await,
                socket_path: paths.socket.clone(),
            });
            Ok(None)
        }
        Err(TryLockError::Error(err)) => Err(io_error(&paths.lock, &err)),
    }
}

async fn bind_socket(paths: &DaemonPaths) -> Result<UnixListener, ServiceError> {
    if paths.socket.exists() {
        let live = timeout(
            Duration::from_millis(300),
            UnixStream::connect(&paths.socket),
        )
        .await
        .is_ok_and(|res| res.is_ok());
        if live {
            return Err(ServiceError::new(
                ErrorCode::DaemonVersionMismatch,
                format!(
                    "an older itts daemon without instance locking is listening at `{}`",
                    paths.socket.display()
                ),
            )
            .with_hint("stop it with `itts daemon stop`, then start again"));
        }
        fs::remove_file(&paths.socket)
            .await
            .map_err(|err| io_error(&paths.socket, &err))?;
    }
    let listener =
        UnixListener::bind(&paths.socket).map_err(|err| io_error(&paths.socket, &err))?;
    fs::set_permissions(&paths.socket, Permissions::from_mode(0o600))
        .await
        .map_err(|err| io_error(&paths.socket, &err))?;
    Ok(listener)
}

async fn load_engine(
    config: EngineConfig,
    reporter: &mut dyn StartupReporter,
    signals: &mut Signals,
) -> Result<IndexTtsEngine, ServiceError> {
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<StartupEvent>();
    let mut load = spawn_blocking(move || {
        IndexTtsEngine::start_with_observer(
            config,
            Box::new(move |event| {
                let _ = event_tx.send(event.clone());
            }),
        )
        .map_err(ServiceError::from)
    });
    let result = loop {
        tokio::select! {
            Some(event) = event_rx.recv() => reporter.report(&startup_progress(&event)),
            res = &mut load => break res,
            () = signals.recv() => signals.stop_requested = true,
        }
    };
    while let Ok(event) = event_rx.try_recv() {
        reporter.report(&startup_progress(&event));
    }
    result.map_err(|err| ServiceError::internal(format!("engine startup task failed: {err}")))?
}

fn report_ready(shared: &Shared, engine: &IndexTtsEngine, reporter: &mut dyn StartupReporter) {
    let voices: Vec<String> = shared
        .voices
        .list()
        .unwrap_or_default()
        .into_iter()
        .map(|entry| entry.voice_id)
        .collect();
    let listed = if voices.is_empty() {
        "none".to_owned()
    } else {
        voices.join(", ")
    };
    reporter.report(&progress("voices", Some(listed), None));
    let load_ms = shared.started.elapsed().as_secs_f64() * 1000.0;
    info!(
        pid = process::id(),
        socket = %shared.opts.paths.socket.display(),
        model_dir = %engine.model_dir().display(),
        voices = voices.len(),
        load_ms,
        "itts daemon ready"
    );
    reporter.report(&StartupLine::Ready(Box::new(ReadyInfo {
        pid: process::id(),
        version: env!("CARGO_PKG_VERSION").to_owned(),
        protocol_version: PROTOCOL_VERSION,
        socket_path: shared.opts.paths.socket.clone(),
        log_path: shared.opts.paths.log.clone(),
        model_dir: engine.model_dir().to_path_buf(),
        voices_dir: shared.voices.voices_dir().to_path_buf(),
        voices,
        load_ms,
        idle_timeout_secs: shared.opts.idle_timeout_secs,
    })));
}

async fn wait_for_shutdown(
    shared: &Shared,
    shutdown_rx: &mut watch::Receiver<bool>,
    signals: &mut Signals,
) {
    let mut idle = interval(IDLE_TICK);
    idle.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let timeout_secs = shared.opts.idle_timeout_secs;
    loop {
        tokio::select! {
            _ = shutdown_rx.wait_for(|stop| *stop) => {
                info!("Stop requested by client");
                return;
            }
            () = signals.recv() => return,
            _ = idle.tick() => {
                if timeout_secs > 0 && shared.idle_secs() >= timeout_secs {
                    info!(idle_timeout_secs = timeout_secs, "Idle timeout reached; shutting down");
                    return;
                }
            }
        }
    }
}

async fn drain_in_flight(shared: &Shared) {
    let deadline = Instant::now() + DRAIN_WAIT;
    while shared.in_flight.load(Ordering::Acquire) > 0 && Instant::now() < deadline {
        sleep(Duration::from_millis(100)).await;
    }
}

async fn accept_loop(listener: UnixListener, shared: Arc<Shared>) {
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    let shared = Arc::clone(&shared);
                    connections.spawn(async move {
                        if let Err(err) = handle_connection(stream, shared).await {
                            warn!(error = %err, "Daemon connection error");
                        }
                    });
                }
                Err(err) => {
                    warn!(error = %err, "Daemon accept error");
                    sleep(Duration::from_millis(50)).await;
                }
            },
            Some(joined) = connections.join_next() => {
                if let Err(err) = joined && err.is_panic() {
                    warn!(error = %err, "Daemon connection task panicked");
                }
            }
        }
    }
}

type Outbound = mpsc::Sender<ResponseEnvelope>;

async fn send(tx: &Outbound, id: u64, response: Response) -> bool {
    tx.send(ResponseEnvelope { id, response }).await.is_ok()
}

async fn handle_connection(stream: UnixStream, shared: Arc<Shared>) -> io::Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let (tx, mut rx) = mpsc::channel::<ResponseEnvelope>(CONNECTION_QUEUE);
    let writer = tokio::spawn(async move {
        while let Some(envelope) = rx.recv().await {
            if write_json(&mut write_half, &envelope).await.is_err() {
                break;
            }
        }
    });

    let mut reader = BufReader::new(read_half);
    let mut buf = Vec::new();
    let mut jobs: HashMap<u64, AbortHandle> = HashMap::new();
    let mut tasks = JoinSet::new();
    loop {
        match read_line(&mut reader, &mut buf).await {
            Ok(true) => {}
            Ok(false) => break,
            Err(err) => {
                send(
                    &tx,
                    0,
                    error_response(ServiceError::invalid(err.to_string())),
                )
                .await;
                break;
            }
        }
        if buf.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let envelope: RequestEnvelope = match serde_json::from_slice(&buf) {
            Ok(envelope) => envelope,
            Err(err) => {
                let error = ServiceError::invalid(format!("invalid request: {err}"));
                send(&tx, 0, error_response(error)).await;
                continue;
            }
        };
        jobs.retain(|_, handle| !handle.is_finished());
        dispatch(&shared, envelope, &tx, &mut jobs, &mut tasks).await;
    }

    for handle in jobs.values().filter(|h| !h.is_finished()) {
        handle.abort();
        shared.cancelled.fetch_add(1, Ordering::Relaxed);
    }
    tasks.shutdown().await;
    drop(tx);
    let _ = writer.await;
    Ok(())
}

async fn dispatch(
    shared: &Arc<Shared>,
    envelope: RequestEnvelope,
    tx: &Outbound,
    jobs: &mut HashMap<u64, AbortHandle>,
    tasks: &mut JoinSet<()>,
) {
    let id = envelope.id;
    match envelope.request {
        Request::Hello {
            protocol_version, ..
        } => {
            let response = if protocol_version == PROTOCOL_VERSION {
                Response::Welcome {
                    server_version: env!("CARGO_PKG_VERSION").to_owned(),
                    protocol_version: PROTOCOL_VERSION,
                }
            } else {
                error_response(version_mismatch(PROTOCOL_VERSION, protocol_version))
            };
            send(tx, id, response).await;
        }
        Request::Status => {
            send(tx, id, Response::Status(Box::new(shared.status()))).await;
        }
        Request::Stop => {
            send(tx, id, Response::Stopping).await;
            shared.request_shutdown();
        }
        Request::Cancel { target } => {
            let found = jobs.remove(&target).is_some_and(|handle| {
                handle.abort();
                true
            });
            if found {
                shared.cancelled.fetch_add(1, Ordering::Relaxed);
                let error = ServiceError::new(ErrorCode::Cancelled, "cancelled by client");
                send(tx, target, error_response(error)).await;
            }
            send(tx, id, Response::Cancelled { found }).await;
        }
        Request::Synth { job, include_pcm } => {
            let Some(slot) = shared.try_acquire_slot() else {
                send(tx, id, error_response(busy(shared))).await;
                return;
            };
            let (shared, tx) = (Arc::clone(shared), tx.clone());
            let handle = tasks.spawn(async move {
                let result = synth(&shared, id, *job, include_pcm, &tx).await;
                finish_request(
                    &shared,
                    &tx,
                    id,
                    result.map(|s| Response::Done(Box::new(s))),
                )
                .await;
                drop(slot);
            });
            jobs.insert(id, handle);
        }
        Request::Annotate { job, api_key } => {
            let Some(slot) = shared.try_acquire_slot() else {
                send(tx, id, error_response(busy(shared))).await;
                return;
            };
            let (shared, tx) = (Arc::clone(shared), tx.clone());
            let handle = tasks.spawn(async move {
                let result = job.run(DirectorConfig::default(), api_key.as_ref()).await;
                finish_request(
                    &shared,
                    &tx,
                    id,
                    result.map(|r| Response::Annotated(Box::new(r))),
                )
                .await;
                drop(slot);
            });
            jobs.insert(id, handle);
        }
        Request::VoiceAdd {
            voice_id,
            ref_audio,
            emo_ref,
            emo_alpha,
            emo,
        } => {
            let Some(slot) = shared.try_acquire_slot() else {
                send(tx, id, error_response(busy(shared))).await;
                return;
            };
            let (shared, tx) = (Arc::clone(shared), tx.clone());
            let handle = tasks.spawn(async move {
                let result = voice_add(&shared, voice_id, ref_audio, emo_ref, emo_alpha, emo).await;
                let response = result.map(|voice| Response::Voice {
                    voice: Box::new(voice),
                });
                finish_request(&shared, &tx, id, response).await;
                drop(slot);
            });
            jobs.insert(id, handle);
        }
    }
}

async fn finish_request(
    shared: &Shared,
    tx: &Outbound,
    id: u64,
    result: Result<Response, ServiceError>,
) {
    let response = match result {
        Ok(response) => {
            shared.completed.fetch_add(1, Ordering::Relaxed);
            response
        }
        Err(error) => {
            warn!(code = error.code.as_str(), error = %error.message, "Daemon request failed");
            shared.record_failure(&error);
            error_response(error)
        }
    };
    send(tx, id, response).await;
}

async fn synth(
    shared: &Shared,
    id: u64,
    job: SpeechJob,
    include_pcm: bool,
    tx: &Outbound,
) -> Result<JobSummary, ServiceError> {
    let engine = shared.ready_engine().await?;
    let prepared = job.prepare(engine.config(), PathPolicy::LocalAbsolute)?;
    let disconnected = || ServiceError::new(ErrorCode::Cancelled, "client disconnected");
    if !send(
        tx,
        id,
        Response::Started(JobStarted::new(&prepared, Backend::Daemon)),
    )
    .await
    {
        return Err(disconnected());
    }
    let started = Instant::now();
    let output_path = prepared.output_path.clone();
    let mut acc = JobAccumulator::new(&prepared, output_path.is_some());
    let mut run = JobRun::new(engine, prepared);
    while let Some(chunk) = run.next_chunk().await? {
        let info = chunk.info(include_pcm);
        acc.push(&chunk);
        if !send(tx, id, Response::Chunk(Box::new(info))).await {
            return Err(disconnected());
        }
    }
    acc.finish(started.elapsed(), output_path.as_deref())
}

async fn voice_add(
    shared: &Shared,
    voice_id: String,
    ref_audio: PathBuf,
    emo_ref: Option<PathBuf>,
    emo_alpha: Option<f32>,
    emo: Option<String>,
) -> Result<VoiceEntry, ServiceError> {
    for path in once(&ref_audio).chain(emo_ref.as_ref()) {
        if !path.is_absolute() {
            return Err(ServiceError::invalid(format!(
                "voice audio paths must be absolute, got `{}`",
                path.display()
            )));
        }
    }
    let engine = shared.ready_engine().await?;
    let emo_vector = match emo.as_deref() {
        Some(spec) => EmotionVector::parse_spec(spec)?,
        None => None,
    };
    let request = VoiceExtractRequest::builder()
        .voice_id(VoiceId::new(voice_id)?)
        .ref_audio_path(ref_audio)
        .emo_audio_path(emo_ref)
        .emo_alpha(EmoAlpha::new(emo_alpha.unwrap_or(1.0))?)
        .emo_vector(emo_vector)
        .build();
    let entry = engine.extract_voice(request).await;
    // Stage 0 encoders cost ~1.4 GB; keep the warm footprint at the synthesis baseline.
    let _ = engine.evict_voice_extractor().await;
    Ok(entry?)
}

fn error_response(error: ServiceError) -> Response {
    Response::Error { error }
}

fn busy(shared: &Shared) -> ServiceError {
    ServiceError::new(
        ErrorCode::Busy,
        format!(
            "daemon already has {} requests in flight",
            shared.opts.max_in_flight
        ),
    )
    .with_hint("retry after the current jobs finish")
}

/// Builds the mismatch error; protocol `1` denotes daemons that predate the handshake.
pub(crate) fn version_mismatch(daemon_protocol: u32, client_protocol: u32) -> ServiceError {
    ServiceError::new(
        ErrorCode::DaemonVersionMismatch,
        format!(
            "the daemon speaks protocol {daemon_protocol} but this itts speaks protocol \
             {client_protocol}"
        ),
    )
    .with_hint("run `itts daemon restart` so the daemon matches the installed itts")
}

fn startup_progress(event: &StartupEvent) -> StartupLine {
    match event {
        StartupEvent::ModelResolved { model_dir } => {
            progress("model", Some(model_dir.display().to_string()), None)
        }
        StartupEvent::ComponentLoaded {
            component,
            elapsed_ms,
        } => progress(component, None, Some(*elapsed_ms)),
        StartupEvent::EngineReady { elapsed_ms } => progress("engine", None, Some(*elapsed_ms)),
        _ => progress("engine", None, None),
    }
}

fn progress(step: &str, detail: Option<String>, elapsed_ms: Option<f64>) -> StartupLine {
    StartupLine::Progress {
        step: step.to_owned(),
        detail,
        elapsed_ms,
    }
}

/// Termination signals. Registering `SIGHUP` replaces its default action, so closing the
/// terminal that launched a foreground daemon does not kill it mid-job.
struct Signals {
    terminate: Signal,
    interrupt: Signal,
    _hangup: Signal,
    stop_requested: bool,
}

impl Signals {
    fn install() -> Result<Self, ServiceError> {
        let install = |kind: SignalKind| {
            signal(kind).map_err(|err| {
                ServiceError::internal(format!("failed to install signal handler: {err}"))
            })
        };
        Ok(Self {
            terminate: install(SignalKind::terminate())?,
            interrupt: install(SignalKind::interrupt())?,
            _hangup: install(SignalKind::hangup())?,
            stop_requested: false,
        })
    }

    async fn recv(&mut self) {
        tokio::select! {
            _ = self.terminate.recv() => info!("Received SIGTERM; stopping daemon"),
            _ = self.interrupt.recv() => info!("Received SIGINT; stopping daemon"),
        }
    }
}

async fn remove_runtime_files(paths: &DaemonPaths) {
    for path in [&paths.socket, &paths.pid] {
        if let Err(err) = fs::remove_file(path).await
            && err.kind() != io::ErrorKind::NotFound
        {
            warn!(path = %path.display(), error = %err, "Failed to remove daemon runtime file");
        }
    }
}

/// Reads the pid recorded by a running daemon.
async fn read_pid(path: &Path) -> Option<u32> {
    fs::read_to_string(path).await.ok()?.trim().parse().ok()
}

fn io_error(path: &Path, err: &io::Error) -> ServiceError {
    ServiceError::new(ErrorCode::Io, format!("`{}`: {err}", path.display()))
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "milliseconds since 1970 fit in u64"
)]
fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicU64, AtomicUsize, Ordering},
        },
        time::Instant,
    };

    use arc_swap::ArcSwapOption;
    use index_tts_rs_core::voice::VoiceStore;
    use index_tts_rs_service::{ErrorCode, ServiceError};
    use tokio::sync::watch;

    use super::{DaemonOptions, Phase, Shared, now_unix_ms};
    use crate::daemon::{DaemonPaths, protocol::DaemonState};

    fn shared(max_in_flight: usize, voices: &Path) -> Result<Arc<Shared>, ServiceError> {
        let (_phase_tx, phase) = watch::channel(Phase::Loading);
        let (shutdown, _) = watch::channel(false);
        Ok(Arc::new(Shared {
            opts: DaemonOptions {
                paths: DaemonPaths::new(&voices.join("d.sock")),
                idle_timeout_secs: 600,
                max_in_flight,
            },
            voices: VoiceStore::new(voices)?,
            started: Instant::now(),
            last_active_ms: AtomicU64::new(now_unix_ms().saturating_sub(30_000)),
            phase,
            shutdown,
            in_flight: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            last_error: ArcSwapOption::empty(),
        }))
    }

    #[test]
    fn test_should_reject_jobs_beyond_max_in_flight_and_release_on_drop()
    -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let shared = shared(2, dir.path())?;
        let first = shared.try_acquire_slot();
        let second = shared.try_acquire_slot();
        assert!(first.is_some() && second.is_some());
        assert!(shared.try_acquire_slot().is_none());
        assert_eq!(shared.idle_secs(), 0, "busy daemons are never idle");
        drop(first);
        assert_eq!(shared.in_flight.load(Ordering::Acquire), 1);
        assert!(shared.try_acquire_slot().is_some());
        Ok(())
    }

    #[test]
    fn test_should_report_status_without_resetting_idle_timer() -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let shared = shared(4, dir.path())?;
        let status = shared.status();
        assert_eq!(status.state, DaemonState::Loading);
        assert!(shared.status().idle_secs >= 29);
        Ok(())
    }

    #[test]
    fn test_should_count_cancellations_separately_from_failures() -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let shared = shared(4, dir.path())?;
        shared.record_failure(&ServiceError::new(ErrorCode::Cancelled, "gone"));
        shared.record_failure(&ServiceError::new(ErrorCode::VoiceNotFound, "ghost"));
        let status = shared.status();
        assert_eq!((status.jobs.cancelled, status.jobs.failed), (1, 1));
        assert_eq!(
            status.last_error.map(|e| e.error.code),
            Some(ErrorCode::VoiceNotFound)
        );
        Ok(())
    }
}

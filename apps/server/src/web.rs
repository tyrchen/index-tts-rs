//! Browser assets, registered voice discovery, and bounded speech synthesis requests.
//!
//! Request bodies are [`SpeechJob`]s, the same model used by the CLI, daemon, and MCP server, so
//! inline controls and parameters behave identically everywhere. File references are rejected
//! ([`PathPolicy::Denied`]); only registered voices can be used over HTTP.

use std::{fmt::Display, io::Error as IoError, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Request, State, rejection::JsonRejection},
    http::{HeaderValue, StatusCode, header},
    middleware::{Next, from_fn_with_state},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::stream::unfold;
use index_tts_rs_core::{dsp::encode_wav_i16_bytes, engine::IndexTtsEngine};
use index_tts_rs_director::{Director, DirectorConfig};
use index_tts_rs_service::{
    ErrorCode, JobChunk, JobRun, PathPolicy, PreparedJob, ServiceError, SpeechJob,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json, to_vec};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    task::spawn_blocking,
    time::{Instant, timeout, timeout_at},
};
use tracing::{error, instrument};

const MAX_HTTP_BODY_BYTES: usize = 32_768;
const MAX_CONCURRENT_REQUESTS: usize = 8;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Longest audio a single request may produce, in seconds.
const MAX_AUDIO_SECS: usize = 600;
const INDEX_HTML: &str = include_str!("../static/index.html");
const APP_CSS: &str = include_str!("../static/app.css");
const APP_JS: &str = include_str!("../static/app.js");

#[derive(Debug, Clone)]
struct AppState {
    engine: Arc<IndexTtsEngine>,
    synthesis: Arc<Semaphore>,
    director_config: Arc<DirectorConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VoiceSummary {
    id: String,
    duration_secs: f32,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: ErrorCode,
    message: String,
    hint: Option<String>,
}

impl ApiError {
    fn new(status: StatusCode, code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            hint: None,
        }
    }

    fn internal(err: impl Display) -> Self {
        error!(error = %err, "speech studio request failed");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorCode::Internal,
            "语音服务处理失败，请检查服务日志后重试。",
        )
    }

    fn rejected(rejection: &JsonRejection) -> Self {
        Self::new(
            rejection.status(),
            ErrorCode::InvalidArgument,
            rejection.body_text(),
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut body = json!({ "error": self.message, "code": self.code });
        if let (Some(hint), Some(object)) = (self.hint, body.as_object_mut()) {
            object.insert("hint".to_owned(), Value::String(hint));
        }
        (self.status, Json(body)).into_response()
    }
}

impl From<ServiceError> for ApiError {
    fn from(err: ServiceError) -> Self {
        let (status, message) = match err.code {
            ErrorCode::InvalidArgument => (StatusCode::UNPROCESSABLE_ENTITY, err.message),
            ErrorCode::VoiceNotFound => (
                StatusCode::NOT_FOUND,
                "找不到这个音色，请刷新音色列表后重试。".to_owned(),
            ),
            ErrorCode::Busy => (StatusCode::SERVICE_UNAVAILABLE, err.message),
            ErrorCode::Timeout => (StatusCode::GATEWAY_TIMEOUT, err.message),
            ErrorCode::AnnotationUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "请先在启动 server 的终端配置 DEEPSEEK_API_KEY。".to_owned(),
            ),
            ErrorCode::AnnotationFailed => {
                error!(error = %err.message, "speech annotation failed");
                (
                    StatusCode::BAD_GATEWAY,
                    "AI 标注失败，请检查模型配置或稍后重试；原文未改动。".to_owned(),
                )
            }
            _ => return Self::internal(err.message),
        };
        Self {
            status,
            code: err.code,
            message,
            hint: err.hint,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AnnotationPayload {
    input: String,
    #[serde(default)]
    poetry: bool,
}

#[instrument(skip(state, payload))]
async fn annotate(
    State(state): State<AppState>,
    payload: Result<Json<AnnotationPayload>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(payload) = payload.map_err(|rejection| ApiError::rejected(&rejection))?;
    let config = state.director_config.as_ref().clone();
    let director = spawn_blocking(move || Director::from_env(config).map_err(ServiceError::from))
        .await
        .map_err(ApiError::internal)??;
    let script = director
        .annotate(&payload.input, payload.poetry)
        .await
        .map_err(ServiceError::from)?;
    Ok(Json(json!({"script": script})))
}

/// Validates a request body for this transport.
fn prepare_job(engine: &IndexTtsEngine, job: &SpeechJob) -> Result<PreparedJob, ApiError> {
    Ok(job.prepare(engine.config(), PathPolicy::Denied)?)
}

pub(super) fn router(engine: Arc<IndexTtsEngine>, director_config: DirectorConfig) -> Router {
    let state = AppState {
        engine,
        synthesis: Arc::new(Semaphore::new(1)),
        director_config: Arc::new(director_config),
    };
    Router::new()
        .route("/", get(|| async { Html(INDEX_HTML) }))
        .route("/favicon.ico", get(|| async { StatusCode::NO_CONTENT }))
        .route(
            "/app.css",
            get(|| async { ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], APP_CSS) }),
        )
        .route(
            "/app.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    APP_JS,
                )
            }),
        )
        .route("/health", get(health))
        .route("/api/voices", get(voices))
        .route("/v1/audio/speech", post(speech))
        .route("/api/speech/stream", post(speech_stream))
        .route("/api/annotate", post(annotate))
        .fallback(|| async {
            ApiError::new(
                StatusCode::NOT_FOUND,
                ErrorCode::InvalidArgument,
                "not found",
            )
        })
        .layer(DefaultBodyLimit::max(MAX_HTTP_BODY_BYTES))
        .layer(from_fn_with_state(
            Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
            guard_request,
        ))
        .with_state(state)
}

async fn guard_request(
    State(requests): State<Arc<Semaphore>>,
    request: Request,
    next: Next,
) -> Response {
    let mut response = match requests.try_acquire() {
        Ok(_permit) => match timeout(REQUEST_TIMEOUT, next.run(request)).await {
            Ok(response) => response,
            Err(_) => ApiError::new(
                StatusCode::GATEWAY_TIMEOUT,
                ErrorCode::Timeout,
                "生成超时，请缩短文本后重试。",
            )
            .into_response(),
        },
        Err(_) => ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorCode::Busy,
            "服务繁忙，请稍后重试。",
        )
        .into_response(),
    };
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; \
             connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        ),
    );
    response
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(
        json!({ "status": "ok", "running": state.engine.is_running(), "sampleRate": state.engine.config().sample_rate }),
    )
}

#[instrument(skip(state))]
async fn voices(State(state): State<AppState>) -> Result<Json<Vec<VoiceSummary>>, ApiError> {
    let store = state.engine.voice_store().clone();
    let entries = spawn_blocking(move || store.list().map_err(ServiceError::from))
        .await
        .map_err(ApiError::internal)??;
    Ok(Json(
        entries
            .into_iter()
            .map(|entry| VoiceSummary {
                id: entry.voice_id,
                duration_secs: entry.duration_secs,
            })
            .collect(),
    ))
}

#[instrument(skip(state, payload))]
async fn speech(
    State(state): State<AppState>,
    payload: Result<Json<SpeechJob>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(job) = payload.map_err(|rejection| ApiError::rejected(&rejection))?;
    let mut session = ReadingSession::start(state, &job)?;
    let sample_rate = session.engine.config().sample_rate;
    let mut pcm = Vec::new();
    while let Some(packet) = session.next_packet().await? {
        pcm.extend(packet.pcm);
    }
    if pcm.is_empty() {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            ErrorCode::InvalidArgument,
            "没有生成可播放的语音，请检查文本。",
        ));
    }
    let wav =
        spawn_blocking(move || encode_wav_i16_bytes(&pcm, sample_rate).map_err(ServiceError::from))
            .await
            .map_err(ApiError::internal)??;
    Ok((
        [
            (header::CONTENT_TYPE, "audio/wav"),
            (
                header::CONTENT_DISPOSITION,
                "inline; filename=\"speech.wav\"",
            ),
        ],
        wav,
    )
        .into_response())
}

#[derive(Debug)]
struct ReadingSession {
    engine: Arc<IndexTtsEngine>,
    run: JobRun,
    _permit: OwnedSemaphorePermit,
    started: Instant,
    samples: usize,
}

#[derive(Debug)]
struct PcmPacket {
    pcm: Vec<i16>,
    paragraph_index: usize,
    total_paragraphs: usize,
}

impl ReadingSession {
    fn start(state: AppState, job: &SpeechJob) -> Result<Self, ApiError> {
        let permit = state.synthesis.try_acquire_owned().map_err(|_| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                ErrorCode::Busy,
                "已有语音正在生成，请稍后重试。",
            )
        })?;
        let prepared = prepare_job(&state.engine, job)?;
        Ok(Self {
            run: JobRun::new(state.engine.as_ref().clone(), prepared.plan),
            engine: state.engine,
            _permit: permit,
            started: Instant::now(),
            samples: 0,
        })
    }

    async fn next_packet(&mut self) -> Result<Option<PcmPacket>, ApiError> {
        let next = timeout_at(self.started + REQUEST_TIMEOUT, self.run.next_chunk())
            .await
            .map_err(|_| {
                ApiError::new(
                    StatusCode::GATEWAY_TIMEOUT,
                    ErrorCode::Timeout,
                    "生成超时，请分段提交较长文本。",
                )
            })??;
        next.map(|chunk| self.packet(chunk)).transpose()
    }

    fn packet(&mut self, chunk: JobChunk) -> Result<PcmPacket, ApiError> {
        self.samples = self.samples.saturating_add(chunk.audio.pcm_i16.len());
        let max_samples = usize::try_from(chunk.audio.sample_rate)
            .map_err(ApiError::internal)?
            .saturating_mul(MAX_AUDIO_SECS);
        if self.samples > max_samples {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                ErrorCode::InvalidArgument,
                "音频超过十分钟，请分段提交。",
            ));
        }
        Ok(PcmPacket {
            pcm: chunk.audio.pcm_i16,
            paragraph_index: chunk.utterance_index,
            total_paragraphs: chunk.total_utterances,
        })
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "Audio is capped to ten minutes, at most 28.8M samples, exactly representable in \
                  f64"
    )]
    fn summary(&self) -> Value {
        let elapsed = self.started.elapsed().as_secs_f64();
        let duration = self.samples as f64 / f64::from(self.engine.config().sample_rate);
        json!({ "event": "done", "wallSecs": elapsed, "durationSecs": duration,
            "sampleRate": self.engine.config().sample_rate, "samples": self.samples,
            "paragraphs": self.run.total_utterances() })
    }
}

impl PcmPacket {
    fn into_event(self, sample_rate: u32, elapsed: f64) -> Value {
        let mut bytes = Vec::with_capacity(self.pcm.len().saturating_mul(2));
        for sample in self.pcm {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        json!({ "event": "chunk", "pcmBase64": STANDARD.encode(bytes), "sampleRate": sample_rate,
            "utteranceIndex": self.paragraph_index, "totalUtterances": self.total_paragraphs, "elapsedSecs": elapsed })
    }
}

fn event_bytes(event: &Value) -> Result<Bytes, IoError> {
    let mut bytes = to_vec(event).map_err(IoError::other)?;
    bytes.push(b'\n');
    Ok(Bytes::from(bytes))
}

#[instrument(skip(state, payload))]
async fn speech_stream(
    State(state): State<AppState>,
    payload: Result<Json<SpeechJob>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(job) = payload.map_err(|rejection| ApiError::rejected(&rejection))?;
    let session = ReadingSession::start(state, &job)?;
    let stream = unfold(Some(session), |session| async move {
        let mut session = session?;
        match session.next_packet().await {
            Ok(Some(packet)) => {
                let event = packet.into_event(
                    session.engine.config().sample_rate,
                    session.started.elapsed().as_secs_f64(),
                );
                Some((event_bytes(&event), Some(session)))
            }
            Ok(None) => {
                let event = if session.samples == 0 {
                    json!({"event": "error", "error": "没有生成可播放的语音，请检查文本。"})
                } else {
                    session.summary()
                };
                Some((event_bytes(&event), None))
            }
            Err(error) => Some((
                event_bytes(&json!({"event": "error", "error": error.message, "code": error.code})),
                None,
            )),
        }
    });
    Ok((
        [(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")],
        Body::from_stream(stream),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use std::{path::Path, sync::Arc};

    use anyhow::{Context, Result, ensure};
    use axum::{
        Router,
        body::{Body, to_bytes},
        http::{Request as HttpRequest, StatusCode, header},
        middleware::from_fn_with_state,
        routing::get,
    };
    use base64::{Engine, engine::general_purpose::STANDARD};
    use index_tts_rs_core::{
        config::{EngineConfig, SamplingParams, VoiceId},
        engine::IndexTtsEngine,
        voice::SpeakerContext,
    };
    use index_tts_rs_director::DirectorConfig;
    use index_tts_rs_service::{ErrorCode, PathPolicy, ServiceError, SpeechJob};
    use rstest::rstest;
    use serde_json::{Value, from_slice, from_str};
    use tempfile::tempdir;
    use tokio::{sync::Semaphore, task::spawn_blocking};
    use tower::ServiceExt;

    use super::{ApiError, MAX_HTTP_BODY_BYTES, guard_request, router};

    #[rstest]
    #[case(r#"{"input":"你好","voice":"tyr","refAudio":"/tmp/ref.wav"}"#)]
    #[case(r#"{"input":"你好","voice":"tyr","outputPath":"/tmp/out.wav"}"#)]
    #[case(r#"{"input":"你好","voiceFile":"/tmp/a.itts_voice"}"#)]
    #[case(r#"{"input":"你好","voice":"tyr","speed":5}"#)]
    fn test_should_reject_jobs_invalid_over_http(#[case] json: &str) -> Result<()> {
        let job: SpeechJob = from_str(json)?;
        let err = job.prepare(&EngineConfig::default(), PathPolicy::Denied);
        assert!(matches!(
            err,
            Err(ServiceError {
                code: ErrorCode::InvalidArgument,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn test_should_reject_unknown_fields() {
        assert!(from_str::<SpeechJob>(r#"{"input":"你好","voice":"tyr","extra":1}"#).is_err());
    }

    #[test]
    fn test_should_map_service_errors_to_http_status() {
        let cases = [
            (ErrorCode::InvalidArgument, StatusCode::UNPROCESSABLE_ENTITY),
            (ErrorCode::VoiceNotFound, StatusCode::NOT_FOUND),
            (ErrorCode::Busy, StatusCode::SERVICE_UNAVAILABLE),
            (ErrorCode::Internal, StatusCode::INTERNAL_SERVER_ERROR),
        ];
        for (code, status) in cases {
            assert_eq!(ApiError::from(ServiceError::new(code, "x")).status, status);
        }
    }

    #[tokio::test]
    async fn test_should_reject_excess_requests_and_apply_response_headers() -> Result<()> {
        let slots = Arc::new(Semaphore::new(1));
        let permit = slots.acquire().await?;
        let app = Router::new()
            .route("/", get(|| async { "ok" }))
            .layer(from_fn_with_state(Arc::clone(&slots), guard_request));
        let response = app
            .clone()
            .oneshot(HttpRequest::get("/").body(Body::empty())?)
            .await?;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(header::CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("no-store")
        );
        drop(permit);
        let response = app
            .oneshot(HttpRequest::get("/").body(Body::empty())?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires local IndexTTS model weights and golden voice fixture"]
    async fn test_should_serve_studio_list_voices_and_generate_wav() -> Result<()> {
        let directory = tempdir()?;
        let voices_dir = directory.path().to_path_buf();
        let engine = Arc::new(
            spawn_blocking(move || -> Result<IndexTtsEngine> {
                let config = EngineConfig::builder()
                    .voices_dir(Some(voices_dir))
                    .sampling(
                        SamplingParams::builder()
                            .greedy(true)
                            .max_mel_tokens(32)
                            .build(),
                    )
                    .build();
                Ok(IndexTtsEngine::start(config)?)
            })
            .await??,
        );
        let result = verify_studio_routes(Arc::clone(&engine)).await;
        spawn_blocking(move || engine.stop().context("failed to stop test engine")).await??;
        result
    }

    async fn verify_studio_routes(engine: Arc<IndexTtsEngine>) -> Result<()> {
        let app = router(Arc::clone(&engine), DirectorConfig::default());
        let response = app
            .clone()
            .oneshot(HttpRequest::get("/api/voices").body(Body::empty())?)
            .await?;
        let empty = to_bytes(response.into_body(), MAX_HTTP_BODY_BYTES).await?;
        assert_eq!(empty.as_ref(), b"[]");
        spawn_blocking(move || -> Result<()> {
            let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../crates/core/fixtures/golden_speaker.itts_voice");
            engine.voice_store().register(
                VoiceId::new("test_voice")?,
                SpeakerContext::load_from_path(fixture)?,
            )?;
            Ok(())
        })
        .await??;
        for (path, content_type) in [
            ("/", "text/html"),
            ("/app.js", "text/javascript"),
            ("/app.css", "text/css"),
            ("/health", "application/json"),
        ] {
            let response = app
                .clone()
                .oneshot(HttpRequest::get(path).body(Body::empty())?)
                .await?;
            assert_eq!(response.status(), StatusCode::OK);
            ensure!(
                response
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .is_some_and(|value| value.starts_with(content_type))
            );
        }
        let response = app
            .clone()
            .oneshot(HttpRequest::get("/api/voices").body(Body::empty())?)
            .await?;
        let body = to_bytes(response.into_body(), MAX_HTTP_BODY_BYTES).await?;
        let voices: Value = from_slice(&body)?;
        assert_eq!(
            voices.pointer("/0/id").and_then(Value::as_str),
            Some("test_voice")
        );
        assert!(voices.pointer("/0/path").is_none());

        verify_error_responses(app.clone()).await?;

        let response = app
            .clone()
            .oneshot(
                HttpRequest::post("/v1/audio/speech")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"input":"你好。","voice":"test_voice"}"#))?,
            )
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("audio/wav")
        );
        let wav = to_bytes(response.into_body(), 10_000_000).await?;
        assert!(wav.starts_with(b"RIFF"));
        assert!(wav.len() > 44);
        verify_streaming_routes(app).await
    }

    async fn verify_error_responses(app: Router) -> Result<()> {
        for (payload, status) in [
            (
                r#"{"input":" ","voice":"test_voice"}"#.to_owned(),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                r#"{"input":"Hello","voice":"../secret"}"#.to_owned(),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
            (
                r#"{"input":"Hello","voice":"missing_voice"}"#.to_owned(),
                StatusCode::NOT_FOUND,
            ),
            ("{".to_owned(), StatusCode::BAD_REQUEST),
            (
                "x".repeat(MAX_HTTP_BODY_BYTES + 1),
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    HttpRequest::post("/v1/audio/speech")
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(payload))?,
                )
                .await?;
            assert_eq!(response.status(), status);
            let body = to_bytes(response.into_body(), MAX_HTTP_BODY_BYTES).await?;
            let error: Value = from_slice(&body)?;
            assert!(error.get("error").and_then(Value::as_str).is_some());
        }
        Ok(())
    }

    async fn verify_streaming_routes(app: Router) -> Result<()> {
        let make_request = || {
            HttpRequest::post("/api/speech/stream")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"input":"你好。[[pause=750]][[speed=0.9]]再见。","voice":"test_voice"}"#,
                ))
        };
        let pending = app.clone().oneshot(make_request()?).await?;
        assert_eq!(pending.status(), StatusCode::OK);
        let competing = app.clone().oneshot(make_request()?).await?;
        assert_eq!(competing.status(), StatusCode::SERVICE_UNAVAILABLE);
        drop(pending);
        let response = app.oneshot(make_request()?).await?;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 4_000_000).await?;
        let events = bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(from_slice::<Value>)
            .collect::<Result<Vec<_>, _>>()?;
        let chunks: Vec<_> = events
            .iter()
            .filter(|event| event.get("event").and_then(Value::as_str) == Some("chunk"))
            .collect();
        assert_eq!(chunks.len(), 2);
        let mut samples = 0;
        for (index, chunk) in chunks.into_iter().enumerate() {
            assert_eq!(
                chunk.get("utteranceIndex").and_then(Value::as_u64),
                Some(u64::try_from(index)?)
            );
            let encoded = chunk
                .get("pcmBase64")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("missing PCM"))?;
            let pcm = STANDARD.decode(encoded)?;
            samples += pcm.len() / 2;
            if index == 1 {
                let leading_bytes = 22050 * 750 / 1000 * 2;
                assert!(
                    pcm.get(..leading_bytes)
                        .is_some_and(|bytes| bytes.iter().all(|byte| *byte == 0))
                );
            }
        }
        let done = events
            .last()
            .ok_or_else(|| anyhow::anyhow!("missing done event"))?;
        assert_eq!(done.get("event").and_then(Value::as_str), Some("done"));
        assert_eq!(
            done.get("samples").and_then(Value::as_u64),
            Some(u64::try_from(samples)?)
        );
        Ok(())
    }
}

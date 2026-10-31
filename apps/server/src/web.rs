//! Browser assets, registered voice discovery, and bounded speech synthesis requests.

use std::{
    fmt::Display, io::Error as IoError, num::NonZeroU32, sync::Arc, time::Duration, vec::IntoIter,
};

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
use index_tts_rs_core::{
    config::{
        DurationFactor, EngineConfig, Language, SpeakerSource, SynthesisRequest, SynthesisText,
        VoiceId,
    },
    dsp::encode_wav_i16_bytes,
    engine::{AudioChunk, IndexTtsEngine, SynthesisStream},
    error::IndexTtsError,
};
use index_tts_rs_director::{
    Director, DirectorConfig, DirectorError, PlannedUtterance, plan_utterances, prepend_silence,
};
use serde::{Deserialize, Deserializer, Serialize, de::Error};
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
const INDEX_HTML: &str = include_str!("../static/index.html");
const APP_CSS: &str = include_str!("../static/app.css");
const APP_JS: &str = include_str!("../static/app.js");

#[derive(Debug, Clone)]
struct AppState {
    engine: Arc<IndexTtsEngine>,
    synthesis: Arc<Semaphore>,
    director_config: Arc<DirectorConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SpeechPayload {
    input: SynthesisText,
    voice: VoiceId,
    #[serde(default)]
    lang: Option<Language>,
    #[serde(default, rename = "speed", deserialize_with = "deserialize_speed")]
    duration_factor: DurationFactor,
    #[serde(default)]
    paragraph_pause_ms: ParagraphPause,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(try_from = "u32")]
struct ParagraphPause(u32);

impl Default for ParagraphPause {
    fn default() -> Self {
        Self(600)
    }
}

impl TryFrom<u32> for ParagraphPause {
    type Error = &'static str;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value > 5000 {
            return Err("paragraphPauseMs must be between 0 and 5000");
        }
        Ok(Self(value))
    }
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
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn internal(err: impl Display) -> Self {
        error!(error = %err, "speech studio request failed");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "语音服务处理失败，请检查服务日志后重试。",
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

impl From<IndexTtsError> for ApiError {
    fn from(err: IndexTtsError) -> Self {
        match err {
            IndexTtsError::Validation { .. } => {
                Self::new(StatusCode::UNPROCESSABLE_ENTITY, err.to_string())
            }
            IndexTtsError::VoiceNotFound { .. } => Self::new(
                StatusCode::NOT_FOUND,
                "找不到这个音色，请刷新音色列表后重试。",
            ),
            _ => Self::internal(err),
        }
    }
}

impl From<DirectorError> for ApiError {
    fn from(error: DirectorError) -> Self {
        match error {
            DirectorError::Invalid(_) | DirectorError::Speech(_) => {
                Self::new(StatusCode::UNPROCESSABLE_ENTITY, error.to_string())
            }
            DirectorError::MissingKey => Self::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "请先在启动 server 的终端配置 DEEPSEEK_API_KEY。",
            ),
            _ => {
                error!(error = %error, "speech annotation failed");
                Self::new(
                    StatusCode::BAD_GATEWAY,
                    "AI 标注失败，请检查模型配置或稍后重试；原文未改动。",
                )
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AnnotationPayload {
    input: SynthesisText,
    #[serde(default)]
    poetry: bool,
}

#[instrument(skip(state, payload))]
async fn annotate(
    State(state): State<AppState>,
    payload: Result<Json<AnnotationPayload>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(payload) =
        payload.map_err(|error| ApiError::new(error.status(), error.body_text()))?;
    let config = state.director_config.as_ref().clone();
    let director = spawn_blocking(move || Director::from_env(config))
        .await
        .map_err(ApiError::internal)??;
    let script = director
        .annotate(payload.input.as_str(), payload.poetry)
        .await?;
    Ok(Json(json!({"script": script})))
}

impl SpeechPayload {
    fn into_request(self, config: &EngineConfig) -> Result<SynthesisRequest, ApiError> {
        let mut cfm = config.cfm.clone();
        cfm.duration_factor = self.duration_factor;
        let token_limit = NonZeroU32::new(
            u32::try_from(config.max_text_tokens_per_segment).map_err(ApiError::internal)?,
        )
        .ok_or_else(|| ApiError::internal("text token limit must not be zero"))?;
        Ok(SynthesisRequest::builder()
            .text(self.input)
            .speaker(SpeakerSource::VoiceId(self.voice))
            .language(self.lang.unwrap_or(config.default_language))
            .sampling(config.sampling.clone())
            .cfm(cfm)
            .max_text_tokens_per_segment(token_limit)
            .interval_silence_ms(config.interval_silence_ms)
            .build())
    }
}

fn deserialize_speed<'de, D>(deserializer: D) -> Result<DurationFactor, D::Error>
where
    D: Deserializer<'de>,
{
    let speed = Option::<f32>::deserialize(deserializer)?.unwrap_or(1.0);
    if !speed.is_finite() || !(0.25..=4.0).contains(&speed) {
        return Err(D::Error::custom("speed must be between 0.25 and 4.0"));
    }
    DurationFactor::new(speed.recip()).map_err(D::Error::custom)
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
        .fallback(|| async { ApiError::new(StatusCode::NOT_FOUND, "not found") })
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
            Err(_) => ApiError::new(StatusCode::GATEWAY_TIMEOUT, "生成超时，请缩短文本后重试。")
                .into_response(),
        },
        Err(_) => {
            ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "服务繁忙，请稍后重试。").into_response()
        }
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
    let entries = spawn_blocking(move || store.list().map_err(ApiError::from))
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
    payload: Result<Json<SpeechPayload>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(payload) = payload.map_err(|err| ApiError::new(err.status(), err.body_text()))?;
    let mut session = ReadingSession::start(state, payload)?;
    let sample_rate = session.engine.config().sample_rate;
    let mut pcm = Vec::new();
    while let Some(packet) = session.next_chunk().await? {
        pcm.extend(packet.pcm);
    }
    if pcm.is_empty() {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "没有生成可播放的语音，请检查文本。",
        ));
    }
    let wav =
        spawn_blocking(move || encode_wav_i16_bytes(&pcm, sample_rate).map_err(ApiError::from))
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
struct ReadingCursor {
    requests: IntoIter<PlannedUtterance>,
    active: Option<SynthesisStream>,
    paragraph_index: usize,
    total_paragraphs: usize,
    active_pause_ms: u32,
}

#[derive(Debug)]
struct ReadingSession {
    engine: Arc<IndexTtsEngine>,
    cursor: ReadingCursor,
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
    fn start(state: AppState, payload: SpeechPayload) -> Result<Self, ApiError> {
        let permit = state.synthesis.try_acquire_owned().map_err(|_| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "已有语音正在生成，请稍后重试。",
            )
        })?;
        let pause_ms = payload.paragraph_pause_ms.0;
        let base = payload.into_request(state.engine.config())?;
        let requests = plan_utterances(&base, pause_ms)?;
        let total_paragraphs = requests.len();
        Ok(Self {
            engine: state.engine,
            cursor: ReadingCursor {
                requests: requests.into_iter(),
                active: None,
                paragraph_index: 0,
                total_paragraphs,
                active_pause_ms: 0,
            },
            _permit: permit,
            started: Instant::now(),
            samples: 0,
        })
    }

    async fn next_chunk(&mut self) -> Result<Option<PcmPacket>, ApiError> {
        loop {
            if let Some(active) = &self.cursor.active {
                let next = timeout_at(self.started + REQUEST_TIMEOUT, active.next_chunk())
                    .await
                    .map_err(|_| {
                        ApiError::new(
                            StatusCode::GATEWAY_TIMEOUT,
                            "生成超时，请分段提交较长文本。",
                        )
                    })?;
                match next {
                    Some(chunk) => return self.packet(chunk?).map(Some),
                    None => self.cursor.active = None,
                }
            } else if let Some(part) = self.cursor.requests.next() {
                self.cursor.paragraph_index = self
                    .cursor
                    .total_paragraphs
                    .saturating_sub(self.cursor.requests.len())
                    .saturating_sub(1);
                self.cursor.active_pause_ms = part.pause_before_ms;
                self.cursor.active = Some(self.engine.synthesize_stream(part.request)?);
            } else {
                return Ok(None);
            }
        }
    }

    fn packet(&mut self, chunk: AudioChunk) -> Result<PcmPacket, ApiError> {
        let pause_ms = if chunk.segment_index > 0 {
            self.engine.config().interval_silence_ms
        } else {
            self.cursor.active_pause_ms
        };
        let mut pcm = chunk.pcm_i16;
        prepend_silence(&mut pcm, chunk.sample_rate, pause_ms)?;
        self.samples = self.samples.saturating_add(pcm.len());
        let max_samples = usize::try_from(chunk.sample_rate)
            .map_err(ApiError::internal)?
            .saturating_mul(600);
        if self.samples > max_samples {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "音频超过十分钟，请分段提交。",
            ));
        }
        Ok(PcmPacket {
            pcm,
            paragraph_index: self.cursor.paragraph_index,
            total_paragraphs: self.cursor.total_paragraphs,
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
            "paragraphs": self.cursor.total_paragraphs })
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
    payload: Result<Json<SpeechPayload>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(payload) = payload.map_err(|err| ApiError::new(err.status(), err.body_text()))?;
    let session = ReadingSession::start(state, payload)?;
    let stream = unfold(Some(session), |session| async move {
        let mut session = session?;
        match session.next_chunk().await {
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
                event_bytes(&json!({"event": "error", "error": error.message})),
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
        config::{EngineConfig, Language, SamplingParams, VoiceId},
        engine::IndexTtsEngine,
        voice::SpeakerContext,
    };
    use index_tts_rs_director::DirectorConfig;
    use rstest::rstest;
    use serde_json::{Value, from_slice, from_str, from_value, json};
    use tempfile::tempdir;
    use tokio::{sync::Semaphore, task::spawn_blocking};
    use tower::ServiceExt;

    use super::{MAX_HTTP_BODY_BYTES, SpeechPayload, guard_request, router};

    #[rstest]
    #[case(r#"{"input":"你好","voice":"tyr"}"#, 1.0)]
    #[case(r#"{"input":"Hello","voice":"tyr","lang":"en","speed":2.0}"#, 0.5)]
    #[case(r#"{"input":"你好","voice":"tyr","speed":0.25}"#, 4.0)]
    fn test_should_validate_speech_payload_and_convert_speed(
        #[case] json: &str,
        #[case] duration: f32,
    ) -> Result<()> {
        let payload: SpeechPayload = from_str(json)?;
        assert!((payload.duration_factor.get() - duration).abs() < f32::EPSILON);
        Ok(())
    }

    #[rstest]
    #[case(r#"{"input":" ","voice":"tyr"}"#)]
    #[case(r#"{"input":"你好","voice":"../secret"}"#)]
    #[case(r#"{"input":"你好","voice":"tyr","speed":0}"#)]
    #[case(r#"{"input":"你好","voice":"tyr","speed":5}"#)]
    #[case(r#"{"input":"你好","voice":"tyr","lang":"invalid"}"#)]
    #[case(r#"{"input":"你好","voice":"tyr","extra":"ignored"}"#)]
    fn test_should_reject_invalid_speech_payload(#[case] json: &str) {
        assert!(from_str::<SpeechPayload>(json).is_err());
    }

    #[test]
    fn test_should_reject_oversized_utf8_text() {
        let payload = json!({ "input": "你".repeat(5462), "voice": "tyr" });
        assert!(from_value::<SpeechPayload>(payload).is_err());
    }

    #[test]
    fn test_should_preserve_server_synthesis_configuration() -> Result<()> {
        let mut config = EngineConfig::default();
        config.default_language = Language::En;
        config.max_text_tokens_per_segment = 60;
        config.interval_silence_ms = 450;
        config.cfm.n_timesteps = 12;
        config.sampling.greedy = true;
        let payload: SpeechPayload = from_str(r#"{"input":"Hello","voice":"tyr"}"#)?;
        let request = payload
            .into_request(&config)
            .map_err(|err| anyhow::anyhow!("{}", err.message))?;
        assert_eq!(request.language, Language::En);
        assert_eq!(request.max_text_tokens_per_segment.get(), 60);
        assert_eq!(request.interval_silence_ms, 450);
        assert_eq!(request.cfm.n_timesteps, 12);
        assert!(request.sampling.greedy);
        Ok(())
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

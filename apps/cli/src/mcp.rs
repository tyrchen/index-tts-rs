//! Model Context Protocol server over stdio (`itts mcp`), built on `rmcp`.
//!
//! The MCP process never loads models itself: every tool call goes through the warm daemon,
//! which is started on first use. Any number of agent sessions therefore share one resident
//! model, and `initialize` answers immediately. Tool results carry `structuredContent`; failures
//! are `isError` results whose structured body is a [`ServiceError`] (`code`, `message`, `hint`).

use std::{env, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use index_tts_rs_core::{
    config::{EngineConfig, VoiceId},
    voice::{VoiceStore, default_voices_dir},
};
use index_tts_rs_director::{Director, DirectorConfig};
use index_tts_rs_service::{
    Catalog, ErrorCode, JobEvent, JobSummary, ServiceError, SpeechJob, catalog, resolve_input_path,
};
use rmcp::{
    ErrorData, RoleServer, ServerHandler, ServiceExt,
    handler::server::tool::{schema_for_input, schema_for_output},
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, Implementation, ListToolsResult,
        PaginatedRequestParams, ProgressNotificationParam, ServerCapabilities, ServerConfig, Tool,
        ToolAnnotations,
    },
    schemars::JsonSchema,
    service::RequestContext,
    transport::stdio,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json, to_value};
use tokio::{sync::mpsc, task::spawn_blocking};

use crate::{
    GlobalPaths,
    commands::doctor::{self, DoctorReport},
    daemon::{
        DEFAULT_IDLE_TIMEOUT_SECS, DEFAULT_MAX_IN_FLIGHT, DaemonPaths,
        client::{DaemonClient, VoiceAddRequest},
        lifecycle::{LaunchSpec, spawn_detached},
    },
    player::Playback,
};

const INSTRUCTIONS: &str = "Expressive text-to-speech and voice cloning (IndexTTS-2.5). Call \
                            `status` first to see registered voices and readiness, and \
                            `describe_controls` for inline control syntax ([[pause=600]], \
                            [[speed=0.9]], [[emo=calm:0.6]], [[temp=0.7]]) and emotion names. \
                            `synthesize` writes a WAV (outputPath required); `speak` plays aloud. \
                            Relative paths resolve against the server's working directory. Errors \
                            are structured {code, message, hint}; follow the hint.";

/// Arguments for `register_voice`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RegisterVoiceArgs {
    /// Voice identifier (`^[a-zA-Z0-9_-]{1,64}$`).
    voice_id: String,
    /// Reference audio: 3-15 s of clean speech (`.wav`/`.mp3`/`.flac`/`.m4a`).
    ref_audio: PathBuf,
    /// Optional separate emotion reference audio.
    #[serde(default)]
    emo_ref: Option<PathBuf>,
    /// Emotion blending weight (`0.0..=2.0`, default `1.0`).
    #[serde(default)]
    emo_alpha: Option<f32>,
    /// Emotion mix baked into the snapshot, e.g. `calm:0.6`.
    #[serde(default)]
    emo: Option<String>,
}

/// Arguments naming one voice.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct VoiceIdArgs {
    /// Voice identifier.
    voice_id: String,
}

/// Arguments for `annotate`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AnnotateArgs {
    /// Source text; it is never rewritten, only control tags are added.
    input: String,
    /// Use poetry pacing.
    #[serde(default)]
    poetry: bool,
}

/// No arguments.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

/// The `itts` MCP server.
#[derive(Debug, Clone)]
struct IttsMcp {
    config: Arc<EngineConfig>,
    launch: Arc<LaunchSpec>,
    cwd: PathBuf,
}

/// Runs the MCP server on stdio until the client disconnects.
///
/// # Errors
///
/// Returns a transport failure.
pub async fn run(config: EngineConfig, globals: GlobalPaths) -> Result<()> {
    let server = IttsMcp {
        launch: Arc::new(LaunchSpec {
            config: globals.config,
            model_dir: globals.model_dir,
            voices_dir: globals.voices_dir,
            paths: DaemonPaths::default_paths(),
            idle_timeout_secs: DEFAULT_IDLE_TIMEOUT_SECS,
            max_in_flight: DEFAULT_MAX_IN_FLIGHT,
        }),
        config: Arc::new(config),
        cwd: env::current_dir().context("cannot read current directory")?,
    };
    let service = server
        .serve(stdio())
        .await
        .context("MCP initialization failed")?;
    service.waiting().await.context("MCP server task failed")?;
    Ok(())
}

fn tool<In: JsonSchema + 'static>(
    name: &'static str,
    description: &'static str,
    annotations: ToolAnnotations,
) -> Tool {
    let schema = schema_for_input::<In>().unwrap_or_default();
    Tool::new(name, description, schema).with_annotations(annotations)
}

fn tools() -> Vec<Tool> {
    let read_only = ToolAnnotations::new().read_only(true).destructive(false);
    let writes = ToolAnnotations::new().read_only(false).destructive(false);
    vec![
        tool::<SpeechJob>(
            "synthesize",
            "Synthesize speech to a 22.05 kHz mono WAV at outputPath. Supports inline controls \
             and emotion; returns duration, RTF, and the absolute output path.",
            writes.clone(),
        )
        .with_raw_output_schema(schema_for_output::<JobSummary>()),
        tool::<SpeechJob>(
            "speak",
            "Synthesize speech and play it through the default audio output (optionally also \
             saving outputPath). Returns when playback finishes.",
            writes.clone(),
        )
        .with_raw_output_schema(schema_for_output::<JobSummary>()),
        tool::<NoArgs>("list_voices", "List registered voices.", read_only.clone()),
        tool::<VoiceIdArgs>(
            "inspect_voice",
            "Show one voice's metadata.",
            read_only.clone(),
        ),
        tool::<RegisterVoiceArgs>(
            "register_voice",
            "Clone a voice from a 3-15 s reference clip into a reusable voice ID.",
            writes,
        ),
        tool::<VoiceIdArgs>(
            "remove_voice",
            "Delete a registered voice.",
            ToolAnnotations::new().read_only(false).destructive(true),
        ),
        tool::<AnnotateArgs>(
            "annotate",
            "Ask DeepSeek to add pacing/emotion control tags to text without changing its words \
             (requires DEEPSEEK_API_KEY). Feed the result to synthesize.",
            read_only.clone().open_world(true),
        ),
        tool::<NoArgs>(
            "status",
            "Environment and daemon readiness: platform, model, voices, daemon, annotation, audio \
             output, each with a fix hint.",
            read_only.clone(),
        )
        .with_raw_output_schema(schema_for_output::<DoctorReport>()),
        tool::<NoArgs>(
            "describe_controls",
            "Inline control syntax, emotion names, languages, parameter ranges, and error codes.",
            read_only,
        )
        .with_raw_output_schema(schema_for_output::<Catalog>()),
    ]
}

impl ServerHandler for IttsMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("itts", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(tools()))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        tools().into_iter().find(|tool| tool.name == name)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        if self.get_tool(&request.name).is_none() {
            return Err(ErrorData::invalid_params(
                format!("unknown tool `{}`", request.name),
                None,
            ));
        }
        let args = Value::Object(request.arguments.clone().unwrap_or_default());
        let outcome = tokio::select! {
            result = self.dispatch(&request.name, args, &context) => result,
            () = context.ct.cancelled() => {
                Err(ServiceError::new(ErrorCode::Cancelled, "cancelled by client"))
            }
        };
        let result = match outcome {
            Ok(value) => CallToolResult::structured(value),
            Err(error) => CallToolResult::structured_error(to_value(&error).unwrap_or_default()),
        };
        Ok(result.into())
    }
}

impl IttsMcp {
    async fn dispatch(
        &self,
        name: &str,
        args: Value,
        context: &RequestContext<RoleServer>,
    ) -> Result<Value, ServiceError> {
        match name {
            "synthesize" => {
                let job: SpeechJob = parse(name, args)?;
                if job.output_path.is_none() {
                    return Err(ServiceError::invalid("outputPath is required")
                        .with_hint("use `speak` to play without saving"));
                }
                json_value(&self.synthesize(job, false, context).await?)
            }
            "speak" => json_value(&self.synthesize(parse(name, args)?, true, context).await?),
            "list_voices" => {
                parse::<NoArgs>(name, args)?;
                Ok(json!({ "voices": self.store()?.list()? }))
            }
            "inspect_voice" => {
                let args: VoiceIdArgs = parse(name, args)?;
                json_value(&self.store()?.inspect(&VoiceId::new(args.voice_id)?)?)
            }
            "remove_voice" => {
                let args: VoiceIdArgs = parse(name, args)?;
                let id = VoiceId::new(args.voice_id)?;
                if !self.store()?.remove(&id)? {
                    return Err(ServiceError::voice_not_found(id.as_str()));
                }
                Ok(json!({ "voiceId": id.as_str(), "removed": true }))
            }
            "register_voice" => {
                let args: RegisterVoiceArgs = parse(name, args)?;
                let request = VoiceAddRequest {
                    voice_id: args.voice_id,
                    ref_audio: resolve_input_path(&self.cwd, &args.ref_audio, "refAudio")?,
                    emo_ref: args
                        .emo_ref
                        .map(|path| resolve_input_path(&self.cwd, &path, "emoRef"))
                        .transpose()?,
                    emo_alpha: args.emo_alpha,
                    emo: args.emo,
                };
                json_value(&self.client().await?.voice_add(request).await?)
            }
            "annotate" => {
                let args: AnnotateArgs = parse(name, args)?;
                let director = Director::from_env(DirectorConfig::default())?;
                let script = director.annotate(&args.input, args.poetry).await?;
                Ok(json!({ "script": script }))
            }
            "status" => {
                parse::<NoArgs>(name, args)?;
                json_value(&doctor::collect(&self.config, &self.launch.paths.socket).await)
            }
            "describe_controls" => {
                parse::<NoArgs>(name, args)?;
                json_value(&catalog())
            }
            other => Err(ServiceError::invalid(format!("unknown tool `{other}`"))),
        }
    }

    async fn synthesize(
        &self,
        mut job: SpeechJob,
        play: bool,
        context: &RequestContext<RoleServer>,
    ) -> Result<JobSummary, ServiceError> {
        job.resolve_paths(&self.cwd)?;
        let playback = if play { Some(Playback::start()?) } else { None };
        let mut client = self.client().await?;

        let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<(f64, f64)>();
        let synth = async {
            // Owning the sender here closes the channel when synthesis ends, which ends `forward`.
            let progress_tx = progress_tx;
            let mut done = 0.0_f64;
            client
                .synth(&job, play, |event| {
                    if let JobEvent::Chunk(info) = event {
                        done += 1.0;
                        let total = (info.total_segments * info.total_utterances).max(1);
                        #[allow(clippy::cast_precision_loss, reason = "segment counts are small")]
                        let _ = progress_tx.send((done, total as f64));
                        if let Some(playback) = &playback {
                            let chunk = info.into_job_chunk()?;
                            playback.enqueue(chunk.audio.pcm_i16, chunk.audio.sample_rate)?;
                        }
                    }
                    Ok(())
                })
                .await
        };
        let token = context.meta.get_progress_token();
        let forward = async {
            while let Some((done, total)) = progress_rx.recv().await {
                if let Some(token) = &token {
                    let _ = context
                        .peer
                        .notify_progress(
                            ProgressNotificationParam::new(token.clone(), done)
                                .with_total(total.max(done))
                                .with_message(format!("synthesized segment {done:.0}")),
                        )
                        .await;
                }
            }
        };
        let (summary, ()) = tokio::join!(synth, forward);
        let summary = summary?;
        if let Some(playback) = playback {
            spawn_blocking(move || playback.finish())
                .await
                .map_err(|err| ServiceError::internal(format!("playback task failed: {err}")))??;
        }
        Ok(summary)
    }

    async fn client(&self) -> Result<DaemonClient, ServiceError> {
        if let Some(client) = DaemonClient::connect(&self.launch.paths.socket).await? {
            return Ok(client);
        }
        spawn_detached(&self.launch, |_| {}).await?;
        DaemonClient::connect(&self.launch.paths.socket)
            .await?
            .ok_or_else(|| {
                ServiceError::new(
                    ErrorCode::DaemonUnavailable,
                    "daemon started but is not reachable",
                )
                .with_hint("check `itts daemon logs`")
            })
    }

    fn store(&self) -> Result<VoiceStore, ServiceError> {
        Ok(VoiceStore::new(
            self.config
                .voices_dir
                .clone()
                .unwrap_or_else(default_voices_dir),
        )?)
    }
}

fn parse<T: DeserializeOwned>(tool: &str, args: Value) -> Result<T, ServiceError> {
    serde_json::from_value(args)
        .map_err(|err| ServiceError::invalid(format!("invalid arguments for `{tool}`: {err}")))
}

fn json_value<T: Serialize>(value: &T) -> Result<Value, ServiceError> {
    to_value(value).map_err(|err| ServiceError::internal(err.to_string()))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::tools;

    #[test]
    fn test_should_expose_unique_tools_with_object_schemas() {
        let tools = tools();
        let names: HashSet<_> = tools.iter().map(|t| t.name.to_string()).collect();
        assert_eq!(names.len(), tools.len());
        for expected in [
            "synthesize",
            "speak",
            "status",
            "describe_controls",
            "register_voice",
        ] {
            assert!(names.contains(expected), "missing tool {expected}");
        }
        for tool in &tools {
            assert_eq!(
                tool.input_schema.get("type").and_then(|t| t.as_str()),
                Some("object"),
                "tool {} must take an object",
                tool.name
            );
        }
    }
}

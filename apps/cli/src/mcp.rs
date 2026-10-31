//! Model Context Protocol (MCP) JSON-RPC 2.0 server over `stdin`/`stdout` (`index-tts mcp`).
//!
//! Exposes 4 standard MCP tools for AI coding agents (Claude Code, Cursor, Antigravity):
//! - `speak`: Synthesize speech and play directly through macOS `CoreAudio`.
//! - `synthesize_to_file`: Synthesize speech and save to a `.wav` file on disk.
//! - `register_voice`: Pre-extract a reference audio file into a `.itts_voice` snapshot.
//! - `list_voices`: List all registered `.itts_voice` snapshots in the store.

use std::{path::PathBuf, str::FromStr};

use anyhow::{Context, Result};
use index_tts_rs_core::{
    config::{
        CfmParams, DurationFactor, EmoAlpha, EngineConfig, Language, SpeakerSource,
        SynthesisRequest, SynthesisText, VoiceExtractRequest, VoiceId,
    },
    engine::IndexTtsEngine,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::player::AudioPlayer;

/// Runs the MCP JSON-RPC 2.0 server over `stdin` and `stdout`.
///
/// # Errors
///
/// Returns an error if `IndexTtsEngine` initialization or stdio I/O fails.
pub async fn run_mcp_stdio_server(config: EngineConfig) -> Result<()> {
    let engine = IndexTtsEngine::start(config)
        .context("failed to start IndexTtsEngine for MCP stdio server")?;

    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut lines = BufReader::new(stdin).lines();

    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(resp) = handle_mcp_jsonrpc_line(&engine, trimmed).await {
            let mut out = serde_json::to_vec(&resp)?;
            out.push(b'\n');
            stdout.write_all(&out).await?;
            stdout.flush().await?;
        }
    }

    engine.stop().context("failed to stop IndexTtsEngine")?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsonRpcError {
    code: i32,
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SpeakArgs {
    text: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default, alias = "ref_audio_path")]
    ref_audio_path: Option<PathBuf>,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SynthesizeToFileArgs {
    text: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default, alias = "ref_audio_path")]
    ref_audio_path: Option<PathBuf>,
    #[serde(alias = "output_path")]
    output_path: PathBuf,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RegisterVoiceArgs {
    #[serde(alias = "voice_id")]
    voice_id: String,
    #[serde(alias = "ref_audio_path")]
    ref_audio_path: PathBuf,
    #[serde(default, alias = "emo_audio_path")]
    emo_audio_path: Option<PathBuf>,
    #[serde(default, alias = "emo_alpha")]
    emo_alpha: Option<f32>,
}

/// Processes a single JSON-RPC 2.0 request line and returns an optional [`JsonRpcResponse`]
/// (`None` for notifications without an `id`).
pub async fn handle_mcp_jsonrpc_line(
    engine: &IndexTtsEngine,
    line: &str,
) -> Option<JsonRpcResponse> {
    let req: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(err) => {
            return Some(JsonRpcResponse {
                jsonrpc: "2.0",
                id: None,
                result: None,
                error: Some(JsonRpcError {
                    code: -32700,
                    message: format!("Parse error: {err}"),
                }),
            });
        }
    };

    // Notifications (e.g., `notifications/initialized`) have no `id` and expect no reply
    if req.id.is_none() || req.method.starts_with("notifications/") {
        return None;
    }

    let id = req.id;
    match req.method.as_str() {
        "initialize" => Some(JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "index-tts",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
            error: None,
        }),
        "ping" => Some(JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({})),
            error: None,
        }),
        "tools/list" => Some(JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(mcp_tools_schema()),
            error: None,
        }),
        "tools/call" => {
            let tool_name = req
                .params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let args = req
                .params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));

            match dispatch_tool_call(engine, tool_name, args).await {
                Ok(payload) => Some(JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string(&payload).unwrap_or_default()
                        }],
                        "isError": false
                    })),
                    error: None,
                }),
                Err(err) => Some(JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [{
                            "type": "text",
                            "text": format!("Error: {err}")
                        }],
                        "isError": true
                    })),
                    error: None,
                }),
            }
        }
        other => Some(JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {other}"),
            }),
        }),
    }
}

async fn dispatch_tool_call(
    engine: &IndexTtsEngine,
    tool_name: &str,
    args: Value,
) -> Result<Value> {
    match tool_name {
        "speak" => {
            let parsed: SpeakArgs =
                serde_json::from_value(args).context("invalid arguments for `speak`")?;
            let synth_req = build_mcp_synthesis_request(
                &parsed.text,
                parsed.voice.as_deref(),
                parsed.ref_audio_path,
                parsed.lang.as_deref(),
                parsed.speed,
            )?;
            let res = engine.synthesize(synth_req).await?;
            let played = if let Ok(player) = AudioPlayer::try_new() {
                let _ = player.enqueue_pcm_i16(&res.pcm_i16, res.sample_rate);
                player.drain();
                true
            } else {
                false
            };
            Ok(json!({
                "status": "ok",
                "played": played,
                "durationSecs": res.duration_secs,
                "synthesisSecs": res.synthesis_secs,
                "rtf": res.rtf,
                "segmentsCount": res.segments_count,
                "timings": res.timings
            }))
        }
        "synthesize_to_file" => {
            let parsed: SynthesizeToFileArgs = serde_json::from_value(args)
                .context("invalid arguments for `synthesize_to_file`")?;
            let synth_req = build_mcp_synthesis_request(
                &parsed.text,
                parsed.voice.as_deref(),
                parsed.ref_audio_path,
                parsed.lang.as_deref(),
                parsed.speed,
            )?;
            let res = engine.synthesize(synth_req).await?;
            res.save_wav(&parsed.output_path)?;
            Ok(json!({
                "status": "ok",
                "outputPath": parsed.output_path,
                "durationSecs": res.duration_secs,
                "synthesisSecs": res.synthesis_secs,
                "rtf": res.rtf,
                "segmentsCount": res.segments_count,
                "timings": res.timings
            }))
        }
        "register_voice" => {
            let parsed: RegisterVoiceArgs =
                serde_json::from_value(args).context("invalid arguments for `register_voice`")?;
            let voice_id = VoiceId::new(parsed.voice_id)?;
            let emo_alpha = EmoAlpha::new(parsed.emo_alpha.unwrap_or(1.0))?;
            let req = VoiceExtractRequest::builder()
                .voice_id(voice_id)
                .ref_audio_path(parsed.ref_audio_path)
                .emo_audio_path(parsed.emo_audio_path)
                .emo_alpha(emo_alpha)
                .build();
            let entry = engine.extract_voice(req).await?;
            let _ = engine.evict_voice_extractor().await;
            Ok(serde_json::to_value(entry)?)
        }
        "list_voices" => {
            let voices = engine.voice_store().list()?;
            Ok(json!({
                "voices": voices
            }))
        }
        other => Err(anyhow::anyhow!("unknown MCP tool `{other}`")),
    }
}

fn build_mcp_synthesis_request(
    raw_text: &str,
    voice: Option<&str>,
    ref_audio_path: Option<PathBuf>,
    lang: Option<&str>,
    speed: Option<f32>,
) -> Result<SynthesisRequest> {
    let text = SynthesisText::new(raw_text)?;
    let speaker = match (voice, ref_audio_path) {
        (Some(v), _) => {
            let path = PathBuf::from(v);
            if path.extension().and_then(|e| e.to_str()) == Some("itts_voice") {
                SpeakerSource::VoiceFile(path)
            } else {
                SpeakerSource::VoiceId(VoiceId::new(v)?)
            }
        }
        (None, Some(ref_path)) => SpeakerSource::ReferenceAudio(ref_path),
        (None, None) => {
            return Err(anyhow::anyhow!(
                "either `voice` or `ref_audio_path` must be provided"
            ));
        }
    };
    let language = match lang {
        Some(l) => Language::from_str(l)?,
        None => Language::Zh,
    };
    let duration_factor = match speed {
        Some(s) if s > 0.0 => DurationFactor::new(1.0 / s)?,
        _ => DurationFactor::default(),
    };
    let cfm = CfmParams::builder()
        .duration_factor(duration_factor)
        .build();

    Ok(SynthesisRequest::builder()
        .text(text)
        .speaker(speaker)
        .language(language)
        .cfm(cfm)
        .build())
}

fn mcp_tools_schema() -> Value {
    json!({
        "tools": [
            {
                "name": "speak",
                "description": "Synthesize speech from text using IndexTTS-2.5 on Apple Silicon and play it aloud immediately.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "Text to speak" },
                        "voice": { "type": "string", "description": "Registered voice ID or .itts_voice path" },
                        "refAudioPath": { "type": "string", "description": "Optional reference audio file path (.wav/.mp3)" },
                        "lang": { "type": "string", "description": "Language code (zh, en, ja, yue, es, ar, zh/en)" },
                        "speed": { "type": "number", "description": "Speaking speed multiplier (default 1.0)" }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "synthesize_to_file",
                "description": "Synthesize speech from text and save as a 22050 Hz 16-bit mono WAV file.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "Text to synthesize" },
                        "voice": { "type": "string", "description": "Registered voice ID or .itts_voice path" },
                        "refAudioPath": { "type": "string", "description": "Optional reference audio file path (.wav/.mp3)" },
                        "outputPath": { "type": "string", "description": "Output WAV file path" },
                        "lang": { "type": "string", "description": "Language code (zh, en, ja, yue, es, ar, zh/en)" },
                        "speed": { "type": "number", "description": "Speaking speed multiplier (default 1.0)" }
                    },
                    "required": ["text", "outputPath"]
                }
            },
            {
                "name": "register_voice",
                "description": "Extract speaker conditioning tensors from a reference audio file and register a persistent .itts_voice snapshot.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "voiceId": { "type": "string", "description": "Voice identifier (^[a-zA-Z0-9_-]{1,64}$)" },
                        "refAudioPath": { "type": "string", "description": "Reference audio file path (3s..15s)" },
                        "emoAudioPath": { "type": "string", "description": "Optional separate emotion reference audio file" },
                        "emoAlpha": { "type": "number", "description": "Emotion blending weight in 0.0..=2.0 (default 1.0)" }
                    },
                    "required": ["voiceId", "refAudioPath"]
                }
            },
            {
                "name": "list_voices",
                "description": "List all registered .itts_voice speaker snapshots and their metadata.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    })
}

#[cfg(test)]
mod tests {
    use index_tts_rs_core::models::resolve_model_dir;

    use super::*;

    #[test]
    fn test_should_expose_four_mcp_tools_in_schema() {
        let schema = mcp_tools_schema();
        let tools = schema
            .get("tools")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        assert_eq!(tools, 4);
    }

    #[tokio::test]
    async fn test_should_handle_mcp_initialize_and_tools_list() {
        if resolve_model_dir(None, "yunfengwang/IndexTTS-2.5-mlx").is_err() {
            return;
        }
        let Ok(temp) = tempfile::tempdir() else {
            return;
        };
        let cfg = EngineConfig::builder()
            .voices_dir(Some(temp.path().to_path_buf()))
            .build();
        let Ok(engine) = IndexTtsEngine::start(cfg) else {
            return;
        };

        let init_resp = handle_mcp_jsonrpc_line(
            &engine,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        )
        .await;
        assert!(init_resp.is_some());

        let notif_resp = handle_mcp_jsonrpc_line(
            &engine,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        )
        .await;
        assert!(notif_resp.is_none());

        let list_resp = handle_mcp_jsonrpc_line(
            &engine,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_voices","arguments":{}}}"#,
        )
        .await;
        assert!(list_resp.is_some());

        let _ = engine.stop();
    }
}

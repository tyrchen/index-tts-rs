# itts Agent Integration Guide

`itts` is designed to be driven by other programs and AI agents. This guide covers the
machine-readable contract: output modes, errors, the warm daemon, and the MCP server. Design
rationale lives in [`specs/itts-agent-usability-design.md`](../../specs/itts-agent-usability-design.md).

## 1. Discover the environment

```bash
itts doctor --output json
```

```json
{
  "version": "0.1.0",
  "ready": true,
  "checks": [
    { "name": "platform", "status": "ok", "summary": "macOS on Apple Silicon (Metal)" },
    { "name": "model", "status": "ok", "summary": "yunfengwang/IndexTTS-2.5-mlx (4.63 GiB) at ~/.cache/..." },
    { "name": "voices", "status": "ok", "summary": "2 in ~/.cache/index-tts-rs/voices — demo_voice, tyr" },
    { "name": "daemon", "status": "warn", "summary": "not running (each synth loads models)",
      "hint": "run `itts daemon start` to keep models warm" }
  ]
}
```

`ready` is `false` only when a check has `status: "error"` (unsupported platform, missing model).
`warn` marks optional capabilities (daemon, DeepSeek key, audio output).

Self-description:

| Command | Content |
| --- | --- |
| `itts schema controls` | Inline control tags, emotion names, languages, speed range, error codes with exit codes |
| `itts schema speech-job` | JSON Schema of the request accepted by the daemon, MCP tools, and HTTP |
| `itts schema events` | JSON Schema of `started` / `chunk` / `done` / `error` events |
| `itts schema error` | JSON Schema of the error body |

## 2. Output modes

Every command accepts `--output text|json|jsonl` (or `ITTS_OUTPUT=json`).

- `text` (default): human-readable results on stdout, progress on stderr.
- `json`: one pretty JSON document on stdout, for both success and failure.
- `jsonl`: compact lines; `synth` streams `started` → `chunk`… → `done`, `models pull` streams
  download `progress`, and `daemon start` streams startup progress followed by `ready`.

With `synth --out -`, stdout carries raw 16-bit little-endian mono PCM at 22050 Hz and the
summary goes to stderr.

## 3. Errors and exit codes

```json
{ "status": "error",
  "error": { "code": "VOICE_NOT_FOUND", "message": "voice `ghost` not found",
             "hint": "run `itts voice list` to see registered voices, or `itts voice add <ID> --ref <AUDIO>` to register one" } }
```

| Exit | Code | Meaning |
| --- | --- | --- |
| 1 | `INTERNAL` | Unexpected failure |
| 2 | `INVALID_ARGUMENT` | Bad flag, value range, control tag, or config (CLI usage errors also exit 2) |
| 3 | `VOICE_NOT_FOUND` | Voice not registered (checked for every `[[voice=...]]` before synthesis starts) |
| 4 | `MODEL_UNAVAILABLE` | Weights missing or unloadable |
| 5 | `AUDIO_DECODE_FAILED` / `AUDIO_OUTPUT_UNAVAILABLE` | Reference audio unreadable / no playback device |
| 6 | `DAEMON_UNAVAILABLE` / `DAEMON_VERSION_MISMATCH` | Daemon unreachable / older protocol (run `itts daemon restart`) |
| 7 | `BUSY` | Daemon already has `--max-in-flight` requests |
| 8 | `CANCELLED` | Client disconnected or cancelled |
| 9 | `TIMEOUT` | Startup, stop, or model loading exceeded its budget |
| 10 | `ANNOTATION_UNAVAILABLE` / `ANNOTATION_FAILED` | `DEEPSEEK_API_KEY` missing / provider failure |
| 11 | `IO` | Filesystem failure |

## 4. Warm daemon

```bash
itts daemon start          # detaches; exits 0 once ready (or already running)
itts daemon ensure         # idempotent start, ideal at the beginning of an agent session
itts daemon status --output json
itts daemon logs -n 100 -f
itts daemon restart        # after upgrading the binary
itts daemon stop
```

- A lock file guarantees one daemon (and one model copy) per socket. The socket is bound before
  models load, so `status` reports `"state": "loading"` and jobs submitted meanwhile wait instead
  of loading their own copy.
- `start` prints each loaded component with its load time, the registered voices, socket, and
  log path. `--foreground` runs it attached (for launchd or debugging).
- The daemon stops after `--idle-timeout-secs` (default 600) without jobs; `status` polling does
  not reset the timer. `SIGTERM`/`SIGINT` shut it down gracefully; `SIGHUP` is ignored.
- Files live next to the socket: `daemon.sock` (mode 0600), `daemon.lock`, `daemon.pid`,
  `daemon.log` (JSON lines, rotated at 10 MiB).
- `synth` and `voice add` use the daemon when present. Relative paths are resolved by the client,
  so they mean the caller's working directory. Disconnecting a client cancels its job at the next
  sentence boundary.

## 5. MCP server

```json
{ "mcpServers": { "itts": { "command": "itts", "args": ["mcp"] } } }
```

The MCP process never loads models; it connects to the daemon and starts it on first use, so
any number of sessions share one model and `initialize` answers immediately.

| Tool | Purpose |
| --- | --- |
| `status` | The `itts doctor` report |
| `describe_controls` | The `itts schema controls` catalog |
| `synthesize` | Write a WAV (`outputPath` required); returns duration, RTF, absolute path |
| `speak` | Play through the default output device |
| `list_voices`, `inspect_voice`, `register_voice`, `remove_voice` | Voice management |
| `annotate` | DeepSeek pacing and emotion tags (text is never rewritten) |

Synthesis tools take `SpeechJob` fields: `input`, `voice` | `voiceFile` | `refAudio`, `emoRef`,
`emoAlpha`, `emo`, `lang`, `speed`, `paragraphPauseMs`, `intervalSilenceMs`, `outputPath`,
`advanced` (`greedy`, `seed`, `temperature`, `topK`, `topP`, `repetitionPenalty`,
`maxMelTokens`, `maxTextTokens`, `nTimesteps`, `cfgRate`). Results include `structuredContent`;
failures are `isError` results whose structured content is the error body above. Requests with a
progress token receive `notifications/progress` per synthesized segment, and MCP cancellation
stops the daemon job.

## 6. Daemon wire protocol

For direct integrations, the socket speaks newline-delimited JSON (protocol version 2):

```text
→ {"id":1,"method":"hello","clientVersion":"0.1.0","protocolVersion":2}
← {"id":1,"event":"welcome","serverVersion":"0.1.0","protocolVersion":2}
→ {"id":2,"method":"synth","job":{"input":"你好","voice":"tyr","outputPath":"/abs/out.wav"},"includePcm":false}
← {"id":2,"event":"started",...}  {"id":2,"event":"chunk",...}  {"id":2,"event":"done",...}
→ {"id":3,"method":"cancel","target":2}
```

Other methods: `status`, `stop`, `voiceAdd`. Paths must be absolute. Requests on one connection
are multiplexed by `id`; lines are limited to 256 KiB.

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
  "version": "0.3.0",
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

With `synth --out -`, stdout carries raw 16-bit little-endian mono PCM (22050 Hz, or 48000 Hz with
`--sample-rate 48000`) and the summary goes to stderr.

## 3. Timestamps, sentences, and reproducible takes

Add `--timestamps` (CLI) or `"timestamps": true` (daemon, MCP, HTTP) to receive when every
character is heard. Timings come from the GPT decoder's own text attention, snapped to the
synthesized waveform's pauses; they cost about 15–45 ms per job and need no extra model.

```bash
itts synth --voice narrator --timestamps --output json \
  --text '大家好！[[pause=400]]今天是2026年10月31日。' --out hello.wav
```

```jsonc
{
  "seed": 17127547225857486438,
  "alignment": {
    "text": "大家好！今天是2026年10月31日。",   // the input without [[...]] controls
    "start": [0.62, 0.79, 0.99, 1.17, 2.40, ...],  // seconds, one per Unicode scalar
    "end":   [0.79, 0.99, 1.17, 2.40, 2.55, ...]
  },
  "sentences": [
    { "text": "大家好！", "chars": { "start": 0, "end": 4 }, "start": 0.62, "end": 1.17 },
    { "text": "今天是2026年10月31日。", "chars": { "start": 4, "end": 19 }, "start": 2.40, "end": 5.20 }
  ]
}
```

- Indices are Unicode scalars: in JavaScript use `Array.from(alignment.text)`.
- Numbers, symbols, and English words are mapped back to the characters you wrote (`2026` gets
  the time of its spoken form). Punctuation and whitespace span the pause between neighbours.
- Accuracy against a forced aligner (MMS): median 16–18 ms, 96–98% of word/character onsets within
  100 ms across Chinese and English, two voices, emotions, and 0.8–1.3× speed. See
  [the research note](../research/gpt-attention-alignment-research.md).

Every job reports the `seed` that drove GPT sampling and CFM noise (in `started` and in the
summary). When `--seed` / `advanced.seed` is omitted a random one is drawn; passing the reported
seed back reproduces the audio bit for bit on the same machine.

`--sample-rate 48000` (`"sampleRate": 48000`) returns 48 kHz audio: the native 22.05 kHz band is
kept and 10–24 kHz is added by a bandwidth extender whose weights `itts models pull` fetches
(`itts doctor` reports them as `hifi`). Durations and timestamps do not change.

Each `chunk` event carries `text` (the part of the input it reads, closing punctuation included;
consecutive chunks partition the text), `textOffset` (its character index), `startSecs` (where
the chunk, including its leading pause, starts in the job audio), and with timestamps an
`alignment` for that chunk on the job timeline.

## 4. Errors and exit codes

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

## 5. Warm daemon

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

## 6. MCP server

```json
{ "mcpServers": { "itts": { "command": "itts", "args": ["mcp"] } } }
```

The MCP process never loads models; it connects to the daemon and starts it on first use, so
any number of sessions share one model and `initialize` answers immediately.

| Tool | Purpose |
| --- | --- |
| `status` | The `itts doctor` report |
| `describe_controls` | The `itts schema controls` catalog |
| `synthesize` | Write a WAV (`outputPath` required); returns duration, RTF, absolute path, seed, and with `timestamps` the alignment and sentences |
| `speak` | Play through the default output device |
| `list_voices`, `inspect_voice`, `register_voice`, `remove_voice` | Voice management |
| `annotate` | DeepSeek pacing and emotion tags (text is never rewritten) |
| `annotate_units` | One delivery per caller-supplied line (`AnnotateJob` → `AnnotateResult`; see the speech-controls guide) |

Synthesis tools take `SpeechJob` fields: `input`, `voice` | `voiceFile` | `refAudio`, `emoRef`,
`emoAlpha`, `emo`, `lang`, `speed`, `paragraphPauseMs`, `intervalSilenceMs`, `outputPath`,
`sampleRate`, `timestamps`, `continuity`, `previousText`, `advanced` (`greedy`, `seed`, `temperature`, `topK`, `topP`, `repetitionPenalty`,
`maxMelTokens`, `maxTextTokens`, `nTimesteps`, `cfgRate`). Results include `structuredContent`;
failures are `isError` results whose structured content is the error body above. Requests with a
progress token receive `notifications/progress` per synthesized segment (the message names the
sentence just finished), and MCP cancellation stops the daemon job.

## 7. Daemon wire protocol

For direct integrations, the socket speaks newline-delimited JSON (protocol version 3):

```text
→ {"id":1,"method":"hello","clientVersion":"0.3.0","protocolVersion":3}
← {"id":1,"event":"welcome","serverVersion":"0.3.0","protocolVersion":3}
→ {"id":2,"method":"synth","job":{"input":"你好","voice":"tyr","outputPath":"/abs/out.wav"},"includePcm":false}
← {"id":2,"event":"started",...}  {"id":2,"event":"chunk",...}  {"id":2,"event":"done",...}
→ {"id":3,"method":"cancel","target":2}
→ {"id":4,"method":"annotate","job":{"units":[{"id":"l1","text":"你好。"}]},"apiKey":"sk-..."}
← {"id":4,"event":"annotated","deliveries":[{"id":"l1","speed":0.95,...}]}
```

`apiKey` is optional (the daemon's `DEEPSEEK_API_KEY` is the fallback) and is never logged; a key
the provider rejects yields `ANNOTATION_UNAVAILABLE`. Other methods: `status`, `stop`, `voiceAdd`. Paths must be absolute. Requests on one connection
are multiplexed by `id`; lines are limited to 256 KiB.

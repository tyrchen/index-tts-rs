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
  "version": "0.3.1",
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
| `itts schema align-job` … `verify-result` | JSON Schemas of the `align`, `listen`, and `verify` requests and results |

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
  "seed": 3071825384613407,
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

## 3a. Aligning any recording (`itts align`)

`itts align` times a recording you already have (another TTS engine, a human take) against the
text it speaks, on the same 0 s as the file (MP3 encoder delay is trimmed exactly as ffmpeg and
browsers do). The alignment has the same shape and filling rules as `synth --timestamps`.

```bash
itts align --audio line.mp3 --lang zh --stdin --phones --output json < say.txt
itts align --audio line.wav --lang en --text "The engine runs hot." --voice-file tyr.itts_voice --output json
```

```jsonc
{
  "status": "ok", "backend": "daemon",
  "aligner": { "name": "itts-attn", "version": 1, "model": "yunfengwang/IndexTTS-2.5-mlx",
               "heads": "L4H10+L5H10", "conditioning": "recording" },
  "alignment": { "text": "鲲化而为鸟。", "start": [0.21, 0.52, …], "end": [0.52, 0.77, …] },
  "sentences": [ { "text": "鲲化而为鸟。", "chars": { "start": 0, "end": 6 }, "start": 0.21, "end": 1.64 } ],
  "quality": { "confidence": 0.99, "coverage": 1.0, "weak": [] },
  "phones": [ { "i": 0, "p": "k", "k": "ini", "start": 0.21, "end": 0.27 },
              { "i": 0, "p": "un", "t": 1, "k": "fin", "start": 0.27, "end": 0.52 } ],
  "phonesMethod": "split",
  "durationSecs": 2.03, "alignSecs": 0.09,
  "timings": { "audioMs": 2, "featuresMs": 21, "encodeMs": 4, "conditionMs": 25, "attentionMs": 35, "totalMs": 88 }
}
```

- Input: WAV / MP3 / FLAC / M4A at any sample rate, at most about 72 s and 600 text tokens
  (otherwise `TOO_LONG`). `<字|pinyin>` / `<word|ARPAbet>` overrides are honoured; `[[...]]`
  controls are removed. Without `--voice` / `--voice-file` the recording itself conditions the
  decoder (the more accurate choice in the evaluation).
- `quality` is computed over characters, words, and overrides. On clean speech `confidence` and
  `coverage` stay above 0.97; below 0.95 the audio most likely does not say the text (omitted,
  extra, or substituted words). `weak` lists characters (a word's first letter) the audio barely
  supports; send those lines to speech recognition. Numbers read in another form that shares
  syllables (`2026年` read 两千零二十六年) are not detected. A result is returned whatever the
  quality, with exit 0.
- `phones` (with `--phones`): Mandarin initials / finals in pinyin (`ü` as `v`, `y`/`w` spellings
  as finals, `t` = tone 1–5), English ARPAbet without stress digits; `i` indexes
  `alignment.text`. They are cut inside each character or word (`phonesMethod: "split"`).
- Accuracy (Apple M5 Pro, 166 lines): Chinese character onsets against MMS 18–29 ms mean error,
  ≥96% within 100 ms for Gemini, ElevenLabs, and Edge speech; English word onsets about 70 ms
  (median −35 ms, early). About 0.11 s per line up to 10 s on a warm daemon. See
  [the evaluation](../research/itts-align-evaluation-research.md).
- `itts schema align-job` / `align-result` print the JSON Schemas; MCP exposes the same as the
  `align` tool.

## 3b. Hearing and checking a recording (`itts listen`, `itts verify`)

`itts listen` transcribes a recording verbatim with a local, non-autoregressive recognizer
(sherpa-onnx; no network after the first-use model download). Unlike Whisper it keeps stutters
and repeats ("从北北边", "the the") and reads numbers back as spoken (no inverse text
normalization). `itts verify` aligns the recording to its script and recognizes it at the same
time, then reports where they differ.

```bash
itts listen line.wav --output json                      # Mandarin (Paraformer)
itts listen take.mp3 --lang en --output json            # English (Parakeet)
itts verify line.mp3 --lang zh --stdin --output json < say.txt
```

```jsonc
{
  "status": "ok", "backend": "daemon",
  "verdict": "error",                                   // clean | info | warning | error
  "findings": [
    { "kind": "repeat", "severity": "error", "expected": "", "heard": "北",
      "chars": { "start": 1, "end": 1 }, "start": 0.50, "end": 0.66, "sounds": 1, "alignWeak": false }
  ],
  "heard": { "recognizer": { "name": "sherpa-onnx", "version": 1, "model": "paraformer", … },
             "text": "从北北边吹来的风越来越冷", "tokens": [ { "text": "从", "start": 0.26, "end": 0.50 }, … ], … },
  "align": { …the `itts align` result for the same recording (no phones)… },
  "durationSecs": 2.55, "verifySecs": 0.17,
  "timings": { "alignMs": 120.4, "listenMs": 95.0, "compareMs": 0.2, "totalMs": 168.9 }
}
```

- Recognizers: `paraformer` (default for `zh`, `zh/en`), `sense-voice` (default for `yue`, `ja`;
  also Mandarin and English), `parakeet` (default for `en`). `es` / `ar` have no local
  recognizer (`INVALID_ARGUMENT`). Each model downloads once from Hugging Face on first use
  (232 / 228 / 631 MiB, pinned revisions, BLAKE3-checked); `itts doctor` shows which are cached.
- Finding kinds: `repeat` (a stutter or re-read), `insertion`, `omission`, `substitution`, and
  `number` (a number read with another value; `2026年` read 两千零二十六年 is the same value
  and not reported). Script characters are compared by toneless pinyin, so a recognizer writing a
  homophone is not a finding; Latin words compare letter by letter.
- Severity: `repeat` and `number` are `error`; two or more differing units are `warning`; a
  single unit is `warning` only when the aligner also finds it weak (`alignWeak`), else `info` —
  most single-character differences on clean speech are the recognizer mishearing rare words or
  product names. The command exits 0 whatever the verdict.
- `chars` indexes `align.alignment.text` (insertions are a point, `start == end`); times come
  from the recognizer tokens, or for omissions from the alignment. `listen` token times are
  approximate (token starts; an end is the next start, at most 0.5 s later).
- Limits: `listen` up to 120 s, `verify` up to the alignment limit (~72 s, 600 tokens).
  Same-character misreadings (行业 read xíng yè) and fragments of the next line after the end
  of a take are not detected.
- Evaluation (352 lines, Apple M5 Pro): every stutter (15/15) and every number read as another
  value (8/8) as `error`, 19/20 omissions, no `warning` on 282 clean lines; about 0.17 s per line
  up to 10 s on a warm daemon. See [the evaluation](../research/itts-asr-evaluation.md).
- `itts schema listen-job` / `listen-result` / `verify-job` / `verify-result` print the JSON
  Schemas; MCP exposes `listen` and `verify`. Builds without the `asr` Cargo feature have
  neither command and answer daemon requests with `MODEL_UNAVAILABLE`.

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
| 4 | `MODEL_UNAVAILABLE` | Weights missing or unloadable (recognizer models: download or check failed; build without `asr`) |
| 5 | `AUDIO_DECODE_FAILED` / `AUDIO_OUTPUT_UNAVAILABLE` | Reference or aligned audio unreadable / no playback device |
| 6 | `DAEMON_UNAVAILABLE` / `DAEMON_VERSION_MISMATCH` | Daemon unreachable / older protocol (run `itts daemon restart`) |
| 7 | `BUSY` | Daemon already has `--max-in-flight` requests |
| 8 | `CANCELLED` | Client disconnected or cancelled |
| 9 | `TIMEOUT` | Startup, stop, or model loading exceeded its budget |
| 10 | `ANNOTATION_UNAVAILABLE` / `ANNOTATION_FAILED` | `DEEPSEEK_API_KEY` missing / provider failure |
| 11 | `IO` | Filesystem failure |
| 12 | `TOO_LONG` | Input beyond one request (`itts align` / `verify`: more than ~72 s of audio or 600 text tokens; `itts listen`: more than 120 s) |

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
- `synth`, `align`, `listen`, `verify`, and `voice add` use the daemon when present. The first
  `align` loads the Stage 0 and codec encoders (about 1.5 GB), which then stay resident until the
  daemon exits. Speech recognition runs on its own ONNX Runtime thread: `listen` does not wait for
  the MLX models, each recognizer loads on first use (0.5–1.1 GB), and `status` lists the
  resident ones under `asr.loaded`. Relative paths are resolved by the client,
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
| `align` | Force-align an existing recording to its text (`AlignJob` → `AlignResult`, see §3a) |
| `listen` | Verbatim local transcription with timed tokens (`ListenJob` → `ListenResult`, see §3b) |
| `verify` | Check a recording against its script: findings with severity and times, plus the alignment (`VerifyJob` → `VerifyResult`, see §3b) |
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

For direct integrations, the socket speaks newline-delimited JSON (protocol version 5):

```text
→ {"id":1,"method":"hello","clientVersion":"0.3.1","protocolVersion":5}
← {"id":1,"event":"welcome","serverVersion":"0.3.1","protocolVersion":5}
→ {"id":2,"method":"synth","job":{"input":"你好","voice":"tyr","outputPath":"/abs/out.wav"},"includePcm":false}
← {"id":2,"event":"started",...}  {"id":2,"event":"chunk",...}  {"id":2,"event":"done",...}
→ {"id":3,"method":"cancel","target":2}
→ {"id":4,"method":"annotate","job":{"units":[{"id":"l1","text":"你好。"}]},"apiKey":"sk-..."}
← {"id":4,"event":"annotated","deliveries":[{"id":"l1","speedScale":0.95,...}]}
→ {"id":5,"method":"align","job":{"audio":"/abs/line.mp3","input":"你好。","lang":"zh","phones":true}}
← {"id":5,"event":"aligned","aligner":{...},"alignment":{...},"quality":{...},...}
→ {"id":6,"method":"listen","job":{"audio":"/abs/line.wav","lang":"zh"}}
← {"id":6,"event":"heard","recognizer":{...},"text":"...","tokens":[...],...}
→ {"id":7,"method":"verify","job":{"audio":"/abs/line.wav","input":"你好。","lang":"zh"}}
← {"id":7,"event":"verified","verdict":"clean","findings":[],"heard":{...},"align":{...},...}
```

`apiKey` is optional (the daemon's `DEEPSEEK_API_KEY` is the fallback) and is never logged; a key
the provider rejects yields `ANNOTATION_UNAVAILABLE`. Other methods: `status`, `stop`, `voiceAdd`. Paths must be absolute. Requests on one connection
are multiplexed by `id`; lines are limited to 256 KiB.

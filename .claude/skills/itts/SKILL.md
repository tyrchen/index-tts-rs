---
name: itts
description: >-
  Synthesize expressive speech, clone voices, and direct emotional prosody on
  Apple Silicon using the `itts` CLI (`index-tts-rs`). Use this skill whenever
  the user asks to generate TTS audio, read poetry or prose aloud, clone or
  manage `.itts_voice` speaker snapshots, control 8-class speech emotions
  (`[[emo=...]]`), tune vocal inflection temperature (`[[temp=...]]`), speed
  (`[[speed=...]]`), pauses (`[[pause=...]]`), manage the warm `itts daemon`,
  or annotate reading scripts with DeepSeek (`itts annotate`).
---

# `itts` Speech Synthesis & Emotional Prosody Director Skill

`itts` is the native Rust + MLX Metal CLI for **IndexTTS-2.5** on Apple Silicon (`22050 Hz` 16-bit mono WAV, or `48000 Hz` with `--sample-rate 48000`; zero-shot voice cloning, 8-class emotion conditioning, inline prosody controls, and per-character timestamps).

## 1. Start Here: Diagnose, Then Use JSON

Run one command to learn everything about the environment (platform, model, registered voices, daemon, DeepSeek key, audio output), each with a fix hint:

```bash
itts doctor --output json
```

**Always pass `--output json`** (or set `ITTS_OUTPUT=json`) when you need to parse results. In JSON mode both results and failures go to stdout, and failures look like this:

```json
{"status":"error","error":{"code":"VOICE_NOT_FOUND","message":"voice `ghost` not found","hint":"run `itts voice list` ..."}}
```

Branch on `error.code` and follow `error.hint`. Exit codes are stable:

| Exit | Codes | Typical fix |
|---|---|---|
| `0` | success | — |
| `2` | `INVALID_ARGUMENT` | Fix the flag, control tag, or value range named in `message` |
| `3` | `VOICE_NOT_FOUND` | `itts voice list`, or `itts voice add <ID> --ref <AUDIO>` |
| `4` | `MODEL_UNAVAILABLE` | `itts models pull` |
| `5` | `AUDIO_DECODE_FAILED`, `AUDIO_OUTPUT_UNAVAILABLE` | Use a 3–15 s wav/mp3/flac/m4a clip; use `--out` instead of `--play` |
| `6` | `DAEMON_UNAVAILABLE`, `DAEMON_VERSION_MISMATCH` | `itts daemon restart` |
| `7` | `BUSY` | Retry after current jobs finish |
| `8`/`9` | `CANCELLED` / `TIMEOUT` | Retry; check `itts daemon logs` |
| `10` | `ANNOTATION_UNAVAILABLE`, `ANNOTATION_FAILED` | Set `DEEPSEEK_API_KEY`; retry |
| `11` / `1` | `IO` / `INTERNAL` | Read `message`; check `itts daemon logs` |

Machine-readable references: `itts schema controls` (control tags, emotions, languages, ranges, error codes), `itts schema speech-job` (request JSON Schema), `itts schema events` (streaming event schema).

## 2. Keep the Model Warm (Daemon)

```bash
itts daemon start      # returns once models are loaded; prints pid, socket, model, voices, log
itts daemon ensure     # idempotent: start only if not running (use at the start of a session)
itts daemon status     # state, memory, voices, job counters, last error
itts daemon logs -n 50 # daemon log (add -f to follow)
itts daemon restart    # after upgrading itts (fixes DAEMON_VERSION_MISMATCH)
itts daemon stop
```

- `start` detaches by itself — **never append `&`**. It exits `0` when ready (or already running) and non-zero with the failure otherwise.
- `synth` and `voice add` use the daemon automatically when it is running and fall back to an in-process engine (≈0.5 s extra model load) when it is not.
- The daemon stops after 600 s without jobs (`--idle-timeout-secs 0` disables this). Polling `status` does not keep it alive.

## 3. Voice Management

```bash
itts voice list --output json
itts voice add narrator --ref /path/to/reference.wav          # 3–15 s of clean speech
itts voice add narrator_warm --ref ref.wav --emo "calm:0.45,happy:0.2"
itts voice inspect narrator
itts voice rm narrator                                         # VOICE_NOT_FOUND (exit 3) if absent
```

Set `default_voice: narrator` in a YAML config (`--config`) to make `--voice` optional.

## 4. Synthesis

Choose exactly one speaker: `--voice <ID>` (registered), `--voice-file <PATH.itts_voice>`, or `--ref <AUDIO>` (zero-shot). Choose at least one sink: `--out file.wav`, `--out -` (raw 16-bit LE PCM on stdout), or `--play`. Relative paths are resolved against your working directory, even when the daemon does the work.

Always respect the user's shell. In **Nushell**, `<` input redirection is not supported; use `open --raw` and `save -f` pipelines instead.

### Nushell (`nu`)
```nu
itts synth --voice tyr --text "你好，欢迎来到我的声音世界。" --out hello.wav --play

# DeepSeek poetry emotion/pacing annotation -> editable script -> speech
open --raw poem.txt | itts annotate --stdin --poetry | save -f poem.itts.txt
open --raw poem.itts.txt | itts synth --voice tyr --stdin --out poem.wav --output json
```

### Bash / Zsh
```bash
itts annotate --stdin --poetry < poem.txt > poem.itts.txt
itts synth --voice tyr --stdin --out poem.wav --output json < poem.itts.txt
```

Useful flags: `--speed 0.9` (larger is faster), `--emo "calm:0.6"`, `--lang en`, `--paragraph-pause 800`, `--seed 7` / `--greedy` for reproducibility, `--sample-rate 48000` for full-band audio (bandwidth-extended above 10 kHz; needs `itts models pull`). Unset options use the config defaults.

`--output json` prints one summary (`status`, `backend`, `durationSecs`, `synthesisSecs`, `rtf`, `segments`, `outputPath`, `seed`). `--output jsonl` streams `started` → `chunk`… → `done` events, one per line; each `chunk` carries the `text` it reads and its `startSecs` in the output.

- **Reproducible takes:** every result reports `seed`. Passing it back with `--seed` reproduces the audio bit for bit, so audition several takes and keep the seed of the one you like.
- **Continuity:** `--continuity` makes each sentence continue after the previous one (no "fresh start" at every sentence); for line-by-line scripts pass the previous line with `--previous-text`. Off by default — audition before adopting.
- **Per-line direction:** `itts annotate --units --stdin --output json < lines.json` takes `{"units":[{"id","text"}],"brief"?}` and returns per-line `emo`, `temperature`, `speedScale`, `pauseAfterMs`, and `controls` (MCP tool `annotate_units`). Prefix each line with `controls`, or copy the fields into its synth call with `speed` = base speed × `speedScale`.
- **Timestamps:** `--timestamps` adds `alignment` (`text` without control tags, plus per-character `start`/`end` seconds indexed by Unicode scalar) and `sentences` (`text`, `start`, `end`). Use them for subtitles, cutting one file into per-sentence clips, or syncing visuals to a word.

## 5. Directing Emotion, Tone Temperature, Speed & Pauses

When generating expressive speech (poetry, storytelling, audiobooks, dialogues), **do not pass flat unannotated text** if emotional nuance is desired. Either:
1. Run `itts annotate --stdin [--poetry]` (requires `DEEPSEEK_API_KEY`), **or**
2. As an LLM, **write the inline control tags directly** into the text before calling `itts synth`.

Every voice named by `--voice` or `[[voice=...]]` is checked before synthesis starts, so a typo fails fast with `VOICE_NOT_FOUND` instead of after partial audio.

### Inline Control Tags Reference

Controls are stateful and apply to all subsequent utterances until changed or reset via `[[reset]]`:

| Tag | Range / Syntax | Acoustic Effect |
|---|---|---|
| `[[emo=...]]` | `name:weight,...` (`0.0..=1.0`, sum auto-capped `<= 0.8`) or `none` | Blends the 8 IndexTTS-2.5 emotion prototypes matched to the speaker's timbre |
| `[[temp=...]]` | `0.05..=2.0` (default `0.80`) | GPT sampling temperature: `0.68..0.75` restrained/intimate; `0.78..0.84` warm narrative; `0.85..0.92` dramatic/exclamatory |
| `[[speed=...]]` | `0.25..=4.0` (relative to the job speed, default `1.2`) | Speaking rate multiplier (`0.82..0.94` for slow poetic/solemn lines; `1.02..1.15` for brisk lines) |
| `[[pause=...]]` | `0..=5000` (`ms` optional) | Exact digital silence inserted before the next spoken unit (`350..650` intra-stanza turn; `750..1200` stanza break) |
| `[[voice=...]]` | `<voice_id>` | Switches speaker mid-script (e.g. multi-character dialogue) |
| `[[lang=...]]` | `zh`, `en`, `zh/en`, `ja`, `yue`, `es`, `ar` | Switches synthesis language |
| `[[reset]]` | *(no value)* | Restores initial `voice`, `lang`, `emo`, `temp`, and `speed` |
| `<字|pinyin4>` | e.g. `<荇|xing4>`, `<行|hang2>` | Forces exact Chinese pinyin (with tone `1..5`) or English ARPAbet phoneme |

### The 8 Supported Emotion Categories (`[[emo=...]]` & `--emo`)

Both English and Chinese names are accepted:
- `happy` / `高兴` (`快乐`, `喜悦`)
- `angry` / `愤怒` (`生气`, `恼怒`)
- `sad` / `悲伤` (`伤心`, `哀伤`)
- `afraid` / `恐惧` (`害怕`, `惊恐`)
- `disgusted` / `反感` (`厌恶`, `嫌弃`)
- `melancholic` / `低落` (`忧郁`, `惆怅`, `感伤`, `眷恋`)
- `surprised` / `惊讶` (`惊奇`, `诧异`)
- `calm` / `平静` (`自然`, `温和`, `宁静`, `舒缓`)
- `none` / `无` / `默认` — clears emotion override and restores the reference voice's base emotion.

For detailed calibration guidelines and complete poetry/dialogue templates (such as 《再别康桥》), read [references/emotion-and-controls.md](./references/emotion-and-controls.md).

## 6. MCP Server

`itts mcp` serves the Model Context Protocol over stdio and shares the daemon (starting it on first use), so multiple agent sessions share one resident model. Tools: `status`, `describe_controls`, `synthesize` (requires `outputPath`), `speak`, `list_voices`, `inspect_voice`, `register_voice`, `remove_voice`, `annotate`, `annotate_units`. Tool arguments use the same fields as `itts schema speech-job` (`input`, `voice`, `voiceFile`, `refAudio`, `emo`, `lang`, `speed`, `paragraphPauseMs`, `outputPath`, `sampleRate`, `timestamps`, `continuity`, `previousText`, `advanced`). Failures are `isError` results whose structured content is `{code, message, hint}`. Long jobs report `notifications/progress` per segment and honor cancellation.

## 7. Verification

A successful synthesis exits `0`; with `--output json` check `status == "ok"`, a positive `durationSecs`, and that `outputPath` exists and is non-empty. On failure, read `error.code` and act on `error.hint` rather than retrying blindly.

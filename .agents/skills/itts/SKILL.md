---
name: itts
description: >-
  Synthesize expressive speech, clone voices, and direct emotional prosody on
  Apple Silicon using the `itts` CLI (`index-tts-rs`). Use this skill whenever
  the user asks to generate TTS audio, read poetry or prose aloud, clone or
  manage `.itts_voice` speaker snapshots, control 8-class speech emotions
  (`[[emo=...]]`), tune vocal inflection temperature (`[[temp=...]]`), speed
  (`[[speed=...]]`), pauses (`[[pause=...]]`), or annotate reading scripts with
  DeepSeek (`itts annotate`).
---

# `itts` Speech Synthesis & Emotional Prosody Director Skill

`itts` is the native Rust + MLX Metal CLI for **IndexTTS-2.5** on Apple Silicon (`22050 Hz` 16-bit mono WAV, zero-shot voice cloning, 8-class emotion conditioning, and inline prosody controls).

## 1. Quick Discovery & Voice Management

Before synthesizing, check registered voices so you use a valid `<voice_id>`:

```bash
# List registered voices (human-readable or JSON for agents)
itts voice list --json

# Register a new reusable .itts_voice snapshot (3s..15s reference audio)
itts voice add narrator --ref /path/to/reference.wav

# Optionally bake a default emotion vector or emotion reference audio into the snapshot
itts voice add narrator_warm --ref /path/to/reference.wav --emo "calm:0.45,happy:0.2"
```

## 2. Core Synthesis Workflows (Nushell & POSIX Shell)

Always respect the user's shell. In **Nushell**, `<` input redirection is not supported; use `open --raw` and `save -f` pipelines instead.

### Nushell (`nu`)
```nu
# Direct text synthesis + live CoreAudio playback + WAV output
itts synth --voice tyr --text "你好，欢迎来到我的声音世界。" --out hello.wav --play

# Pipeline: DeepSeek poetry emotion/pacing annotation -> editable script -> speech
open --raw poem.txt | itts annotate --stdin --poetry | save -f poem.itts.txt
open --raw poem.itts.txt | itts synth --voice tyr --stdin --out poem.wav --play

# One-liner pipeline without intermediate file
open --raw poem.txt | itts annotate --stdin --poetry | itts synth --voice tyr --stdin --out poem.wav --play
```

### Bash / Zsh
```bash
itts annotate --stdin --poetry < poem.txt > poem.itts.txt
itts synth --voice tyr --stdin --out poem.wav --play < poem.itts.txt
```

## 3. Directing Emotion, Tone Temperature, Speed & Pauses

When generating expressive speech (poetry, storytelling, audiobooks, dialogues), **do not pass flat unannotated text** if emotional nuance is desired. Either:
1. Run `itts annotate --stdin [--poetry]` (requires `DEEPSEEK_API_KEY`), **or**
2. As an LLM, **write the inline control tags directly** into the text before calling `itts synth`!

### Inline Control Tags Reference

Controls are stateful and apply to all subsequent utterances until changed or reset via `[[reset]]`:

| Tag | Range / Syntax | Acoustic Effect |
|---|---|---|
| `[[emo=...]]` | `name:weight,...` (`0.0..=1.0`, sum auto-capped `<= 0.8`) or `none` | Blends the 8 IndexTTS-2.5 emotion prototypes matched to the speaker's timbre |
| `[[temp=...]]` | `0.05..=2.0` (default `0.80`) | GPT sampling temperature: `0.68..0.75` restrained/intimate; `0.78..0.84` warm narrative; `0.85..0.92` dramatic/exclamatory |
| `[[speed=...]]` | `0.25..=4.0` (default `1.0`) | Speaking rate multiplier (`0.82..0.94` for slow poetic/solemn lines; `1.02..1.15` for brisk lines) |
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

## 4. Warm Daemon & Verification

- For repeated CLI calls, start the warm-model background daemon so weights stay resident in Metal GPU memory:
  ```bash
  itts daemon start
  itts daemon status --json
  ```
- Verify generated audio by checking exit code `0`, inspecting `--json` output (`durationSecs`, `rtf`, `segmentsCount`), and ensuring the output `.wav` file exists and is non-empty.

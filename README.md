# index-tts-rs

High-performance, 100% pure-Rust inference engine, CLI, warm-model daemon, and MCP server for **IndexTTS-2.5** (`yunfengwang/IndexTTS-2.5-mlx`) on Apple Silicon (`mlx-rs` + Metal GPU + Accelerate).

---

## Highlights

- **Zero Python / Zero PyTorch / Zero Librosa Runtime**: Self-contained Rust workspace (`index-tts-rs-core`, `itts` CLI, `index-tts-rs-server`) running natively on Apple Silicon Unified Memory.
- **4.01x Faster End-to-End & 2.60x Real-Time Synthesis (`RTF = 0.384`)**:
  - **Warm Daemon (`synth --voice`)**: Synthesizes `10.02s` of cloned speech in **`3.85s`** (`RTF = 0.384`), vs `15.43s` for `uvx index-tts-2.5-mlx` and `6.15s` for warm Python.
  - **Single-Shot `.itts_voice` (`synth --voice --no-daemon`)**: Synthesizes `10.02s` of audio in **`4.36s`** (including `483ms` model load) with a **peak RSS of `2.50 GB`** (**-53.5% memory** vs Python's `5.38 GB`).
  - **Single-Shot Zero-Shot Reference Audio (`synth --ref --no-daemon`)**: Sequentially streams and immediately frees Stage 0 encoders (`w2v-bert-2.0` -> `EmoVec` -> `CAMPPlus`), finishing in **`4.93s`** with a **peak RSS of `3.94 GB`** (**-26.8% memory** and **1.25x faster** than warm Python).
- **1.85s Time-To-First-Audio (TTFA) Sentence Streaming**: Streams [`AudioChunk`]s sentence-by-sentence over a bounded channel to `cpal` CoreAudio playback, `--jsonl` events, or `--out -` raw PCM stdout (>6x lower first-audio latency than batch Python synthesis).
- **14/14 Golden Checkpoints (`CP-01`..`CP-14`) Numerically Verified**: Bit-exact tokenizer & 32-step greedy GPT codes, `cos_sim >= 0.99998` across DSP, Stage 0 encoders, CFM Euler ODE, and `BigVGANv2` (>58 dB SNR).
- **Strict Safety & Code Quality**: `#![forbid(unsafe_code)]` across all crates, zero `.unwrap()` / `.expect()` / `panic!()` / unchecked indexing (`[]`), validated domain newtypes, unit, integration, and doc-tests covering inference and application flows.

---

## Workspace Structure

```text
index-tts-rs/
├── crates/
│   ├── core/          # index-tts-rs-core: DSP, Frontend, Stage 0..3 MLX models, VoiceStore, Actor Engine
│   └── director/      # Validated speech controls and optional DeepSeek pacing annotation
├── apps/
│   ├── cli/           # itts: CLI (`synth`, `voice`, `models`, `daemon`, `mcp`)
│   └── server/        # index-tts-rs-server: HTTP /v1/audio/speech server
├── specs/             # PRD, Architecture Design, Implementation Plan, Verification Plan
└── docs/              # Deep-dive research & benchmark verification reports
```

---

## Quick Start

### 1. Build Release Binaries

```bash
cargo build --release --workspace

# Install the CLI as `itts` in Cargo's bin directory (usually ~/.cargo/bin)
make install
itts --help
```

### 2. Pull or Inspect Local Model Weights

```bash
# Inspect local Hugging Face cache snapshot & dead-weight stripping stats
itts models info

# Or pull weights from Hugging Face Hub (`yunfengwang/IndexTTS-2.5-mlx`)
itts models pull
```

### 3. Pre-Extract a Reusable `.itts_voice` Snapshot (`1.60 MiB`)

```bash
itts voice add narrator --ref /path/to/voice.wav
itts voice list
```

### 4. Synthesize Speech

```bash
# Fast synthesis from pre-computed `.itts_voice` snapshot
itts synth \
  --voice narrator \
  --text "你好，欢迎使用 IndexTTS Rust 原生语音合成引擎！" \
  --out output.wav

# Zero-shot synthesis directly from a reference audio file + live CoreAudio playback
itts synth \
  --ref /path/to/voice.wav \
  --text "Hello from Apple Silicon Metal GPU!" \
  --play \
  --out output_en.wav

# Start the warm-model daemon in the background
itts daemon start &
itts synth --voice narrator --text "秒级热启动推理测试。" --out warm.wav
itts daemon stop
```

---

## Speech Controls & AI Pacing

```bash
itts synth --voice tyr \
  --text '[[speed=0.9]]轻轻的我走了，[[pause=600]]正如我轻轻的来。' \
  --out poem.wav --play

# Optional: uses DEEPSEEK_API_KEY to annotate pacing, without rewriting the words
itts annotate --stdin --poetry < poem.txt > poem.itts.txt
itts synth --voice tyr --stdin --out poem.wav --play < poem.itts.txt
```

Controls include `pause`, `speed`, `voice`, `lang`, and `reset`, along with existing
`<word|pronunciation>` overrides. Blank lines preserve paragraphs and poem stanzas. See the
[speech control guide](docs/usage/itts-speech-controls.md) for syntax, provider settings, and
limits. Emotion and pitch tags are rejected because the current pipeline cannot execute them.

## Browser Speech Studio

```bash
make serve
# Optional: make serve SERVER_ARGS="--bind 127.0.0.1:8081 --voices-dir /path/to/voices"
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080), enter text, select a registered voice,
and generate speech. The page includes language, speed, and paragraph-pause controls, live
playback while generating, WAV downloads, and optional DeepSeek annotation with editable tags. It loads voices from the same local voice library as `itts`, including newly
registered voices after clicking **刷新列表**. Register a voice with `itts voice add` first if the
library is empty. No frontend build or external assets are required.

The HTTP server also exposes:

| Endpoint | Response |
| --- | --- |
| `GET /health` | Engine status and sample rate |
| `GET /api/voices` | Registered voices as `[{"id":"tyr","durationSecs":8.14}]` |
| `POST /api/annotate` | Editable pacing controls from indexed source paragraphs |
| `POST /api/speech/stream` | Streaming PCM chunks and completion statistics as JSON Lines |
| `POST /v1/audio/speech` | WAV audio from `{"input":"你好。","voice":"tyr","lang":"zh","speed":1.0}` |

The HTTP speech endpoint accepts registered voice IDs; use `itts synth --ref` for synthesis
directly from reference audio. Text is limited to 16,384 UTF-8 bytes. `speed` ranges from 0.25
to 4.0 (larger values speak faster); the page offers 0.5–2.0. Requests have a 32 KiB body limit,
a 120-second timeout, and an eight-request concurrency limit with a busy response when full.

---

## Documentation & Specifications

- **Speech Controls & AI Pacing**: [`docs/usage/itts-speech-controls.md`](docs/usage/itts-speech-controls.md)
- **Benchmark & Verification Report**: [`docs/benchmarks/indextts-rust-verification-and-benchmark-report.md`](docs/benchmarks/indextts-rust-verification-and-benchmark-report.md)
- **Architecture & Backend Research**: [`docs/research/indextts-2.5-mlx-architecture-and-backend-research.md`](docs/research/indextts-2.5-mlx-architecture-and-backend-research.md)
- **Specs Index**: [`specs/index.md`](specs/index.md) (`PRD`, `Architecture Design`, `Implementation Plan`, `Verification Plan`)

---

## License

Distributed under the terms of the [MIT License](LICENSE.md).

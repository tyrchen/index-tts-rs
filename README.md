# index-tts-rs

High-performance, 100% pure-Rust inference engine, CLI, warm-model daemon, and MCP server for **IndexTTS-2.5** (`yunfengwang/IndexTTS-2.5-mlx`) on Apple Silicon (`mlx-rs` + Metal GPU + Accelerate).

---

## Highlights

- **Zero Python / Zero PyTorch / Zero Librosa Runtime**: Self-contained Rust workspace (`index-tts-rs-core`, `index-tts` CLI, `index-tts-server`) running natively on Apple Silicon Unified Memory.
- **4.01x Faster End-to-End & 2.60x Real-Time Synthesis (`RTF = 0.384`)**:
  - **Warm Daemon (`synth --voice`)**: Synthesizes `10.02s` of cloned speech in **`3.85s`** (`RTF = 0.384`), vs `15.43s` for `uvx index-tts-2.5-mlx` and `6.15s` for warm Python.
  - **Single-Shot `.itts_voice` (`synth --voice --no-daemon`)**: Synthesizes `10.02s` of audio in **`4.36s`** (including `483ms` model load) with a **peak RSS of `2.50 GB`** (**-53.5% memory** vs Python's `5.38 GB`).
  - **Single-Shot Zero-Shot Reference Audio (`synth --ref --no-daemon`)**: Sequentially streams and immediately frees Stage 0 encoders (`w2v-bert-2.0` -> `EmoVec` -> `CAMPPlus`), finishing in **`4.93s`** with a **peak RSS of `3.94 GB`** (**-26.8% memory** and **1.25x faster** than warm Python).
- **1.85s Time-To-First-Audio (TTFA) Sentence Streaming**: Streams [`AudioChunk`]s sentence-by-sentence over a bounded channel to `cpal` CoreAudio playback, `--jsonl` events, or `--out -` raw PCM stdout (>6x lower first-audio latency than batch Python synthesis).
- **14/14 Golden Checkpoints (`CP-01`..`CP-14`) Numerically Verified**: Bit-exact tokenizer & 32-step greedy GPT codes, `cos_sim >= 0.99998` across DSP, Stage 0 encoders, CFM Euler ODE, and `BigVGANv2` (>58 dB SNR).
- **Strict Safety & Code Quality**: `#![forbid(unsafe_code)]` across all crates, zero `.unwrap()` / `.expect()` / `panic!()` / unchecked indexing (`[]`), validated domain newtypes, 91 unit/integration tests + 92 doc-tests passing.

---

## Workspace Structure

```text
index-tts-rs/
├── crates/
│   └── core/          # index-tts-rs-core: DSP, Frontend, Stage 0..3 MLX models, VoiceStore, Actor Engine
├── apps/
│   ├── cli/           # index-tts: CLI (`synth`, `voice`, `models`, `daemon`, `mcp`)
│   └── server/        # index-tts-server: Dedicated MCP stdio server binary
├── specs/             # PRD, Architecture Design, Implementation Plan, Verification Plan
└── docs/              # Deep-dive research & benchmark verification reports
```

---

## Quick Start

### 1. Build Release Binaries

```bash
cargo build --release --workspace
```

### 2. Pull or Inspect Local Model Weights

```bash
# Inspect local Hugging Face cache snapshot & dead-weight stripping stats
./target/release/index-tts models info

# Or pull weights from Hugging Face Hub (`yunfengwang/IndexTTS-2.5-mlx`)
./target/release/index-tts models pull
```

### 3. Pre-Extract a Reusable `.itts_voice` Snapshot (`1.60 MiB`)

```bash
./target/release/index-tts voice add narrator --ref /path/to/voice.wav
./target/release/index-tts voice list
```

### 4. Synthesize Speech

```bash
# Fast synthesis from pre-computed `.itts_voice` snapshot
./target/release/index-tts synth \
  --voice narrator \
  --text "你好，欢迎使用 IndexTTS Rust 原生语音合成引擎！" \
  --out output.wav

# Zero-shot synthesis directly from a reference audio file + live CoreAudio playback
./target/release/index-tts synth \
  --ref /path/to/voice.wav \
  --text "Hello from Apple Silicon Metal GPU!" \
  --play \
  --out output_en.wav

# Start warm-model background daemon for zero cold-start latency
./target/release/index-tts daemon start
./target/release/index-tts synth --voice narrator --text "秒级热启动推理测试。" --out warm.wav
./target/release/index-tts daemon stop
```

---

## Documentation & Specifications

- **Benchmark & Verification Report**: [`docs/benchmarks/indextts-rust-verification-and-benchmark-report.md`](docs/benchmarks/indextts-rust-verification-and-benchmark-report.md)
- **Architecture & Backend Research**: [`docs/research/indextts-2.5-mlx-architecture-and-backend-research.md`](docs/research/indextts-2.5-mlx-architecture-and-backend-research.md)
- **Specs Index**: [`specs/index.md`](specs/index.md) (`PRD`, `Architecture Design`, `Implementation Plan`, `Verification Plan`)

---

## License

Distributed under the terms of the [MIT License](LICENSE.md).

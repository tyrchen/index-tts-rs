# Changelog

All notable changes to this project are documented here.

## 0.3.0

- Added per-character and per-sentence timestamps from GPT attention alignment
  (`--timestamps`, `POST /v1/audio/speech/timestamps`), and report the seed of
  every synthesis for reproducible output.
- Added optional 48 kHz output through LavaSR bandwidth extension.
- Added per-line annotation (`AnnotateJob`, daemon `annotate`, `POST /v1/annotate`,
  MCP `annotate_units`) with per-request DeepSeek keys; the daemon protocol is now 3.
- Added opt-in cross-sentence continuity (`continuity`, `previousText`).
- HTTP synthesis now queues for up to 60 s instead of returning `BUSY` at once.
- Changed defaults: `top_k` 10 (fewer Mandarin tone errors) and speed 1.2; inline
  `[[speed=X]]` is relative to the job speed. Removed the unused
  `max_ref_seconds` setting.

## 0.2.0

- Added a shared speech-job service layer, stable runtime errors, JSON / JSONL
  output, environment diagnostics, and machine-readable schemas.
- Added daemon lifecycle management and MCP tools that share a resident model.
- Added editable emotion, temperature, speed, voice, pronunciation, and pause
  controls, with optional DeepSeek annotation and poetry pacing.
- Added the browser speech studio, sentence streaming, and HTTP speech routes.
- Added English and Chinese usage, product rationale, architecture, and qualified
  historical benchmark documentation.
- Prepared versioned workspace source packages, self-contained core configuration,
  upstream model-asset notices, and a checked draft-release workflow.

## Earlier development

The initial implementation introduced the Rust / MLX inference pipeline, DSP and
multilingual frontend, reusable speaker snapshots, and golden parity checkpoints.
Historical measurements are in the [benchmark report](docs/benchmarks/indextts-rust-verification-and-benchmark-report.md).

# Changelog

All notable changes to this project are documented here.

## 0.3.1

- Changed the default `top_k` from 10 to 3. Sampling wider candidate sets
  occasionally picks tone-shifted codes that whisper-based checks miss (e.g.
  困境 read as 捆近 in 3 of 8 seeds at 10, none at 3); listening tests found no
  loss of naturalness and long text does not loop.

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

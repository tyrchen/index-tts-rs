# Documentation Index

## Research

- [IndexTTS-2.5-mlx 深度逆向分析与 Rust 推理后端调研 (`mlx-rs` vs `candle`)](research/indextts-2.5-mlx-architecture-and-backend-research.md) — 针对本机 `yunfengwang/IndexTTS-2.5-mlx` 权重结构、`index-tts-2.5-mlx` Python 源码瓶颈、以及 Apple Silicon 下 `mlx-rs` 与 `candle` 的深度对比分析。

## Benchmarks & Verification

- [IndexTTS-2.5-mlx Rust (`index-tts-rs`) 验证与性能基准对比报告](benchmarks/indextts-rust-verification-and-benchmark-report.md) — 14 项黄金检查点 (`CP-01`..`CP-14`) 数值对齐结果、死权重裁剪与 `.itts_voice` 显存优化审计、以及基于 `~/Downloads/voice.wav` 对比原版 `uvx index-tts-2.5-mlx` 的端到端耗时、RTF、多句流式首包延迟 (TTFA) 与峰值 RSS 基准测试报告。

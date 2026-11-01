# Documentation Index

## Usage

- [English README](../README.md) / [中文 README](../README.zh-CN.md) — Installation, product decisions, architecture, and historical benchmark context.
- [Release guide](usage/releasing.md) — Source packaging, release gates, model verification, registry publication, and draft GitHub releases.

- [itts Agent Integration Guide](usage/itts-agent-integration.md) — Output modes, stable error codes and exit codes, `itts doctor` / `itts schema`, warm daemon lifecycle, MCP tools, and the daemon wire protocol.
- [itts Speech Controls and AI Pacing](usage/itts-speech-controls.md) — Inline controls, optional DeepSeek annotation, poem paragraph handling, and browser streaming.

## Research

- [IndexTTS-2.5-mlx 深度逆向分析与 Rust 推理后端调研 (`mlx-rs` vs `candle`)](research/indextts-2.5-mlx-architecture-and-backend-research.md) — 针对本机 `yunfengwang/IndexTTS-2.5-mlx` 权重结构、`index-tts-2.5-mlx` Python 源码瓶颈、以及 Apple Silicon 下 `mlx-rs` 与 `candle` 的深度对比分析。
- [基于 GPT 文本注意力的字级时间戳：实验与选型](research/gpt-attention-alignment-research.md) — 24 层 × 20 头注意力扫描、与 MMS 强制对齐器对比（中位数 17 ms、96–98% ≤ 100 ms）、Viterbi + 能量修正的解码设计与复现脚本。
- [跨句衔接：标注与声学续写实验](research/sentence-continuity-research.md) — 标注（宏观走势）与 GPT 前缀续写（声学衔接）的对比，续写立刻停止的失败模式与最小长度修复，3 组句对 × 6 seed 的长度与可懂度统计。
- [普通话"洋腔"：声调错误的来源与采样参数](research/mandarin-tone-accuracy-research.md) — 与上游 Python 实现对比排除移植问题，重复惩罚与拼音标注实验，top-k/温度网格（top-k 10 声调错误减半），默认语速调整的测量依据。
- [语音带宽扩展（22.05 kHz → 48 kHz）选型与验证](research/speech-bandwidth-extension-research.md) — 为什么不能直接 44.1 kHz、BWE 候选对比、LavaSR v2 权重可信度验证、交叉频点扫描、与参考实现及真实录音的数值对比。

## Benchmarks & Verification (`itts` CLI)

- [IndexTTS-2.5-mlx Rust (`index-tts-rs`) 验证与性能基准对比报告](benchmarks/indextts-rust-verification-and-benchmark-report.md) — 14 项黄金检查点 (`CP-01`..`CP-14`) 数值对齐结果、死权重裁剪与 `.itts_voice` 显存优化审计、以及基于 `~/Downloads/voice.wav` 对比原版 `uvx index-tts-2.5-mlx` 的端到端耗时、RTF、多句流式首包延迟 (TTFA) 与峰值 RSS 基准测试报告。

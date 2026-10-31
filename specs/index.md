# Specifications Index

本目录包含 `index-tts-rs` 的产品需求文档（PRD）、详细架构设计（Design Spec）、分步实施计划（Implementation Plan）与验证计划（Verification Plan），供研发与 AI Agent 生成及验证代码使用。

## Core Specifications (`itts` CLI & Server)

- [IndexTTS-2.5 Rust 高性能语音合成系统产品需求文档 (`indextts-rust-prd`)](indextts-rust-prd.md) — 定义产品愿景、双核形态（Rust Library Crate + AI Agent CLI/Daemon/MCP）、功能需求（FR-1..FR-5）、非功能 SLA 指标与安全边界。
- [IndexTTS-2.5 Rust 高性能推理引擎与 Agent CLI 详细架构设计 (`indextts-rust-engine-design`)](indextts-rust-engine-design.md) — 定义 Workspace 模块分层、两阶段音色快照（`.itts_voice`）架构、纯 Rust DSP/Frontend 算法规格、`mlx-rs` 算子映射与全 GPU 驻留优化、Actor 并发流式管线及验证策略。
- [IndexTTS-2.5 Rust 高性能推理引擎与 Agent CLI 分步实施计划 (`indextts-rust-impl-plan`)](indextts-rust-impl-plan.md) — 将系统开发拆解为 6 个可独立交付与验证的阶段（Phase 1 基建与 Fixture -> Phase 2 纯 Rust DSP/Frontend -> Phase 3 S2M/BigVGAN -> Phase 4 GPT Int8 -> Phase 5 Stage 0 音色提取 -> Phase 6 Actor/CLI/Daemon/MCP），明确每阶段文件清单与准出标准。
- [IndexTTS-2.5 Rust 高性能推理引擎与 Agent CLI 验证计划 (`indextts-rust-verification-plan`)](indextts-rust-verification-plan.md) — 定义 14 个黄金检查点张量（CP-01..CP-14）的余弦相似度与误差阈值、`rstest`/`proptest` 测试矩阵、端到端集成测试以及性能/安全门禁。

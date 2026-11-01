# Specifications Index

本目录包含 `index-tts-rs` 的产品需求文档（PRD）、详细架构设计（Design Spec）、分步实施计划（Implementation Plan）与验证计划（Verification Plan），供研发与 AI Agent 生成及验证代码使用。

## Core Specifications (`itts` CLI & Server)

- [IndexTTS-2.5 Rust 高性能语音合成系统产品需求文档 (`indextts-rust-prd`)](indextts-rust-prd.md) — 定义产品愿景、双核形态（Rust Library Crate + AI Agent CLI/Daemon/MCP）、功能需求（FR-1..FR-5）、非功能 SLA 指标与安全边界。
- [IndexTTS-2.5 Rust 高性能推理引擎与 Agent CLI 详细架构设计 (`indextts-rust-engine-design`)](indextts-rust-engine-design.md) — 定义 Workspace 模块分层、两阶段音色快照（`.itts_voice`）架构、纯 Rust DSP/Frontend 算法规格、`mlx-rs` 算子映射与全 GPU 驻留优化、Actor 并发流式管线及验证策略。
- [IndexTTS-2.5 Rust 高性能推理引擎与 Agent CLI 分步实施计划 (`indextts-rust-impl-plan`)](indextts-rust-impl-plan.md) — 将系统开发拆解为 6 个可独立交付与验证的阶段（Phase 1 基建与 Fixture -> Phase 2 纯 Rust DSP/Frontend -> Phase 3 S2M/BigVGAN -> Phase 4 GPT Int8 -> Phase 5 Stage 0 音色提取 -> Phase 6 Actor/CLI/Daemon/MCP），明确每阶段文件清单与准出标准。
- [IndexTTS-2.5 Rust 高性能推理引擎与 Agent CLI 验证计划 (`indextts-rust-verification-plan`)](indextts-rust-verification-plan.md) — 定义 14 个黄金检查点张量（CP-01..CP-14）的余弦相似度与误差阈值、`rstest`/`proptest` 测试矩阵、端到端集成测试以及性能/安全门禁。
- [IndexTTS-2.5 情绪向量推理与细粒度朗读导演架构设计 (`indextts-emotion-director-design`)](indextts-emotion-director-design.md) — 定义上游 8 类情绪原型矩阵（`spk_matrix` / `emo_matrix`）与零 Stage-0 开销的 `.itts_voice` 动态 `conds` 混合算法、`[[emo=...]]` 与 `[[temp=...]]` 控制标签、以及基于诗句联（Couplet）切分的 DeepSeek 细粒度情绪与韵律标注架构。

## Agent Usability & Service Layer

- [`itts` Agent-First 可用性与服务化能力设计 (`itts-agent-usability-design`)](itts-agent-usability-design.md) — 审计并修复 CLI / Daemon / MCP / HTTP 的可观测性缺口与一致性 Bug（daemon 无就绪输出、相对路径、voice 缓存失效、MCP 不解析内联标签等）；定义统一 `SpeechJob` 请求模型与 `index-tts-rs-service` crate、14 个稳定错误码与退出码、daemon 后台化 / 单例锁 / 协议 v2（握手、多路复用、取消、BUSY）、`itts doctor` / `itts schema`，以及基于 `rmcp` 且复用 daemon 的 MCP 服务。早期规范中关于 `--json`、`synthesize_to_file` 等 CLI/MCP 接口的描述以本文档为准。
- [`itts` 字级时间戳、可复现种子与 48 kHz 高保真输出设计 (`itts-timestamps-and-hifi-design`)](itts-timestamps-and-hifi-design.md) — 基于 GPT 自身文本注意力（L4H10+L5H10）的零额外权重字级对齐、字符 diff 坐标映射、`alignment`/`sentences`/chunk 文本输出；缺省随机并回报的 seed；`EngineConfig.sample_rate` 类型化与 LavaSR v2 带宽扩展的 48 kHz 输出。
- [`itts` 逐行朗读指令协议与跨句衔接设计 (`itts-direction-and-continuity-design`)](itts-direction-and-continuity-design.md) — 按行 ID 返回结构化朗读指令的 `AnnotateJob`（CLI / daemon / MCP / HTTP）、按请求提供并脱敏的 DeepSeek key、HTTP 合成排队；基于 GPT 前缀续写的 `continuity` / `previousText` 跨句衔接（含最小生成长度、续写对齐）与 daemon 协议 v3。

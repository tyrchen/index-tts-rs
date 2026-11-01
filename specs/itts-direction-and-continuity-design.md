# `itts` 逐行朗读指令协议与跨句衔接设计 (`itts-direction-and-continuity-design`)

- **状态**: 已实施 (Implemented)
- **版本**: 1.0
- **日期**: 2026-10-31
- **关联组件**: `index-tts-rs-director`、`index-tts-rs-service`、`index-tts-rs-core`、`itts`（CLI / Daemon / MCP）、`index-tts-rs-server`（HTTP）
- **调研**: [跨句衔接实验](../docs/research/sentence-continuity-research.md)
- **前置规范**: [`itts-timestamps-and-hifi-design`](itts-timestamps-and-hifi-design.md)、[`indextts-emotion-director-design`](indextts-emotion-director-design.md)

---

## 1. 背景

外部 SDK（首个用户是视频管线 reshape）按"脚本行"驱动合成，此前有三个协议缺口和一个能力缺口：

| # | 缺口 | 影响 |
|---|------|------|
| P1 | 标注只接受整段文本、返回带 `[[...]]` 标签的脚本 | 调用方无法把结果对应回自己的行 ID，也无法直接填进逐行请求 |
| P2 | DeepSeek key 只能来自进程环境变量 | 多租户 / SDK 场景无法按请求提供 key |
| P3 | HTTP 已有合成在跑时，新请求立即 `BUSY` | 客户端必须自己实现重试排队 |
| C1 | 每句独立生成，句首总像重新起头 | 逐行旁白缺乏连贯性（ElevenLabs 用 `previous_text` 解决） |

## 2. 逐行朗读指令（P1）

### 2.1 线上模型（`index-tts-rs-service::annotation`）

```jsonc
// AnnotateJob（itts schema annotate-job）
{
  "units": [ { "id": "s1.l1", "text": "三年前，我们的推理服务每天都在超时。" }, ... ],  // 1..=128，合计 ≤ 16 KiB
  "style": "natural",                     // natural（默认）| poetry
  "brief": "科技纪录片旁白，克制、自信"      // 可选，≤ 1 KiB，作为数据传给模型
}
// AnnotateResult（itts schema annotate-result）
{ "deliveries": [ {
    "id": "s1.l1",
    "emo": "melancholic:0.2,calm:0.45",   // 与 SpeechJob.emo 同语法；省略表示基础情绪
    "temperature": 0.76,                  // → SpeechJob.advanced.temperature
    "speed": 0.9,                         // → SpeechJob.speed
    "pauseAfterMs": 520,                  // 建议的行后停顿；最后一行为 0
    "controls": "[[emo=melancholic:0.2,calm:0.45]][[temp=0.76]][[speed=0.9]]"
} ] }
```

- ID 白名单 `^[A-Za-z0-9_.:-]{1,64}$` 且唯一；文本不得含 `[[` 或 NUL。
- 模型只返回按序号对应的数值，原文与 ID 都在本地拼回，提示注入无法改写文字或 ID。数值按既有范围校验（speed 0.8–1.2、temperature 0.5–1.2、pause 0–2000、情绪权重和 ≤ 0.8），最后一行停顿必须为 0。
- 提示词按风格分别校准。`natural`（旁白/叙述）：叙述者投入但不表演内容——讲困难、失败或成功本身不是悲伤/愤怒/高兴；叙述句返回空情绪，只有文本本身是情绪化表达（引语、感叹、明确的情感描述）时才给 1–2 个情绪、权重 0.15–0.35；语速 0.92–1.10（相对叙述者常速 1.0，任务语速另行叠加）；温度 0.75–0.88。`poetry` 沿用诗朗诵校准。（一度尝试过"带情绪的句子必须用非 calm 情绪"，试听反馈为"为什么要悲伤"——给事实性叙述套情绪是错误方向，已回退为克制策略。）
- 数值保留两位小数输出（避免 f32→JSON 的 `0.9399999`）。

### 2.2 director 重构

`Director::annotate`（整段→脚本）与新增的 `Director::direct_units`（单元→`Direction`）共用同一个 provider 调用与解析/校验路径；脚本渲染改为基于 `Direction`。`Direction::controls()` 生成可前置到行文本的内联控制。

### 2.3 入口

| 入口 | 形式 |
|------|------|
| CLI | `itts annotate --units`（输入为 `AnnotateJob` JSON，`--output json` 返回 `deliveries`） |
| Daemon | 方法 `annotate`：`{"method":"annotate","job":{...},"apiKey":"..."}` → `{"event":"annotated","deliveries":[...]}` |
| MCP | 工具 `annotate_units`（只用环境变量中的 key——不应让 LLM 在工具参数里传密钥） |
| HTTP | `POST /v1/annotate`（`AnnotateJob` → `AnnotateResult`） |

## 3. 按请求提供 provider key（P2）

- `Director::new(config, SecretString)` 公开；`from_env` 复用它。key 需为 1–512 个可见 ASCII 字节。
- service 定义 `ApiKey` 新类型：`Debug` 输出 `ApiKey([REDACTED])`，仅在调用 provider 时暴露。
- HTTP：请求头 `X-DeepSeek-Api-Key`（`/v1/annotate` 与 `/api/annotate` 均支持）；handler 使用 `#[instrument(skip_all)]`，key 不进日志。Daemon：`annotate` 请求的 `apiKey` 字段（socket 为 0600）。未提供时回退到 `DEEPSEEK_API_KEY`。
- 错误分类：provider 返回 401/403 → `ANNOTATION_UNAVAILABLE`（"provider rejected the API key"），与网络/服务故障的 `ANNOTATION_FAILED` 区分。

## 4. HTTP 排队（P3）

合成许可由 `try_acquire`（立即 `BUSY`）改为先校验请求、再最多等待 60 s 获取 GPU；超时才返回 `BUSY`。准入仍受 8 个并发请求的信号量限制，所以排队深度 ≤ 7；排队时间计入 120 s 的请求预算。

## 5. 跨句衔接（C1）

### 5.1 机制：GPT 前缀续写

训练序列是 `[conds, 文本 A+B, codes A+B]`，所以以 `文本 A + 文本 B` 和 `codes A` 为前缀继续生成，是在分布内的"接着说"。实验（调研文档）显示：续写正确读出 B、无复读/漏读，节奏与整句合成一致（独立生成系统性长 15–30%）。

core 新增：

- `Gpt::prefill_after(text, conds, lang, prior_codes, cache)`：前缀含 prior codes（Mel 位置从 2 起，与解码步一致）；`prefill` 是前缀为空的特例。
- `CodePrompt { text_tokens, conds, lang_id, prior_codes, min_codes }` + `generate_codes_after`：prior codes 计入位置容量但**不计入重复惩罚**（计入时续写被迫避开上一句的常见音素 codes，听感像外国口音，字错率 4.6% vs 2.9%）；`min_codes` 之前屏蔽 STOP。续写取 **2 × 当前段可读 token 数**——否则约 1/6 的续写会在 prior 的句末立刻停止。
- `Gpt::alignment_attention` 接受 prior codes，只返回新 codes 的行；actor 再把列映射回当前段（前缀中的 prior token 与被丢弃的语言前缀 token 不参与），对齐算法不变。
- `SynthesisRequest.context: SpeechContext`（`None` | `Text(上一句文本)` | `Spoken { token_ids, codes }`）与 `continuity: bool`；`AudioChunk.spoken` 带回该段的 token 与 codes（不序列化）。
- 限制：只续写上一段（不累积）；prior codes > 900 或拼接文本 > 600 token 时回退为独立生成。

### 5.2 服务层语义

| 字段 | 语义 |
|------|------|
| `SpeechJob.continuity` | 同一任务内：相邻 segment 续写；相邻 utterance（段落、控制标签边界）在**音色与语言相同**时以上一段 codes 续写。句间静音默认改为 0（模型自带停顿），显式 `intervalSilenceMs` 仍生效；段落停顿与 `[[pause]]` 不变 |
| `SpeechJob.previousText` | 只作用于第一个 utterance：引擎用相同 seed 先生成上一句（取最后一个 segment）的 codes，再续写。不朗读、不得含 `[[`。跨请求的逐行合成用它，结果仍由 `(input, previousText, seed, …)` 唯一确定 |

CLI：`--continuity`、`--previous-text`。HTTP / MCP / Daemon 通过 `SpeechJob` 透传。

### 5.3 默认关闭的理由

续写段句首一两个音节偏弱的情况在测试中出现过（3 次/约 25 段），且自然度提升需要人工试听确认。默认保持此前行为（输出逐字节不变），由使用者 A/B 后启用。

## 6. 协议版本

Daemon 协议升到 **3**：上一轮 `chunk`/`done` 事件新增了必填字段，本轮新增 `annotate` 方法。版本不一致时客户端得到 `DAEMON_VERSION_MISMATCH` 与 `itts daemon restart` 提示，而不是反序列化错误。

## 7. 验证

- director：脚本渲染、`Direction::controls`（含情绪复位）、最后停顿校验、坏 key 拒绝、Debug 脱敏。
- service：单元 ID 校验、`ApiKey` 脱敏、默认风格与序列化、401/403 分类、`previousText` 只作用于首个 utterance、续写时句间静音默认值、拒绝带控制标签的 `previousText`。
- CLI / server：daemon `annotate` 请求反序列化且 Debug 不含 key；`X-DeepSeek-Api-Key` 解析与长度上限；HTTP 排队（第二个请求排队，第一个释放后成功）。
- 引擎模型测试：`SpeechContext::Text` + `continuity` 下两段合成，第二段对齐覆盖其全部可读字。
- 端到端（真实 DeepSeek）：CLI、HTTP、daemon 原始套接字三条路径的 `annotate`；错误 key 返回 `ANNOTATION_UNAVAILABLE`；daemon 日志不含 key。
- 回归：不开续写时，同 seed 输出与改动前逐字节一致。

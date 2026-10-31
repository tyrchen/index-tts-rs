# `itts` Agent-First 可用性与服务化能力设计 (`itts-agent-usability-design`)

- **状态**: 已实施 (Implemented)
- **版本**: 1.0
- **日期**: 2026-10-31
- **关联组件**: `itts`（`apps/cli`：CLI / Daemon / MCP）、`index-tts-rs-server`（`apps/server`：HTTP / Web Studio）、`index-tts-rs-core`、`index-tts-rs-director`、`index-tts-rs-service`（新增）
- **使用文档**: [`docs/usage/itts-agent-integration.md`](../docs/usage/itts-agent-integration.md)
- **前置规范**: [`indextts-rust-prd`](indextts-rust-prd.md)、[`indextts-rust-engine-design`](indextts-rust-engine-design.md)、[`indextts-emotion-director-design`](indextts-emotion-director-design.md)

---

## 1. 背景与目标

`itts` 的主要使用者不是坐在终端前的人，而是 **其他 AI Agent**（Claude Code、Cursor、自研 orchestrator）以及以编程方式集成它的服务。对这类使用者来说，“能不能用”取决于以下几件事：

1. **可发现**：不读源码就能知道装了哪些模型、有哪些 voice、支持哪些参数和控制标签。
2. **可观测**：daemon 是否启动成功、模型是否加载完成、正在做什么、为什么失败。
3. **可预测**：CLI、Daemon、MCP、HTTP 四个入口对同一请求给出同样的语义和同样的结果。
4. **可编程**：所有输出（包括错误）都能被机器稳定解析，错误有稳定的 code，退出码有语义。

当前实现的推理核心已经很扎实（性能、内存、数值对齐都达标），短板集中在 **外壳层**。本文先审计现状，再给出按优先级排列的改进设计。

### 1.1 非目标

- 远程部署与异步任务队列（Lambda / SQS / S3 等）不在本轮范围内。
- 不改动推理管线（GPT / CFM / BigVGAN）的数值行为。
- 不引入鉴权体系；HTTP 服务默认仍只绑定 `127.0.0.1`。

---

## 2. 现状审计

审计基于 `master@393ad94`。问题分为 **Bug**（行为错误）与 **UX**（行为正确但不好用）两类。下表所有条目均已在本轮修复，对应方案见 §5。

### 2.1 Daemon 生命周期（`apps/cli/src/daemon.rs`）

| # | 类型 | 问题 | 位置 |
|---|------|------|------|
| D1 | UX | `itts daemon start` 完全无输出。tracing 默认过滤级别是 `warn`，而启动日志（模型加载耗时、监听地址）都是 `info!`，全部被吞掉。用户无法判断它是在加载、已就绪还是卡死。 | `main.rs:363`、`daemon.rs:183` |
| D2 | UX | 只能前台运行。README 建议 `itts daemon start &`，这在 Agent 的 shell 工具里没法用（后台进程会随 shell 会话结束，或让工具调用一直挂起）。缺少 `--detach`、pidfile、日志文件。 | `README.md:82` |
| D3 | Bug | 模型加载（约 0.5–2 s，峰值 RSS 2.5 GB）发生在 `bind` **之前**。在加载窗口内：(a) 再执行一次 `daemon start` 会通过 `socket_path.exists()` 检查，并行加载第二份模型；(b) 客户端连不上 socket，于是在进程内各自再加载一份模型。没有加锁，也没有“启动中”状态。 | `daemon.rs:131-150` |
| D4 | Bug | 任何连接都会刷新 idle 计时，包括 `status`。Agent 若轮询 `itts daemon status`，daemon 永远不会因空闲退出；文档注释也与实现不符。 | `daemon.rs:195`、`daemon.rs:52` |
| D5 | Bug | 只处理 `SIGINT`，不处理 `SIGTERM`。`kill <pid>`、launchd / 进程管理器停止 daemon 时不会清理 socket 和 engine。 | `daemon.rs:222` |
| D6 | UX | `status` 只返回 pid / uptime / idle / `voicesCount`。缺少：版本、协议版本、模型目录与各子模型状态、Stage 0 extractor 是否常驻、voice 列表、已处理请求数、队列深度、最近一次错误、RSS。 | `daemon.rs:43-58` |
| D7 | Bug | **相对路径在 daemon 的 cwd 中解析**。CLI 原样转发 `--voice ./a.itts_voice`、`--ref ./ref.wav`、`--emo-ref` 的相对路径；daemon 的 cwd 是它启动时所在的目录，结果要么找不到文件，要么读到错误的文件。 | `main.rs:576-590`、`daemon.rs:523` |
| D8 | Bug | **Voice 缓存不一致**。daemon 的 `VoiceStore` 用 `DashMap` 缓存 `SpeakerContext`。另一个进程执行 `itts voice add <id>`（覆盖同名 voice）或 `voice rm <id>` 后，daemon 仍然返回旧的或已删除的 voice，直到重启。 | `voice/store.rs:488-505` |
| D9 | UX | 协议没有版本号，没有 `hello` 握手。CLI 与 daemon 二进制版本不一致时（例如升级后旧 daemon 仍在跑），会出现难以诊断的 JSON 反序列化错误。 | `daemon.rs:78-98` |
| D10 | UX | 无法取消正在进行的合成；客户端断开后 actor 仍会把整段文本合成完。 | `daemon.rs:343-410` |
| D11 | UX | `event` / `method` 用的是裸 `String`，而不是带标签的 enum；错误只有字符串，没有 code。 | `daemon.rs:78-98` |

### 2.2 MCP 服务（`apps/cli/src/mcp.rs`）

| # | 类型 | 问题 | 位置 |
|---|------|------|------|
| M1 | Bug | **MCP 不支持内联控制标签**。CLI 和 HTTP 都会先经过 `plan_utterances`，处理 `[[pause]]` / `[[speed]]` / `[[emo]]` / `[[temp]]`；MCP 直接调用 `engine.synthesize`，标签被当作普通文本送进前端。而 Agent 恰恰是最需要这些标签的使用者（`itts` skill 也在教 Agent 使用它们）。 | `mcp.rs:248-256` |
| M2 | UX | 每个 `itts mcp` 进程都自己加载一份完整模型（约 2.5 GB），不复用 daemon。3 个 Agent 会话就是 7.5 GB；而且 MCP 握手要等模型加载完成之后才开始。 | `mcp.rs:31` |
| M3 | UX | 手写 JSON-RPC，协议版本固定为 `2024-11-05`。workspace 已经声明了 `rmcp = 3.8.0`，但没有使用（dead dependency）。缺少 `outputSchema` / `structuredContent`、进度通知和取消支持。 | `mcp.rs:165`、`Cargo.toml:37` |
| M4 | UX | 请求串行处理：一次长合成会阻塞同一会话里的 `list_voices` / `ping`。 | `mcp.rs:38-49` |
| M5 | UX | 工具集不完整：没有 `status` / `doctor`、`inspect_voice`、`remove_voice`、`annotate`，也没有用于发现能力的 `describe_controls`（情绪类别、参数范围）。 | `mcp.rs:377-438` |
| M6 | Bug | `speed <= 0` 被静默忽略并回退到默认值；`AudioPlayer` 失败只返回 `played: false`，不给原因。 | `mcp.rs:356-359`、`mcp.rs:257-263` |
| M7 | UX | `outputPath` / `refAudioPath` 的相对路径基于 MCP 进程的 cwd 解析，与 Agent 的工作目录不一定一致；返回值里也不是规范化后的绝对路径。 | `mcp.rs:286` |

### 2.3 请求语义在各入口不一致

同一个“合成请求”在 4 个地方分别构造，规则互不相同（违反 DRY，也让 Agent 无法预测行为）：

| 维度 | CLI `synth` | Daemon 协议 | MCP | HTTP `/v1/audio/speech` |
|------|-------------|-------------|-----|--------------------------|
| 构造位置 | `main.rs:574` | 透传 core `SynthesisRequest` | `mcp.rs:327` | `web.rs` `SpeechPayload::into_request` |
| 语速 | `--duration-factor`（时长倍数） | `durationFactor` | `speed`（取倒数，≤0 被忽略） | `speed`（取倒数，校验 `0.25..=4.0`） |
| 内联标签 | ✅ | ✅（由 CLI 先展开） | ❌ | ✅ |
| 段落停顿 | `--paragraph-pause` | — | ❌ | `paragraphPauseMs` |
| `--ref` 零样本 | ✅ | ✅ | ✅ | ❌ |
| 采样参数 | 全部可调 | 全部 | ❌ | ❌（只读配置） |
| voice 路径判定 | 扩展名或 `is_file()` | — | 仅扩展名 | 仅 `VoiceId` |
| 默认语言 | `--lang zh` | — | `zh` 硬编码 | `config.default_language` |

### 2.4 CLI 通用体验（`apps/cli/src/main.rs`）

| # | 类型 | 问题 |
|---|------|------|
| C1 | UX | 每个子命令都有自己的 `--json`，但 **错误从不输出 JSON**：失败时由 `anyhow` 往 stderr 打一行文本，退出码固定为 1。Agent 无法区分“voice 不存在”“模型没下载”“参数非法”“daemon 版本不符”。 |
| C2 | UX | 没有一站式诊断命令。首次使用时要依次执行 `models info`、`voice list`、`daemon status`，才能拼出环境全貌；`models info` 在模型缺失时直接报错，而不是给出可执行的修复建议。 |
| C3 | UX | `models pull` 下载约 2 GB 的权重时完全没有进度；已存在时也不校验完整性（只看必需文件是否存在）。 |
| C4 | UX | `synth` 必须显式指定 `--voice` 或 `--ref`；配置中没有 `default_voice`。 |
| C5 | UX | `--jsonl` 流在第一个 `chunk` 之前没有任何事件：不报 `started`，也不报“使用 daemon / 在进程内加载模型中”。 |
| C6 | UX | `--voice` 的歧义解析：`--voice narrator` 会在 cwd 下恰好存在名为 `narrator` 的文件时被当作文件路径。 |

## 3. 设计原则

1. **单一请求模型（One DTO）**：唯一一个对外的、`serde` 可序列化的 `SpeechJob` 请求类型。CLI、Daemon、MCP、HTTP 都只做“传输层 → `SpeechJob`”的映射，校验与标签展开只在 service 层做一次。
2. **机器优先，人类友好**：每个命令都有稳定的 JSON 形态（成功和失败都有）；人类可读输出只是一种渲染。stdout 只放结果，日志和进度放 stderr（或 JSONL 事件）。
3. **稳定错误码**：错误由 `code`（稳定的 `SCREAMING_SNAKE_CASE` 枚举）+ `message`（人读）+ `hint`（可执行的修复建议）组成，并映射到确定的进程退出码。
4. **显式就绪（Readiness）**：任何“启动”类操作都明确报告 *启动中 → 就绪 / 失败*，失败时带原因。
5. **单例重资源**：一台机器上每个 socket 只有一份模型常驻；CLI 与 MCP 都作为 daemon 的客户端。
6. **尽早失败**：参数、路径、voice 是否存在都在提交 GPU 工作之前校验，避免生成一半音频后才报错。

---

## 4. 架构

### 4.1 新增 `crates/service`（`index-tts-rs-service`）

请求构造、`plan_utterances`、chunk 合并、静音拼接等逻辑原本在 `apps/cli/src/main.rs`、`apps/cli/src/mcp.rs`、`apps/server/src/web.rs` 中重复了三份，现已下沉到一个传输无关的 crate：

```text
crates/service/src/
├── lib.rs
├── request.rs   # SpeechJob DTO + AdvancedOptions + PathPolicy + prepare() → PreparedJob
├── run.rs       # JobRun（拉取式游标，应用段落停顿/句间静音）、JobAccumulator、run_job
├── events.rs    # JobEvent { Started, Chunk, Done, Error }、ChunkInfo（可选 base64 PCM）、JobSummary
├── error.rs     # ErrorCode（14 个稳定码）+ ServiceError { code, message, hint } + exit code
└── catalog.rs   # describe_controls 目录 + JSON Schema（schemars）
```

依赖方向：`apps/* → service → {director, core}`。

### 4.2 `SpeechJob`：统一请求模型

字段全部是 JSON 原生类型（便于生成清晰的 JSON Schema），在 `SpeechJob::prepare(&EngineConfig, PathPolicy)` 中统一校验并编译为领域类型：

| 字段 | 说明 |
| --- | --- |
| `input` | 文本（≤ 16 KiB），支持内联控制标签 |
| `voice` / `voiceFile` / `refAudio` | 三选一；都省略时使用配置的 `default_voice`，否则返回带已注册 voice 列表的 hint |
| `emoRef`、`emoAlpha`、`emo` | 情绪参考音频（要求同时给出 `refAudio`）、混合权重、8 类情绪向量 |
| `lang`、`speed` | 语言；语速 `0.25..=4.0`（越大越快，全入口统一语义） |
| `paragraphPauseMs`、`intervalSilenceMs` | 段落停顿、句间静音（`0..=5000`） |
| `outputPath` | 写出 WAV 的路径（daemon 端直接写文件，不经 socket 回传 PCM） |
| `advanced` | `greedy`、`seed`、`temperature`、`topK`、`topP`、`repetitionPenalty`、`maxMelTokens`、`maxTextTokens`、`nTimesteps`、`cfgRate`，未给出的使用配置默认值 |

- `deny_unknown_fields`，拼错字段名直接报错。
- `PathPolicy::LocalAbsolute`（CLI / daemon / MCP）要求路径为绝对路径；客户端用 `SpeechJob::resolve_paths(cwd)` 按调用方工作目录规范化输入文件（并在本地验证文件存在），修复 D7 / M7。`PathPolicy::Denied`（HTTP）拒绝一切文件字段。
- `prepare` 会检查计划中所有 `VoiceId`（包括文中 `[[voice=...]]` 切换）是否已注册，未注册直接返回 `VOICE_NOT_FOUND`。
- CLI 未显式给出的采样参数不再硬编码默认值，而是取自配置文件（原实现会忽略配置中的 `sampling` / `cfm`）。
- JSON Schema 通过 `itts schema speech-job` 输出，MCP 工具的 `inputSchema` 也由同一类型生成。

### 4.3 事件流

所有入口共用同一事件序列，CLI `--output jsonl`、daemon 协议、MCP progress 都是它的不同编码：

```text
started  { speaker, language, utterances, backend: "daemon" | "inProcess" }
chunk    { utteranceIndex, totalUtterances, segmentIndex, totalSegments, samples, sampleRate,
           durationSecs, melTokens, timings, pcmBase64? }
done     { outputPath?, sampleRate, samples, durationSecs, synthesisSecs, wallSecs, rtf,
           utterances, segments, melTokens, timings }
error    { error: { code, message, hint? } }
```

### 4.4 稳定错误码与退出码

| Exit | Code |
| --- | --- |
| 1 | `INTERNAL` |
| 2 | `INVALID_ARGUMENT`（clap 的用法错误同样以 2 退出） |
| 3 | `VOICE_NOT_FOUND` |
| 4 | `MODEL_UNAVAILABLE` |
| 5 | `AUDIO_DECODE_FAILED`、`AUDIO_OUTPUT_UNAVAILABLE` |
| 6 | `DAEMON_UNAVAILABLE`、`DAEMON_VERSION_MISMATCH` |
| 7 | `BUSY` |
| 8 | `CANCELLED` |
| 9 | `TIMEOUT` |
| 10 | `ANNOTATION_UNAVAILABLE`、`ANNOTATION_FAILED` |
| 11 | `IO` |

`IndexTtsError` / `DirectorError` 通过 `From<&T>` 映射到 `ServiceError`。CLI 顶层沿 `anyhow` 的 context 链向下查找可分类的错误，`message` 保留完整的 context 链。全局 `--output text|json|jsonl`（环境变量 `ITTS_OUTPUT`）取代了原来各子命令的 `--json` / `--jsonl`。

---

## 5. 实施内容

### 5.1 Daemon 生命周期与可观测性（D1–D6）

```text
$ itts daemon start
  ▸ socket               ~/.cache/index-tts-rs/daemon.sock
  ▸ model                ~/.cache/huggingface/hub/models--yunfengwang--IndexTTS-2.5-mlx/snapshots/660a…
  ▸ frontend                   6 ms
  ▸ gpt                      140 ms
  ▸ semantic_codec            21 ms
  ▸ length_regulator           5 ms
  ▸ cfm                       77 ms
  ▸ bigvgan                  232 ms
  ▸ emotion_projection         0 ms
  ▸ engine                   514 ms
  ▸ voices               demo_voice, tyr
✔ itts daemon 0.1.0 ready in 515 ms
  pid      41320
  socket   ~/.cache/index-tts-rs/daemon.sock
  model    ~/.cache/huggingface/hub/models--yunfengwang--IndexTTS-2.5-mlx/snapshots/660a…
  voices   2 — demo_voice, tyr
  idle     auto-stops after 600 s without jobs
  log      ~/.cache/index-tts-rs/daemon.log
```

- **后台化**：`daemon start` 以隐藏子命令 `itts daemon run` re-exec 自身：子进程放入独立进程组（`CommandExt::process_group(0)`，无需 `unsafe`），stdin 关闭，stdout 回传 JSONL 启动事件，stderr 追加写入日志文件，工作目录为 `/`。父进程渲染进度，收到 `ready` 后以 0 退出，收到 `failed` 则以对应错误码退出，子进程意外退出时附上日志尾部。`--foreground` 保留前台运行（供 launchd 托管或调试）。
- **引擎加载进度**：core 新增 `IndexTtsEngine::start_with_observer` 与 `StartupEvent`（`ModelResolved` / `ComponentLoaded` / `EngineReady`），由 worker 线程逐组件回调。
- **单例与先 bind 后加载（D3）**：`<socket>.lock` 上的 `File::try_lock` 保证单实例；拿到锁后立即 bind socket、写 pidfile，然后才加载模型。加载期间 `status` 返回 `state: "loading"`，`synth` 在 daemon 端等待就绪（上限 300 s），不会在客户端再加载一份。第二个 `start` 返回 “already running” 并以 0 退出（幂等）。
- **空闲判定（D4）**：只有真实任务刷新空闲计时，`status` / `hello` 不刷新；有任务在跑时空闲时间恒为 0。
- **信号（D5）**：`SIGTERM` / `SIGINT` 触发优雅关闭（停止 accept → 等待在途任务最多 10 s → 清理 socket 与 pidfile → `engine.stop()`）；`SIGHUP` 被忽略，关闭终端不会杀掉 daemon。
- **status（D6）**：`state`、`version`、`protocolVersion`、`pid`、`socketPath`、`logPath`、`modelDir`、`voicesDir`、`uptimeSecs`、`idleSecs`、`idleTimeoutSecs`、`voices[]`、`engine`（MLX active / peak 内存、Stage 0 extractor 是否常驻，由 worker 写入的原子量提供，查询无需排队等 GPU）、`jobs`（inFlight / maxInFlight / completed / failed / cancelled）、`lastError`。
- **新命令**：`daemon restart`、`daemon ensure`（未运行则启动）、`daemon logs [-n N] [-f]`。日志为 JSON 行，启动时超过 10 MiB 会轮转为 `.log.1`。

### 5.2 Daemon 协议 v2（D7–D11）

- 首条消息 `hello { clientVersion, protocolVersion }` → `welcome`；版本不符返回 `DAEMON_VERSION_MISMATCH`（hint：`itts daemon restart`）。对于没有握手的 v1 daemon，其回复无法按 v2 解析，同样识别为版本不匹配；`daemon stop` / `restart` 会改用 v1 的 `stop` 报文并等待回复，因此升级后也能替换掉旧 daemon。
- `method` / `event` 改为带标签的 enum，错误携带 `ServiceError`。
- 同一连接按 `id` 多路复用；`cancel { target }` 取消指定请求；客户端断开会取消该连接上的全部请求。取消时丢弃 `JobRun`，engine actor 在下一个 segment 边界检测到 receiver 已断开后停止，这一机制原本就存在。
- 在途请求数达到 `--max-in-flight`（默认 16，范围 1..=256）时返回 `BUSY`。
- `includePcm: false` 时 chunk 事件不带 PCM；若同时给出 `outputPath`，WAV 由 daemon 直接写盘。CLI 只写文件时走这条路径，只有 `--play` 或 `--out -` 才回传 PCM。
- `voiceAdd` 复用 daemon 常驻的模型，提取完成后驱逐 Stage 0 编码器，回到合成基线内存。
- 行长度上限 256 KiB，超出直接拒绝且不缓冲。

### 5.3 Voice 缓存一致性（D8）

`VoiceStore` 缓存项记录文件的 `(mtime, len)`，每次 `get` 先做一次 `stat`：文件被替换则重新加载，文件被删除则驱逐缓存并返回 `VoiceNotFound`。磁盘是唯一事实来源。

### 5.4 MCP（M1–M7）

- 基于 `rmcp`（关闭 `macros` feature，手写 `ServerHandler`），协议版本由 rmcp 协商。
- **不再持有 engine**：所有工具通过 daemon 执行，首次使用时 detach 启动 daemon；多个 Agent 会话共享一份模型，`initialize` 立即返回。
- 合成走 `SpeechJob`，因此与 CLI 一样支持内联标签和段落停顿（修复 M1）。
- 工具：`status`（即 `DoctorReport`）、`describe_controls`、`synthesize`（`outputPath` 必填）、`speak`、`list_voices`、`inspect_voice`、`register_voice`、`remove_voice`、`annotate`。`synthesize`、`speak`、`status`、`describe_controls` 声明了 `outputSchema`。
- 成功结果带 `structuredContent`；失败返回 `isError` 结果，其结构化内容为 `{code, message, hint}`。
- 带 progress token 的请求每合成一段发送一次 `notifications/progress`；MCP 取消会断开 daemon 连接，从而取消对应任务。rmcp 并发处理请求（修复 M4）。
- `speak` 的播放在专用线程上进行（`Playback`）：`CoreAudio` stream 不是 `Send`，不能跨 `.await` 持有。播放失败返回 `AUDIO_OUTPUT_UNAVAILABLE` 及原因（修复 M6）。

### 5.5 CLI（C1–C6）

- 全局 `--output`；结构化错误与退出码（C1）。
- `itts doctor`：platform / model / voices（含 `default_voice` 校验）/ daemon / director / audio，每项给出 `ok | warn | error` 和 hint（C2）。
- `models pull` 通过 hf-hub `ProgressHandler` 按 10% 步进显示进度（`jsonl` 模式输出 `progress` 事件），下载后用快照校验确认完整（C3）。
- `default_voice` 配置项（C4）。
- `--output jsonl` 的首个事件为 `started`（C5）。
- `--voice` / `--voice-file` / `--ref` 互斥（C6）；`--speed` 取代 `--duration-factor`；未指定 `--out` / `--play` 时直接报错，不再白白合成后丢弃。
- `itts schema speech-job|events|error|controls`。
- daemon 不可用（如版本不符）时向 stderr 打印一行提示并回退到进程内合成，不影响 stdout 上的结构化结果。

### 5.6 HTTP 服务

`/v1/audio/speech` 与 `/api/speech/stream` 的请求体直接使用 `SpeechJob`（`PathPolicy::Denied`），合成走 `JobRun`；错误响应增加 `code` 与 `hint` 字段，原有的中文提示保持不变。

---

## 6. 验证

| 场景 | 结果 |
| --- | --- |
| `itts daemon start` 输出逐组件加载耗时与就绪摘要；重复执行 `start` 幂等 | ✅ |
| 在任意 cwd 下，`synth --voice-file ./x.itts_voice -o out.wav` 经 daemon 成功（daemon cwd 为 `/`） | ✅ |
| `VOICE_NOT_FOUND` / `INVALID_ARGUMENT` 的 text 与 JSON 形态及退出码 3 / 2 | ✅ |
| 客户端被 kill 后，daemon 记一次 `cancelled`，`inFlight` 立即归零 | ✅ |
| 轮询 `status` 期间 `idleSecs` 持续增长 | ✅ |
| `kill -TERM <daemon>` 后 socket 与 pidfile 被清理 | ✅ |
| 旧版 v1 daemon：`status` 报 `DAEMON_VERSION_MISMATCH`，`synth` 回退到进程内，`stop` 能正常停止 | ✅ |
| daemon 运行时 `voice add` → `synth` → `voice rm` → `synth` 返回 `VOICE_NOT_FOUND` | ✅ |
| MCP：`initialize`、`tools/list`、`describe_controls`、带 progress 的 `synthesize`、结构化错误、`status`；daemon 未运行时自动拉起 | ✅ |
| `cargo test --workspace`（含模型的 server 集成测试 `--ignored`）、`cargo clippy -D warnings`、`cargo +nightly fmt` | ✅ |

## 7. 后续可选项

- `itts daemon install`：生成 launchd plist，实现开机常驻（目前可直接用 `daemon start --foreground` 交给 launchd 托管）。
- 压缩输出格式（mp3 / opus）。

# IndexTTS-2.5 Rust 高性能语音合成系统产品需求文档 (`indextts-rust-prd`)

## 1. 产品愿景与定位

**`index-tts-rs`** 是基于 Apple Silicon (M 系列芯片) 原生 GPU 加速的工业级零样本多语种情感语音合成（TTS）与声音克隆系统。它以 [`yunfengwang/IndexTTS-2.5-mlx`](https://huggingface.co/yunfengwang/IndexTTS-2.5-mlx) 模型为算法基座，采用 **Rust 2024 + `mlx-rs`** 完全重写，旨在解决现有 Python 推理方案在生产集成与 AI Agent 调用中的三大痛点：

1. **内存与启动过重**：原版 Python 方案常驻内存高达 **4.8 GB**，每次 CLI 启动需加载庞大的 Python 解释器与 `transformers`/`librosa` 依赖，冷启动耗时数秒。
2. **难以被其他应用内嵌集成**：Python 依赖复杂的虚拟环境（`uv`/`conda`），难以作为轻量级、类型安全的原生库（Crate）直接编译进桌面端应用、游戏引擎或高性能后端服务。
3. **对 AI Agent 不友好**：缺乏面向 Agent 的毫秒级常驻响应（Daemon）、结构化 JSON/JSONL 遥测输出、标准输入流式朗读（Streaming Stdin）以及原生 MCP (Model Context Protocol) 协议支持。

### 1.1 双核产品形态
- **形态一：可嵌入式 Rust 核心库（Crate: `index-tts-rs-core`）**
  供第三方 Rust 开发者通过 `cargo add` 集成，提供编译期类型安全、零 `unwrap`、支持异步流式（`Stream<Item = Result<AudioChunk>>`）与 Actor 并发调度的原生 API。
- **形态二：面向 AI Agent 与开发者的全能命令行与服务套件（App: `itts` CLI & Server）**
  提供单二进制 CLI，内置 **音色快照管理（`.itts_voice`）**、**自动驻留后台守护进程（Auto-Warm Daemon）**、**MCP Server (stdio)** 以及 **OpenAI 兼容 HTTP/WebSocket 语音接口**。

---

## 2. 目标用户与核心使用场景 (Personas & Use Cases)

| 目标用户角色 | 典型使用场景 | 核心痛点与诉求 |
| :--- | :--- | :--- |
| **1. AI Agent 系统（Claude Code / OpenClaw / 本地语音助手）** | Agent 在完成任务、播报状态、或与用户进行实时语音对话时，通过 CLI、Unix Socket 或 MCP Tool 触发语音合成与实时播放。 | 1. **零冷启动等待**（连续多次发声不能每次重加载 2GB 模型）；<br>2. **低首包延迟（TTFA < 350ms）**，边生成文字边开口说话；<br>3. **结构化 JSON 输出与确定性退出码**，方便 Agent 解析结果与耗时。 |
| **2. Rust 应用与系统开发者** | 在 macOS 原生应用（Tauri / GPUI / Swift-Rust FFI）、数字人管线或本地 AI 服务中直接集成语音合成与零样本克隆能力。 | 1. **干净的 Cargo 依赖**，完全脱离 Python 运行时；<br>2. **低内存占用**（合成模式常驻 $\le 1.75\text{ GB}$）；<br>3. **优雅的 Builder + Async Stream API**，支持中途取消（Cancellation）。 |
| **3. 内容创作者与多语种配音工作流** | 使用一段 3~15 秒参考音频克隆音色，批量合成中英日粤混合长文本有声书、播客或视频配音，并对多音字/专有名词精确注音。 | 1. **音色一次提取、永久秒级复用**（`.itts_voice` 音色包仅 1.5MB）；<br>2. **支持 `<汉字\|PIN1 YIN1>` 与 `glossary.yaml` 精准发音控制**；<br>3. **高实时率（RTF < 0.20，即 5x 实时速度以上）**。 |

---

## 3. 功能需求说明 (Functional Requirements)

### FR-1：两阶段解耦与 `.itts_voice` 音色包管理（Voice Snapshot System）
- **FR-1.1 零样本音色提取**：支持输入任意 `3s ~ 15s` 的常见音频文件（WAV / MP3 / FLAC / M4A，任意采样率与声道），自动完成重采样、特征提取并生成 `SpeakerContext`。
- **FR-1.2 轻量级 `.itts_voice` 音色快照持久化**：
  - 将提取出的 `SpeakerContext`（包含 `conds`、`prompt_condition`、`ref_mel`、`style` 四个张量）及元数据（源音频哈希、采样率、时长、创建时间、音色别名）保存为单文件 `<name>.itts_voice`（体积约 **1.5 MB**）。
- **FR-1.3 纯合成免编码器加载（Encoder-Free Synthesis）**：
  - 当传入预编译的 `.itts_voice` 音色名或文件路径时，引擎**完全跳过加载** `w2v-bert-2.0`（2.2GB）、`EmoVec`（622MB）和 `CampPlus`（27MB），仅加载 1.68 GB 合成主干权重即可直接合成。
- **FR-1.4 独立情感参考与混合控制（Emotion Blending）**：
  - 支持默认「说话人音频兼作情感参考（`emo_alpha = 1.0`）」，同时支持可选传入第二段独立的情感参考音频（`--emo-ref`）与情感强度系数（`--emo-alpha 0.0..=2.0`），在提取阶段合成定制化的 `conds = spk_cond + emo_alpha * emovec`。

### FR-2：多语种文本前端与精细发音控制（Multilingual Frontend）
- **FR-2.1 多语种与混合语种支持**：支持中文（`zh`）、英文（`en`）、日文（`ja`）、粤语（`yue`）及中英混合（`zh/en`）合成。
- **FR-2.2 显式注音覆盖语法（Pronunciation Override）**：
  - 支持在文本中使用 `<文本|注音>` 标记：
    - 中文拼音带声调（如 `请把<行|HANG2>业报告发给<行|XING2>长`），自动将 `jqx` 后的 `u/ü` 规范化为 `v` 并包裹 `<|SPECIAL_TOKEN_2|>`；
    - 英文音素标注包裹 `<|SPECIAL_TOKEN_1|>`；
    - 日文平假名/片假名注音直接内联替换。
- **FR-2.3 术语词典与文本正则化**：
  - 支持加载 YAML 格式的 `glossary.yaml` 进行多语种专有名词替换；
  - 纯 Rust 实现技术术语（如 `IndexTTS-2.5`、`UTF-8`）、人名间隔号、拼音声调保护与数字/标点规范化。
- **FR-2.4 智能分句与 Token 预算控制**：
  - 根据 `max_text_tokens_per_segment`（默认 `120`，硬上限 `600`）与标点符号自动切分长文本，保证 `<|SPECIAL_TOKEN_*|>` 原子块不被截断。

### FR-3：高性能合成与流式音频输出（Synthesis & Streaming）
- **FR-3.1 完整采样与扩散超参控制**：
  - T2S GPT 参数：`greedy`、`seed`、`top_k` (默认 30)、`top_p` (默认 0.8)、`temperature` (默认 0.8)、`repetition_penalty` (默认 10.0)、`max_mel_tokens` (默认 1500)。
  - S2M & CFM 参数：`duration_factor` (语速控制，默认 1.0)、`n_timesteps` (ODE 步数，默认 25，支持 10~50 调节以平衡速度与音质)、`cfg_rate` (默认 0.7)、`interval_silence_ms` (句间静音毫秒数，默认 200ms)。
- **FR-3.2 分句流式生成（Sentence-Level Pipelined Streaming）**：
  - 长文本合成时，第一句完成 `GPT -> Codec -> Regulator -> CFM -> BigVGAN` 后**立即通过异步 Stream 产出首个 `AudioChunk`**，同时后台流水线并行处理后续句子。
- **FR-3.3 原生音频播放与多格式落盘**：
  - 支持直接写入 `22050 Hz, 16-bit PCM Mono` `.wav` 文件或裸 PCM 流（stdout）；
  - 支持 `--play` 通过 macOS CoreAudio (`cpal`) 实时低延迟流式播放。

### FR-4：Rust Library Crate (`index-tts-rs-core`) 集成体验
- **FR-4.1 编译期安全的 Builder API**：使用 `typed-builder` 构建 `EngineConfig`、`SynthesisRequest` 与 `VoiceExtractRequest`。
- **FR-4.2 同步/异步双模与流式接口**：
  - `engine.synthesize(req).await -> Result<SynthesisOutput>`（一次性返回完整 PCM 与各阶段耗时统计）；
  - `engine.synthesize_stream(req).await -> Result<SynthesisStream>`（逐句返回 `AudioChunk`，支持 Drop 即取消后续计算）。
- **FR-4.3 线程安全与 Actor 隔离**：
  - 由于底层 Metal/MLX 计算图执行具有设备上下文亲和性，Crate 内部采用专用的推理 Worker 线程/Actor 持有 MLX 模型状态，对外暴露可自由 `Clone + Send + Sync` 的轻量级 `IndexTtsEngine` 句柄（通过 `flume` 通道通信）。

### FR-5：面向 AI Agent 的 CLI、常驻 Daemon 与 MCP Server
- **FR-5.1 CLI 子命令体系 (`itts`)**：
  - `itts models pull / info`：自动从 Hugging Face 下载或校验模型权重。
  - `itts voice add / list / inspect / rm`：管理本地 `.itts_voice` 音色库。
  - `itts synth`：核心合成命令，支持 `--voice <name>` 或 `--ref <wav>`、`| itts synth --stdin`、`--play`、`--out <path>`、`--json` / `--jsonl`。
  - `itts daemon start / stop / status`：管理后台常驻 Unix Domain Socket 守护进程（支持空闲超时自动退出，如 `--idle-timeout 10m`）。
  - `itts mcp`：启动标准 MCP (Model Context Protocol) stdio Server，向 Agent 暴露 `speak`、`synthesize_to_file`、`register_voice`、`list_voices` 四个工具。
- **FR-5.2 自动守护进程透明加速（Auto-Daemon Fast Path）**：
  - 当 Agent 执行 `itts synth` 时，若本地 Daemon 正在运行，CLI 自动通过 Unix Socket 将请求转发给 Daemon 执行，省去 100% 的模型加载时间（响应延迟降至毫秒级）。

---

## 4. 非功能需求与性能指标 (Non-Functional Requirements & SLAs)

### 4.1 性能与资源指标（以 Apple Silicon M2/M3/M4 Max/Pro MBP 为基准）

| 指标项 | 原版 Python `index-tts-2.5-mlx` | `index-tts-rs` 目标值 | 核心实现手段 |
| :--- | :--- | :--- | :--- |
| **常驻内存占用（缓存音色合成）** | ~4.80 GB | **$\le$ 1.75 GB** (降低 >63%) | 剥离 Stage 0 音色编码器 (2.85GB) + 剔除 `semantic_codec` 编码端死权重 (101MB) |
| **全量内存峰值（首次提取新音色）** | ~5.10 GB | **$\le$ 3.90 GB** | 剔除 `w2v-bert` 第 17~23 层死权重 (677MB) + 提取完毕后可立即释放 Stage 0 |
| **合成引擎冷启动耗时（SSD 热页）** | ~2.80 s | **$\le$ 0.25 s** | 纯 Rust 二进制 + SafeTensors 按需加载，无 Python / `transformers` 导入开销 |
| **Daemon 热态请求响应开销** | 不支持（每次重跑进程） | **$\le$ 5 ms** | Unix Domain Socket + 内存常驻 `DashMap` 音色池 |
| **实时率（RTF，越低越快）** | ~0.35 - 0.50 | **$\le$ 0.20** (>5x 实时速度) | GPT In-Place KV Cache + CFM 全 GPU 驻留 ODE 循环（消除 25 次 CPU/NumPy 往返） |
| **流式首包音频延迟（TTFA）** | 整段合成完才输出（数秒） | **$\le$ 400 ms**（首短句） | 标点切句 + 三级 Actor 流水线并发 |

### 4.2 数值精度与音质验收标准
- 与原版 Python `index-tts-2.5-mlx` 在相同输入与固定随机种子/噪声（`greedy=True, z=fixed`）下逐模块对比：
  1. **Tokenizer 输出 IDs**：100% 比特级一致（Bit-exact match）。
  2. **DSP 特征（`mel_spectrogram`、`campplus_fbank`、`w2v_features`）**：余弦相似度 $\ge 0.9999$，最大绝对误差 $< 10^{-4}$。
  3. **`SpeakerContext` 各张量**：余弦相似度 $\ge 0.9995$。
  4. **Greedy GPT 生成的 Semantic Codes**：100% 序列一致（或在浮点边界步重合率 $\ge 99\%$）。
  5. **CFM Mel 频谱（给定相同初始噪声 `z`）与 BigVGAN 输出波形**：余弦相似度 $\ge 0.999$，听感完全无退化、无拼接爆音。

### 4.3 安全性与代码质量要求（遵循 [AGENTS.md](../../AGENTS.md)）
- **零 `unsafe` 业务代码**：所有 Workspace Crate 根节点强制声明 `#![forbid(unsafe_code)]`。
- **零 Panic 边界防护**：外部输入（文本长度、音频时长/大小、文件路径、超参数范围）在进入引擎前通过 `validator` 与 Newtype 严格校验，严禁在非测试代码中使用 `unwrap()`、`expect()` 或无界切片 `[]`。
- **路径穿越防护**：CLI 与 Server 处理 `--ref`、`--out` 与音色名时，严格校验字符白名单与规范化路径，拒绝 `..` 与非法字节。

# `itts` 字级时间戳、可复现种子与 48 kHz 高保真输出设计 (`itts-timestamps-and-hifi-design`)

- **状态**: 已实施 (Implemented)
- **版本**: 1.0
- **日期**: 2026-10-31
- **关联组件**: `index-tts-rs-core`、`index-tts-rs-director`、`index-tts-rs-service`、`itts`（CLI / Daemon / MCP）、`index-tts-rs-server`（HTTP）
- **调研**: [GPT 注意力对齐实验](../docs/research/gpt-attention-alignment-research.md)、[语音带宽扩展选型](../docs/research/speech-bandwidth-extension-research.md)
- **前置规范**: [`itts-agent-usability-design`](itts-agent-usability-design.md)

---

## 1. 背景与目标

外部应用（首个用户是视频生产管线 reshape，目前用 ElevenLabs）要把 `itts` 当作逐句旁白引擎使用。相对 ElevenLabs，`itts` 有四个能力缺口：

| # | 缺口 | 下游影响 |
|---|------|----------|
| G1 | 没有字级时间戳 | 无法把动画/字幕锚定到某个词；无法按"首末有声字"裁剪句尾杂音；无法对齐文本做漏读/口吃筛查 |
| G2 | 事件不带文本 | 流式 `chunk` 只有序号，调用方无法知道这段音频读的是哪句话 |
| G3 | 不指定 seed 时不可复现 | GPT 采样用固定默认种子而 CFM 噪声取全局 RNG：每次结果不同，且无法得知用了什么随机性，"挑一个满意的 take 锁定"无从谈起 |
| G4 | 只有 22.05 kHz | 有效带宽 ≤11 kHz，视频旁白听感偏闷；简单重采样不增加内容 |

非目标：

- 跨句韵律衔接（ElevenLabs 的 `previous_text`/`next_text`）。IndexTTS 每个 segment 独立生成，模型本身没有这个输入；以固定 seed + 情绪向量保持一致性。
- 训练新模型。所有新增能力只用现有权重或公开的预训练权重。

## 2. 设计总览

```text
SpeechJob{input, timestamps:true, sampleRate:48000, seed?}
  │ service: 解析 seed（缺省则随机生成并回报）
  ▼
director::plan_utterances ──► PlannedUtterance{request, pause_before_ms, source: Range<input bytes>}
  │                           + 控制标签区间（剥离后得到 spoken text）
  ▼
core actor（每个 segment）
  GPT 生成 codes ─► [align] teacher-forcing 前 6 层 → L4H10+L5H10 注意力
                   → 单调 Viterbi → 波形能量修正静音 → token 时间
  CFM → BigVGAN(22.05k) ─► [hifi] LavaSR v2 带宽扩展 → 48k
  AudioChunk{pcm, alignment: token 时间 + 对应 request.text 的字符区间}
  ▼
service JobRun
  chunk 时间轴偏移（含停顿/句间静音）+ request.text → input 的字符 diff 映射
  → ChunkInfo{text, startSecs, alignment}  /  JobSummary{seed, alignment, sentences}
```

## 3. 字级时间戳（G1）

### 3.1 对齐信号：GPT 自身的文本注意力

本版本管线中 S2Mel 的输入只来自 GPT 生成的语义 codes：`codes → codec(2× repeat) → LengthRegulator(最近邻线性展开) → CFM → BigVGAN`。因此第 *i* 个 code 精确对应音频的 `[i·Δ, (i+1)·Δ)`，`Δ = segment 时长 / codes 数`（≈40 ms）。只要知道每个 code 在"读"哪个文本 token，就得到了时间戳。

实验（见调研文档）对 24 层 × 20 头逐一评估，与 MMS 强制对齐器（torchaudio `MMS_FA`）对比：

- 第 4 层第 10 头（L4H10）单头即最优；与 L5H10 平均后，生产实现的中文字起点平均绝对误差 19–23 ms、英文词起点 42–44 ms，整体中位数 17 ms、96% ≤ 100 ms；情绪与 0.8–1.3× 变速条件下 98% ≤ 100 ms。
- 静音期间注意力落在 `<BOS>` 上（attention sink），停顿内注意力会提前看向下一个字——必须用波形能量修正边界。

因此只需 teacher-forcing 重放 GPT 的前 6 层（`0..=5`），取 2 个头，成本约为一次 prefill 的 1/4（实测见 §8）。不引入任何新模型或权重。

### 3.2 解码算法（`core::align`）

输入：注意力 `p[N, T]`（N 个 code 查询 × `<BOS>, tokens…, <EOS>`），token 列表（带字节区间），segment PCM。

1. **可读 token**：排除 `<|lang|>` 等特殊 token、空白和标点；剩余 K 个 token 作为状态。
2. **发射**：对每帧在 K 个可读列上重新归一化后取对数。
3. **单调 Viterbi**：状态只能停留或前进 1；从第一个可读 token 开始、在最后一个结束；每个 token 至少 1 帧。O(N·K)，N≤1815、K≤600。
4. **能量修正**：PCM 以 10 ms 帧计算 RMS 包络，阈值 `max(1e-3, P95 × 0.05)`；连续 ≥80 ms、且不触及首尾的低能量区为"停顿"。首 token 起点钉到第一个有声帧，末 token 终点钉到最后一个有声帧；每个停顿归属距其最近（≤0.2 s）的 token 边界：前一 token 终点**设为**停顿开始、后一 token 起点**设为**停顿结束（双向，因为解码边界可能落在停顿任一侧）。
5. **塌缩修复**：注意力整段落在静音里的 token 会被压成零长度，与相邻 token 平分时长，保证每个 token 时长为正、单调不重叠。
6. **输出**：每个可读 token 的 `[start, end)` 秒（相对 segment 音频起点）。

### 3.3 坐标映射

时间戳必须落在调用方给出的文本上，而文本经过了三层变换：

| 层 | 文本 | 变换 |
|----|------|------|
| input | `SpeechJob.input` | 原文，可含 `[[...]]` 控制标签 |
| utterance | `PlannedUtterance.request.text` | 剥离控制标签、按空行分段、行尾补 `，`/`。`、trim |
| normalized | 前端 `working` 文本 | 标点归一、数字/符号读法展开、大小写折叠、`<字\|拼音>` 注音、glossary |

不侵入 normalizer（700 行规则），统一用 **字符级 diff 映射**（`core::align::TextMap`，Myers 差分，比较时对大小写与全/半角标点折叠）：相等段逐字映射，替换段（如 `2026`↔`二零二六`）整体对应并按字符数比例分摊时间。编辑距离超过上限时退化为按长度比例映射，不报错。

- core：normalized ↔ request.text，产出 `AudioChunk.alignment`（时间相对该 segment，区间是 request.text 的字符下标）。
- service：request.text ↔ `input[source]`，再叠加 chunk 在整段音频时间轴上的偏移。

### 3.4 输出格式

与 ElevenLabs `with-timestamps` 及 reshape 现有 `Alignment` 结构一致：

```jsonc
"alignment": {
  "text": "你好，世界。",          // input 去掉 [[...]] 控制标签后的文本
  "start": [0.31, 0.47, 0.62, 0.66, 0.80, 0.95],   // 按 Unicode 标量（JS: Array.from(text)）索引，单位秒
  "end":   [0.47, 0.62, 0.66, 0.80, 0.95, 0.95]
},
"sentences": [ { "text": "你好，世界。", "start": 0.31, "end": 0.95 } ]
```

- 可读字：取其 token 时间；一个 token 覆盖多个字（英文子词）时按字符数等分。
- 标点、空白、无法映射的字：`start` = 前一可读字终点，`end` = 后一可读字起点（即占据停顿）。
- `sentences`：在 `。！？.!?` 与换行处切分 alignment 文本，`start`/`end` 取句内首末可读字。
- 时间单调不减，均落在 `[0, durationSecs]`。

### 3.5 接口

| 入口 | 变化 |
|------|------|
| `SpeechJob` | 新增 `timestamps: bool`（默认 false）；`advanced.seed` 语义见 §5 |
| `ChunkInfo` | 新增 `text`（本 chunk 覆盖的可读文本片段，相邻 chunk 恰好切分全文）、`textOffset`、`startSecs`（在整段音频中的起点，含前置静音）；`timestamps` 时附 `alignment`（整段时间轴） |
| `JobStarted` / `JobSummary` | 新增 `seed`；`JobSummary` 在 `timestamps` 时附 `alignment`、`sentences` |
| CLI | `itts synth --timestamps`：JSON/JSONL 输出带对齐；text 模式打印逐句时间表 |
| HTTP | 新增 `POST /v1/audio/speech/timestamps` → JSON `{audioBase64, sampleRate, durationSecs, seed, alignment, sentences}`（WAV base64）；`/api/speech/stream` 事件随请求带出 |
| Daemon / MCP | 透传 `timestamps`；MCP `synthesize` 的 structuredContent 包含 `alignment` / `sentences` |

`timestamps` 关闭时不跑对齐计算，无额外开销。

## 4. 事件带文本（G2）

`ChunkInfo.text` / `textOffset` / `startSecs` 不依赖 `timestamps`，总是返回。chunk 文本由 segment 的归一化文本经同一 `TextMap` 映射回可读文本，再规范为"从首个可读字开始、延伸到下一个可读字之前"（含收尾标点与其后空白），所以相邻 chunk 恰好切分全文，成本只有一次 diff。

## 5. 可复现种子（G3）

- 实测（M5 Pro）：同一 seed 下，进程内与 daemon、daemon 处理过其他任务之后、带情绪向量时，输出 WAV **逐字节一致**；不设 seed 时每次不同。
- 新语义：service 在 `prepare` 时解析 seed：请求给了就用；没给就生成随机 `u64`。该 seed 同时驱动 GPT 采样与 CFM 噪声，并在 `JobSummary.seed` 返回。任何一次输出都可用返回的 seed 复现。
- 跨芯片型号（M1 vs M5）的位级一致性不作承诺：Metal 内核的归约顺序可能不同。

## 6. 48 kHz 高保真输出（G4）

### 6.1 为什么不能直接 44.1 kHz

CFM 产出 22.05 kHz、80 band 的 mel，BigVGAN（22k/80band/256x）只能还原到 22.05 kHz，内容上限 11.025 kHz。没有现成的"22k mel → 44.1k/48k"声码器；重训 S2Mel 或声码器超出范围。上游 IndexTTS 也只给 22.05 kHz（issue #89 维护者答复"重采样即可"）。

### 6.2 方案：LavaSR v2 带宽扩展（可选阶段）

选型结论见调研文档：LavaSR v2（Apache-2.0，约 14 M 参数，单次前向 Vocos GAN，原生 48 kHz）。流程：

```text
22.05k PCM → 重采样到 48k → Slaney mel(按 sr=44100, n_fft 2048, hop 512, 80 band, fmax 8000)
 → log → Conv1d(k7)+LN → 8× ConvNeXt(512/1536) → LN → Linear(512→2050) → |X|,∠X → iSTFT(Hann 2048, hop 512)
 → 全信号 FFT 交叉合并：低于 ~11 kHz 保留原始上采样信号，过渡带 smoothstep，上方用模型输出
```

- 原始 0–11 kHz 频带逐样本保留，模型只补 11–24 kHz，不会改变已有音色与对齐。
- 与参考实现的差异：参考代码先把任意输入降到 16 kHz，会丢掉 8–11 kHz；我们直接 22.05k→48k，截止频率取 11 kHz。
- 权重：官方仓库只有 torch pickle；使用 `AEmotionStudio/lavasr-models` 的 safetensors 转换版，固定 revision 并校验 SHA-256；接入前用官方 `.bin` 验证张量逐位相等（记录于调研文档）。
- 接口：`SpeechJob.sampleRate`（`22050` | `48000`，默认取配置 `sample_rate`）；CLI `--sample-rate 48000`；`itts models pull` 同时拉取扩展权重，`itts doctor` 增加 `hifi` 检查，`describe_controls` 列出可选采样率。`EngineConfig.sample_rate` 从裸 `u32` 改为 `SampleRate` 枚举（此前设成非 22050 会把 22.05k 数据错标为其他采样率）。`48000` 时惰性加载 BWE 权重（约 56 MB）。时间戳在原生音频上计算，不受影响（BWE 不改变时长）。
- 质量风险：LavaSR 只在 VCTK（英文朗读）上训练。默认关闭，由使用者 A/B 试听后启用。

## 7. 分期

| 阶段 | 内容 |
|------|------|
| P1 | `core::align`（TextMap、Viterbi、能量修正）、GPT 部分层 teacher-forcing、actor 产出 segment 对齐 |
| P2 | director 记录 source 区间与控制标签；service 映射、`ChunkInfo.text/startSecs`、`JobSummary.seed/alignment/sentences`；seed 语义 |
| P3 | CLI / daemon / MCP / HTTP 接口与文档、schema |
| P4 | LavaSR v2 BWE 移植、`sampleRate: 48000` |

## 8. 验证

实施结果见两份调研文档；以下为验证项清单（均已通过）。

- 单元测试：TextMap（相等/替换/插入/删除/退化路径，proptest：映射单调且覆盖全部字符）；Viterbi（合成注意力矩阵上的已知路径）；能量修正（合成 PCM 上的静音段）；sentences 切分；alignment 单调性与边界。
- 模型测试（`#[ignore]`，需权重）：真实合成后对齐的单调性、覆盖率，与 `forced_text_attention` 的头选择一致。
- 端到端：调研文档中的 MMS_FA 对比脚本复跑，确认生产实现与原型误差一致。
- 确定性：同 seed 逐字节一致（CLI 进程内与 daemon）；缺省 seed 时返回的 seed 可复现。
- 性能：`timestamps` 开启前后 RTF 对比。

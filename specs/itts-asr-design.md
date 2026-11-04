# `itts listen` / `itts verify`：本机逐字识别与稿件核对设计 (`itts-asr-design`)

- **状态**: 已实施 (Implemented)；评测结论见 §12
- **版本**: 1.0
- **日期**: 2026-11-03
- **关联组件**: `index-tts-rs-asr`（新 crate：sherpa-onnx 识别 actor、模型缓存、稿件比对）、`index-tts-rs-service`（`ListenJob` / `VerifyJob`）、`itts`（CLI / Daemon / MCP）
- **上游契约**: reshape `specs/33-free-voice-alignment.md` §4（没有时间戳的句子整句送识别）、§5.6（`quality.weak` 送识别、A014）、§7（ASR：whisper 粗筛、Scribe 升级）；reshape `specs/09-automation-quality.md` §2（A004 重复、A005 漏字 / 多字、误报 ≤ 5%）
- **前置规范**: [`itts-align-design`](itts-align-design.md)（`AlignJob`、`quality`、daemon v4、`TOO_LONG`）、[`itts-agent-usability-design`](itts-agent-usability-design.md)（错误码、daemon 协议、MCP）
- **评测**: [`itts` 本机识别评测](../docs/research/itts-asr-evaluation.md)

---

## 1. 背景与目标

reshape 的配音 QA 要回答"录音是不是照稿念的"：口吃（"北北边""从……从"）、漏念、多念、念错字、数字念成别的值。现在的做法（reshape 09 §2、33 §7）：

- 粗筛挑出可疑句，交给 **whisper**（本机、免费）。whisper 是自回归、带语言模型的识别器，会把重复的音节"顺"掉——reshape 自己的回归样本 `stutter-spliced`（"从……从 vLLM"）whisper 报不出 A004，只能再交 **ElevenLabs Scribe**（付费）。
- `itts align` 的 `quality`（似然增益）能抓住大多数漏念 / 多念 / 替换，但抓不住口吃（多出的音节与相邻字共享注意力）和共享音节的数字读法（[对齐评测](../docs/research/itts-align-evaluation-research.md) §5.4）。

本设计给 itts 加本机、非自回归的逐字识别，并把它与强制对齐合成一个核对命令：

```bash
itts listen line.wav --output json                                   # 逐字识别：文本 + 每个字 / 词的起止时间
itts verify line.wav --text "从北边吹来的风越来越冷。" --output json    # 对齐 + 识别 + 稿件比对 → findings
```

`verify` 是 reshape 音频 QA 取代 Scribe 那一步要调用的命令。

非目标：

- 流式识别、说话人分离、标点恢复、逆文本归一化（ITN 关闭：要的是念出来的样子）。
- 发音 / 声调评测。识别器输出汉字，同字异读（"行业"念成 xíng yè、"长大"念成 cháng dà）看不出来，见 §12。
- 把识别结果当对齐用。字级时间仍以 `itts align` 为准；`listen` 的时间是识别器的近似值。
- HTTP 端点（与 `align` 一致，reshape 只经 CLI 调用）。

## 2. 选型（写设计前已验证）

| 项 | 结论 |
|---|---|
| 运行时 | **sherpa-onnx 官方 Rust 绑定** `sherpa-onnx` 2.0.1（k2-fsa，Apache-2.0；`sherpa-onnx-sys` 构建时下载官方预编译静态库 `osx-arm64-static-lib`，内含 ONNX Runtime）。不用社区的 `sherpa-rs`。绑定是安全封装，本仓库仍 `#![forbid(unsafe_code)]` |
| 中文模型 | **Paraformer-large**（`csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14`，int8，232 MiB）为 `zh` / `zh/en` 缺省；**SenseVoice-Small**（`csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`，int8，228 MiB）为 `yue` / `ja` 缺省、中文可选。二者都是非自回归（CIF / CTC），不"顺"口吃：评测里 13 / 13 个中文口吃样本两者都逐字报出，whisper 只报出 8 个；召回相同时 Paraformer 的一字误听少（干净句 `info` 21 对 27） |
| 英文模型 | **Parakeet-TDT 0.6B v2**（`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8`，int8，630 MiB，CC-BY-4.0）。同一个 transducer 配置即可接入；SenseVoice 和 whisper 都会把英文的 "the the" "to to" 去掉，Parakeet 保留 |
| 时间戳 | 三个模型经 sherpa-onnx 都给每个 token 的起点（SenseVoice / Paraformer 帧级，Parakeet 另给时长）；**不给置信度**（离线 API 的结果只有 `text` / `tokens` / `timestamps` / `durations`），所以输出没有 `confidence` 字段 |
| 执行器 | CPU（4 线程）。CoreML EP 对 int8 模型更慢（同一句 32 ms → 43 ms，加载 0.3 s → 2.0 s），不用 |
| 速度 | 静态链接的 ORT：中文平均 144 ms / 句（Paraformer）、英文 336 ms（Parakeet），RTF 0.034；首次加载 0.26–0.50 s。同版本的动态库（Python wheel / `shared` 预编译）快 3 倍，但要随二进制分发 dylib，不用（评测 §3.4） |
| 长度 | SenseVoice 在 126 s 的拼接录音上字准确率与单句相同；`listen` 上限 120 s（解码器 `MAX_DECODED_AUDIO_SAMPLES`）；`verify` 受 `align` 约束（约 72 s） |

许可：SenseVoice 权重按 FunASR 模型许可（可商用，需署名）、Paraformer Apache-2.0、Parakeet CC-BY-4.0；权重不进仓库，首次使用时从 Hugging Face 下载（§5）。

## 3. 设计总览

```text
CLI / daemon / MCP
  │ ListenJob{audio, lang?, model?}            VerifyJob{audio, input, lang?, model?, voice?|voiceFile?}
  ▼ service：校验、路径策略
  ├─ run_listen ── 解码录音 → 16 kHz 单声道 ─────────────► AsrEngine（ORT actor 线程，惰性加载模型）
  │                                                       └─ ListenResult{recognizer, text, tokens[], …}
  └─ run_verify ── tokio::join!(run_align, run_listen)     ← MLX actor 与 ORT actor 并行
                   → compare(稿件, 识别 token, align.quality.weak)
                   → VerifyResult{verdict, findings[], heard, align, …}
```

- 新 crate `crates/asr`（`index-tts-rs-asr`）不依赖 `index-tts-rs-core`（不连 MLX）：只接收 16 kHz 单声道 `f32`，解码与重采样在 service 里用 core 已有的 `decode_audio_file` + `resample_sinc_hann`。
- ORT 在自己的 actor 线程（`index-tts-asr-worker`）里跑，与 MLX actor 互不阻塞：daemon 里 `listen` 不必等 MLX 模型加载完成；`verify` 的对齐与识别同时进行。
- Cargo feature **`asr`**，`index-tts-rs-service` 与 `index-tts-rs-cli` 缺省开启。关掉时不编译 sherpa-onnx；`ListenJob` / `VerifyJob` 等**线上类型照常编译**（daemon 协议与特性无关），执行它们返回 `MODEL_UNAVAILABLE`（提示用 `--features asr` 重新编译）；CLI 没有 `listen` / `verify` 子命令，MCP 不列这两个工具。

## 4. 识别 actor（`index-tts-rs-asr::AsrEngine`）

- `AsrEngine::start()` 启动工作线程，不加载模型。命令：`Transcribe{files, samples}`、`Evict`、`Shutdown`；常驻模型列表由 worker 写进 `ArcSwap`，`loaded()` 无锁读取。`stop()` / `restart()` / `Drop` 与 `IndexTtsEngine` 同样的生命周期（运行标志 + 关停回执 + join）。首次下载由一个许可的信号量串行化，等到许可后先重查缓存。
- 每个模型第一次被用到时创建 `OfflineRecognizer`（Paraformer；SenseVoice：`language = auto`、`use_itn = false`；Parakeet：`model_type = nemo_transducer`），之后常驻；daemon 空闲超时退出时一并释放。`itts daemon status` 的 `asr.loaded` 列出常驻的模型。
- 线程数 `ASR_THREADS = 4`（编译期常量，评测里 4 线程比 2 线程快 1.6 倍，再多无益）。
- sherpa-onnx 的 `OfflineRecognizer::create` 失败只返回 `None`：报 `MODEL_UNAVAILABLE`（"sherpa-onnx failed to load it"）。

### 4.1 token 后处理

识别器的原始 token 合成输出单位：

| 来源 | 规则 |
|---|---|
| 汉字（以及假名、谚文） | 一字一个 token；一个 token 含多个汉字时按字均分它的时长 |
| 拉丁字母 / 数字 | 合成词：SenseVoice / Parakeet 以前导空格（`▁` 或 ` `）开新词，Paraformer 以 `@@` 结尾表示未完；词内的 `.` `'` `%` 保留（`5.4.0`、`he's`、`50%`），词两端的 `.` `'` 去掉 |
| 标点、空白、`<…>` 特殊 token | 丢弃（Parakeet 输出标点和大小写） |

时间：`start` = token 起点；`end` = Parakeet 用起点 + 时长，其余用下一个 token 的起点，但最长 `MAX_TOKEN_SECS = 0.5` s（停顿不计入前一个字）；最后一个 token 到录音结尾、同样受 0.5 s 限制。

## 5. 模型与下载（`index-tts-rs-asr::models`）

| `model` | 仓库 / 固定提交 | 文件（字节） | 语言 |
|---|---|---|---|
| `sense-voice` | `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` @ `2365baea` | `model.int8.onnx` 239 233 841、`tokens.txt` 315 894 | zh、yue、ja、zh/en、en |
| `paraformer` | `csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14` @ `def02708` | `model.int8.onnx` 243 371 218、`tokens.txt` 75 756 | zh、zh/en |
| `parakeet` | `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` @ `1ab93235` | `encoder.int8.onnx` 652 184 296、`decoder.int8.onnx` 7 257 753、`joiner.int8.onnx` 1 739 080、`tokens.txt` 9 384 | en |

- 缺省模型：`zh`、`zh/en` → `paraformer`；`yue`、`ja` → `sense-voice`；`en` → `parakeet`。`es`、`ar`、`common` 没有本机识别器：`INVALID_ARGUMENT`。显式 `model` 与语言不配（`parakeet` + `zh`）也是 `INVALID_ARGUMENT`。
- 解析：在 Hugging Face 缓存（`HF_HUB_CACHE` / `HF_HOME` / `~/.cache/huggingface/hub`，与 core 相同的候选目录）里找固定提交的快照，每个文件大小必须相符。
- **首次使用时下载**（`hf-hub`，固定 revision，与 `itts models pull` 同一个客户端）：下载完对每个文件做 BLAKE3 校验（摘要编进二进制），不符报 `MODEL_UNAVAILABLE` 并提示删掉该模型的缓存目录重试。之后每次解析只核对大小。下载在调用方的 async 任务里完成，不占 actor 线程；daemon 里第一次 `listen` 会多花下载时间（日志记一行开始下载），结果里 `modelDownloaded: true`。
- 不发送任何用户标识：请求头只有 `hf-hub` 自己的 User-Agent（以及用户已配置的 HF token，与模型下载一致）。

## 6. 稿件比对（`index-tts-rs-asr::compare`）

输入：稿件（去掉 `[[…]]` 控制标签后的念的文本，与 `align` 相同）、识别 token、对齐的 `weak` 字下标。输出：按稿件顺序的 findings。

### 6.1 单位

稿件与识别结果用同一套规则切成**单位**（标点、空白不成单位）：

| 单位 | 来源 | 相等的条件 |
|---|---|---|
| 汉字 | 一个汉字 | 同一个字，或**不带声调的拼音读音集合有交集**（`pinyin` crate 多音字表）。识别器写别的同音字（"鹏"→"朋"、"化"→"画"）不是念错 |
| 数字 | 阿拉伯数字串（`2026`、`1.5`、`5.4.0`，`85%`）、汉字数字串（`两千零二十六`、`二零二六`、`一点五`、`百分之五十八`）、英文数词串（`fifteen`、`twenty fourteen`、`five point four point oh`） | **数值**相等（每个数字单位带一组候选值：英文成对的年份读法另加一个值）。数值相同而读法不同（`2026年` 念成"两千零二十六年"或"二零二六年"、`3日` 念成"三号"）不算差异 |
| 拉丁字母 | 拉丁词的每个字母（小写），记所属词 | 同一个字母。按字母比，"vLLM"↔"v l l m"、"Homebrew"↔"home brew" 的切分差异自然消失 |

- 单个汉字数字（"一共""第三"的"一""三"）既是数字也保留读音，与同音汉字相等。
- 读音覆盖 `<字|pinyin>` / `<word|ARPAbet>` 只取被覆盖的字 / 词。
- "号"在数字之后与"日"相等。

### 6.2 对齐与归类

1. 两串单位做编辑距离对齐（替换、插入、删除代价都是 1，回溯优先匹配），得到匹配之间的**缺口**：稿件 `[i0, i1)` 对识别 `[j0, j1)`。
2. 缺口碰到拉丁词时扩到整个词（两边各自扩，并带上与这些字母匹配的对面的词），扩完重叠的缺口合并：一个英文词念错报一条，不按字母报多条。合并后两端整词相同的词再剥掉（`the final quarterly report` 听成 `the report` 报漏念 `final quarterly`）。
3. 只含"儿"的缺口丢弃（儿化可有可无）。
4. 归类：

| `kind` | 条件 |
|---|---|
| `repeat` | 只有多出的单位，且它们与紧邻的稿件单位或识别单位（前或后，同样长度）相等：口吃、重读 |
| `insertion` | 只有多出的单位，不是重复 |
| `omission` | 只有缺的单位 |
| `number` | 两边都有数字单位（值对不上），或缺口里有多于一个字的数字（`2026`、`三十`、`十五`）。单个汉字数字的漏念 / 多念（"这一回"听成"这回"）不算 |
| `substitution` | 其余（两边都有单位） |

5. `sounds` = 缺口两边实际不匹配的单位数的较大者（一个汉字、一个数字、一个拉丁词各算一；为扩成整词而带进来的字母不计）。

### 6.3 严重度

| `severity` | 条件 |
|---|---|
| `error` | `repeat`，或 `number` |
| `warning` | `sounds ≥ 2`；或 `sounds = 1` 且**对齐佐证**（`alignWeak`）：缺口及其左右各一个稿件单位里有 `align` 的 `weak` 字 |
| `info` | 其余（`sounds = 1` 且对齐不佐证：多半是识别器对生僻字、专名的误听） |

`verdict` = 最高的严重度，没有 finding 时为 `clean`。不论 verdict，命令都成功退出（退出码 0），与 `align` 的 `quality` 一样由调用方决定阈值。

为什么这样定（评测 §4.2）：一个字 / 一个词的替换或漏念，识别器单独报出时干净句里约 10% 有这种"误听"（生僻字、专名、`Rust`→`rest`），而 `align` 的似然增益对它们不佐证；真正念错的一个字，对齐多半也认为它弱。两个以上单位的差异在 282 句干净句里没有出现过。

### 6.4 时间

- 有识别单位的 finding（`repeat` / `insertion` / `substitution` / `number`）：`start` = 第一个识别单位的起点，`end` = 最后一个的终点。
- `omission`：取 `align` 给这些稿件字的区间（强制对齐总会给它们分一段时间，就在漏念的位置附近）；对齐区间为空时退回识别缺口两侧单位之间的空当。
- `chars` = 稿件（`align.alignment.text`，即去掉控制标签的念的文本）里的字下标 `[start, end)`；`insertion` / `repeat` 是插入点（`start = end`）。

## 7. 输出

### 7.1 CLI

```text
itts listen <AUDIO> [--lang <CODE>] [--model paraformer|sense-voice|parakeet]
            [--no-daemon] [--socket <PATH>] [--output text|json|jsonl]
itts verify <AUDIO> (--text <TEXT> | --stdin) [--lang <CODE>] [--model …]
            [--voice <ID> | --voice-file <PATH>] [--no-daemon] [--socket <PATH>] [--output …]
```

- 录音格式、路径解析、daemon 优先 / 进程内退回（stderr 提示）与 `align` 相同。`listen --no-daemon` 只启动识别 actor，不加载 MLX 模型。
- text 模式：`listen` 打印识别文本和 token 时间；`verify` 打印 verdict、每条 finding（`[error] repeat "" → "北" @0.42–0.60`）、识别文本、对齐 quality。

### 7.2 `ListenResult`（`itts schema listen-result`）

```jsonc
{
  "status": "ok", "backend": "daemon",
  "recognizer": { "name": "sherpa-onnx", "version": 1, "model": "paraformer",
                  "repo": "csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14",
                  "revision": "def027084691107096b5ebba69785756d63de6c5" },
  "text": "从北北边吹来的风越来越冷",
  "tokens": [ { "text": "从", "start": 0.26, "end": 0.50 }, { "text": "北", "start": 0.50, "end": 0.66 }, … ],
  "modelDownloaded": true,                // 仅首次使用、这次请求下载了模型时出现
  "durationSecs": 2.55, "listenSecs": 0.09,
  "timings": { "audioMs": 8.6, "loadMs": 0.0, "recognizeMs": 80.2, "totalMs": 89.1 }
}
```

`recognizer.version` 是比对 / 后处理规则的主版本（reshape 缓存键用 `name/version/model`）。SenseVoice 识别出的语言不输出（sherpa-onnx 的 Rust 结果类型不带它）。

### 7.3 `VerifyResult`（`itts schema verify-result`）

```jsonc
{
  "status": "ok", "backend": "daemon",          // 进程内时另有 modelLoadMs
  "verdict": "error",                          // clean | info | warning | error
  "findings": [
    { "kind": "repeat", "severity": "error", "expected": "", "heard": "北",
      "chars": { "start": 1, "end": 1 }, "start": 0.42, "end": 0.60, "sounds": 1, "alignWeak": false }
  ],
  "heard": { …ListenResult 去掉外层 status / backend… },
  "align": { …AlignResult（aligner / alignment / sentences / quality / durationSecs / alignSecs / timings）… },
  "durationSecs": 2.62, "verifySecs": 0.21,
  "timings": { "alignMs": 180.3, "listenMs": 48.9, "compareMs": 0.4, "totalMs": 205.0 }
}
```

- `align` 就是同一录音的 `itts align` 结果（不含音素）：reshape 补齐时间戳与核对可以一次调用完成。
- `expected` 是稿件原文（`chars` 区间的字，含其间的标点）；`heard` 是识别单位的原文。

### 7.4 错误码与退出码

沿用现有表，不新增：

| 情形 | 码 | 退出码 |
|---|---|---|
| 录音无法解码 | `AUDIO_DECODE_FAILED` | 5 |
| 文本为空、录音太短（< 0.1 s）、语言没有本机识别器、模型与语言不配 | `INVALID_ARGUMENT` | 2 |
| `listen` 超过 120 s；`verify` 超过对齐上限 | `TOO_LONG` | 12 |
| 识别模型下载 / 校验 / 加载失败；构建时没开 `asr` | `MODEL_UNAVAILABLE` | 4 |
| `voice` 未注册（verify） | `VOICE_NOT_FOUND` | 3 |
| daemon 忙 / 超时 / 版本不符 | `BUSY` / `TIMEOUT` / `DAEMON_VERSION_MISMATCH` | 7 / 9 / 6 |

### 7.5 Daemon / MCP / schema / doctor

- Daemon 协议 **v5**：新请求 `{"method":"listen","job":ListenJob}` → `{"event":"heard", …ListenResult}`；`{"method":"verify","job":VerifyJob}` → `{"event":"verified", …VerifyResult}`；失败 → `error`。与其他请求共用并发槽（`BUSY`）与断线取消。`status` 多一个 `asr: { loaded: [...] }`。v4 daemon 不认识这两个方法，所以协议号加一：新 CLI 遇到旧 daemon 得到 `DAEMON_VERSION_MISMATCH` 并退回进程内运行（提示 `itts daemon restart`）。
- daemon 启动时同时启动识别 actor（只是一个空闲线程），模型惰性加载。`listen` 不等待 MLX 就绪；`verify` 等待。
- MCP 新工具 `listen`、`verify`（输入即 job，`structuredContent` 为结果，路径按服务器工作目录解析），经 daemon 执行。
- `itts schema listen-job | listen-result | verify-job | verify-result`。
- `itts doctor` 新增 `asr` 项：构建是否含 `asr`；每个模型是否已缓存（未缓存是 `info`："首次使用时下载 N MiB"，不是失败）。

## 8. reshape 集成要点（reshape 侧改动，不在本仓库）

| reshape 现在 | 建议 |
|---|---|
| `asr: { provider: whisper, escalate: elevenlabs }`（33 §7）：粗筛 → whisper → 拿不准的交 Scribe | 新 provider `itts`：`itts verify <mp3> --stdin --output json`（`say` 经 stdin），一次拿到对齐（补时间戳，§5.6 的 `align:itts-attn/1` 来源）与 findings |
| 只有粗筛挑中的句子送识别 | verify 每句约 0.2 s、免费：所有句子都送（不再需要粗筛与 A006 的"可疑但识别干净"） |
| A004 / A005 由 `compare()`（不带声调拼音的 LCS）判定 | 直接映射 findings：`repeat` → **A004（E）**；`omission` / `insertion` / `substitution` 的 `warning` → **A005（W）**、`info` → **A005（I）**；`number` → 建议新码 **A015（E）**"数字念成了别的值"（或并入 A005 W） |
| 含阿拉伯数字的补齐句子一律送识别（33 §5.5） | 由 `number` 覆盖 |
| A014（补齐对齐不可信） | 不变：读 `align.quality`；verify 的 finding 已用 `weak` 做佐证 |
| Scribe 升级 | 中文不再需要。英文 Parakeet 的结果足够做 A004 / A005；Scribe 仅在用户显式要求时保留 |
| 缓存键 | `itts/<recognizer.version>/<model>` + 录音键 + 稿件 |

句尾"多出的声音"（下一句的开头漏进本句录音，Scribe 偶尔听成"所以""辱"）识别器不报，仍由 A009 的剪尾处理。

## 9. 与 reshape 契约的偏差

| 契约 | 本实现 | 理由 |
|---|---|---|
| 33 §7：本机只有 whisper，逐字识别要 Scribe | 本机 Paraformer / SenseVoice / Parakeet 逐字 | 非自回归识别器保留重复音节（评测 §4.1） |
| 09 §2.3：A005 按"差 ≥ 2 字或 > 5%"定 W / I | 按单位数 + 对齐佐证定 warning / info | 一个字的差异在识别器单独判断时误报多；对齐的似然增益能区分真错与误听 |
| 09 §2.3：数字读法用 lexicon 归一 | 按数值比较，读法不同而值相同不报 | itts 自己把 `2026年` 念成"二千零二十六年"；读法是风格，值才是错误 |

## 10. 分期

| 阶段 | 内容 |
|---|---|
| P1 | 评测原型（sherpa-onnx Python + 比对原型），定模型、严重度规则 |
| P2 | `crates/asr`：模型表、缓存解析与下载校验、`AsrEngine` actor、token 后处理、`compare` |
| P3 | service `ListenJob` / `VerifyJob` / `run_listen` / `run_verify`、feature `asr`；CLI `itts listen` / `itts verify`、daemon v5、MCP、schema、doctor |
| P4 | 用实现重跑评测（`tools/asr_eval.py`、`make asr-eval`），结果写进评测文档（本文 §12） |

## 11. 验证

- 单元测试（`crates/asr`）：数字解析（汉字位值、逐位、小数、百分数、英文数词与年份读法、版本号）；单位切分（读音覆盖、拉丁字母、"号/日"）；同音字相等；归类与严重度全表（重复在前 / 在后、部分重复"编编译器"、漏念、替换、数字值不同、读法不同值相同、儿化、英文整词合并）；时间（漏念取对齐区间、空区间退回）；token 后处理（`@@`、前导空格、多字 token 均分、0.5 s 上限、标点丢弃）；模型表与语言缺省、不配报错；缓存解析（大小不符视为缺失）。
- service：`ListenJob` / `VerifyJob` 校验（相对路径、空文本、未知模型、语言无识别器）、`deny_unknown_fields`、结果序列化字段名；无 `asr` 特性时返回 `MODEL_UNAVAILABLE`。
- CLI：协议 v5 的 `listen` / `verify` / `heard` / `verified` 序列化；MCP 工具列表含 `listen` / `verify`。
- 模型测试（模型没缓存时跳过）：SenseVoice 识别一段合成音，token 单调、落在 `[0, duration]`，加载后 `loaded` 列出它、`evict` 后为空；actor 的启动 / 重启 / 停止。真实录音上的检出由评测（下一条）覆盖。
- 评测（结果写进评测文档）：每类错误的召回、干净句误报、速度、模型大小；对照 whisper、`align` quality、Scribe 历史结果。

## 12. 评测结论

详见 [评测](../docs/research/itts-asr-evaluation.md)（352 句：干净 282、有错 70；Apple M5 Pro，warm daemon）。`verify` 缺省模型，`warning` 及以上：

| 类型 | verify | whisper + reshape 比对 | 只用 align quality |
|---|---|---|---|
| 口吃 / 重读（中 13、英 2） | **15 / 15**（全部 `error`，含 reshape 回归样本 `stutter-spliced`） | 8 / 15 | 0 / 15 |
| 漏念（16 + 4） | **19 / 20** | 12 / 20 | 18 / 20 |
| 多念（6，含 4 句 Scribe 报过的句尾残音） | 2 / 6（残音 0 / 4，本机识别器都听不出） | 2 / 6 | 2 / 6 |
| 替换（9 + 2，含 2 句同字异读） | 7 / 11（同字异读 0 / 2，识别器看不出） | 0 / 11 | 4 / 11 |
| 数字值不同（6 + 2） | **8 / 8**（全部 `error`） | 0 / 8 | 3 / 8 |
| 数字值相同、读法不同（6） | 0 / 6 报出（按设计不报） | 0 / 6 | 1 / 6 |
| 干净句误报（221 + 61） | **0 / 282**（`info` 27 / 282） | 7 / 282 | 6 / 282 |

速度：≤ 10 s 的句子 `verify` 平均 173 ms、P95 294 ms（对齐与识别并行；whisper-cli 每句 536 ms）。模型：Paraformer 232 MiB、SenseVoice 228 MiB、Parakeet 631 MiB，首次使用时下载。

过关线（评测文档 §1，评测前定下：中文口吃召回 ≥ 90% 且含回归样本；其余错误合计不低于 whisper 与 align；干净句 `warning` 误报 ≤ 5%；≤ 10 s 的句子 ≤ 0.3 s）全部通过：**go**。

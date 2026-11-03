# `itts align`：任意录音的强制对齐（字级时间 + 可选音素）设计 (`itts-align-design`)

- **状态**: 已实施 (Implemented)；评测结论见 §11
- **版本**: 1.0
- **日期**: 2026-11-03
- **关联组件**: `index-tts-rs-core`（`models::codec`、`align`、`engine`、`voice`）、`index-tts-rs-director`、`index-tts-rs-service`、`itts`（CLI / Daemon / MCP）
- **上游契约**: reshape `specs/33-free-voice-alignment.md` §5.1（命令契约）、§5.2 方案 (a)、§5.5（评测与过关线）、§6（音素数据）
- **前置规范**: [`itts-timestamps-and-hifi-design`](itts-timestamps-and-hifi-design.md)（注意力对齐器、`Alignment` 形状）、[`itts-agent-usability-design`](itts-agent-usability-design.md)（错误码、daemon 协议、MCP）
- **调研 / 评测**: [GPT 注意力对齐实验](../docs/research/gpt-attention-alignment-research.md)、[`itts align` 评测](../docs/research/itts-align-evaluation-research.md)

---

## 1. 背景与目标

reshape 以后优先用 Gemini 3.8 Flash TTS 配音，它不返回时间戳；而锚点、A007/A009 句尾检查、ASR 粗筛、口型（reshape 27 §4.4、29）都靠逐字时间。`itts synth --timestamps` 只能给 itts 自己合成的音频对齐。本设计新增：

```bash
itts align --audio line.mp3 --lang zh --stdin --phones --output json < say.txt
itts align --audio line.wav --lang en --text "The engine runs hot." --voice-file tyr.itts_voice --output json
```

给一段**任意**录音和它念的文本，本机强制对齐，输出与 `synth` 摘要同形状的 `alignment` / `sentences`，加上 `quality`，以及可选的音素级时间（`phones`）。

非目标：

- 识别（ASR）。文本由调用方给定；文本与录音不一致只能由 `quality` 暴露，不纠正。
- 新模型权重。只用 IndexTTS-2.5 已有权重 + 一份 CMUdict 词典。
- 真正的音素边界（声学音素对齐，reshape 33 方案 b）。本版音素是字区间内的切分（`split`）。
- HTTP 端点（`index-tts-rs-server`）。reshape 只经 CLI 调用；需要时按本设计的 `AlignJob` 另加。

## 2. 可行性核对（写设计前已验证）

| 假设（reshape 33 §5.2） | 核对结果 |
|---|---|
| 语义 codec 编码器权重在 `raw/semantic_codec.safetensors` 里，加载时被跳过 | **属实**。文件含 `encoder.0.*`（Vocos 骨干，12 个 ConvNeXt 块，384/2048）、`encoder.1.*`（Linear 384→1024）、`down.*`（Conv1d 1024→1024，k3 s2 p1）、`quantizer.quantizers.0.in_project.*`（1024→8，weight-norm 已折叠）。`loader.rs` 的 `SemanticCodec` 过滤器跳过它们 |
| 量化 = 编码器 + 码本最近邻 | 上游 `indextts/codec/models.py::EnhancedCodec.quantize` 与 `amphion_codec/quantize/factorized_vector_quantize.py`：`down → GELU → VocosBackbone → Linear → in_project → L2 归一化后余弦最近邻`（`use_l2_normlize=True`），8 维 × 8192 码 |
| 编码器输入 = w2v-BERT 语义特征 | 上游 `infer_v2_5.py::get_emb`：`hidden_states[17]` 再 `(x - mean) / std`，与 `models::semantic::SemanticModel::forward` 的输出完全相同（音色快照已在用） |
| codes 帧率 | w2v-BERT 输出 50 Hz（fbank 10 ms 帧、两两拼接），`down` 步长 2 → **25 Hz，每个 code 40 ms**，与 GPT 生成的 codes 同一套 |
| 音素的拼音"来自 itts 前端" | **不成立**。itts 前端直接把汉字送进 GPT 的字级分词器，没有字→拼音（只有 `<字\|拼音>` 读音覆盖）。需新增 G2P：中文用 `pinyin` crate（MIT，字级默认读音），英文用 CMUdict（见 §7） |

所以方案 (a) 不需要任何新权重，只需移植编码器 + 量化（约 25 M 参数，与解码器对称），其余（teacher forcing 重放、L4H10+L5H10、Viterbi、能量修正、`TextMap`）复用。

## 3. 设计总览

```text
AlignJob{audio, input, lang?, voice?|voiceFile?, phones}
  │ service: 校验、去掉 [[…]] 控制标签 → 念的文本；解码录音（symphonia，任意采样率、混成单声道）
  ▼
core actor（Align 命令）
  录音 → 16 kHz → w2v-BERT 特征 [1, T50, 1024]（SemanticModel，前 17 层 + 归一化）
        ├─ SemanticEncoder::quantize → codes [ceil(T50/2)]（25 Hz）
        └─ 条件 conds：录音本身（前 15 s：EmoVec + CAMPPlus → spk_proj）或给定快照
  文本 → Frontend::prepare_text（单段，≤ 600 token）
  Gpt::teacher_forced(text, conds, codes)        ← teacher forcing 24 层：L4H10+L5H10 注意力 + 每个 code 的对数概率
  Gpt::code_log_probs(语言前缀, conds, codes)    ← 基线：没有文本时每个 code 的对数概率
  → 单调 Viterbi（40 ms 网格）→ 能量修正 → token 区间 + 每个字 / 词的注意力峰值与似然增益
  → TextMap 投影到念的文本（字下标）；quality；phones（split）
  ▼
service AlignResult{aligner, alignment, sentences, quality, phones?, phonesMethod?, durationSecs, alignSecs, timings}
```

## 4. 语义编码器（`core::models::codec::SemanticEncoder`）

- 新权重种类 `ModelWeightKind::SemanticCodecEncoder`：同一文件 `raw/semantic_codec.safetensors`，只保留 `encoder.*`、`down.*`、`quantizer.quantizers.0.{in_project,codebook}.*`；Conv1d 权重按现有规则转为 MLX `[out, K, in]`。解码器种类不变（仍跳过编码器），所以合成的常驻内存不增加。
- 前向（MLX 通道在后 `[B, T, C]`）：

  ```text
  x[1, T50, 1024] → Conv1d(k3, s2, p1) → GELU(erf)        → [1, T25, 1024]
    → Conv1d(1024→384, k7, p3) → LN → 12 × ConvNeXt → LN    （VocosBackbone，复用 ConvNeXtBlock）
    → Linear(384→1024) → Conv1d(1024→8, k1)                 → z_e [T25, 8]
    → argmax_k ( normalize(z_e) · normalize(codebook)ᵀ )     → codes [T25]
  ```

- 惰性加载：第一次 `Align` 时加载（约 100 MB），之后常驻。w2v-BERT / EmoVec / CAMPPlus 复用 actor 已有的惰性 `VoiceExtractor`（约 1.4 GB）。align 之后二者都常驻（reshape 一次要对齐一整片的句子，逐句重载会远超 0.2 s/句的目标）；`itts daemon status` 的 `voiceExtractorLoaded` 反映这一点，daemon 空闲超时退出时一并释放。
- 正确性：对上游 torch `EnhancedCodec.quantize`（同一份权重）做 codes 比对（评测文档 §2）；在 itts 自己合成的音频上，重新编码的 codes 与 GPT 生成的 codes 是否落在同一语义邻域（经解码器逐帧余弦）作为模型测试（逐码一致率很低，见 §8）。

## 5. 外来录音对齐（`core::align` + actor）

### 5.1 请求与限制

`core::align::AlignmentRequest`：`samples`（单声道 f32）、`sample_rate`、`text`（念的文本，`SynthesisText`）、`language`、`conditioning`（`Recording` | `Speaker(SpeakerSource)`）、`phones`。

| 限制 | 值 | 超出时 |
|---|---|---|
| 录音时长 | ≥ 0.1 s；codes ≤ 1815（约 72.6 s，`GPT_MEL_POS_LEN - 3`） | 太短 `INVALID_ARGUMENT`；太长 `TOO_LONG` |
| 文本 | 去掉控制标签后非空、有可读字；前端单段（≤ 600 token） | 空 `INVALID_ARGUMENT`；多段 `TOO_LONG` |
| 序列 | `3 + (token + 2) + codes ≤ 2048`（KV 缓存） | `TOO_LONG` |

### 5.2 GPT 条件（`conds`）

- **缺省 `Recording`**：用录音本身。w2v-BERT 只跑一次（全长），codes 与条件共用它：`EmoVec` 取前 15 s 的特征帧，`CAMPPlus` 取前 15 s 的 fbank，`spk_proj` 投影后 `build_conds`（与音色快照的算法相同，`emo_alpha = 1`）。
- `--voice <ID>` / `--voice-file <PATH>`：取快照的 `conds`（不混情绪）。
- 选哪个作缺省由评测决定（reshape 33 §5.5"条件来源 self vs 中性快照"），结果记在评测文档；输出的 `aligner.conditioning` 写 `recording` 或 `voice`。

### 5.3 解码

- `Gpt::teacher_forced(tokens, conds, lang, codes)` 重放 `[conds, <BOS>, text, <EOS>, <START_MEL>, codes[..N-1]]`，N 个查询 × (token + 2) 列。
- 时间网格：外来录音的第 i 个 code 固定是 `[0.04·i, 0.04·(i+1))` 秒（不再像合成那样用"时长 / codes 数"），所以 `decode` 改为显式接收帧长。能量修正在 16 kHz PCM 上做，规则与合成路径相同（首末字钉到首末有声帧、≥80 ms 停顿吸附到 0.2 s 内最近的边界、塌缩平分）。
- 时间以录音第 0 秒为基准，单调不减，落在 `[0, durationSecs]`。

### 5.4 `quality`

只看注意力不够：GPT 的文本注意力是"软"的，少念一个字时会平滑地跨过去（评测里 12 句不一致的录音只标出 2 句）。所以对齐时做两遍 teacher forcing：

1. **完整一遍**（`Gpt::teacher_forced`，24 层）：同时读 L4H10+L5H10 的注意力（用于对齐，与只跑 6 层的结果相同）和每个 code 的对数概率 `log p(code_i | conds, text, codes<i)`。
2. **基线一遍**（`Gpt::code_log_probs`）：同样的 codes，文本只有语言前缀（`<|zh|>`）。

每帧的**增益** = 两者之差（nats）：文本对这一帧有多大解释力。录音没说的字，它分到的帧上增益为负。

判断以**单位**进行：一个汉字、一个英文词（BPE 切开的几段合并）、一个读音覆盖（`<axon|AE1 K S AA0 N>` 的音素 token 合并）。单位的证据：注意力峰值取最大、"被注意"取任一、增益取平均。对 Viterbi 之后、能量修正之前的 40 ms 网格：

- **峰值** `peak`：单位被分到的帧（左右各扩 1 帧）上，"在可读列上重新归一化的注意力"对它的最大值。
- **被注意**：这些帧里至少有一帧的可读列 argmax 是它。
- **增益** `gain`：它被分到的帧上增益的平均。

| 字段 | 定义 |
|---|---|
| `confidence` | 所有单位 `peak` 的平均（0–1） |
| `coverage` | 被注意的单位比例 |
| `weak` | `peak < 0.3`（`WEAK_PEAK`）、未被注意、或 `gain < −1`（`WEAK_GAIN`）的单位，取其首个可读字下标（覆盖取被覆盖的字），升序 |

建议调用方的阈值（评测 §5.4，154 句干净录音最低 confidence 0.974、coverage 0.982）：`confidence < 0.95` 或 `coverage < 0.95` → 补齐不可信（reshape A014）；`weak` 非空 → 送识别复查。阈值以下照常输出、退出码 0。代价：每句约 +15～40 ms。

## 6. 输出

### 6.1 CLI

```text
itts align --audio <FILE> (--text <TEXT> | --stdin) [--lang <CODE>] [--voice <ID> | --voice-file <PATH>]
           [--phones] [--no-daemon] [--socket <PATH>] [--output text|json|jsonl]
```

- `--audio`：WAV / MP3 / FLAC / M4A，任意采样率，多声道混成单声道（与 `voice add` 同一个解码器）。
- `--text` / `--stdin`：可含 `<字|拼音>` / `<word|ARPAbet>` 读音覆盖（与 synth 同一前端）；`[[…]]` 控制标签被去掉、不执行（语法错误照样报 `INVALID_ARGUMENT`）。
- 有 daemon 时走 daemon（与 synth 相同：连不上就进程内跑并在 stderr 提示；`--no-daemon` 强制进程内）。
- text 模式打印逐句时间表与 `quality`；`json` / `jsonl` 打印一个摘要文档。

### 6.2 JSON 摘要（`itts schema align-result`）

```jsonc
{
  "status": "ok", "backend": "daemon",                 // 与 synth 相同的外层字段（进程内时另有 modelLoadMs）
  "aligner": { "name": "itts-attn", "version": 1, "model": "yunfengwang/IndexTTS-2.5-mlx",
               "heads": "L4H10+L5H10", "conditioning": "recording" },
  "alignment": { "text": "鲲化而为鸟。", "start": [0.21, …], "end": [0.52, …] },   // 与 synth 的 alignment 同形状、同填充规则
  "sentences": [ { "text": "鲲化而为鸟。", "chars": { "start": 0, "end": 6 }, "start": 0.21, "end": 1.64 } ],  // 与 synth 相同
  "quality": { "confidence": 0.81, "coverage": 0.97, "weak": [4] },
  "phones": [ { "i": 0, "p": "k", "k": "ini", "start": 0.21, "end": 0.27 },
              { "i": 0, "p": "un", "t": 1, "k": "fin", "start": 0.27, "end": 0.52 } ],   // 仅 --phones
  "phonesMethod": "split",                                                            // 仅 --phones
  "durationSecs": 2.03, "alignSecs": 0.09,
  "timings": { "audioMs": 3.1, "featuresMs": 21.0, "encodeMs": 4.2, "conditionMs": 9.8, "attentionMs": 30.5, "totalMs": 88.0 }
}
```

- `alignment` / `sentences` 由 service 的同一套函数生成（`Transcript` 用的 `fill_gaps` 抽成公共的 `Alignment::from_spans`），标点、空白占据两侧可读字之间的停顿。
- `phones[].i` 是 `alignment.text` 的字下标；英文一个词的音素都用词首字母下标；读音覆盖 `<字|拼音>` 指向被覆盖的字（不是 `<`）。
- `aligner.version` 是主版本：编码器、头选择、解码规则或 `quality` 定义改变时加一（reshape 的缓存键只含 `name/version`）。

### 6.3 错误码与退出码

与 synth 共用一张表（`itts schema controls` 的 `errors`）：

| 情形 | 码 | 退出码 |
|---|---|---|
| 录音无法解码、格式不支持 | `AUDIO_DECODE_FAILED` | 5 |
| 文本为空 / 没有可读字、录音太短、参数冲突 | `INVALID_ARGUMENT` | 2 |
| 录音或文本超过单次上限（§5.1） | **`TOO_LONG`（新增）** | 12 |
| 权重缺失（含编码器张量） | `MODEL_UNAVAILABLE` | 4 |
| `voice` 未注册 | `VOICE_NOT_FOUND` | 3 |
| daemon 忙 / 超时 / 版本不符 | `BUSY` / `TIMEOUT` / `DAEMON_VERSION_MISMATCH` | 7 / 9 / 6 |

### 6.4 Daemon / MCP / schema / doctor

- Daemon 协议 **v4**：新请求 `{"method":"align","job":AlignJob}`，回复 `{"event":"aligned", …AlignResult}` 或 `error`；与 synth 共用并发槽（`BUSY`）、断线取消。旧 v3 daemon 不认识 `align`，所以协议号加一：新 CLI 遇到旧 daemon 得到 `DAEMON_VERSION_MISMATCH` 并退回进程内运行（提示 `itts daemon restart`）。
- MCP 新工具 `align`（输入 `AlignJob`，`structuredContent` 为 `AlignResult`，路径按服务器工作目录解析），经 daemon 执行。
- `itts schema align-job` / `align-result`。
- `itts doctor` 新增 `align` 项：编码器张量（`encoder.0.embed.weight`、`down.weight`、`quantizer.quantizers.0.in_project.weight`）在权重文件里，且 Stage 0 编码器权重齐全。

## 7. 音素：`split`（`core::align::phones`）

音素在**归一化文本**上切分（数字、符号已展开成念出的字词），用 token 时间得到每个"音节单元"的区间，再经 `TextMap` 投影回 `alignment.text` 的字下标。

| 单元 | 读音来源 | 输出 |
|---|---|---|
| 汉字 | `pinyin` crate 的默认读音（`with_tone_num_end`；无数字 = 轻声 5） | 声母 `ini` + 韵母 `fin`（带 `t`） |
| `<字\|拼音>` 覆盖 | 覆盖里的拼音（多字覆盖按空格分音节，依次对应各字） | 同上 |
| 英文词 `[a-z']+` | CMUdict（首选读音，去重音数字）；不在词典里的词按字母规则近似 | `ph`（ARPAbet，无重音数字） |
| `<word\|ARPAbet>` 覆盖 | 覆盖里的音素 | `ph` |
| 其他（假名、阿拉伯文、未展开的数字等） | — | 不出音素 |

拼音拆分（reshape 33 §6.1 的约定）：声母 `b p m f d t n l g k h j q x zh ch sh r z c s`；韵母写全拼书写形式、`ü` 写 `v`（`ju` → `j` + `v`，`lü` → `l` + `v`）；`y` / `w` 开头的音节按韵母记（`yi`→`i`、`wu`→`u`、`yu`→`v`、`ya`→`ia`、`ye`→`ie`、`yao`→`iao`、`you`→`iu`、`yan`→`ian`、`yin`→`in`、`yang`→`iang`、`ying`→`ing`、`yong`→`iong`、`yue`→`ve`、`yuan`→`van`、`yun`→`vn`、`wa`→`ua`、`wo`→`uo`、`wai`→`uai`、`wei`→`ui`、`wan`→`uan`、`wen`→`un`、`wang`→`uang`、`weng`→`ueng`）；零声母字只有韵母；`zhi chi shi ri zi ci si` 的韵母记 `i`。

区间内切分（reshape 33 §6.2）：

- 清声母（`p t k q c ch s sh x f h z zh j`，以及不送气塞音 `b d g`——普通话里它们同样是清音，闭塞段无低频能量）：声母终点 = 区间内**低频能量**（二阶低通 1 kHz，10 ms 帧 RMS）第一次超过区间峰值 30% 的帧；夹在 `[区间起点 + 最短段, 区间起点 + 60% 时长]` 内，最短段 = `min(20 ms, 时长 / 4)`。
- 浊声母（`m n l r`）：按区间的 25% 切。
- 零声母：整段是韵母。
- 英文：按音素类别的典型时长权重（元音 1.0、双元音 1.3、塞音 0.5、塞擦音 0.8、擦音 0.8、鼻音 0.6、流音 / 滑音 0.6）分摊词区间。

局限（写进评测）：多音字取字级默认读音（"银行"的"行"会读成 `xing2`），可用读音覆盖纠正；`split` 只是近似，reshape 33 §5.5 的音素过关线（声母 / 韵母分界对 MMS 的拼音字母区间 MAE ≤ 30 ms）不过就启动方案 (b)。

CMUdict：取 `cmusphinx/cmudict` 固定提交，只保留每词首选读音、去掉重音数字和注释（约 3.0 MB，crate 包内压缩后约 0.8 MB），连同其 BSD 许可放在 `crates/core/assets/`，首次使用时解析成表。`make update-cmudict` 重新生成。

## 8. 与 reshape 33 契约的偏差

| 契约 | 本实现 | 理由 |
|---|---|---|
| 错误码 `unsupported_audio` / `text_empty` / `too_long` / `model_missing` | `AUDIO_DECODE_FAILED` / `INVALID_ARGUMENT` / **`TOO_LONG`** / `MODEL_UNAVAILABLE`（§6.3） | 30 要求"与 synth 的错误表合并"：沿用 itts 已有的 `SCREAMING_SNAKE` 码与退出码，只新增没有对应项的 `TOO_LONG` |
| 声母 / 韵母"来自 itts 前端的拼音" | `pinyin` crate（字级默认读音）+ 读音覆盖优先 | itts 前端没有字→拼音（§2）；多音字是已知局限 |
| §6.2 清声母列表 `p t k q c ch s sh x f h z zh j` | 另含 `b d g` | 普通话不送气塞音也是清音，闭塞段同样没有低频能量；否则这三个声母没有规则 |
| §5.1 摘要字段 | 另有外层 `status` / `backend`（与 synth 一致）、`aligner.conditioning`、`timings`，`sentences[].chars` | 与 synth 摘要保持同形状；多出的字段只增不改 |
| `--voice-file` | 另支持 `--voice <ID>` | 与 synth 对称；二者互斥 |
| §11 第 1 步：重新编码 itts 录音，codes 与 GPT 生成的"一致率预期很高" | 逐码一致只有约 2%（1/60）；解码后逐帧余弦 0.66（错位 2 帧 0.42） | GPT → S2Mel → 声码器 → w2v-BERT 不是恒等映射；codes 落在语义邻域内。对齐不受影响，模型测试改为检查语义邻域 |
| §5.5"itts 录音：外来路径 vs 原生，字起点平均差 ≤ 10 ms" | 13.5 ms（中位 10 ms） | 两条路径的帧网格不同（约 33 ms vs 40 ms），约 10 ms 是下限；建议改为"对 MMS 不劣于原生路径 5 ms 以内"（实测 20.9 vs 16.7 ms） |
| §5.1 `quality` 只有注意力峰值与覆盖率、`confidence` 初值 0.5 | 加入似然增益，按字 / 词 / 覆盖为单位判断；建议阈值 0.95 | 只用注意力识别不了大多数漏念；0.5 在评测里只标出 2/12 |

## 9. 分期

| 阶段 | 内容 |
|---|---|
| P1 | `SemanticEncoder` + 权重种类；对上游 torch 实现比对 codes |
| P2 | `decode` 显式帧长 + 峰值；actor `Align` 命令（特征、codes、条件、注意力、quality）；`IndexTtsEngine::align` |
| P3 | `phones`（拼音拆分、CMUdict、低频起振切分） |
| P4 | service `AlignJob` / `AlignResult` / `TOO_LONG`、director `control_ranges`；CLI `itts align`、daemon v4、MCP、schema、doctor |
| P5 | 评测（reshape 33 §5.5）与调研文档；按结果定 `quality` 阈值与条件缺省；加入似然增益（§5.4） |

## 10. 验证

- 单元测试：编码器权重过滤（保留 / 跳过的键）；`decode` 的峰值与覆盖（合成注意力矩阵：干净对角、整列缺失）；拼音拆分全表（`y/w`、`j/q/x + u`、`zhi` 类、`er`、轻声、覆盖大写）；CMUdict 查表与规则回退；低频起振切分在合成 PCM 上的已知答案（静音 → 噪声 → 低频正弦）；音素下标投影（数字展开、读音覆盖）；`Alignment::from_spans` 单调有界；`control_ranges` 与 `plan_utterances` 一致；`AlignJob` 校验（空文本、互斥参数、相对路径）；协议 v4 的 `align` / `aligned` 序列化；MCP 工具列表含 `align`。
- 模型测试（没有权重时跳过，有权重时随 `make test` 运行）：合成一句（1.0 倍速）→ 重新编码，codes 与生成 codes 的语义邻域；`align` 合成音频的外来路径对原生对齐的起点平均差；输出单调、落在 `[0, duration]`、覆盖每个可读字。
- 评测（reshape 33 §5.5，结果写进评测文档）：itts 录音（外来 vs 原生）、Gemini（zh / en）、ElevenLabs（zh / en，原生对齐 vs `itts align` vs MMS_FA 三方）、难例（数字读法、漏念、长停顿、中英混排、喊叫）、条件来源、速度、音素分界。

## 11. 评测结论

详见 [评测调研](../docs/research/itts-align-evaluation-research.md)（166 句，Apple M5 Pro）。

| reshape 33 §5.5 过关线 | 结果 | 判定 |
|---|---|---|
| 中文字起点对 MMS ≤ 40 ms、≥ 95% ≤ 100 ms | Gemini 18.7 / 29.4 ms，ElevenLabs 21.7 ms，Edge 17.8 ms；96–100% | 过 |
| 英文词起点 ≤ 60 ms | ElevenLabs 69.7、Gemini 70.2 ms（中位 −35 ms，偏早） | 不过 |
| itts 录音外来 vs 原生 ≤ 10 ms | 13.5 ms | 不过（见 §8，线本身依赖已否定的假设） |
| 句末终点 P95 ≤ 80 ms | Gemini / itts 65–80 ms；ElevenLabs 96 / 163、Edge 111 ms（偏晚，A009 安全方向） | 部分不过 |
| 难例被标为可疑 ≥ 80% | 75%（漏念 / 多念 / 替换 8/8，数字读法 1/4），干净句误报 0/154 | 不过（数字读法需 ASR） |
| 条件来源 | `recording` 在 8 组中 7 组更准 | 缺省 `recording` |
| 一句 ≤ 0.2 s | ≤ 10 s 的句子平均 0.11 s；24 s 的句子 0.47 s | ≤ 10 s 过 |
| 音素分界 ≤ 30 ms | 14.5–25.9 ms | 过 |

中文（reshape 的主要缺口）go；英文有条件使用；数字读法不一致交给 ASR。不转方案 (b)。


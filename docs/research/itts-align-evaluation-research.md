# `itts align` 评测：外来录音的注意力强制对齐

- **日期**: 2026-11-03
- **结论落地**: [`specs/itts-align-design.md`](../../specs/itts-align-design.md)，代码 `crates/core/src/align/`、`crates/core/src/models/codec.rs`、`crates/core/src/engine/actor/align.rs`
- **契约与过关线**: reshape `specs/30-free-voice-alignment.md` §5.5
- **硬件**: Apple M5 Pro，macOS 27.0.1；模型 `yunfengwang/IndexTTS-2.5-mlx`（GPT int8）；`itts` 0.3.1（分支 `feat/itts-align`，提交 `05dcaab`），warm daemon
- **脚本**: `tools/align_eval.py`（`make align-eval`），参照为 torchaudio 2.8 `MMS_FA`（CC-BY-NC 4.0，只作评测参照，不进生产）

## 1. 问题

reshape 改用 Gemini TTS 后，大部分句子没有时间戳。方案 (a)：把外来录音经 w2v-BERT + 语义 codec 编码器量化成 GPT 的语义 codes，teacher forcing 复用 itts 合成时的注意力对齐器（L4H10+L5H10、单调 Viterbi、能量修正）。本文回答两件事：编码器移植是否正确；在 itts 以外的录音上，精度、可疑句识别、音素切分和速度是否达到 reshape 30 §5.5 的过关线。

## 2. 编码器移植

- **对上游逐码比对**：同一份 w2v-BERT 特征（itts 合成的 4 s 中文句，198 帧）分别送进 Rust `SemanticEncoder::quantize` 与上游 torch `indextts/codec/models.py::EnhancedCodec.quantize`（同一份 `raw/semantic_codec.safetensors`，先 `remove_weight_norm` 再加载折叠后的权重）：**99/99 个 code 完全一致**。
- **重新编码 itts 自己的录音**（速度 1.0，GPT 生成 60 个 code）：重新编码得到 60 个 code，但**逐码一致只有 1/60**。逐帧经 codec 解码器还原成连续特征后比较：对齐位置余弦 0.66，错开 2 帧 0.42，错开 10 帧 0.22。即 GPT → S2Mel → BigVGAN → w2v-BERT → 量化这一圈不是恒等映射，重新编码的 code 落在原 code 的语义邻域里而不是同一个 code。reshape 30 §11 第 1 步"一致率预期很高"的假设不成立，但对对齐没有影响（§4.1：外来路径与原生对齐的字起点差 13.5 ms）。模型测试 `test_should_re_encode_synthesized_audio_into_semantically_matching_codes` 检查的是语义邻域（余弦 > 0.5 且比错位高 0.1 以上），不是逐码一致。
- **MP3 时间基准**：symphonia 解码 MP3 与 ffmpeg 结果长度相同、互相关滞后 0 样本（ElevenLabs 44.1 kHz、Gemini 48 kHz 各一例）：编码器延迟与尾部填充被同样裁掉，`itts align` 的 0 秒就是播放器的 0 秒。

## 3. 样本与方法

| 组 | 来源 | 句数 | 平均时长 | 参照 |
|---|---|---|---|---|
| `itts-zh` / `itts-en` | 新合成（`tyr` / `demo_voice` 交替，默认 1.2 倍速） | 20 / 20 | 3.3 s | 原生对齐 + MMS_FA |
| `eleven-zh` | 庄子 10 + 追赶卡诺 10（ElevenLabs v4 克隆，reshape 缓存） | 20 | 9.1 s（最长 23.8 s） | ElevenLabs 原生对齐 + MMS_FA |
| `eleven-en` | 庄子 10 + 安全史话 10 | 20 | 9.3 s | 同上 |
| `gemini-zh` | axon 的 voice.lock 录音（Gemini 3.8 Flash TTS，Aoede） | 14 | 5.5 s | MMS_FA（2 句含数字，无参照） |
| `gemini-zh-more` | products 其余 Gemini 句子（含 `<axon\|AE1 K S AA0 N>` 读音覆盖、中英混排） | 19 | 6.5 s | MMS_FA（4 句含数字，无参照） |
| `edge-zh` | 追赶卡诺的 Edge 草稿配音 | 15 | 4.6 s | MMS_FA |
| `gemini-en` | 新合成：Flash 10 句 + Flash Lite 10 句（Kore / Puck） | 20 | 4.2 s | MMS_FA |
| `hard-mismatch` | 录音与文本不一致：数字读法 4、漏念 5、多念 2、替换 1（itts 8 句、Gemini 4 句） | 12 | 3.1 s | 应被标为可疑 |
| `hard-match` | 一致但难：句中 1.5 s 停顿、中英混排 `Rust`、喊叫（itts / Gemini 各 3） | 6 | 3.9 s | 原生对齐（itts）+ MMS_FA |

- 指标：中文按字、英文按词的**起点**绝对误差（MAE、P95、≤100 ms 比例、偏置 = 平均有符号误差，负值表示 `itts align` 更早）；句末最后一个可读字的**终点**误差；音素为中文字的声母 / 韵母分界对 MMS 拼音字母区间（取两边声母相同的字）。
- MMS 转写：中文用 pypinyin（按汉字连写取词组读音）、英文按词；含阿拉伯数字的句子不出参照。
- 条件来源：缺省为录音本身（`recording`）；对照组用固定快照 `demo_voice`（`--voice demo_voice`）。

## 4. 结果

### 4.1 对过关线（条件来源 = 录音本身）

| 指标（reshape 30 §5.5） | 过关线 | 结果 | 判定 |
|---|---|---|---|
| itts 录音：外来路径 vs 原生，字起点平均差 | ≤ 10 ms | 中文 **13.5 ms**（中位 10 ms，P95 35 ms）；英文词 23.5 ms | **不过**（见 §5.1：两套帧网格不同，约 10 ms 是下限） |
| Gemini / ElevenLabs / Edge vs MMS：中文字起点 MAE、≤100 ms 比例 | ≤ 40 ms，≥ 95% | Gemini（axon）**18.7 ms / 98.8%**；Gemini 其余 **29.4 / 96.0%**；ElevenLabs **21.7 / 98.9%**；Edge **17.8 / 100%**；itts 20.9 / 100% | **过** |
| 英文词起点 MAE | ≤ 60 ms | ElevenLabs **69.7 ms**、Gemini **70.2 ms**（中位约 −35 ms）；itts 44.2 ms | **不过**（见 §5.2） |
| 句末可读字终点误差 P95 | ≤ 80 ms | 对 MMS：itts 65 / 79、Gemini zh 65 / 66、Gemini en 80、ElevenLabs en 96、ElevenLabs zh 163、Edge 111 ms | **部分不过**（见 §5.3：误差偏向"晚"，是 A009 安全的方向） |
| 难例里漏念 / 读法不一致被 `quality` 标为可疑 | ≥ 80% | **9/12 = 75%**（`confidence < 0.95` 或 `coverage < 0.95`）：漏念、多念、替换 8/8；数字读法 1/4 | **不过**（数字读法） |
| 条件来源 `recording` vs 中性快照 | 取误差小者 | `recording` 在 8 组里 7 组更准（中文差 0.4–1.7 ms），只有英文 itts 组相反 | 缺省 **`recording`** |
| 速度（warm daemon） | 一句 ≤ 0.2 s | ≤ 10 s 的句子平均 **0.11 s**；全部 166 句平均 0.125 s、P95 0.27 s、最长 23.8 s 的句子 0.47 s | **≤ 10 s 的句子过**；长句不过（w2v-BERT 约 7 ms/秒音频） |
| 音素 `split`：声母 / 韵母分界对 MMS 拼音字母 | MAE ≤ 30 ms | itts 14.5、Gemini 20.1 / 25.9、ElevenLabs 23.5、Edge 21.8、难例 25.8 ms | **过** |
| QA 回归（axon 的 A007 / A009） | 人工确认 | 未做：属 reshape 侧（30 §11 reshape 第 5 步） | — |

### 4.2 全表

起点误差（ms，MAE / P95 / ≤100 ms / 偏置），"原生"一列是对该录音提供方自己的对齐：

| 组 | 对 MMS | 对原生 | 原生对 MMS | 终点对 MMS（P95） | 音素分界 MAE |
|---|---|---|---|---|---|
| itts-zh | 20.9 / 41 / 100% / +14 | 13.5 / 35 / 99.7% / +6 | 16.7 / 37 / 99.7% / +9 | 65 | 14.5 |
| itts-en | 44.2 / 134 / 88.7% / −27 | 23.5 / 52 / 96.7% / +14 | 55.3 / 183 / 84.5% / −41 | 79 | — |
| eleven-zh | 21.7 / 59 / 98.9% / −4 | 106.5 / 240 / 51.3% / −102 | 101.0 / 215 / 51.6% / +97 | 163 | 23.5 |
| eleven-en | 69.7 / 216 / 79.4% / −61 | 111.2 / 371 / 61.9% / −108 | 57.6 / 185 / 85.4% / +47 | 96 | — |
| gemini-zh | 18.7 / 54 / 98.8% / −3 | — | — | 65 | 20.1 |
| gemini-zh-more | 29.4 / 87 / 96.0% / −16 | — | — | 66 | 25.9 |
| edge-zh | 17.8 / 39 / 100% / −1 | — | — | 111 | 21.8 |
| gemini-en | 70.2 / 237 / 78.9% / −64 | — | — | 80 | — |
| hard-match | 24.5 / 55 / 97.3% / −9 | 12.9 / 24 / 97.3% / −2 | 23.1 / 54 / 97.1% / +4 | 45 | 25.8 |

条件来源换成 `demo_voice` 快照：中文各组 MAE 变化 +0.4～+1.7 ms，itts-zh 句末 P95 从 65 升到 124 ms；英文 itts 组 −2.8 ms。

## 5. 分析

### 5.1 itts 录音的外来路径为什么到不了 10 ms

原生对齐的帧长是"时长 / GPT codes 数"（默认 1.2 倍速下约 33 ms），外来路径的帧长固定 40 ms（w2v-BERT 50 Hz ÷ 2）。两套网格上的边界本来就各自量化，误差在一帧内近似均匀分布，平均差的下限约 10 ms；实测中位数正好 10 ms、偏置 +5.6 ms。外来路径对 MMS 的误差（20.9 ms）与原生对 MMS（16.7 ms）同一量级。这条过关线建立在"重新编码得到同一套 codes"的假设上（§2 已否定），建议改为"对 MMS 的误差不劣于原生路径 5 ms 以内"。

### 5.2 英文词起点偏早

偏置集中在两类词上（对 MMS，ElevenLabs / Gemini）：

| 词的位置 / 首音 | 偏置 | MAE | 中位 |
|---|---|---|---|
| 句中连读的词 | −54 / −57 ms | 64 / 63 ms | −35 ms |
| 逗号、句号后的词 | −126 / −211 ms | 130 / 214 ms | −31 / −44 ms |
| 元音开头的词 | −109 / −113 ms | 109 / 114 ms | −65 / −59 ms |
| 塞音开头的词 | −33 / −36 ms | 44 / 43 ms | −11 / −16 ms |

- 停顿后的词：ElevenLabs / Gemini 在停顿里有换气声，能量修正把换气算作有声，词首被提前到吸气处（均值大、中位数正常）。
- 元音开头的词（`and`、`it`、`a`）：注意力在前一个词收尾时就移到带前导空格的 BPE token 上；CTC 参照又倾向把词首放晚（ElevenLabs 自己的对齐也比 MMS 晚 47 ms），两边叠加。
- 试过对句中英文词起点统一后移：后移 40 ms 时 ElevenLabs / Gemini 降到 60.7 / 58.8 ms，但 itts 录音对原生的误差从 23.5 升到 49.3 ms——只是在两种参照之间搬误差，没有采用。
- 中文不受影响（偏置 −1～−16 ms），reshape 的主要场景（Gemini 中文旁白、角色对白）都在中文。

### 5.3 句末终点

终点由能量包络决定（最后一个有声帧）。ElevenLabs 中文和 Edge 对 MMS 的偏置是 +57 / +68 ms，即 `itts align` 的终点**更晚**：录音尾部的气声、混响被算作有声；CTC 的终点本身偏"尖峰"、偏早（上一份调研已指出不作主指标）。对 A009（剪掉句尾之后多出的声音）而言，终点偏晚只会少剪一点，不会剪到字，是安全的方向；reshape 30 §5.6 对补齐的对齐另加的 0.2 s 余量足以覆盖 P95。

### 5.4 可疑句识别（`quality`）

最初只用注意力（路径峰值、覆盖率）：12 句不一致的录音只标出 2 句（阈值 0.5 / 0.9）。原因是 GPT 的文本注意力是"软"的：少一个字时注意力会平滑地跨过去。改进后的 `quality`（设计 §5.4）：

1. **似然增益**：同一组 codes 在完整文本下与只有语言前缀时的 teacher-forced 对数概率之差（每帧，nats）。干净句子里绝大多数字的增益为正（多在 +0.5～+3）；漏念的字（"下午"−2.4～−5.6、"里"−5.3、`by` −3.2）明显为负。字的增益 < −1 即为 `weak`。代价是把只跑前 6 层的注意力重放换成完整的 24 层，再加一遍只有前缀的重放，每句约 +15～40 ms。
2. **按单位判断**：一个汉字、一个英文词（BPE 切开的几段合在一起）、一个读音覆盖为一个单位。之前 `Homebrew` 的 `brew`、`<axon|AE1 K S AA0 N>` 的音素 token 会单独被判为弱。

结果（154 句干净录音 + 12 句不一致）：

| 规则 | 不一致被标出 | 干净句被标出 |
|---|---|---|
| `confidence < 0.95` 或 `coverage < 0.95` | **9/12**（漏念 5/5、多念 2/2、替换 1/1、数字 1/4） | **0/154**（最低 confidence 0.974、coverage 0.982） |
| 再加 `weak` 非空 | 9/12 | 5/154 |

`weak` 非空的 5 句干净录音用 whisper 复听：1 句是真错（Gemini 把"Axon **用**伪终端"的"用"读成英文 `used`），1 句是 ElevenLabs 把人名 `Hway-dzuh's` 念成 `Huayza's`，其余 3 句（"查一查"、`has`、"方"）是误报。所以 `weak` 适合作为"送识别复查"的信号（reshape 30 §5.6 ASR 粗筛"`weak` 非空即可疑"），A014 用 `confidence` / `coverage` < 0.95。

**没解决的**：数字读法不一致且共享音节时（文本 `2026年` 归一化为"二零二六年"，录音念"两千零二十六年"；`11月3日` 念成"十一月三号"）注意力和似然都把多出的音节吸收掉，4 句只标出 1 句（英文 `2026` 念成 `twenty twenty-six`）。这类句子要靠识别，reshape 可以对含阿拉伯数字的补齐句子一律送 whisper。

### 5.5 速度

warm daemon，166 句：≤ 10 s 的句子平均 0.11 s（音频解码 1–6 ms、w2v-BERT 特征约 7 ms/秒音频、编码器 3–7 ms、录音条件 20–70 ms、两遍 teacher forcing 30–110 ms）。24 s 的句子 0.47 s，其中 w2v-BERT 约 175 ms。daemon 启动后第一次 `align` 额外加载 Stage 0 编码器与 codec 编码器（约 0.3 s），之后常驻。用 `--voice` 快照条件可省去 20–70 ms。

### 5.6 音素

中文声母 / 韵母分界对 MMS 拼音字母边界 MAE 14.5–25.9 ms，全部低于 30 ms；外来录音的偏置 −14～−21 ms（我们的韵母起点略早；itts 录音约 0），来源是"低频能量超过峰值 30%"的起振点在元音真正稳定之前。英文的 MMS 只有字母级，不能直接评估 ARPAbet 音素，未评。

## 6. 结论（go / no-go）

**中文：go。** reshape 的主要缺口（Gemini / Edge 中文旁白与对白）在所有中文组都过了精度线（MAE 18–29 ms、≥96% 在 100 ms 内），比 ElevenLabs 自带的中文对齐还准（ElevenLabs 原生对 MMS 偏晚约 100 ms）；`split` 音素过线；≤ 10 s 的句子在 0.2 s 内。

**英文：有条件。** 词起点 MAE 70 ms，比 60 ms 线高 10 ms，中位数 −35 ms；主要误差在换气后的词和元音开头的词。可以用于锚点（偏早几十毫秒）和口型，但 reshape 的 QA 对英文补齐句子应放宽阈值。

**可疑句识别：未达 80%（75%）。** 漏念、多念、替换全部识别，误报为 0；数字读法不一致基本识别不了，需要 ASR 兜底。

不需要转方案 (b)：(b) 的 CTC 同样无法发现共享音节的数字读法差异，而中文精度、音素都已过线。建议的后续工作：英文停顿后的换气检测（区分换气与清辅音）；把 w2v-BERT 的长句耗时降下来；reshape 侧完成 axon 的 A007 / A009 人工回归。

## 7. 复现

```bash
# manifest：每行 {id, set, audio, text, lang, native?, mismatch?}；native 指向带 ElevenLabs 形状 alignment 的 JSON
itts daemon start
make align-eval ALIGN_EVAL_MANIFEST=/path/to/manifest.jsonl
# 条件来源对照：ALIGN_EVAL_ARGS='--voice demo_voice'
```

- itts 样本：`itts synth --voice tyr|demo_voice --seed N --timestamps --output json --out X.wav`。
- Gemini 样本：Vertex AI `generateContent`（`gemini-3.8-flash-tts` / `gemini-3.8-flash-lite-tts`，`responseModalities: ["AUDIO"]`），返回体是 24 kHz WAV。
- ElevenLabs / Edge / Gemini（axon）样本：reshape 项目 `.reshape/cache/tts/<key>.{mp3,json}`，文本取 `alignment.text`（ElevenLabs）或 `spec.say`。

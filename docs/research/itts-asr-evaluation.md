# `itts listen` / `itts verify` 评测：本机逐字识别能否取代 Scribe

- **日期**: 2026-11-03
- **结论落地**: [`specs/itts-asr-design.md`](../../specs/itts-asr-design.md)，代码 `crates/asr/`、`crates/service/src/{listen,verify}.rs`、`apps/cli/src/commands/{listen,verify}.rs`
- **上游**: reshape `specs/33-free-voice-alignment.md` §7（ASR）、`specs/09-automation-quality.md` §2（A004 / A005、误报 ≤ 5%）
- **硬件**: Apple M5 Pro，macOS 27.0.1；`itts` 0.3.1（分支 `feat/itts-asr`），warm daemon（协议 v5）；sherpa-onnx 2.0.1（ONNX Runtime 1.28.2，静态链接，CPU 4 线程）
- **脚本**: `tools/asr_eval.py`（`make asr-eval`）；对齐评测的样本与方法沿用 [`itts align` 评测](itts-align-evaluation-research.md)
- **付费接口**: 没有调用 ElevenLabs Scribe 或任何付费 API。Scribe 只用 reshape 缓存里的历史转写（`zhuangzi/.reshape/cache/asr/*.json`）作参照

## 1. 问题

reshape 的配音 QA（09 §2）要找出录音和稿件不一致的地方：口吃 / 重读（A004）、漏念 / 多念（A005）。现在本机只有 whisper，它是带语言模型的自回归识别器，会把"从……从"顺成"从"——reshape 自己的回归样本 whisper 报不出，只能再花钱交 Scribe（33 §7）。`itts align` 的 `quality` 抓得住大多数漏念，但抓不住口吃和共享音节的数字读法（对齐评测 §5.4）。

问题是：非自回归的本机识别器（SenseVoice-Small、Paraformer-large；英文 Parakeet-TDT）能不能逐字听出这些错误，和强制对齐合起来（`verify`）是否足以取代 Scribe 那一步，代价多少。

**过关线（评测前定下）**：中文上，

1. 口吃 / 重读的召回 ≥ 90%（以 `error` 报出），且报出 reshape 的回归样本 `stutter-spliced`（whisper 报不出的那条）；
2. 漏念、多念、替换、数字值不同合计召回（`warning` 及以上）不低于 whisper + reshape 现在的比对，也不低于只用 `align` quality；
3. 干净句在 `warning` 及以上的误报 ≤ 5%（09 §2.4 的线）；
4. 一句（≤ 10 s）额外耗时 ≤ 0.3 s。

## 2. 样本

| 组 | 来源 | 句数 | 标注 |
|---|---|---|---|
| `synth-zh` / `synth-en` | 新合成（`itts synth`，`tyr` / `demo_voice` 交替，固定 seed）：TTS 照"念的文本"读，稿件是正确文本 | 46 / 10 | 口吃 8 + 2（"北北边""我们，我们""编译器，编译器"、"The the"）、漏念 8 + 2、替换 8 + 2（含 2 句同字异读："行业"念 xíng、"长大"念 cháng）、数字值不同 6 + 2（`3`→三十、`1.5`→十五、`85%`→五十八、`3000`→三百、`5`→十五、`12`→二十一）、数字读法不同而值相同 2（"两千零二十六年""一千八百二十四年"）、含数字的干净句 6 + 2、干净句 8 |
| `spliced-zh` | 上面 8 句干净录音按原生字时间剪接：复制一个字 / 词并留 0.15–0.3 s 空隙（口吃 4），或剪掉一个字 / 词（漏念 4） | 8 | 剪接位置即真值 |
| `regression-zh` | reshape `llm-intro/tests/voice`：deepinfer `colophon.l2` 的干净版、剪掉"SGLang 的水系"版、开头"从"剪接重复版 | 3 | reshape 的 A004 / A005 回归样本 |
| `scribe-zh` | 庄子所有被 Scribe 转写过的中文句（ElevenLabs v4 克隆，原始 TTS 缓存，含句尾） | 115 | 4 句 Scribe 在句尾听出多余音节（"所以"×2、"辱"、"笑"：下一句开头漏进本句录音），记为多念；另 2 句（"鹏要等"↔"朋友等"）听不出真假，剔除；其余记为干净 |
| 对齐评测的 166 句 | itts、ElevenLabs、Gemini Flash / Lite、Edge 的干净句，难例（漏念 5、多念 2、替换 1、数字读法 4、停顿 / 中英混排 / 喊叫 6） | 166 | 沿用；对齐评测里 whisper 复听确认的 4 句真错（Gemini 把"用"念成 `used` 3 句、ElevenLabs 把 `Hway-dzuh's` 念成 `Huayza's`）记为 `real-error`；4 句数字读法里 3 句值相同（"两千零二十六""十一月三号"、`twenty twenty-six`）记为读法不同 |

合计 352 句：干净 282（中文 221、英文 61）、错误 70。追赶卡诺的 `qa.json` 和缓存里没有 Scribe 转写（那时还没开识别），只作为干净句出现在对齐评测的组里。

判定："被标出" = 该句至少有一条 finding 达到给定严重度。对照方法：

- **align quality**：`confidence` 或 `coverage` < 0.95 记 `warning`（reshape A014），`weak` 非空记 `info`。
- **whisper + reshape 比对**：`whisper-cli large-v3-turbo-q5_0`（reshape 的缺省命令和中文提示词），比对是 reshape `rs-audio/src/asr.rs::compare` 的移植：不带声调的拼音 LCS，重复 → A004（`error`），漏 / 多 ≥ 3 个音 → A005 W（`warning`），否则 I（`info`）。
- **Scribe（历史）+ reshape 比对**：同上，转写换成 Scribe 的历史结果（只有 `scribe-zh` 组有）。
- **listen + reshape 比对**：itts 的识别结果交给 reshape 现在的比对，用来分开"换识别器"和"换比对规则"的效果。
- **verify**：`itts verify`（对齐 + 识别 + 设计 §6 的比对）。另列"不用对齐佐证"（一个音的差异不因 `weak` 升级）。

## 3. 结果

### 3.1 中文（221 句干净、58 句有错），格子 = `warning` 及以上 / `info` 及以上 / 句数

| 方法 | 口吃 | 漏念 | 多念 | 替换 | 数字值不同 | 读法不同值相同 | 干净 | 真错 |
|---|---|---|---|---|---|---|---|---|
| align quality | 0/0/13 | 14/15/16 | 2/3/6 | 4/7/9 | 3/4/6 | 0/0/5 | 6/14/221 | 0/1/3 |
| whisper + reshape 比对 | 8/8/13 | 8/16/16 | 2/2/6 | 0/1/9 | 0/4/6 | 0/0/5 | 4/11/221 | 1/3/3 |
| Scribe（历史）+ reshape 比对 | — | — | 0/4/4 | — | — | — | 0/0/115 | — |
| listen + reshape 比对 | 13/13/13 | 7/16/16 | 2/3/6 | 0/1/9 | 0/5/6 | 0/0/5 | 6/11/221 | 1/3/3 |
| verify，不用对齐佐证 | 13/13/13 | 14/16/16 | 2/3/6 | 3/7/9 | 6/6/6 | 0/0/5 | 0/21/221 | 1/3/3 |
| **verify（Paraformer，缺省）** | **13/13/13** | **15/16/16** | 2/3/6 | **6/7/9** | **6/6/6** | 0/0/5 | **0/21/221** | 1/3/3 |
| verify（SenseVoice） | 13/13/13 | 15/16/16 | 2/3/6 | 6/7/9 | 6/6/6 | 0/0/5 | 0/27/221 | 1/3/3 |

- 口吃 13 句全部以 `error`（`repeat`）报出，包括 reshape 的 `stutter-spliced`；whisper 报出 8 句，漏掉的 5 句里有 `stutter-spliced` 和 3 句剪接的口吃——正是 33 §7 说要交 Scribe 的那一类。
- 数字值不同 6 句全部以 `error`（`number`）报出；读法不同而值相同的 5 句一句不报（设计 §6.1：按数值比）。itts 自己把 `2026年` 念成"二千零二十六年"，按读法报会把 itts 的干净句全部误报。
- 漏念唯一没到 `warning` 的是剪掉一个"化"（`info`，对齐不认为它弱）；多念里 4 句 Scribe 历史上报过的句尾残音，三个本机识别器（Paraformer、SenseVoice、whisper）都听不出，只有"笑"一句被 Paraformer / SenseVoice 以 `info` 报出。这类声音落在句子剪点之后，reshape 由 A009 剪尾处理。
- 替换没到 `warning` 的 3 句：2 句同字异读（"行业"念 xíng yè、"长大"念 cháng dà）——识别器输出汉字，按字比对看不出读音，Scribe 也一样；1 句"庄子"→"孔子"（`info`：一个字、对齐不佐证）。
- 干净句在 `warning` 以上**零误报**；`info` 21 句（9.5%），见 §4.2。
- reshape 现在的比对对替换一句都不报：LCS 上等长的差异被当作"识别器听错"直接丢掉（09 §2.3 的规则），所以即使换成 itts 的识别结果（"listen + reshape 比对"），替换和数字仍然是 0。verify 的改进一半来自识别器（口吃），一半来自比对规则（替换、数字）。

### 3.2 英文（61 句干净、11 句有错，Parakeet）

| 方法 | 口吃 | 漏念 | 替换 | 数字值不同 | 读法不同值相同 | 干净 | 真错 |
|---|---|---|---|---|---|---|---|
| align quality | 0/0/2 | 4/4/4 | 0/1/2 | 0/0/2 | 1/1/1 | 0/2/61 | 0/1/1 |
| whisper + reshape 比对 | 0/0/2 | 4/4/4 | 0/1/2 | 0/2/2 | 0/0/1 | 3/4/61 | 0/1/1 |
| verify，不用对齐佐证 | 2/2/2 | 1/4/4 | 0/2/2 | 2/2/2 | 0/0/1 | 0/6/61 | 1/1/1 |
| **verify（Parakeet）** | **2/2/2** | **4/4/4** | 1/2/2 | **2/2/2** | 0/0/1 | **0/6/61** | **1/1/1** |

SenseVoice 和 whisper 都把 "The the engine" "to to finish" 去掉了（原型评测里 SenseVoice 的英文口吃 0/2），Parakeet 两句都保留，所以英文缺省用 Parakeet。英文一个词的漏念（`cold`、`final quarterly` 里被对齐认为弱的词）靠对齐佐证升到 `warning`：不用佐证时只有 1/4。干净句的 6 句 `info` 是专名拼写（`Xu You`→`Hsu Yo`、`Wu`→`U`、`criticised`→`criticized`）。

### 3.3 汇总（中英合计，`warning` 及以上）

| 错误类型 | verify | whisper + reshape 比对 | align quality |
|---|---|---|---|
| 口吃 / 重读 | **15 / 15（100%，全部 `error`）** | 8 / 15（53%） | 0 / 15 |
| 漏念 | **19 / 20（95%）** | 12 / 20（60%） | 18 / 20（90%） |
| 多念 | 2 / 6（33%；句尾残音 0 / 4） | 2 / 6 | 2 / 6 |
| 替换 | **7 / 11（64%；去掉 2 句同字异读 7 / 9）** | 0 / 11 | 4 / 11 |
| 数字值不同 | **8 / 8（100%，全部 `error`）** | 0 / 8 | 3 / 8 |
| 干净句误报 | **0 / 282（0%）**；`info` 27 / 282 | 7 / 282 | 6 / 282 |

finding 的类型也对：口吃 15 句都有 `repeat`、数字 8 句都有 `number`；漏念 18 / 20 句报为 `omission`，另 1 句漏的是"十五"（归为 `number`）、1 句只到 `info`；替换报出的 7 句都是 `substitution`。

### 3.4 速度与体积

warm daemon，352 句（平均 5.4 s）：

| 阶段 | 平均 | P95 |
|---|---|---|
| `verify` 合计（CLI → daemon → 结果） | 213 ms | 465 ms |
| ≤ 10 s 的句子（322 句），daemon 内 | **173 ms** | 294 ms |
| 其中 `align`（MLX，与识别并行） | 145 ms | 287 ms |
| 其中 `listen`（ORT） | 201 ms | 438 ms |
| 识别本身 | 中文 144 ms、英文 336 ms；RTF 0.034 | |
| 比对 | < 1 ms | |
| whisper-cli（reshape 的调法，每句起一个进程） | 536 ms | 660 ms |

| 模型 | 下载 | 首次加载 | 进程峰值 RSS（`listen --no-daemon`，一句 3 s） |
|---|---|---|---|
| Paraformer-large int8 | 232 MiB | 0.31 s | 0.52 GiB |
| SenseVoice-Small int8 | 228 MiB | 0.26 s | 0.54 GiB |
| Parakeet-TDT 0.6B v2 int8 | 631 MiB | 0.50 s | 1.08 GiB |

（不加载模型时进程 RSS 12 MiB；daemon 里识别器与 MLX 引擎的约 1.8 GiB 分开计。）

- **静态 ORT 比动态库慢 3 倍**：同一句 3.2 s 的录音、同样 4 线程，sherpa-onnx 的 `static` 预编译库识别 110 ms，`shared` 预编译库（以及 Python wheel）36 ms；单线程 369 ms 对 98 ms，是同一版本（1.28.2）不同构建的算子差别，与线程无关。itts 用静态库：`cargo install` 出来是一个自包含的二进制；动态库要把 `libonnxruntime.dylib` / `libsherpa-onnx-c-api.dylib` 随二进制分发并设 rpath。现在的速度已在过关线内，所以没有换；要再快 3 倍可以改用 `shared` 并在发布包里带上两个 dylib。
- CoreML EP 对 int8 模型更慢（32 → 43 ms，加载 0.3 → 2.0 s），不用。
- SenseVoice 在 126 s 的拼接录音上与单句准确率相同（`listen` 上限 120 s）。

## 4. 分析

### 4.1 为什么非自回归识别器能听出口吃

whisper 的解码器按语言模型生成文本，"从从"这种不合语法的重复概率极低，会被顺掉；SenseVoice（CTC）、Paraformer（CIF）逐帧或逐段输出，没有跨 token 的语言模型，听到两个"从"就输出两个。13 句中文口吃（TTS 读出的 8 句 + 剪接 4 句 + reshape 回归 1 句）两个模型都逐字报出，连"鲲化化"被听成"昆画画"也没关系——比对按拼音，`画画` 重复了紧邻的 `化`。英文同理，Parakeet（transducer，无外部语言模型）保留 "the the"；SenseVoice 的英文部分会去掉重复，所以英文不用它。

### 4.2 一个字的差异：识别器误听还是真念错

干净句里 `info` 的 21 句（Paraformer）几乎都是一个字 / 一个词的替换：

- 生僻字、古文：`鴳`→`娟` / `冠`（3 句）、`蟪`→`过`、`散`→`善`、`楚`→`蜀`、`列`→`别`；
- 产品名、英文词：`axon`→`x on`、`Homebrew`→`home bro`、`TestFlight`→`text flight`、`Rust`→`rest`、`SGLang`→`s g land`、`Claude`→`cloud`；
- 少数常用字：`志`→`记`、`进`→`睛`、`大`→`道`。

其中 `鴳`→`娟`、`蟪`→`过` 在 Scribe 的历史转写里也是一样的（"赤娟""过孤"），可能是 ElevenLabs 真念错了生僻字；没有试听无法确认，按干净句计。这些差异对齐几乎都不认为弱（`weak` 为空），而合成的真替换（"北京"→"南京"、"蓝"→"红"、"杯"→"盘"）对齐多半认为弱。所以设计里一个音的差异只有在对齐佐证时才升为 `warning`：替换的 `warning` 召回从 3 / 9 升到 6 / 9，干净句的 `warning` 仍是 0。两个以上单位的差异在 282 句干净句里没有出现过，直接记 `warning`。

Paraformer 与 SenseVoice 召回完全相同，`info` 少 6 句（21 对 27，主要在 Gemini 的中英混排句和庄子的古文句），识别还快一点（144 对 177 ms），所以 `zh` / `zh/en` 缺省用 Paraformer；SenseVoice 留给粤语、日语，以及想要多语言模型的场合（`--model sense-voice`）。

### 4.3 数字

识别器关掉 ITN 后数字按念出来的样子输出（"两千零二十六""百分之五十八""一点五"），稿件里的阿拉伯数字、汉字数字、英文数词都换成数值再比（英文 `twenty fourteen` 同时有 `34` 和 `2014` 两个候选值，`five point four point oh` 是 `5.4.0`）。值不同报 `number`（`error`），值相同不报。reshape 33 §5.5 里"含阿拉伯数字的补齐句一律送识别"的规则可以由 `number` 取代。

### 4.4 没解决的

- **同字异读**："行业"念成 xíng yè、"长大"念成 cháng dà。识别器输出的是汉字，按字比对相同；要查需要识别器输出拼音（声学模型直出音素），不在本次范围。Scribe 同样查不出。
- **句尾残音**：下一句的开头漏进本句录音（ElevenLabs 的缓存录音在句尾带着 0.2–0.9 s 的下一句开头），Scribe 有时听成"所以""辱"，本机识别器都听不出，由 reshape 的 A009 剪尾处理。
- **一个字的漏念 / 替换且对齐不佐证**（"化"、"庄"→"孔"）：以 `info` 报出，reshape 可在 G2 页面列出但不阻挡。

## 5. 结论（go / no-go）

| 过关线 | 结果 | 判定 |
|---|---|---|
| 中文口吃召回 ≥ 90%，含 `stutter-spliced` | **13 / 13，全部 `error`**，含回归样本（whisper 8 / 13，漏掉回归样本） | 过 |
| 漏念 + 多念 + 替换 + 数字合计不低于 whisper / align | **29 / 37**（whisper 10 / 37，align 23 / 37） | 过 |
| 干净句 `warning` 以上误报 ≤ 5% | **0 / 221**（中文）、0 / 61（英文） | 过 |
| ≤ 10 s 的句子额外耗时 ≤ 0.3 s | verify 合计平均 173 ms、P95 294 ms（其中对齐 145 ms 本来就要跑） | 过 |

**go**：实现 `itts listen` / `itts verify`，中文缺省 Paraformer、粤语 / 日语 SenseVoice、英文 Parakeet。对 reshape：verify 可以取代"whisper 粗筛 → Scribe 升级"整条链路，每句约 0.2 s、免费，所以所有句子都可以送检（设计 §8）。

## 6. 复现

```bash
# 1. 合成样本（需要 itts daemon 和 tyr / demo_voice 两个音色）
uv run -q --python 3.12 --no-project --with pypinyin==0.55.0 --with numpy --with soundfile \
  python -I tools/asr_eval.py synth target/asr-eval/synth > target/asr-eval/manifest.jsonl
# 2. 追加其他样本行（对齐评测的 manifest、reshape 缓存）：{id, set, audio, text, lang, kind, scribe?}
# 3. 识别 + 报告（verify 走 daemon；whisper 需要 ~/.reshape/models/ggml-large-v3-turbo-q5_0.bin）
make asr-eval ASR_EVAL_MANIFEST=target/asr-eval/manifest.jsonl
```

`make asr-eval` 依次运行 `whisper`、`itts`（缺省模型）、`itts --model sense-voice` 与 `report`，结果写在 manifest 旁边。

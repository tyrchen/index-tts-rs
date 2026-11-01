# 基于 GPT 文本注意力的字级时间戳：实验与选型

- **日期**: 2026-10-31
- **结论落地**: [`specs/itts-timestamps-and-hifi-design.md`](../../specs/itts-timestamps-and-hifi-design.md) §3，代码 `crates/core/src/align/`
- **硬件**: Apple M5 Pro；模型 `yunfengwang/IndexTTS-2.5-mlx`（GPT int8）

## 1. 问题

外部应用需要逐字时间戳（字幕、动画锚点、按句切音频、漏读筛查）。候选方案：

| 方案 | 代价 | 结论 |
|------|------|------|
| 外挂强制对齐模型（wav2vec2/MMS-FA、Whisper DTW） | 额外 300M–1.5B 参数权重、移植一个新网络、每次再跑一遍声学模型；中文需要拼音转写 | 作为**评测参照**，不进生产 |
| ASR 词级时间戳（whisper.cpp `-ml 1`） | 同上；实测时间戳粒度粗（±100 ms，句间停顿不可见） | 不采用 |
| **GPT 自身 text↔mel 注意力** | 零新权重；只需重放部分层 | **采用** |

可行的前提：本版本管线中 S2Mel 只消费 GPT 的语义 codes（`codec` 2× repeat → `LengthRegulator` 最近邻线性展开 → CFM → BigVGAN），所以第 *i* 个 code 精确对应音频 `[i·Δ, (i+1)·Δ)`，`Δ = 时长 / codes 数 ≈ 40 ms`。

## 2. 方法

1. 探针：生成 codes 后，用 teacher forcing 重放 `[conds, <BOS>, text, <EOS>, <START_MEL>, codes[..N-1]]`，逐层逐头计算 mel 查询对文本键的 softmax 注意力（`[24, 20, N, T]`）。
2. 参照：同一 seed 重新合成得到逐字节相同的音频，用 torchaudio `MMS_FA`（多语言 wav2vec2 CTC 强制对齐，中文用 pypinyin 转写）得到每个字/词的起点。
3. 指标：字（中文）/词（英文）**起点**绝对误差。CTC 的终点偏"尖峰"，不作主指标。
4. 样本：中文 2 条、英文 2 条，两个音色（`tyr`、`demo_voice`），含数字展开（"二零二六年""百分之四十二"）、逗号/句号停顿、问句感叹句。

## 3. 观察

- 第 4–7 层存在多个"对齐头"：逐帧 argmax 单调率 0.95–0.98，覆盖 ~90% token，峰值概率 ~0.8。
- 静音期间注意力落在 `<BOS>`（attention sink）。停顿内，部分头会**提前**看向下一个字（第 6 层 H12 整体超前一个字），所以停顿常被算进下一个字——必须用波形修正边界。
- 全部 480 个头的单头扫描（含 Viterbi + 能量修正）：

| 头 | 平均起点误差 zh1 / zh2 / en1 / en2 (ms) |
|----|------|
| **L4H10** | 31 / 32 / 52 / 43 |
| L5H10 | 25 / 25 / 173 / 93 |
| L6H15 | 38 / 39 / 156 / 104 |
| L4H1 | 114 / 123 / 256 / 200 |

- 头组合：**L4H10 + L5H10** 最优（19 / 23 / 54 / 42 ms），加更多头无收益或变差。

## 4. 解码与修正

- 可读 token（排除特殊 token、空白、标点）作为状态；每帧在可读列上重新归一化后取对数作发射。
- 单调 Viterbi：停留或前进 1，从首个到最后一个可读 token。
- 能量修正（10 ms 帧 RMS，阈值 `max(1e-3, P95 × 0.05)`）：
  - 首 token 起点钉到第一个有声帧、末 token 终点钉到最后一个有声帧；
  - ≥80 ms 的句中静音吸附到 0.2 s 内最近的 token 边界：左 token 终点**设为**静音起点、右 token 起点**设为**静音终点（双向，因为解码边界可能落在静音任一侧）；
  - 首尾静音不参与吸附（否则会把开头静音错误地算给第二个字）；
  - 被压成零长度的 token 与相邻 token 平分时长。

吸附阈值在 0.12–0.25 s 间对上述样本几乎无差异（平均 29–31 ms），但 0.12 s 无法修正一个真实中英混排样本中 "Rust" 后 0.48 s 停顿被整体算进"重"的错误（偏差 0.6 s），故取 0.2 s。

## 5. 生产实现结果

生产实现（Rust，`IndexTtsEngine` 带 `align: true`）与 Python 原型逐值一致，开启对齐后**音频逐字节不变**。

| 条件 | 平均 | 中位数 | P90 | ≤100 ms |
|------|------|--------|-----|---------|
| 中性（zh×2、en×2） | 29 ms | 17 ms | 63 ms | 96% |
| 情绪/变速（sad 0.8×、angry、1.3×，zh×2、en×2） | 23 ms | 16 ms | — | 98% |

- 中文字起点偏置 +5～+13 ms；英文词起点偏置 −17～−32 ms（BPE token 带前导空格，且 CTC 参照本身倾向于把词首放晚）。
- 开销（warm daemon）：每个任务 13–45 ms，约为 GPT 生成时间的 2–4%。只重放前 6 层、只算 2 个头。
- 坐标映射：归一化文本 ↔ 请求文本 ↔ 原始输入用 Myers 字符 diff（大小写、全/半角标点折叠）；数字、英文缩写、注音标记等替换段按字符数比例分摊。

## 6. 局限

- 呼吸声或环境噪声填充的停顿可能达不到静音阈值，此时停顿两侧的边界可能偏移（见 en1 "shifted" 前 0.3 s）。
- 一个 BPE token 覆盖多个字母时，token 内各字符按等分处理，英文字母级时间只是近似。
- 头选择基于 IndexTTS-2.5 本版权重；换权重需按本文方法重新扫描。

## 7. 复现

参照对齐脚本（需 `uv`；首次运行下载约 1.2 GB 的 MMS 模型）：

```python
# fa.py WAV LANG TEXT OUT_JSON -> 每个字/词的 [start, end] 秒（torchaudio MMS_FA）
import sys, json, re, torch, torchaudio
from torchaudio.pipelines import MMS_FA as bundle
wav, lang, text, out = sys.argv[1:5]
w, sr = torchaudio.load(wav); w = torchaudio.functional.resample(w, sr, bundle.sample_rate)
model, tok, aligner = bundle.get_model(), bundle.get_tokenizer(), bundle.get_aligner()
if lang == "zh":
    from pypinyin import lazy_pinyin
    units = [c for c in text if c.strip() and not re.match(r"[，。！？、；：,.!?;:\"'“”‘’（）()—…-]", c)]
    words = [re.sub(r"[^a-z]", "", lazy_pinyin(c)[0]) for c in units]
else:
    units = re.findall(r"[A-Za-z']+", text); words = [u.lower() for u in units]
with torch.inference_mode():
    em, _ = model(w); spans = aligner(em[0], tok(words))
ratio = w.shape[1] / em.shape[1] / bundle.sample_rate
json.dump([{"u": u, "s": sp[0].start * ratio, "e": sp[-1].end * ratio} for u, sp in zip(units, spans)],
          open(out, "w"), ensure_ascii=False)
```

```bash
uv run --python 3.12 --no-project --with torch==2.8.0 --with torchaudio==2.8.0 \
  --with pypinyin --with soundfile python fa.py out.wav zh "$TEXT" fa.json
itts synth --voice tyr --seed 7 --timestamps --output json --text "$TEXT" --out out.wav > ours.json
```

`torchaudio` 2.9 移除了 `forced_align`，需固定 2.8。

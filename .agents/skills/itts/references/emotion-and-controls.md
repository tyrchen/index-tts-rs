# `itts` Emotion & Prosody Calibration Cookbook

## 1. How IndexTTS-2.5 Emotion Conditioning Works

1. **8-Class Prototype Matching**:
   - `index-tts-rs-core` embeds the 73 upstream emotion-speaker prototypes (`spk_matrix [73, 192]` and `emo_matrix [73, 1280]`).
   - For each active emotion in `[[emo=...]]` or `--emo`, the engine selects the prototype row whose `CAMPPlus` speaker style vector has the highest cosine similarity with the active speaker's `style` vector.
2. **Bias & Sum Cap (`<= 0.8`)**:
   - Raw weights are multiplied by upstream empirical bias (`[0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]`).
   - If the sum of biased weights exceeds `0.8`, the vector is scaled to sum to `0.8` so at least `20%` of the speaker's original reference emotion (`base_emovec`) and full speaker identity (`spk_cond`) are preserved.
3. **Recommended Weight Ranges**:
   - **Subtle / natural shading**: total weights around `0.40..0.65` (e.g. `[[emo=melancholic:0.35,calm:0.25]]`).
   - **Strong emotional turn**: total weights around `0.65..0.80` (e.g. `[[emo=sad:0.4,melancholic:0.35]]` or `[[emo=happy:0.5,surprised:0.25]]`).
   - Avoid setting 4+ conflicting emotions at once; blend **1 to 3 complementary emotions** per couplet or sentence.

---

## 2. Fine-Grained Poetry Example: 《再别康桥》 (Xu Zhimo)

When reading lyrical poetry, split stanzas at strong couplet boundaries (`；`, `。`, `！`, `？`) so intra-stanza emotional transitions receive distinct `[[emo=...]]`, `[[temp=...]]`, `[[speed=...]]`, and `[[pause=...]]` controls:

```text
[[emo=melancholic:0.25,calm:0.45]][[temp=0.72]][[speed=0.88]]
轻轻的我走了，
正如我轻轻的来；
[[pause=450]][[emo=melancholic:0.35,calm:0.35]][[temp=0.74]][[speed=0.9]]
我轻轻的招手，
作别西天的云彩。

[[pause=900]][[emo=happy:0.2,calm:0.45]][[temp=0.78]][[speed=0.92]]
那河畔的金柳，
是夕阳中的新娘；
[[pause=450]][[emo=happy:0.3,melancholic:0.25,calm:0.2]][[temp=0.8]][[speed=0.9]]
波光里的艳影，
在我的心头荡漾。

[[pause=900]][[emo=happy:0.25,calm:0.4]][[temp=0.78]][[speed=0.92]]
软泥上的青<荇|xing4>，
油油的在水底招摇；
[[pause=450]][[emo=melancholic:0.35,calm:0.35]][[temp=0.75]][[speed=0.86]]
在康河的柔波里，
我甘心做一条水草！

[[pause=950]][[emo=calm:0.45,surprised:0.15]][[temp=0.76]][[speed=0.9]]
那榆荫下的一潭，
不是清泉，是天上虹；
[[pause=500]][[emo=melancholic:0.3,calm:0.35]][[temp=0.74]][[speed=0.88]]
揉碎在浮藻间，
沉淀着彩虹似的梦。

[[pause=950]][[emo=happy:0.35,surprised:0.2]][[temp=0.84]][[speed=0.96]]
寻梦？撑一支长篙，
向青草更青处漫溯；
[[pause=450]][[emo=happy:0.5,surprised:0.2]][[temp=0.88]][[speed=0.98]]
满载一船星辉，
在星辉斑斓里放歌。

[[pause=1000]][[emo=melancholic:0.45,sad:0.2]][[temp=0.7]][[speed=0.85]]
但我不能放歌，
悄悄是别离的笙箫；
[[pause=600]][[emo=sad:0.35,melancholic:0.4]][[temp=0.68]][[speed=0.82]]
夏虫也为我沉默，
沉默是今晚的康桥！

[[pause=1100]][[emo=melancholic:0.3,calm:0.45]][[temp=0.72]][[speed=0.86]]
悄悄的我走了，
正如我悄悄的来；
[[pause=500]][[emo=melancholic:0.25,calm:0.5]][[temp=0.7]][[speed=0.84]]
我挥一挥衣袖，
不带走一片云彩。
```

### Running this script in Nushell
```nu
open --raw kangqiao.itts.txt | itts synth --voice tyr --stdin --out kangqiao.wav --play
```

---

## 3. Multi-Speaker Dialogue Example

```text
[[voice=narrator]][[emo=calm:0.5]][[temp=0.75]][[speed=0.95]]
夜色渐深，书房里只剩下一盏台灯。

[[pause=700]][[voice=tyr]][[emo=surprised:0.35,afraid:0.2]][[temp=0.86]][[speed=1.05]]
这么晚了，外面是谁在敲门？

[[pause=550]][[voice=narrator]][[emo=calm:0.45,melancholic:0.2]][[temp=0.74]][[speed=0.9]]
没有人回答，只有风吹过走廊的声音。[[reset]]
```

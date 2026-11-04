# /// script
# requires-python = ">=3.12"
# dependencies = ["pypinyin==0.55.0", "numpy", "soundfile"]
# ///
"""Evaluation harness for `itts listen` / `itts verify` (docs/research/itts-asr-evaluation.md).

Every subcommand reads a JSONL manifest of {id, set, audio, text, lang, kind, scribe?}: `kind` is
`clean`, `stutter`, `omission`, `insertion`, `substitution`, `digits` (a number read with another
value), `digit-form` (same value, another reading) or `real-error`; `scribe` is a historical
ElevenLabs Scribe transcript (reference only — this tool never calls a paid API).

  synth    OUT_DIR             synthesize the stutter / omission / substitution / digit samples
                               with `itts synth` and splice the cut-and-paste ones; prints JSONL
  whisper  MANIFEST OUT        whisper-cli transcripts (reshape's local recognizer and command)
  itts     MANIFEST OUT        `itts verify --output json` per sample (also yields listen + align)
  report   MANIFEST --verify OUT [--alt OUT --alt-label NAME] [--whisper OUT]   recall table
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

RANK = {"info": 1, "warning": 2, "error": 3}
KINDS = ["stutter", "omission", "insertion", "substitution", "digits", "digit-form", "clean", "real-error"]
WHISPER_MODEL = Path.home() / ".reshape/models/ggml-large-v3-turbo-q5_0.bin"

# (id, kind, lang, script, what the TTS reads); None reads the script (a clean line).
SAMPLES = [
    ("st-zh-1", "stutter", "zh", "从北边吹来的风越来越冷。", "从北北边吹来的风越来越冷。"),
    ("st-zh-2", "stutter", "zh", "我们今天下午去公园散步。", "我们，我们今天下午去公园散步。"),
    ("st-zh-3", "stutter", "zh", "这件事情其实没有那么复杂。", "这件事事情其实没有那么复杂。"),
    ("st-zh-4", "stutter", "zh", "鲲化而为鸟，其名为鹏。", "鲲化化而为鸟，其名为鹏。"),
    ("st-zh-5", "stutter", "zh", "每一次笑，都是一把尺子在量人。", "每一次，每一次笑，都是一把尺子在量人。"),
    ("st-zh-6", "stutter", "zh", "编译器会检查所有的类型。", "编译器，编译器会检查所有的类型。"),
    ("st-zh-7", "stutter", "zh", "水往低处流，人往高处走。", "水水往低处流，人往高处走。"),
    ("st-zh-8", "stutter", "zh", "他终于明白了这个道理。", "他终于明明白了这个道理。"),
    ("st-en-1", "stutter", "en", "The engine runs hot today.", "The the engine runs hot today."),
    ("st-en-2", "stutter", "en", "We need to finish this by Friday.", "We need to to finish this by Friday."),
    ("om-zh-1", "omission", "zh", "他放下手里的书，抬头看了看窗外。", "他放下书，抬头看了看窗外。"),
    ("om-zh-2", "omission", "zh", "庄子说，至人无己，神人无功，圣人无名。", "庄子说，至人无己，圣人无名。"),
    ("om-zh-3", "omission", "zh", "这个模型在中文上的表现非常好。", "这个模型在中文上表现好。"),
    ("om-zh-4", "omission", "zh", "风积得不厚，就托不起大翅膀。", "风积得不厚，就托不起翅膀。"),
    ("om-zh-5", "omission", "zh", "列子乘风而行，十五天才回来。", "列子乘风而行，才回来。"),
    ("om-zh-6", "omission", "zh", "第一步是读取配置文件。", "第一步是读取文件。"),
    ("om-zh-7", "omission", "zh", "请把门关上，然后坐下来。", "请把门关上，坐下来。"),
    ("om-zh-8", "omission", "zh", "蒸汽推动活塞，活塞带动飞轮。", "蒸汽推动活塞，带动飞轮。"),
    ("om-en-1", "omission", "en", "Please send me the final quarterly report by tomorrow morning.",
     "Please send me the report by tomorrow morning."),
    ("om-en-2", "omission", "en", "The heat flows from the hot body to the cold one.",
     "The heat flows from the hot body to the one."),
    ("sb-zh-1", "substitution", "zh", "银行明天上午九点开门。", "商店明天上午九点开门。"),
    ("sb-zh-2", "substitution", "zh", "他在北京工作了五年。", "他在南京工作了五年。"),
    ("sb-zh-3", "substitution", "zh", "鹏要等风起才能飞。", "鹏要等雨停才能飞。"),
    ("sb-zh-4", "substitution", "zh", "这本书的作者是庄子。", "这本书的作者是孔子。"),
    ("sb-zh-5", "substitution", "zh", "天空是深蓝色的。", "天空是深红色的。"),
    ("sb-zh-6", "substitution", "zh", "他把杯子放在桌子上。", "他把盘子放在桌子上。"),
    ("sb-zh-7", "substitution", "zh", "行业报告已经发布了。", "<行|xing2>业报告已经发布了。"),
    ("sb-zh-8", "substitution", "zh", "孩子们都长大了。", "孩子们都<长|chang2>大了。"),
    ("sb-en-1", "substitution", "en", "The cat sat quietly on the mat.", "The dog sat quietly on the mat."),
    ("sb-en-2", "substitution", "en", "Carnot asked how much work heat can do.",
     "Carnot asked how much work steam can do."),
    ("dg-zh-1", "digit-form", "zh", "2026年，我们发布了第三个版本。", "两千零二十六年，我们发布了第三个版本。"),
    ("dg-zh-2", "digits", "zh", "第3章讲的是热机。", "第三十章讲的是热机。"),
    ("dg-zh-3", "digits", "zh", "速度提升了1.5倍。", "速度提升了十五倍。"),
    ("dg-zh-4", "digits", "zh", "大约有85%的人同意。", "大约有百分之五十八的人同意。"),
    ("dg-zh-5", "digits", "zh", "这座山高3000米。", "这座山高三百米。"),
    ("dg-zh-6", "digit-form", "zh", "他在1824年发表了论文。", "他在一千八百二十四年发表了论文。"),
    ("dg-zh-7", "digits", "zh", "温度降到了零下5度。", "温度降到了零下十五度。"),
    ("dg-zh-8", "digits", "zh", "一共用了12个小时。", "一共用了二十一个小时。"),
    ("dg-en-1", "digits", "en", "In 2026, sales grew by 15 percent.", "In 2026, sales grew by 50 percent."),
    ("dg-en-2", "digits", "en", "Chapter 3 begins with a steam engine.", "Chapter 30 begins with a steam engine."),
    ("cd-zh-1", "clean", "zh", "2026年，我们发布了第三个版本。", None),
    ("cd-zh-2", "clean", "zh", "第3章讲的是热机。", None),
    ("cd-zh-3", "clean", "zh", "速度提升了1.5倍。", None),
    ("cd-zh-4", "clean", "zh", "大约有85%的人同意。", None),
    ("cd-zh-5", "clean", "zh", "这座山高3000米。", None),
    ("cd-zh-6", "clean", "zh", "一共用了12个小时。", None),
    ("cd-en-1", "clean", "en", "In 2026, sales grew by 15 percent.", None),
    ("cd-en-2", "clean", "en", "Chapter 3 begins with a steam engine.", None),
    ("cl-zh-1", "clean", "zh", "从北边吹来的风越来越冷。", None),
    ("cl-zh-2", "clean", "zh", "我们今天下午去公园散步。", None),
    ("cl-zh-3", "clean", "zh", "他放下手里的书，抬头看了看窗外。", None),
    ("cl-zh-4", "clean", "zh", "蒸汽推动活塞，活塞带动飞轮。", None),
    ("cl-zh-5", "clean", "zh", "每一次笑，都是一把尺子在量人。", None),
    ("cl-zh-6", "clean", "zh", "这个模型在中文上的表现非常好。", None),
    ("cl-zh-7", "clean", "zh", "鲲化而为鸟，其名为鹏。", None),
    ("cl-zh-8", "clean", "zh", "编译器会检查所有的类型。", None),
]
# Cut-and-paste errors on the clean takes: (id, base, op, first char, last char inclusive, gap s).
SPLICES = [
    ("sp-st-1", "cl-zh-1", "dup", 1, 1, 0.2), ("sp-st-2", "cl-zh-2", "dup", 0, 0, 0.3),
    ("sp-st-3", "cl-zh-5", "dup", 0, 2, 0.25), ("sp-st-4", "cl-zh-8", "dup", 0, 0, 0.15),
    ("sp-om-1", "cl-zh-3", "cut", 3, 5, 0.0), ("sp-om-2", "cl-zh-4", "cut", 7, 8, 0.0),
    ("sp-om-3", "cl-zh-6", "cut", 9, 10, 0.0), ("sp-om-4", "cl-zh-7", "cut", 1, 1, 0.0),
]


def load_manifest(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cmd_synth(args: argparse.Namespace) -> None:
    import numpy as np
    import soundfile as sf

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scripts = {}
    for n, (sid, kind, lang, script, read) in enumerate(SAMPLES):
        scripts[sid] = script
        wav = out / f"{sid}.wav"
        if not wav.exists():
            voice = "tyr" if n % 2 == 0 else "demo_voice"
            cmd = [args.bin, "synth", "--voice", voice, "--lang", lang, "--seed", str(100 + n),
                   "--timestamps", "--output", "json", "--out", str(wav), "--stdin"]
            done = subprocess.run(cmd, input=(read or script).encode(), capture_output=True, check=True)
            (out / f"{sid}.json").write_bytes(done.stdout)
        print(json.dumps({"id": sid, "set": f"synth-{lang}", "audio": str(wav), "text": script,
                          "lang": lang, "kind": kind}, ensure_ascii=False))
    for sid, base, op, first, last, gap in SPLICES:
        x, sr = sf.read(out / f"{base}.wav", dtype="float32")
        al = json.loads((out / f"{base}.json").read_text())["alignment"]
        s, e = int(al["start"][first] * sr), int(al["end"][last] * sr)
        ramp = np.linspace(1, 0, int(0.01 * sr), dtype=np.float32)
        if op == "dup":
            head, tail = x[:e].copy(), x[s:].copy()
            mid = np.zeros(int(gap * sr), np.float32)
        else:
            head, tail, mid = x[:s].copy(), x[e:].copy(), np.zeros(0, np.float32)
        head[-len(ramp):] *= ramp
        tail[: len(ramp)] *= ramp[::-1]
        sf.write(out / f"{sid}.wav", np.concatenate([head, mid, tail]), sr)
        kind = "stutter" if op == "dup" else "omission"
        print(json.dumps({"id": sid, "set": "spliced-zh", "audio": str(out / f"{sid}.wav"),
                          "text": scripts[base], "lang": "zh", "kind": kind}, ensure_ascii=False))


def cmd_whisper(args: argparse.Namespace) -> None:
    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    wav = Path(args.out).with_suffix(".tmp.wav")
    for sample in load_manifest(args.manifest):
        if sample["id"] in out:
            continue
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", sample["audio"], "-ac", "1", "-ar", "16000",
                        str(wav)], check=True)
        cmd = ["whisper-cli", "-m", str(WHISPER_MODEL), "-l", sample["lang"], "-nt", "-np", "-f", str(wav)]
        if sample["lang"] == "zh":
            cmd += ["--prompt", "以下是普通话的句子，使用简体中文。"]
        started = time.perf_counter()
        done = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
        out[sample["id"]] = {"text": done.stdout.decode().strip(), "secs": time.perf_counter() - started}
        print(f"whisper {sample['id']}: {out[sample['id']]['text']}", file=sys.stderr)
    wav.unlink(missing_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))


def cmd_itts(args: argparse.Namespace) -> None:
    out = {}
    for sample in load_manifest(args.manifest):
        cmd = [args.bin, "verify", sample["audio"], "--lang", sample["lang"], "--stdin", "--output", "json"]
        if args.socket:
            cmd += ["--socket", args.socket]
        if args.model and sample["lang"] != "en":
            cmd += ["--model", args.model]
        started = time.perf_counter()
        done = subprocess.run(cmd, input=sample["text"].encode(), capture_output=True)
        doc = json.loads(done.stdout or b"{}")
        doc["wallSecs"] = time.perf_counter() - started
        doc["exitCode"] = done.returncode
        out[sample["id"]] = doc
        print(f"itts {sample['id']}: {doc.get('verdict')} {doc.get('heard', {}).get('text')}", file=sys.stderr)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))


# ------------------------------------------------------------ reshape's comparison (the baseline)
# A port of reshape `rs-audio/src/asr.rs::{normalize, sound_tokens, compare, asr_codes}`: toneless
# pinyin LCS, A004 (repeat, E) and A005 (missing / extra, W from 3 sounds, else I).

CN_DIGIT = dict(zip("零〇一二两三四五六七八九", [0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9]))
CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn_number(run: str) -> str:
    if all(c in CN_DIGIT for c in run):
        return "".join(str(CN_DIGIT[c]) for c in run)
    total, section, digit = 0, 0, 0
    for c in run:
        if c in CN_DIGIT:
            digit = CN_DIGIT[c]
        elif c == "万":
            total += (section + digit) * 10**4
            section = digit = 0
        elif c == "亿":
            total = (total + section + digit) * 10**8
            section = digit = 0
        else:
            section += (digit or 1) * CN_UNIT[c]
            digit = 0
    return str(total + section + digit)


def reshape_normalize(s: str) -> str:
    s = re.sub("百分之([〇零一二两三四五六七八九十百千]+)", lambda m: cn_number(m.group(1)) + "%", s).lower()
    s = re.sub("[零〇一二两三四五六七八九十百千万亿]{2,}|[十百千]",
               lambda m: m.group(0) if len(m.group(0)) == 1 and m.group(0) in CN_DIGIT else cn_number(m.group(0)), s)
    s = re.sub(r"\s*(?:%|percent|per cent)", "", s)
    return re.sub(r"[^\w]", "", s)


def reshape_tokens(s: str) -> list[str]:
    from pypinyin import lazy_pinyin

    out: list[str] = []
    for m in re.finditer(r"[一-鿿]+|[^一-鿿]+", reshape_normalize(s)):
        run = m.group(0)
        out += lazy_pinyin(run) if re.match(r"[一-鿿]", run) else list(run)
    return out


def reshape_level(reading: str, heard: str) -> int:
    a, b = reshape_tokens(reading), reshape_tokens(heard)
    level, sounds = 0, 0
    for op, i0, i1, j0, j1 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        miss, extra = i1 - i0, j1 - j0
        if miss > extra:
            sounds += miss - extra
        elif extra > miss:
            k, added = extra - miss, b[j0 + miss:j1]
            if miss == 0 and (added == a[i1:i1 + k] or added == a[max(0, i0 - k):i0]):
                level = 3
            else:
                sounds += k
    if sounds:
        level = max(level, 2 if sounds >= 3 else 1)
    return level


# ------------------------------------------------------------ report


def verify_level(doc: dict, use_align: bool = True) -> int | None:
    if doc.get("status") != "ok":
        return None
    level = 0
    for f in doc["findings"]:
        sev = f["severity"]
        if not use_align and sev == "warning" and f["sounds"] < 2 and f["alignWeak"]:
            sev = "info"
        level = max(level, RANK[sev])
    return level


def align_level(doc: dict) -> int | None:
    if doc.get("status") != "ok":
        return None
    q = doc["align"]["quality"]
    return 2 if q["confidence"] < 0.95 or q["coverage"] < 0.95 else (1 if q["weak"] else 0)


def stats(xs: list[float]) -> str:
    if not xs:
        return "-"
    xs = sorted(xs)
    return f"mean {sum(xs) / len(xs) * 1000:.0f} ms, p95 {xs[int(0.95 * (len(xs) - 1))] * 1000:.0f} ms"


def cmd_report(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    verify = json.loads(Path(args.verify).read_text())
    alt = json.loads(Path(args.alt).read_text()) if args.alt else {}
    whisper = json.loads(Path(args.whisper).read_text()) if args.whisper else {}
    methods = {
        "align quality": lambda s: align_level(verify.get(s["id"], {})),
        "whisper + reshape diff": lambda s: reshape_level(s["text"], whisper[s["id"]]["text"]) if s["id"] in whisper else None,
        "Scribe (historical) + reshape diff": lambda s: reshape_level(s["text"], s["scribe"]) if s.get("scribe") else None,
        "listen + reshape diff": lambda s: reshape_level(s["text"], verify[s["id"]]["heard"]["text"]) if verify.get(s["id"], {}).get("status") == "ok" else None,
        "verify, no align evidence": lambda s: verify_level(verify.get(s["id"], {}), use_align=False),
        "verify": lambda s: verify_level(verify.get(s["id"], {})),
    }
    if alt:
        methods[args.alt_label] = lambda s: verify_level(alt.get(s["id"], {}))
    report = {}
    for lang in sorted({s["lang"] for s in manifest}):
        rows = [s for s in manifest if s["lang"] == lang]
        table = {}
        for name, level in methods.items():
            cells = {}
            for kind in KINDS:
                vals = [v for v in (level(s) for s in rows if s["kind"] == kind) if v is not None]
                if vals:
                    cells[kind] = {"warning": sum(v >= 2 for v in vals), "info": sum(v >= 1 for v in vals),
                                   "error": sum(v >= 3 for v in vals), "n": len(vals)}
            table[name] = cells
        report[lang] = table
        print(f"\n## {lang} (cells: flagged at >= warning / >= info / n)")
        print("| method | " + " | ".join(KINDS) + " |")
        print("|---" * (len(KINDS) + 1) + "|")
        for name, cells in table.items():
            cols = [f"{c['warning']}/{c['info']}/{c['n']}" if (c := cells.get(k)) else "-" for k in KINDS]
            print(f"| {name} | " + " | ".join(cols) + " |")
    docs = [d for d in verify.values() if d.get("status") == "ok"]
    timing = {
        "verify wall (CLI → daemon)": stats([d["wallSecs"] for d in docs]),
        "verify total": stats([d["timings"]["totalMs"] / 1000 for d in docs]),
        "align": stats([d["timings"]["alignMs"] / 1000 for d in docs]),
        "listen": stats([d["timings"]["listenMs"] / 1000 for d in docs]),
        "recognize": stats([d["heard"]["timings"]["recognizeMs"] / 1000 for d in docs]),
        "whisper": stats([w["secs"] for w in whisper.values()]),
    }
    audio = sum(d["durationSecs"] for d in docs)
    rtf = sum(d["heard"]["timings"]["recognizeMs"] / 1000 for d in docs) / audio if audio else 0
    print("\n" + "\n".join(f"- {k}: {v}" for k, v in timing.items()) + f"\n- recognizer RTF: {rtf:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"recall": report, "timing": timing, "rtf": rtf}, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synth")
    p.add_argument("out_dir")
    p.add_argument("--bin", default="itts")
    p.set_defaults(func=cmd_synth)
    p = sub.add_parser("whisper")
    p.add_argument("manifest")
    p.add_argument("out")
    p.set_defaults(func=cmd_whisper)
    p = sub.add_parser("itts")
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--bin", default="itts")
    p.add_argument("--socket")
    p.add_argument("--model", help="recognizer for non-English samples (default: sense-voice)")
    p.set_defaults(func=cmd_itts)
    p = sub.add_parser("report")
    p.add_argument("manifest")
    p.add_argument("--verify", required=True)
    p.add_argument("--alt", help="a second `itts` run (e.g. another --model)")
    p.add_argument("--alt-label", default="verify (alternative model)")
    p.add_argument("--whisper")
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.8.0", "torchaudio==2.8.0", "pypinyin==0.55.0", "numpy"]
# ///
"""Evaluation harness for `itts align` (docs/research/itts-align-evaluation-research.md).

Subcommands (all read a JSONL manifest of {id, set, audio, text, lang, native?, mismatch?}):

  mms     MANIFEST OUT   reference alignment with torchaudio MMS_FA (evaluation only: CC-BY-NC)
  itts    MANIFEST OUT   run `itts align --output json --phones` for every sample
  report  MANIFEST --itts OUT [--mms OUT]   metrics per sample set (JSON)

`native` points at a JSON file with an ElevenLabs-shaped `alignment` (itts synth output or a
reshape TTS cache entry). `mismatch: true` marks hard cases whose audio does not say the text.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PUNCT = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()《》—…\-·「」『』]")
HANZI = re.compile(r"[一-鿿]")
WORD = re.compile(r"[A-Za-z][A-Za-z']*")
OVERRIDE = re.compile(r"<([^<>|]+)\|[^<>]+>")


def load_manifest(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def units_of(text: str, lang: str) -> list[tuple[int, str, str]] | None:
    """(char index, unit text, MMS romanization) per pronounced unit, or None if MMS cannot
    transcribe the text (digits). Overrides `<word|PRON>` count as their word."""
    from pypinyin import lazy_pinyin

    chars = list(text)
    spans: list[tuple[int, int]] = []  # override word spans (char indices) to keep
    skip: set[int] = set()
    for m in OVERRIDE.finditer(text):
        start = len(text[: m.start()])
        word = len(text[: m.start(1)])
        end = len(text[: m.end()])
        spans.append((word, word + len(m.group(1))))
        skip.update(i for i in range(start, end) if not word <= i < word + len(m.group(1)))
    if any(ch.isdigit() for i, ch in enumerate(chars) if i not in skip):
        return None
    out: list[tuple[int, str, str]] = []
    i = 0
    while i < len(chars):
        if i in skip:
            i += 1
            continue
        if HANZI.match(chars[i]):
            j = i
            while j < len(chars) and HANZI.match(chars[j]) and j not in skip:
                j += 1
            for k, syl in enumerate(lazy_pinyin("".join(chars[i:j]))):
                roman = re.sub(r"[^a-z]", "", syl.replace("ü", "u").lower())
                out.append((i + k, chars[i + k], roman))
            i = j
        elif re.match(r"[A-Za-z]", chars[i]):
            j = i
            while j < len(chars) and re.match(r"[A-Za-z']", chars[j]) and j not in skip:
                j += 1
            word = "".join(chars[i:j])
            out.append((i, word, word.lower()))
            i = j
        else:
            i += 1
    return [u for u in out if u[2]]


def load_16k(path: str):
    import numpy as np
    import torch

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())[None]


def cmd_mms(args: argparse.Namespace) -> None:
    import torch
    from torchaudio.pipelines import MMS_FA as bundle

    model, tokenizer, aligner = bundle.get_model(), bundle.get_tokenizer(), bundle.get_aligner()
    out = {}
    for sample in load_manifest(args.manifest):
        units = units_of(sample["text"], sample["lang"])
        if not units:
            out[sample["id"]] = None
            continue
        wav = load_16k(sample["audio"])
        with torch.inference_mode():
            emission, _ = model(wav)
            spans = aligner(emission[0], tokenizer([u[2] for u in units]))
        ratio = wav.shape[1] / emission.shape[1] / bundle.sample_rate
        out[sample["id"]] = [
            {
                "i": i,
                "u": u,
                "roman": roman,
                "s": sp[0].start * ratio,
                "e": sp[-1].end * ratio,
                "letters": [[roman[k], t.start * ratio, t.end * ratio] for k, t in enumerate(sp)],
            }
            for (i, u, roman), sp in zip(units, spans)
        ]
        print(f"mms {sample['id']}: {len(units)} units", file=sys.stderr)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))


def cmd_itts(args: argparse.Namespace) -> None:
    out = {}
    for sample in load_manifest(args.manifest):
        cmd = [args.bin]
        if args.socket:
            cmd += ["align", "--socket", args.socket]
        else:
            cmd += ["align"]
        cmd += ["--audio", sample["audio"], "--lang", sample["lang"], "--stdin", "--phones"]
        cmd += ["--output", "json"]
        if args.voice:
            cmd += ["--voice", args.voice]
        started = time.perf_counter()
        done = subprocess.run(cmd, input=sample["text"].encode(), capture_output=True)
        wall = time.perf_counter() - started
        doc = json.loads(done.stdout or b"{}")
        doc["wallSecs"] = wall
        doc["exitCode"] = done.returncode
        out[sample["id"]] = doc
        q = doc.get("quality", {})
        print(f"itts {sample['id']}: exit {done.returncode} {q}", file=sys.stderr)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))


def spoken(ch: str) -> bool:
    return not PUNCT.match(ch)


def stats(errors: list[float]) -> dict:
    import numpy as np

    if not errors:
        return {}
    a = np.abs(np.array(errors)) * 1000
    return {
        "n": len(errors),
        "mae_ms": round(float(a.mean()), 1),
        "median_ms": round(float(np.median(a)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "le100_pct": round(float((a <= 100).mean() * 100), 1),
        "bias_ms": round(float(np.mean(errors) * 1000), 1),
    }


def word_starts(text: str) -> list[int]:
    return [len(text[: m.start()]) for m in WORD.finditer(text)]


def onset_errors(ours: dict, ref: dict, text: str, lang: str) -> list[float]:
    """Onset differences ours - ref over pronounced characters (zh) or word starts (en)."""
    if ref is None or ref.get("text") != ours.get("text"):
        return []
    idx = word_starts(text) if lang == "en" else [i for i, c in enumerate(text) if spoken(c)]
    return [ours["start"][i] - ref["start"][i] for i in idx if i < len(ref["start"])]


def last_index(text: str) -> int | None:
    found = [i for i, c in enumerate(text) if spoken(c)]
    return found[-1] if found else None


def phone_boundaries(result: dict, mms_units: list[dict]) -> list[float]:
    """Initial/final boundary differences ours - MMS for characters where both agree on the
    initial (pypinyin strict initials)."""
    from pypinyin import Style, pinyin

    ours = {}
    for p in result.get("phones") or []:
        if p["k"] == "ini":
            ours[p["i"]] = (p["p"], p["end"])
    errors = []
    for unit in mms_units:
        if not HANZI.match(unit["u"]) or unit["i"] not in ours:
            continue
        initial = pinyin(unit["u"], style=Style.INITIALS, strict=True)[0][0]
        name, end = ours[unit["i"]]
        if not initial or initial != name or len(unit["letters"]) <= len(initial):
            continue
        errors.append(end - unit["letters"][len(initial)][1])
    return errors


def cmd_report(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    itts = json.loads(Path(args.itts).read_text())
    mms = json.loads(Path(args.mms).read_text()) if args.mms else {}
    by_set: dict[str, list[dict]] = {}
    for sample in manifest:
        by_set.setdefault(sample["set"], []).append(sample)
    report = {}
    for name, samples in by_set.items():
        r: dict = {"samples": len(samples)}
        vs_mms, vs_native, native_vs_mms, end_mms, end_native, phones = [], [], [], [], [], []
        conf, cov, weak, flagged, weak_flagged, secs, failures = [], [], [], [], [], [], 0
        for s in samples:
            res = itts.get(s["id"]) or {}
            if res.get("status") != "ok":
                failures += 1
                continue
            al = res["alignment"]
            text, lang = s["text"], s["lang"]
            q = res["quality"]
            conf.append(q["confidence"])
            cov.append(q["coverage"])
            weak.append(len(q["weak"]))
            flagged.append(q["confidence"] < args.min_confidence or q["coverage"] < args.min_coverage)
            weak_flagged.append(bool(q["weak"]))
            secs.append(res["alignSecs"])
            units = mms.get(s["id"])
            native = None
            if s.get("native"):
                native = json.loads(Path(s["native"]).read_text()).get("alignment")
            if s.get("mismatch"):
                continue
            last = last_index(al["text"])
            if units:
                pick = (lambda u: WORD.fullmatch(u["u"])) if lang == "en" else (lambda u: True)
                for u in units:
                    if pick(u):
                        vs_mms.append(al["start"][u["i"]] - u["s"])
                if last is not None and units[-1]["i"] <= last:
                    end_mms.append(al["end"][last] - units[-1]["e"])
                phones += phone_boundaries(res, units)
                if native and native.get("text") == al["text"]:
                    native_vs_mms += [native["start"][u["i"]] - u["s"] for u in units if pick(u)]
            if native:
                vs_native += onset_errors(al, native, al["text"], lang)
                if last is not None and native.get("text") == al["text"]:
                    end_native.append(al["end"][last] - native["end"][last])
        warm = sorted(secs)[: max(len(secs) - 1, 1)] if secs else []
        r.update(
            failures=failures,
            onset_vs_mms=stats(vs_mms),
            onset_vs_native=stats(vs_native),
            native_vs_mms=stats(native_vs_mms),
            last_end_vs_mms=stats(end_mms),
            last_end_vs_native=stats(end_native),
            phone_boundary_vs_mms=stats(phones),
            confidence_min=round(min(conf), 3) if conf else None,
            confidence_mean=round(sum(conf) / len(conf), 3) if conf else None,
            coverage_min=round(min(cov), 3) if cov else None,
            weak_total=sum(weak),
            flagged=sum(flagged),
            weak_nonempty=sum(weak_flagged),
            align_secs_mean=round(sum(warm) / len(warm), 3) if warm else None,
            align_secs_max=round(max(warm), 3) if warm else None,
        )
        report[name] = r
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mms")
    p.add_argument("manifest")
    p.add_argument("out")
    p.set_defaults(func=cmd_mms)
    p = sub.add_parser("itts")
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--bin", default="itts")
    p.add_argument("--socket")
    p.add_argument("--voice")
    p.set_defaults(func=cmd_itts)
    p = sub.add_parser("report")
    p.add_argument("manifest")
    p.add_argument("--itts", required=True)
    p.add_argument("--mms")
    p.add_argument("--out")
    p.add_argument("--min-confidence", type=float, default=0.95)
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

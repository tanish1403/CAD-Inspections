#!/usr/bin/env python3
"""Compare configurations (classical / DINOv2 / + SigLIP / + SmolVLM / + Qwen2-VL / + InternVL3 ...) against ground truth.

  python scripts/benchmark.py --input_dir data/input --truth examples/ground_truth.json
  python scripts/benchmark.py --input_dir data/input --truth examples/ground_truth.json --configs classical dino dino+internvl3-2b
  python scripts/benchmark.py --list

Writes <out>/benchmark.csv and benchmark.md.  A part counts as correct when PRESENT/ABSENT matches the truth; UNCERTAIN is
counted separately (not right, not wrong).  Use the table to choose the model for your presentation.
"""
import argparse, csv, json, os, sys, time, traceback
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

G = ["--golden", "auto"]
V = lambda *v: ["--vlm", *v, "--use_names"]
ALONE = ["--vlm_weight", "1.0"]                 # decision made by the verifier alone (measures the VLM itself)
CONFIGS = {
    "classical":           ["--method", "classical", *G],
    "dino":                ["--gray", *G],
    "dino+sam":            ["--gray", "--sam", *G],
    "dino+siglip":         ["--gray", *V("siglip"), *G],
    "dino+smolvlm":        ["--gray", *V("smolvlm"), *G],
    "dino+qwen2-vl":       ["--gray", *V("qwen2-vl"), *G],
    "dino+internvl3-2b":   ["--gray", *V("internvl3-2b"), *G],
    "qwen2-vl alone":      ["--method", "classical", *V("qwen2-vl"), *ALONE, *G],
    "internvl3-2b alone":  ["--method", "classical", *V("internvl3-2b"), *ALONE, *G],
    "smolvlm alone":       ["--method", "classical", *V("smolvlm"), *ALONE, *G],
    "qwen2-vl-7b-4bit":    ["--gray", "--vlm", "vlm:Qwen/Qwen2-VL-7B-Instruct", "--vlm_4bit", "--use_names", *G],
}


def truth_for(name, truth):
    keys = [k for k in truth if not k.startswith("_") and (name == k or name.endswith("/" + k))]
    return truth[max(keys, key=len)] if keys else None


def score(out_dir, truth):
    tp = tn = fp = fn = unc = n = 0
    for rp in sorted(__import__("glob").glob(os.path.join(out_dir, "**", "report.json"), recursive=True)):
        rel = os.path.relpath(os.path.dirname(rp), out_dir).replace("__", "/")
        t = truth_for(rel, truth)
        if t is None:
            continue
        for part, r in json.load(open(rp)).items():
            if r["status"] == "UNVERIFIABLE":
                continue
            want = 0 if part in t["absent"] else 1
            n += 1
            if r["status"] == "UNCERTAIN":
                unc += 1
            elif r["status"] == "PRESENT":
                tp, fp = tp + (want == 1), fp + (want == 0)
            else:
                tn, fn = tn + (want == 0), fn + (want == 1)
    done = tp + tn + fp + fn
    return dict(parts=n, correct=tp + tn, wrong=fp + fn, uncertain=unc, missed_absent=fp, false_alarms=fn,
                accuracy=round((tp + tn) / n, 3) if n else None,
                absent_recall=round(tn / (tn + fp), 3) if (tn + fp) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/input")
    ap.add_argument("--truth", default="examples/ground_truth.json")
    ap.add_argument("--out", default="data/benchmark")
    ap.add_argument("--configs", nargs="+", default=["classical", "dino", "dino+siglip", "dino+smolvlm", "dino+qwen2-vl", "dino+internvl3-2b"])
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra run.py args for every config, e.g. --extra --backbone dummy")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(f"{k:22s} {' '.join(v)}" for k, v in CONFIGS.items()))
        return
    import run as cli
    truth = json.load(open(a.truth))
    rows = []
    for name in a.configs:
        if name not in CONFIGS:
            sys.exit(f"unknown config '{name}'. --list shows them")
        d = os.path.join(a.out, name.replace(" ", "_").replace("+", "_"))
        t0 = time.time()
        try:
            cli.main(["check", "--input_dir", a.input_dir, "--output_dir", d, *CONFIGS[name], *a.extra])
            row = dict(config=name, **score(d, truth), seconds=round(time.time() - t0, 1), error="")
        except BaseException as e:                         # noqa: BLE001  (one failing model must not stop the table)
            traceback.print_exc()
            row = dict(config=name, seconds=round(time.time() - t0, 1), error=f"{type(e).__name__}: {str(e)[:120]}")
        rows.append(row)
    cols = ["config", "accuracy", "absent_recall", "correct", "wrong", "uncertain", "missed_absent", "false_alarms", "parts", "seconds", "error"]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
         ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    open(os.path.join(a.out, "benchmark.md"), "w").write("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()

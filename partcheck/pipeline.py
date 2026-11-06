"""End-to-end: samples in input_dir -> overlays + reports in output_dir/<case>/<real image>/."""
import csv, json, os, pickle
import cv2
import numpy as np
from .data import discover, load_golden, load_sample, part_region
from .align import align
from .features import extract, fuse, sig, vec
from .viz import render


def samples_from(args):
    s = discover(args.input_dir, getattr(args, "real_glob", None), getattr(args, "cad_glob", None),
                 getattr(args, "mask_glob", None), getattr(args, "json_glob", None))
    if not s:
        raise SystemExit(f"No samples found in {args.input_dir}. Expected a folder with a CAD image, a mask, a json "
                         f"({{'mask.bmp': [{{'Name','StencilValue'}}]}}) and real photos - see README 'Input'.")
    return s


def prepare(sample, args, with_original=False):
    real, cad, mask, parts = load_sample(sample, args.max_side)
    real_al = align(real, cad, mask, args.align)
    return (cad, real_al, mask, parts, real) if with_original else (cad, real_al, mask, parts)


# ----------------------------------------------------------------- scoring
def base_score(f, a, head, use_ai, use_sam):
    """Score from the core cues (training-free fusion, or the trained head). -> (p, threshold)"""
    if head is not None:
        return float(head.predict_proba([vec(f)])[0, 1]), 0.5
    return float(fuse(f, use_ai, use_sam, a.m0, a.tau, a.ai_weight)), a.thr


def blend(p, f, a):
    """Mix in optional cues: verifier models (vlm_*), few-shot bank (ref_margin), golden reference (gold)."""
    ex = [v for k, v in f.items() if k.startswith("vlm_") and v is not None]
    if ex:
        p = (1 - a.vlm_weight) * p + a.vlm_weight * float(np.mean(ex))
    if f.get("ref_margin") is not None:
        p = (1 - a.ref_weight) * p + a.ref_weight * sig(f["ref_margin"] / 0.05)
    if f.get("gold") is not None and a.golden_weight > 0:
        p = (1 - a.golden_weight) * p + a.golden_weight * sig((f["gold"] - 0.5) / 0.12)
    return p


def decide(feats, a, head=None, use_ai=True, use_sam=False):
    res = {}
    for n, f in feats.items():
        if f.get("skip"):
            why = {"not_visible": "not visible in the CAD view (fully hidden) - cannot be checked",
                   "thin_sliver": "visible only as a hairline in the CAD view (hidden behind other parts) - cannot be checked",
                   "too_small": f"only {f['area']} px visible - too small to judge (raise --max_side / lower --min_area_px)"}
            res[n] = dict(status="UNVERIFIABLE", present=None, score=None, area_px=f["area"], note=why[f["skip"]])
            continue
        p0, t = base_score(f, a, head, use_ai, use_sam)
        p = blend(p0, f, a)
        status = "UNCERTAIN" if abs(p - t) < a.uncertain_band else ("PRESENT" if p >= t else "ABSENT")
        r = dict(status=status, present={"PRESENT": True, "ABSENT": False}.get(status), score=round(p, 3),
                 area_px=f["area"])
        if f.get("gold") is not None and f["gold"] < a.altered_thr and status != "ABSENT":
            r["altered"] = True
            r["note"] = "structure present but pixels differ strongly from the golden reference (tampered / damaged?)"
        r.update({k: (None if v is None else round(float(v), 3)) for k, v in f.items() if k not in ("area", "skip")})
        res[n] = r
    return res


# ----------------------------------------------------------------- output
def out_path(out_dir, s):
    return os.path.join(out_dir, s.case.replace("/", "__"), s.stem)


def save(d, cad, real_al, parts, mask, res, original=None, title=""):
    os.makedirs(d, exist_ok=True)
    card, detail, panels = render(cad, real_al, parts, mask, res, original, title)
    cv2.imwrite(os.path.join(d, "result.png"), card)               # original | result | CAD + status summary
    cv2.imwrite(os.path.join(d, "detail_2x2.png"), detail)         # present / absent on CAD and on the photo
    for k, im in panels.items():
        cv2.imwrite(os.path.join(d, f"{k}.png"), im)
    cv2.imwrite(os.path.join(d, "aligned_real.jpg"), real_al, [cv2.IMWRITE_JPEG_QUALITY, 92])
    json.dump(res, open(os.path.join(d, "report.json"), "w"), indent=2)
    with open(os.path.join(d, "report.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "status", "score", "area_px", "altered", "note"])
        for n, r in res.items():
            w.writerow([n, r["status"], r["score"], r["area_px"], r.get("altered", ""), r.get("note", "")])


def pick_golden(args, samples):
    """--golden PATH, or 'auto': per case, the first real image whose name has no 'test' in it."""
    if not args.golden:
        return {}
    if args.golden != "auto":
        return {s.name: args.golden for s in samples}
    out = {}
    everything = discover(args.input_dir, None, getattr(args, "cad_glob", None), getattr(args, "mask_glob", None),
                          getattr(args, "json_glob", None))           # candidates ignore --real_glob
    for s in samples:
        cands = [x for x in everything if x.case == s.case and "test" not in x.stem.lower()]
        if cands and cands[0].real != s.real:          # a sample is never its own golden
            out[s.name] = cands[0].real
    return out


class Analysis:
    """Everything produced for one photo (images are at the working resolution, BGR)."""
    def __init__(self, sample, res, feats, cad, real_al, original, mask, parts):
        self.sample, self.res, self.feats = sample, res, feats
        self.cad, self.real_al, self.original, self.mask, self.parts = cad, real_al, original, mask, parts

    def render(self):
        return render(self.cad, self.real_al, self.parts, self.mask, self.res, self.original, self.sample.name)


def analyze(s, args, bb=None, sam=None, verifiers=(), bank=None, head=None, golden_path=None, save_to=None):
    """Run the whole check on ONE photo (`s` = data.Sample). Used by the CLI, the Streamlit app, notebooks and benchmark."""
    use_ai, use_sam = bb is not None, sam is not None
    cad, real_al, mask, parts, original = prepare(s, args, with_original=True)
    only = set(args.parts.split(",")) if getattr(args, "parts", None) else None
    if only:
        parts = {n: v for n, v in parts.items() if n in only}
    gold = load_golden(golden_path, (cad.shape[1], cad.shape[0])) if golden_path and golden_path != s.real else None
    feats = extract(parts, mask, cad, real_al, bb, sam, args.gray, bank, gold, args.min_area_px, args.min_thickness_px)
    if verifiers:
        for v in verifiers:
            v.prepare(cad, real_al)
        for n, f in feats.items():
            if f.get("skip"):
                continue
            p0, t0 = base_score(f, args, head, use_ai, use_sam)
            if args.vlm_uncertain_only and abs(blend(p0, f, args) - t0) > args.vlm_band:
                continue                              # confident already - skip the expensive models
            reg = part_region(mask, parts[n])
            for v in verifiers:
                f[f"vlm_{v.name}"] = float(v.score(n, reg))
    res = decide(feats, args, head, use_ai, use_sam)
    if save_to:
        save(save_to, cad, real_al, parts, mask, res, original, s.name)
    return Analysis(s, res, feats, cad, real_al, original, mask, parts)


def run(args, bb, sam, verifiers=(), bank=None):
    head = pickle.load(open(args.head, "rb")) if args.head else None
    samples = samples_from(args)
    golden_for = pick_golden(args, samples)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for s in samples:
        a = analyze(s, args, bb, sam, verifiers, bank, head, golden_for.get(s.name), out_path(args.output_dir, s))
        res = a.res
        by = lambda st: [n for n, r in res.items() if r["status"] == st]
        alt = [n for n, r in res.items() if r.get("altered")]
        print(f"[{s.name}] PRESENT {len(by('PRESENT'))} | ABSENT {by('ABSENT')} | UNCERTAIN {by('UNCERTAIN')} | "
              f"altered {alt} | unverifiable {len(by('UNVERIFIABLE'))}")
        rows += [(s.case, s.stem, n, r["status"], r["score"], r["area_px"], r.get("altered", "")) for n, r in res.items()]
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "image", "part", "status", "score", "area_px", "altered"])
        w.writerows(rows)
    print(f"Results written to {args.output_dir}")

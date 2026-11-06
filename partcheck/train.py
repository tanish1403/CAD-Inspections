"""Learning modes on top of the frozen / unfrozen foundation model.

  train     small classifier head on match scores        (few-shot, seconds on CPU)
  bank      per-part real-image prototype bank           (few-shot, no gradient training)
  finetune  unfreeze the last DINOv2 blocks              (needs a GPU; Colab/Kaggle is fine)
All three create their data from your own photos with synthetic variants (lighting/shift + 'erase a part').
"""
import fnmatch, json, os, pickle
import cv2
import numpy as np
from .data import part_region
from .features import extract, fuse, part_cells, pooled, vec
from .pipeline import prepare, samples_from


# ---------------------------------------------------------------- augmentation
def jitter(img, rng):
    """Lighting change + noise + small shift (simulates domain shift and alignment error)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] *= rng.uniform(0.75, 1.25)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = np.clip(img + rng.normal(0, rng.uniform(0, 6), img.shape), 0, 255).astype(np.uint8)
    H, W = img.shape[:2]
    M = np.float32([[1, 0, rng.uniform(-.015, .015) * W], [0, 1, rng.uniform(-.015, .015) * H]])
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)


def erase(img, region, rng=None):
    """Synthetic 'part missing': fill the part's region with what surrounds it (median colour of a ring + matching noise)."""
    rng = rng or np.random.default_rng(0)
    r8 = region.astype(np.uint8)
    k = max(3, int(0.15 * np.sqrt(region.sum())) | 1)
    cover = cv2.dilate(r8, np.ones((k, k), np.uint8)).astype(bool)
    ring = cv2.dilate(r8, np.ones((3 * k, 3 * k), np.uint8)).astype(bool) & ~cover
    out = img.copy()
    if not ring.any():
        return out
    med, sd = np.median(img[ring], 0), img[ring].std(0)
    fill = np.clip(med + rng.normal(0, 0.5 * sd, (int(cover.sum()), 3)), 0, 255)
    out[cover] = fill.astype(np.uint8)
    return out


# ---------------------------------------------------------------- shared data
def match_key(name, mapping):
    """Value of the longest key that equals `name` or is a '/'-suffix of it ('TestCase2/BG_00' beats 'BG_00')."""
    keys = [k for k in mapping if not k.startswith("_") and (name == k or name.endswith("/" + k))]
    return mapping[max(keys, key=len)] if keys else None


def _labels_for(labels, smp):
    return match_key(smp.name, labels) or labels.get(smp.case) or {}


def is_reference(args, smp):
    """--ref_glob marks photos that are COMPLETE (every verifiable part present), e.g. 'input/BG_00' or 'BG_00'."""
    pat = getattr(args, "ref_glob", None)
    return bool(pat) and (fnmatch.fnmatch(smp.name, pat) or fnmatch.fnmatch(smp.name, "*/" + pat))


def collect_samples(args, bb, sam):
    """Load all samples + labels of the ORIGINAL photos. Priority per part:
       1) --labels json (keys 'case/image', 'image' or 'case')   2) photo matched by --ref_glob -> present
       3) confident classical prediction (|score-0.5| > 0.2)     4) otherwise unlabeled (ignored).
    Edited / test photos are therefore never blindly assumed complete."""
    labels = json.load(open(args.labels)) if getattr(args, "labels", None) else {}
    out = []
    for smp in samples_from(args):
        cad, real_al, mask, parts = prepare(smp, args)
        base = extract(parts, mask, cad, real_al, bb, sam, args.gray, min_area=args.min_area_px,
                       min_thick=args.min_thickness_px)
        lab, ref = _labels_for(labels, smp), is_reference(args, smp)
        orig = {}
        for n, f in base.items():
            if f.get("skip"):
                orig[n] = None
            elif n in lab:
                orig[n] = int(lab[n])
            elif ref:
                orig[n] = 1
            else:
                p = fuse(f, False)                      # pseudo-labels from the validated classical cues only
                orig[n] = int(p >= 0.5) if abs(p - 0.5) > 0.2 else None
        regions = {n: part_region(mask, v) for n, v in parts.items()}
        out.append(dict(pre=smp.name, cad=cad, real_al=real_al, mask=mask, parts=parts, orig=orig, regions=regions,
                        cells={n: part_cells(r, bb.g) for n, r in regions.items() if orig.get(n) is not None}))
        print(f"[{smp.name}] {'REFERENCE ' if ref else ''}labels: present={sum(v == 1 for v in orig.values())} "
              f"absent={sum(v == 0 for v in orig.values())} ignored={sum(v is None for v in orig.values())}")
    if not any(v == 0 for s in out for v in s["orig"].values()):
        print("[warn] no 'absent' examples found - synthetic erasures will provide the negatives; "
              "add --labels for real ones")
    return out


def variant(s, rng, p_erase=0.35):
    """One synthetic variant of a sample: (image, set of erased part names)."""
    img = jitter(s["real_al"], rng)
    erased = {n for n in s["parts"] if s["orig"].get(n) == 1 and rng.random() < p_erase}
    for n in erased:
        img = erase(img, s["regions"][n], rng)
    return img, erased


# ---------------------------------------------------------------- 1) head
def train_head(args, bb, sam):
    from sklearn.ensemble import GradientBoostingClassifier
    rng = np.random.default_rng(0)
    X, y, grp = [], [], []
    for s in collect_samples(args, bb, sam):
        for _ in range(args.iters):
            img, erased = variant(s, rng)
            feats = extract(s["parts"], s["mask"], s["cad"], img, bb, sam, args.gray, min_area=args.min_area_px,
                       min_thick=args.min_thickness_px)
            for n, f in feats.items():
                lab = 0 if n in erased else s["orig"].get(n)
                if not f.get("skip") and lab is not None:
                    X.append(vec(f)); y.append(lab); grp.append(s["pre"])
    X, y = np.array(X), np.array(y)
    if len(set(y)) < 2:
        raise SystemExit("Training data has a single class - provide --labels or more varied samples.")
    print(f"training rows: {len(y)}  present={int(y.sum())} absent={int((1 - y).sum())}")
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=0)
    if len(set(grp)) > 1:
        from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
        acc = cross_val_score(clf, X, y, groups=grp, cv=LeaveOneGroupOut(), error_score=np.nan)
        print("leave-one-sample-out accuracy:", np.nanmean(acc).round(3), f"({int(np.isfinite(acc).sum())} valid folds)")
    clf.fit(X, y)
    os.makedirs(os.path.dirname(args.out_head) or ".", exist_ok=True)
    pickle.dump(clf, open(args.out_head, "wb"))
    print("saved head ->", args.out_head)


# ---------------------------------------------------------------- 2) few-shot bank
def build_bank(args, bb, sam=None):
    """Per-part real-image descriptors: positives (part present) and negatives (part erased / absent).
    Mark complete photos with --ref_glob (e.g. 'BG_00'); do NOT feed edited photos as references."""
    rng = np.random.default_rng(0)
    bank = {}
    for s in collect_samples(args, bb, None):
        for it in range(args.iters + 1):
            img, erased = (s["real_al"], set()) if it == 0 else variant(s, rng)
            fr = bb(img, args.gray)
            for n, cells in s["cells"].items():
                lab = 0 if n in erased else s["orig"].get(n)
                if lab is None:
                    continue
                bank.setdefault(n, {"pos": [], "neg": []})["pos" if lab else "neg"].append(pooled(fr, cells))
    bank = {n: {k: np.array(v) for k, v in d.items()} for n, d in bank.items()}
    os.makedirs(os.path.dirname(args.out_bank) or ".", exist_ok=True)
    pickle.dump(dict(bank=bank, meta=dict(dino=args.dino, size=args.size, gray=args.gray)), open(args.out_bank, "wb"))
    print(f"bank saved -> {args.out_bank}  parts={len(bank)}  "
          f"pos={sum(len(d['pos']) for d in bank.values())} neg={sum(len(d['neg']) for d in bank.values())}")


def load_bank(path, args=None):
    d = pickle.load(open(path, "rb"))
    m = d.get("meta", {})
    if args is not None and (m.get("size") != args.size or bool(m.get("gray")) != bool(args.gray)
                             or m.get("dino") != args.dino):
        print(f"[warn] bank was built with {m} but you run dino={args.dino} size={args.size} gray={args.gray}; "
              "descriptors are only comparable with identical settings.")
    return d["bank"]


# ---------------------------------------------------------------- 3) fine-tuning
def finetune(args, bb, sam=None):
    """Fine-tune the last `--unfreeze` DINOv2 blocks so that a part's CAD descriptor and its real-image
    descriptor agree when the part is there and disagree when it is absent (or when another part is there).
    logit = scale * cos(desc_cad, desc_real) + bias,  BCE loss.  Saves a drop-in backbone folder."""
    t = bb.t
    F = t.nn.functional
    rng = np.random.default_rng(0)
    S = collect_samples(args, bb, sam)
    model = bb.m
    for p in model.parameters():
        p.requires_grad = False
    params = []
    for mod in list(model.encoder.layer[-args.unfreeze:]) + ([model.layernorm] if hasattr(model, "layernorm") else []):
        for p in mod.parameters():
            p.requires_grad = True
            params.append(p)
    scale = t.nn.Parameter(t.tensor(10.0, device=bb.dev))
    bias = t.nn.Parameter(t.tensor(-5.0, device=bb.dev))
    opt = t.optim.AdamW([{"params": params, "lr": args.lr}, {"params": [scale, bias], "lr": 1e-2}], weight_decay=1e-4)
    scaler = t.cuda.amp.GradScaler(enabled=bb.fp16)
    print(f"fine-tuning {sum(p.numel() for p in params) / 1e6:.1f}M params | steps={args.steps} lr={args.lr}")

    def desc(feat, cells):
        idx = t.from_numpy(cells).long().to(bb.dev)
        return F.normalize(feat[idx[:, 0], idx[:, 1]].mean(0), dim=-1)

    hist = []
    for step in range(1, args.steps + 1):
        s = S[rng.integers(len(S))]
        names = [n for n in s["cells"] if s["orig"].get(n) is not None]
        if not names:
            continue
        img, erased = variant(s, rng)
        fc = bb.tensor_features(s["cad"], args.gray)
        fr = bb.tensor_features(img, args.gray)
        dc = t.stack([desc(fc, s["cells"][n]) for n in names])
        dr = t.stack([desc(fr, s["cells"][n]) for n in names])
        y = t.tensor([0.0 if n in erased else float(s["orig"][n]) for n in names], device=bb.dev)
        logit = scale * (dc * dr).sum(-1) + bias
        pres = [i for i in range(len(names)) if y[i] > 0]
        if len(pres) > 1:                      # hard negatives: CAD part i vs the real patch of ANOTHER part
            shift = np.roll(np.array(pres), 1)
            neg = scale * (dc[pres] * dr[shift]).sum(-1) + bias
            logit = t.cat([logit, neg]); y = t.cat([y, t.zeros(len(pres), device=bb.dev)])
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        hist.append((float(loss.detach()), float(((logit.detach() > 0).float() == y).float().mean())))
        if step % 25 == 0 or step == args.steps:
            h = np.mean(hist[-25:], 0)
            print(f"step {step:4d}/{args.steps}  loss {h[0]:.3f}  acc {h[1]:.3f}")
    model.eval()
    os.makedirs(args.out_model, exist_ok=True)
    model.save_pretrained(args.out_model)
    json.dump(dict(base=args.dino, size=args.size, gray=args.gray, scale=float(scale), bias=float(bias),
                   unfreeze=args.unfreeze, steps=args.steps), open(os.path.join(args.out_model, "finetune_meta.json"), "w"))
    print(f"saved fine-tuned backbone -> {args.out_model}\n"
          f"use it with:  python run.py check --dino {args.out_model} --size {args.size}"
          f"{' --gray' if args.gray else ''}   (then optionally `train` a head with the same flags)")

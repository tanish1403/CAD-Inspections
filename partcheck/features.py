"""Per-part evidence.

Classical cues (calibrated on real photos with cluttered backgrounds - no plain-background assumption):
  edge      chamfer match of the part's CAD outline against edges in the photo
  contrast  structure inside the part vs the ring around it, relative to what the CAD shows there
            (a bolt head has edges, the bare plate behind it does not)
Foundation-model cues: DINOv2 patch matching (tight/local/bg/margin), optional SAM overlap (sam_iou).
Optional cues: few-shot bank (ref_*), golden reference photo (gold).
"""
import cv2
import numpy as np
from .align import gradient_map
from .data import part_region

FEATS = ["tight", "local", "bg", "margin", "contrast", "edge", "shape", "sam_iou", "log_area"]


def sig(x):
    return float(1 / (1 + np.exp(-x)))


def odd(n):
    n = int(round(n))
    return n if n % 2 else n + 1


def _crop(reg, pad):
    ys, xs = np.where(reg)
    H, W = reg.shape
    return max(ys.min() - pad, 0), min(ys.max() + pad + 1, H), max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)


# ------------------------------------------------------------ classical cues
def ring_ratio(g, reg, sf):
    """mean gradient inside the part / mean gradient in a ring around it (None if no ring)."""
    k_out, k_gap, k_in = odd(max(9, 41 * sf)), odd(max(3, 9 * sf)), odd(max(3, 5 * sf))
    y0, y1, x0, x1 = _crop(reg, k_out)
    r = reg[y0:y1, x0:x1].astype(np.uint8)
    inner = cv2.erode(r, np.ones((k_in, k_in), np.uint8)).astype(bool)
    if inner.sum() < 10:
        inner = r.astype(bool)
    ring = cv2.dilate(r, np.ones((k_out, k_out), np.uint8)).astype(bool) & \
        ~cv2.dilate(r, np.ones((k_gap, k_gap), np.uint8)).astype(bool)
    if not ring.any():
        return None
    gg = g[y0:y1, x0:x1]
    return float(gg[inner].mean() / (gg[ring].mean() + 1e-3))


def edge_score(region, edge_dt, tau_frac=0.06):
    """Chamfer: how close are the part's CAD contour pixels to real edges? 1 = perfect."""
    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    pts = np.vstack([c.reshape(-1, 2) for c in cnts])
    d = edge_dt[pts[:, 1], pts[:, 0]]
    return float(np.mean(np.exp(-d / max(3.0, tau_frac * np.sqrt(region.sum())))))


def _hog(gray, x0, y0, x1, y1, nb=8, cells=3):
    c = cv2.GaussianBlur(gray[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(c, cv2.CV_32F, 1, 0), cv2.Sobel(c, cv2.CV_32F, 0, 1)
    mag, ang = np.hypot(gx, gy), (np.arctan2(gy, gx) % np.pi) / np.pi * nb
    b0 = np.floor(ang).astype(int) % nb
    b1, w1 = (b0 + 1) % nb, ang - np.floor(ang)
    h, w = c.shape
    d = []
    for i in range(cells):
        for j in range(cells):
            sl = (slice(i * h // cells, (i + 1) * h // cells), slice(j * w // cells, (j + 1) * w // cells))
            d.append(np.bincount(b0[sl].ravel(), (mag[sl] * (1 - w1[sl])).ravel(), nb) +
                     np.bincount(b1[sl].ravel(), (mag[sl] * w1[sl]).ravel(), nb))
    d = np.array(d).ravel()
    return d / (np.linalg.norm(d) + 1e-6)


def shape_score(gray_cad, gray_real, reg, shift=6):
    """Orientation-histogram (HOG-like) similarity of the CAD crop and the real crop (best small shift).
    Not used by the training-free fusion (not separable enough alone) - a feature for the learned head."""
    y0, y1, x0, x1 = _crop(reg, 0)
    p = int(0.15 * max(x1 - x0, y1 - y0)) + 2
    H, W = reg.shape
    X0, Y0, X1, Y1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, W), min(y1 + p, H)
    a, best = _hog(gray_cad, X0, Y0, X1, Y1), -1.0
    for dy in range(-shift, shift + 1, 3):
        for dx in range(-shift, shift + 1, 3):
            if X0 + dx < 0 or Y0 + dy < 0 or X1 + dx > W or Y1 + dy > H:
                continue
            best = max(best, float(a @ _hog(gray_real, X0 + dx, Y0 + dy, X1 + dx, Y1 + dy)))
    return best


def golden_score(ref_gray, test_gray, reg, shift=3, max_px=20000, rng=np.random.default_rng(0)):
    """Pearson correlation of the part's pixels between a complete 'golden' photo and this photo (best small shift)."""
    ys, xs = np.where(reg)
    if len(ys) > max_px:
        k = rng.choice(len(ys), max_px, replace=False)
        ys, xs = ys[k], xs[k]
    a = ref_gray[ys, xs]
    if a.std() < 1e-3:
        return None
    H, W = ref_gray.shape
    best = -1.0
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            b = test_gray[np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)]
            if b.std() > 1e-3:
                best = max(best, float(np.corrcoef(a, b)[0, 1]))
    return best


# ------------------------------------------------------------ foundation-model cues
def to_grid(region, g):
    return cv2.resize(region.astype(np.float32), (g, g), interpolation=cv2.INTER_AREA)


def part_cells(region, g):
    pg = to_grid(region, g)
    cells = np.argwhere(pg > 0.5)
    return cells if len(cells) else np.array([np.unravel_index(pg.argmax(), pg.shape)])


def pooled(feat_grid, cells):
    d = feat_grid[cells[:, 0], cells[:, 1]].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)


def background_cells(regions, areas, shape, g):
    """Cells where a *missing* part would show something else: outside the object, and the main surface
    (largest part) away from every other part. -> flat bool array of g*g."""
    sil = np.zeros(shape, bool)
    for r in regions.values():
        sil |= r
    surf_name = max(areas, key=areas.get)
    surf = regions[surf_name]
    others = sil & ~surf
    far = cv2.dilate((to_grid(others, g) > 0.01).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) == 0
    bg = (to_grid(sil, g) < 0.05) | ((to_grid(surf, g) > 0.9) & far)
    return bg.ravel()


def patch_match(fc, fr, cells, bg_flat, rad):
    """Each CAD-part patch -> best cosine to a real patch within `rad` cells (tolerates misalignment).
    bg = how well the same patches match what a missing part would leave behind; margin = local - bg."""
    g, _, C = fc.shape
    S = fc[cells[:, 0], cells[:, 1]] @ fr.reshape(-1, C).T
    yy, xx = np.divmod(np.arange(g * g), g)
    d = np.maximum(np.abs(cells[:, 0][:, None] - yy[None]), np.abs(cells[:, 1][:, None] - xx[None]))
    out = {k: float(np.where(d <= r, S, -1).max(1).mean()) for k, r in (("tight", 1), ("local", rad))}
    out["bg"] = float(S[:, bg_flat].max(1).mean()) if bg_flat.any() else 0.0
    out["margin"] = out["local"] - out["bg"]
    return out


def ref_cues(bank, name, desc):
    b = bank.get(name)
    if b is None:
        return dict(ref_pos=None, ref_neg=None, ref_margin=None)
    pos = float((b["pos"] @ desc).max()) if len(b["pos"]) else None
    neg = float((b["neg"] @ desc).max()) if len(b["neg"]) else None
    return dict(ref_pos=pos, ref_neg=neg, ref_margin=(pos - neg) if (pos is not None and neg is not None) else None)


# ------------------------------------------------------------ extraction
def extract(parts, mask, cad, real_al, bb=None, sam=None, gray=False, bank=None, golden=None, min_area=400,
            min_thick=6):
    """-> {part: feature dict}. Parts that cannot be judged get f["skip"] in
    {"not_visible" (0 px), "too_small", "thin_sliver" (visible only as a hairline, e.g. a hidden part's rim)}."""
    H, W = cad.shape[:2]
    sf = max(H, W) / 2048
    regions = {n: part_region(mask, v) for n, v in parts.items()}
    areas = {n: int(r.sum()) for n, r in regions.items()}
    gc, gr = gradient_map(cad), gradient_map(real_al)
    gray_r, gray_c = cv2.cvtColor(real_al, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cad, cv2.COLOR_BGR2GRAY)
    edt = cv2.distanceTransform(255 - cv2.Canny(cv2.GaussianBlur(gray_r, (5, 5), 0), 40, 120), cv2.DIST_L2, 3)
    if golden is not None:
        blur = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1)
        g_ref, g_test = blur(golden), blur(real_al)
    if bb is not None:
        g = bb.g
        fc, fr = bb(cad, gray), bb(real_al, gray)
        bg_flat = background_cells(regions, areas, (H, W), g)
        rad = max(2, int(round(0.08 * g)))
    if sam:
        sam.set_image(real_al)
    feats = {}
    for name, reg in regions.items():
        a = areas[name]
        if a == 0:
            feats[name] = dict(area=0, skip="not_visible")
            continue
        if a < min_area:
            feats[name] = dict(area=a, skip="too_small")
            continue
        y0, y1, x0, x1 = _crop(reg, 1)
        thick = 2 * float(cv2.distanceTransform(reg[y0:y1, x0:x1].astype(np.uint8), cv2.DIST_L2, 3).max())
        if thick < max(3.0, min_thick * sf):
            feats[name] = dict(area=a, skip="thin_sliver", thickness=round(thick, 1))
            continue
        f = dict(area=a, skip=None)
        if bb is not None:
            cells = part_cells(reg, g)
            f.update(patch_match(fc, fr, cells, bg_flat, rad))
            if bank is not None:
                f.update(ref_cues(bank, name, pooled(fr, cells)))
        else:
            f.update(tight=0.0, local=0.0, bg=0.0, margin=0.0)
        rr, rc = ring_ratio(gr, reg, sf), ring_ratio(gc, reg, sf)
        big = a > 0.05 * H * W            # a large surface (the body): "structure vs ring" says nothing about it
        f["contrast"] = None if (rr is None or rc is None or big) else float(min(rr / max(rc, 0.3), 3.0))
        f["shape"] = None if big else shape_score(gray_c, gray_r, reg, max(3, int(round(6 * sf))))
        f["edge"] = edge_score(reg, edt)
        f["sam_iou"] = sam.iou(reg) if sam else 0.0
        f["log_area"] = float(np.log(a / (H * W)))
        if golden is not None:
            f["gold"] = golden_score(g_ref, g_test, reg, shift=max(2, int(round(3 * sf))))
        feats[name] = f
    return feats


def vec(f):
    return [0.0 if f.get(k) is None else f[k] for k in FEATS]


def fuse(f, use_ai=True, use_sam=False, m0=0.10, tau=0.06, ai_w=0.4, edge_mid=0.10, con_mid=0.35):
    """Training-free score in [0,1].  classical = sqrt(edge * contrast) ; AI adds the DINOv2 margin (+SAM)."""
    s_edge = sig((f["edge"] - edge_mid) / 0.04)
    # geometric mean: a part counts as present only if BOTH the outline and the inner structure agree
    s_cls = s_edge if f.get("contrast") is None else float(np.sqrt(s_edge * sig((f["contrast"] - con_mid) / 0.10)))
    if not use_ai:
        return s_cls
    s_ai = sig((f["margin"] - m0) / tau)
    if use_sam:
        s_ai = 0.6 * s_ai + 0.4 * f["sam_iou"]
    return (1 - ai_w) * s_cls + ai_w * s_ai

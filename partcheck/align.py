"""Registration of the real photo to the CAD frame.

none        (default) photos and CAD share the camera pose (your dataset) - only resize.
ecc         light refinement (rotation+translation) maximising gradient correlation inside the CAD silhouette.
silhouette  legacy: plain-background photos only (segments the part from the border colour).
"""
import cv2
import numpy as np
from .data import bbox


def gradient_map(img_bgr, sigma=1.5):
    g = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), sigma)
    return np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def refine_ecc(real, cad, mask, max_shift_frac=0.03):
    """Returns the aligned real image (or the input unchanged if refinement does not clearly help)."""
    H, W = cad.shape[:2]
    s = 512 / max(H, W)
    sil = cv2.dilate(((mask if mask.ndim == 2 else mask.max(-1)) > 0).astype(np.uint8),
                     np.ones((31, 31), np.uint8)).astype(np.float32)

    def prep(img):
        g = cv2.resize(gradient_map(img, 2.0) * sil, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return (g / (g.max() + 1e-6)).astype(np.float32)

    gc, gr = prep(cad), prep(real)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(gc, gr, warp, cv2.MOTION_EUCLIDEAN,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        return real
    after = cv2.warpAffine(gr, warp, (gr.shape[1], gr.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    shift = np.hypot(*warp[:, 2]) / s / max(H, W)
    if _ncc(gc, after) < _ncc(gc, gr) + 0.01 or shift > max_shift_frac:
        return real
    w = warp.copy()
    w[:, 2] /= s                                    # translation back to working resolution
    return cv2.warpAffine(real, w, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------- legacy plain-background mode
def segment_real(real_bgr):
    h, w = real_bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(real_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(5, min(h, w) // 40)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(border, 0), axis=2)
    d8 = np.clip(dist / (dist.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    _, fg = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    return fg > 0


def align_silhouette(real, cad_sil):
    H, W = cad_sil.shape
    real_fg = segment_real(real)
    if not real_fg.any():
        return cv2.resize(real, (W, H))
    rx0, ry0, rx1, ry1 = bbox(real_fg)
    cx0, cy0, cx1, cy1 = bbox(cad_sil)
    s = min((cx1 - cx0) / max(rx1 - rx0, 1), (cy1 - cy0) / max(ry1 - ry0, 1))
    M = np.array([[s, 0, (cx0 + cx1) / 2 - s * (rx0 + rx1) / 2],
                  [0, s, (cy0 + cy1) / 2 - s * (ry0 + ry1) / 2]], np.float32)
    try:
        a = cv2.GaussianBlur(cad_sil.astype(np.float32), (0, 0), 5)
        b = cv2.GaussianBlur(cv2.warpAffine(real_fg.astype(np.float32), M, (W, H)), (0, 0), 5)
        warp = np.eye(2, 3, dtype=np.float32)
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
        M = (np.linalg.inv(np.vstack([warp, [0, 0, 1]])) @ np.vstack([M, [0, 0, 1]]))[:2].astype(np.float32)
    except cv2.error:
        pass
    return cv2.warpAffine(real, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def align(real, cad, mask, mode="none"):
    if mode == "ecc":
        return refine_ecc(real, cad, mask)
    if mode == "silhouette":
        return align_silhouette(real, (mask if mask.ndim == 2 else mask.max(-1)) > 0)
    return real

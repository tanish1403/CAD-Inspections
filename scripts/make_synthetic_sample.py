#!/usr/bin/env python3
"""Create small synthetic samples for smoke tests / a quick demo.

  layout "folder": mimics the real dataset - FG_00.png (CAD on black), BG_00.png (complete real photo on a cluttered
                   background), BG_00_TestImage*.png (edited photos), Mask_00_00.bmp (ids in channel 0),
                   0000.json with the StencilValue schema (key deliberately != mask file name).
  layout "prefix": the original <p>_Real.jpeg / _CAD.png / _Mask.png / _PartNamesToPixelMap.json files.
"""
import argparse, json, os
import cv2
import numpy as np

H, W = 600, 800
# name -> (id, kind, cx, cy, size)  ; 'hidden' has no pixels, 'sliver' is a hairline
PARTS = {"body": (1, "body"), "washer_a": (2, "washer", 250, 200, 46), "bolt_a": (3, "bolt", 250, 200, 28),
         "washer_b": (4, "washer", 550, 200, 46), "bolt_b": (5, "bolt", 550, 200, 28),
         "washer_c": (6, "washer", 400, 420, 46), "bolt_c": (7, "bolt", 400, 420, 28),
         "hidden_nut": (8, "hidden"), "rim": (9, "sliver")}


def _mask():
    m = np.zeros((H, W), np.uint8)
    m[100:520, 150:650] = 1
    for n, p in PARTS.items():
        if p[1] == "washer":
            cv2.circle(m, (p[2], p[3]), p[4], p[0], -1)
        if p[1] == "bolt":
            cv2.circle(m, (p[2], p[3]), p[4], p[0], -1)
    for n, p in PARTS.items():                       # bolt heads sit on top of washers
        if p[1] == "bolt":
            cv2.circle(m, (p[2], p[3]), p[4], p[0], -1)
    m[100:101, 300:500] = 9                          # 1-px hairline
    return m


def _cad(mask):
    img = np.zeros((H, W, 3), np.uint8)
    img[mask == 1] = (90, 70, 40)
    for n, p in PARTS.items():
        if p[1] == "washer":
            cv2.circle(img, (p[2], p[3]), p[4], (60, 60, 60), -1); cv2.circle(img, (p[2], p[3]), p[4], (20, 20, 20), 3)
        if p[1] == "bolt":
            hexa = np.array([[p[2] + int(p[4] * np.cos(a)), p[3] + int(p[4] * np.sin(a))] for a in np.arange(6) * np.pi / 3])
            cv2.fillPoly(img, [hexa], (110, 110, 110)); cv2.polylines(img, [hexa], True, (30, 30, 30), 3)
    return img


def _real(mask, rng, missing=(), altered=()):
    bg = cv2.GaussianBlur(rng.integers(60, 200, (H, W, 3)).astype(np.uint8), (0, 0), 3)        # cluttered background
    img = bg.copy()
    img[100:520, 150:650] = (125, 135, 145)
    img[100:520, 150:650] += rng.integers(0, 6, (420, 500, 3), dtype=np.uint8)
    for n, p in PARTS.items():
        if n in missing:
            continue
        if p[1] == "washer":
            cv2.circle(img, (p[2], p[3]), p[4], (200, 200, 205), -1); cv2.circle(img, (p[2], p[3]), p[4], (90, 90, 90), 2)
        if p[1] == "bolt":
            hexa = np.array([[p[2] + int(p[4] * np.cos(a)), p[3] + int(p[4] * np.sin(a))] for a in np.arange(6) * np.pi / 3])
            cv2.fillPoly(img, [hexa], (175, 175, 180)); cv2.polylines(img, [hexa], True, (60, 60, 60), 2)
    for n in altered:                                  # replaced by a rotated, darker bolt: structure present, pixels differ
        q = PARTS[n]
        cv2.circle(img, (q[2], q[3]), q[4], (200, 200, 205), -1)
        hexa = np.array([[q[2] + int(q[4] * np.cos(a + np.pi / 6)), q[3] + int(q[4] * np.sin(a + np.pi / 6))]
                         for a in np.arange(6) * np.pi / 3])
        cv2.fillPoly(img, [hexa], (95, 95, 100)); cv2.polylines(img, [hexa], True, (30, 30, 30), 3)
    return cv2.GaussianBlur(img, (3, 3), 0)


def make_folder(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    mask = _mask()
    cv2.imwrite(f"{out}/Mask_00_00.bmp", np.dstack([np.zeros_like(mask), np.zeros_like(mask), mask]))   # ids in R (= channel 0 for PIL)
    cv2.imwrite(f"{out}/FG_00.png", _cad(mask))
    cv2.imwrite(f"{out}/BG_00.png", _real(mask, np.random.default_rng(1)))
    cv2.imwrite(f"{out}/BG_00_TestImage.png", _real(mask, np.random.default_rng(1), missing=("washer_b", "bolt_b")))
    cv2.imwrite(f"{out}/BG_00_TestImage2.png", _real(mask, np.random.default_rng(1), altered=("bolt_c",)))
    entries = [{"NodeId": 0, "Name": n, "StencilValue": p[0]} for n, p in PARTS.items()]
    json.dump({"0000_00.bmp": entries}, open(f"{out}/0000.json", "w"), indent=1)   # key != mask file name (as in TestCase2)
    return {"BG_00": [], "BG_00_TestImage": ["washer_b", "bolt_b"], "BG_00_TestImage2": []}


def make_prefix(out, prefix="demo"):
    os.makedirs(out, exist_ok=True)
    mask = _mask()
    cv2.imwrite(f"{out}/{prefix}_Mask.png", mask)
    cv2.imwrite(f"{out}/{prefix}_CAD.png", _cad(mask))
    cv2.imwrite(f"{out}/{prefix}_Real.jpeg", _real(mask, np.random.default_rng(1), missing=("washer_a", "bolt_a")))
    json.dump({n: p[0] for n, p in PARTS.items()}, open(f"{out}/{prefix}_PartNamesToPixelMap.json", "w"))
    return ["washer_a", "bolt_a"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/input/synthetic")
    ap.add_argument("--layout", choices=["folder", "prefix"], default="folder")
    a = ap.parse_args()
    print("truth (absent parts per image):", make_folder(a.out) if a.layout == "folder" else make_prefix(a.out))

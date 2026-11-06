"""Sample discovery + loading. Understands two layouts:

A) "folder" layout (your dataset) - one folder per case:
     FG_00.png                 CAD render                      (or *CAD*, --cad_glob)
     BG_00.png, BG_00_TestImage*.png   real photos to check     (every other image, or --real_glob)
     Mask_00_00.bmp            part-id mask (ids in channel 0)   (or the file named in the json, --mask_glob)
     0000.json                 {"Mask_00_00.bmp": [{"Name": "box", "StencilValue": 2}, ...]}
B) "prefix" layout (original assignment): <p>_Real.jpeg <p>_CAD.png <p>_Mask.png <p>_PartNamesToPixelMap.json
"""
import fnmatch, glob, json, os
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class Sample:
    case: str          # folder (or prefix) the sample belongs to
    stem: str          # name of the real image (used for the output folder)
    real: str
    cad: str
    mask: str
    json: str

    @property
    def name(self):
        return f"{self.case}/{self.stem}"


def bbox(m):
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


# ----------------------------------------------------------------- json
def _stencil_entries(data):
    """-> (mask_hint, [entries]) if `data` follows the {"mask.bmp": [{"Name","StencilValue"}]} schema."""
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "StencilValue" in v[0]:
                return k, v
    return None, None


def load_part_map(path):
    """-> {part name: pixel value}.  Accepts the StencilValue schema, {"name": 12}, {"12": "name"},
    [{"name":..,"pixel_value":..}] and RGB triplets."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    _, ent = _stencil_entries(data)
    if ent is not None:
        return {str(e.get("Name", e.get("name"))): int(e["StencilValue"]) for e in ent}
    parts = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (int, float, list, tuple)):
                parts[str(k)] = v
            else:
                parts[str(v)] = int(k) if str(k).isdigit() else k
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = next((item[k] for k in item if "name" in k.lower()), None)
                val = next((item[k] for k in item
                            if any(s in k.lower() for s in ("stencil", "pixel", "value", "id", "color"))), None)
                parts[str(name)] = val
    return parts


def part_region(mask, value):
    """Boolean region of one part. mask is HxW (id image) or HxWx3 (RGB); value int or [r,g,b]."""
    if isinstance(value, (list, tuple)):
        if mask.ndim == 3:
            return (mask[..., :3] == np.array(value[:3], dtype=mask.dtype)).all(-1)
        return mask == int(value[0])
    g = mask if mask.ndim == 2 else mask[..., 0]
    return g == int(value)


# ----------------------------------------------------------------- discovery
def _imgs(files):
    return [f for f in files if f.lower().endswith(IMG_EXT)]


def _match(files, pattern):
    return [f for f in files if fnmatch.fnmatch(f.lower(), pattern.lower())]


def discover(input_dir, real_glob=None, cad_glob=None, mask_glob=None, json_glob=None):
    out = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        files = sorted(files)
        rel = os.path.relpath(root, input_dir)
        case = os.path.basename(os.path.abspath(input_dir)) if rel == "." else rel.replace(os.sep, "/")
        P = lambda f: os.path.join(root, f)
        # --- prefix layout
        for js in [f for f in files if f.endswith("_PartNamesToPixelMap.json")]:
            pre = js[: -len("_PartNamesToPixelMap.json")]
            find = lambda s: next(iter(sorted(glob.glob(os.path.join(root, pre + s)))), None)
            real, cad, mask = find("_Real.*"), find("_CAD.*"), find("_Mask.*")
            miss = [k for k, v in dict(real=real, cad=cad, mask=mask).items() if v is None]
            if miss:
                raise FileNotFoundError(f"sample '{pre}': missing {miss} in {root}")
            out.append(Sample(case, pre, real, cad, mask, P(js)))
        # --- folder layout (json in StencilValue schema)
        jsons = _match(files, json_glob) if json_glob else [f for f in files if f.endswith(".json")]
        for js in jsons:
            if js.endswith("_PartNamesToPixelMap.json"):
                continue
            try:
                hint, ent = _stencil_entries(json.load(open(P(js), encoding="utf-8")))
            except Exception:
                continue
            if ent is None:
                continue
            images = _imgs(files)
            # mask: name in json, else the only/first mask-looking file
            if mask_glob:
                cands = _match(images, mask_glob)
            elif hint and hint in files:
                cands = [hint]
            else:
                cands = [f for f in images if "mask" in f.lower()]
            if not cands:
                raise FileNotFoundError(f"{root}: no mask image found (json refers to '{hint}'). Use --mask_glob.")
            mask = cands[0]
            rest = [f for f in images if f not in cands and "mask" not in f.lower()]
            pats = [cad_glob] if cad_glob else ["fg_*", "*cad*", "*render*"]
            cad = next((f for p in pats for f in _match(rest, p)), None)
            if cad is None:
                raise FileNotFoundError(f"{root}: no CAD image found among {rest}. Use --cad_glob.")
            reals = _match(rest, real_glob) if real_glob else [f for f in rest if f != cad]
            for r in reals:
                out.append(Sample(case, os.path.splitext(r)[0], P(r), P(cad), P(mask), P(js)))
    return out


# ----------------------------------------------------------------- loading
def _read_rgb(path):
    return np.array(Image.open(path).convert("RGB"))        # alpha ignored on purpose (photos are opaque)


def _read_cad_bgr(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):                       # transparent CAD -> composite on black
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 255)), im)
    return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def _read_mask(path):
    a = np.array(Image.open(path))                           # palette images keep their indices
    if a.ndim == 3:
        a = a[..., :3]
        if not a[..., 1:].any():                             # ids stored in channel 0 only (your masks)
            a = a[..., 0]
        elif (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all():
            a = a[..., 0]
    return a


def load_sample(s, max_side=2048):
    """-> real_bgr, cad_bgr, mask, parts - all at the same working resolution (longest side <= max_side)."""
    cad = _read_cad_bgr(s.cad)
    mask = _read_mask(s.mask)
    real = cv2.cvtColor(_read_rgb(s.real), cv2.COLOR_RGB2BGR)
    if mask.shape[:2] != cad.shape[:2]:
        mask = cv2.resize(mask, (cad.shape[1], cad.shape[0]), interpolation=cv2.INTER_NEAREST)
    H, W = cad.shape[:2]
    if abs(real.shape[1] / real.shape[0] - W / H) > 0.02:
        print(f"[warn] {s.name}: real image aspect differs from CAD ({real.shape[1]}x{real.shape[0]} vs {W}x{H}); stretching")
    k = min(1.0, max_side / max(H, W)) if max_side else 1.0
    nW, nH = int(round(W * k)), int(round(H * k))
    if k < 1.0:
        cad = cv2.resize(cad, (nW, nH), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nW, nH), interpolation=cv2.INTER_NEAREST)
    real = cv2.resize(real, (nW, nH), interpolation=cv2.INTER_AREA if real.shape[1] > nW else cv2.INTER_LINEAR)
    parts = load_part_map(s.json)
    if not any(part_region(mask, v).any() for v in parts.values()):
        raise ValueError(f"{s.name}: none of the json pixel values occur in the mask - check json / mask encoding")
    return real, cad, mask, parts


def load_golden(path, size_wh):
    img = cv2.cvtColor(_read_rgb(path), cv2.COLOR_RGB2BGR)
    return cv2.resize(img, size_wh, interpolation=cv2.INTER_AREA if img.shape[1] > size_wh[0] else cv2.INTER_LINEAR)

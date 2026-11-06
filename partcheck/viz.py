"""Overlays.  green = present, red = absent, yellow = uncertain, orange = present but altered vs golden,
grey outline = cannot be verified (not visible / too small)."""
import cv2
import numpy as np
from .data import bbox, part_region

GREEN, RED, YELLOW, ORANGE, GREY = (60, 200, 60), (50, 50, 230), (0, 215, 255), (0, 140, 255), (150, 150, 150)


def _outline(out, reg, color, thick):
    cnts, _ = cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, thick)


def short(name, n=22):
    return name if len(name) <= n else name[:n - 9] + "~" + name[-8:]


def draw(base, parts, mask, res, pick, alpha=0.5, grey_unverifiable=False, label=True, label_if=None):
    """pick(result_dict) -> color or None. Large surfaces (the body) get an outline only, so the photo stays visible."""
    out = base.copy()
    H, W = out.shape[:2]
    sc = max(H, W) / 2048
    thick = max(1, int(round(3 * sc)))
    fs, ft = max(0.4, 0.8 * sc), max(1, int(round(2 * sc)))
    for name, val in parts.items():
        r = res[name]
        reg = part_region(mask, val)
        if not reg.any():
            continue
        color = pick(r)
        if color is None:
            if grey_unverifiable and r["status"] == "UNVERIFIABLE":
                _outline(out, reg, GREY, max(1, thick // 2))
            continue
        big = reg.sum() > 0.03 * H * W
        if not big:
            layer = out.copy()
            layer[reg] = color
            out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)
        _outline(out, reg, color, thick)
        if label and not big and (label_if is None or label_if(r)):
            n, lab, st, _ = cv2.connectedComponentsWithStats(reg.astype(np.uint8))
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))             # largest visible blob
            x0, y0 = st[k, cv2.CC_STAT_LEFT], st[k, cv2.CC_STAT_TOP]
            txt = short(name) + ("*" if r.get("altered") else "")
            p = (int(x0), int(max(y0 - 6 * sc, 14)))
            cv2.putText(out, txt, p, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ft + 2)
            cv2.putText(out, txt, p, cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft)
    return out


def banner(img, text):
    img = img.copy()
    h = max(26, img.shape[0] // 30)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (30, 30, 30), -1)
    cv2.putText(img, text, (8, int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, h / 38, (255, 255, 255), max(1, h // 18))
    return img


def _present_color(r):
    if r["status"] == "PRESENT":
        return ORANGE if r.get("altered") else GREEN
    return YELLOW if r["status"] == "UNCERTAIN" else None


def _absent_color(r):
    return RED if r["status"] == "ABSENT" else (YELLOW if r["status"] == "UNCERTAIN" else None)


def _any_color(r):
    return {"ABSENT": RED, "UNCERTAIN": YELLOW}.get(r["status"]) or \
        ((ORANGE if r.get("altered") else GREEN) if r["status"] == "PRESENT" else None)


def _problem(r):
    return r["status"] in ("ABSENT", "UNCERTAIN") or bool(r.get("altered"))


def object_box(mask, margin=0.05):
    sil = (mask if mask.ndim == 2 else mask.max(-1)) > 0
    H, W = sil.shape
    if not sil.any():
        return 0, 0, W, H
    x0, y0, x1, y1 = bbox(sil)
    m = int(margin * max(x1 - x0, y1 - y0))
    return max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, W), min(y1 + m, H)


def _wrap(text, width, fs, ft):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0][0] > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + [cur]


def summary_panel(res, width, title):
    """Text panel: counts + the parts of every status (problems first)."""
    groups = [("ABSENT", RED, [n for n, r in res.items() if r["status"] == "ABSENT"]),
              ("UNCERTAIN", YELLOW, [n for n, r in res.items() if r["status"] == "UNCERTAIN"]),
              ("ALTERED vs golden (present)", ORANGE, [n for n, r in res.items() if r.get("altered")]),
              ("PRESENT", GREEN, [n for n, r in res.items() if r["status"] == "PRESENT" and not r.get("altered")]),
              ("CANNOT BE VERIFIED from this view", GREY, [n for n, r in res.items() if r["status"] == "UNVERIFIABLE"])]
    fs, ft, lh = max(0.5, width / 2400), 1, max(22, int(width / 80))
    rows = [(title, (255, 255, 255), True)]
    for name, col, items in groups:
        if name.startswith("CANNOT"):
            rows.append((f"{name}: {len(items)} parts (hidden / hairline / too small)", col, True))
            continue
        rows.append((f"{name} ({len(items)})" + (":" if items else ""), col, True))
        if items:
            for ln in _wrap(", ".join(items), width - 40, fs, ft):
                rows.append(("    " + ln, (220, 220, 220), False))
    img = np.full((lh * (len(rows) + 1), width, 3), 30, np.uint8)
    for i, (txt, col, bold) in enumerate(rows):
        y = int(lh * (i + 0.9))
        if bold and i:
            cv2.rectangle(img, (10, y - int(lh * 0.55)), (10 + int(lh * 0.5), y - int(lh * 0.05)), col, -1)
        cv2.putText(img, txt, (14 + (int(lh * 0.7) if bold and i else 0), y), cv2.FONT_HERSHEY_SIMPLEX, fs, col if bold else (220, 220, 220), ft + (1 if bold else 0))
    return img


def report_card(cad, original, real_al, parts, mask, res, title=""):
    """ORIGINAL test image | result on the test image | CAD reference, with a status summary underneath."""
    x0, y0, x1, y1 = object_box(mask)
    ann = draw(real_al, parts, mask, res, _any_color, 0.45, label_if=_problem)
    cad_ann = draw(cad, parts, mask, res, _any_color, 0.45, grey_unverifiable=True, label_if=_problem)
    panels = [banner(original[y0:y1, x0:x1], "ORIGINAL TEST IMAGE"), banner(ann[y0:y1, x0:x1], "RESULT ON TEST IMAGE"),
              banner(cad_ann[y0:y1, x0:x1], "CAD REFERENCE")]
    h = 640
    panels = [cv2.resize(p, (max(8, int(p.shape[1] * h / p.shape[0])), h), interpolation=cv2.INTER_AREA) for p in panels]
    top = np.hstack(panels)
    n = lambda s: sum(r["status"] == s for r in res.values())
    head = f"{title}   PRESENT {n('PRESENT')} | ABSENT {n('ABSENT')} | UNCERTAIN {n('UNCERTAIN')} | UNVERIFIABLE {n('UNVERIFIABLE')}"
    return np.vstack([top, summary_panel(res, top.shape[1], head)])


def render(cad, real_al, parts, mask, res, original=None, title="", margin=0.05):
    """-> (report card, 2x2 detail image, dict of single panels). Everything is cropped to the object."""
    original = real_al if original is None else original
    x0, y0, x1, y1 = object_box(mask, margin)
    ghost = cv2.addWeighted(real_al, 0.6, cad, 0.4, 0)
    P = {
        "present_cad": banner(draw(cad, parts, mask, res, _present_color, grey_unverifiable=True),
                              "CAD: PRESENT green | altered orange | uncertain yellow | grey = cannot verify"),
        "present_real": banner(draw(real_al, parts, mask, res, _present_color), "TEST IMAGE: parts PRESENT"),
        "absent_cad": banner(draw(cad, parts, mask, res, _absent_color, 0.6), "CAD: parts ABSENT in the photo (red)"),
        "absent_real": banner(draw(ghost, parts, mask, res, _absent_color, 0.6),
                              "TEST IMAGE (with CAD ghost): parts ABSENT (red)"),
    }
    P = {k: v[y0:y1, x0:x1] for k, v in P.items()}
    P["original"] = original[y0:y1, x0:x1].copy()
    detail = np.vstack([np.hstack([P["present_cad"], P["present_real"]]), np.hstack([P["absent_cad"], P["absent_real"]])])
    k = min(1.0, 2400 / detail.shape[1])
    if k < 1:
        detail = cv2.resize(detail, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return report_card(cad, original, real_al, parts, mask, res, title), detail, P

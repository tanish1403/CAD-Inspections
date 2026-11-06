#!/usr/bin/env python3
"""Build ONE self-contained HTML report (images embedded) from an output folder - open it in any browser, e-mail it,
or screen-share it during the presentation.   python scripts/make_report.py --output_dir data/output --out report.html"""
import argparse, base64, glob, html, json, os
import cv2


def b64(path, max_w=1600):
    if not os.path.exists(path) and path.endswith(".png"):
        path = path[:-4] + ".jpg"                 # demo results are stored as jpg
    im = cv2.imread(path)
    if im.shape[1] > max_w:
        im = cv2.resize(im, None, fx=max_w / im.shape[1], fy=max_w / im.shape[1], interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


COL = {"PRESENT": "#2ecc71", "ABSENT": "#e74c3c", "UNCERTAIN": "#f1c40f", "UNVERIFIABLE": "#7f8c8d"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="data/output")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--title", default="CAD vs real: part inspection")
    a = ap.parse_args()
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.output_dir, "**", "report.json"), recursive=True))
    parts = [f"<h1>{html.escape(a.title)}</h1><p>green present · red absent · yellow uncertain · orange altered vs golden · grey cannot be verified</p>"]
    for d in dirs:
        res = json.load(open(os.path.join(d, "report.json")))
        name = os.path.relpath(d, a.output_dir)
        n = lambda s: sum(r["status"] == s for r in res.values())
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td style='color:{COL[r['status']]}'><b>{r['status']}</b>"
            f"{' · altered' if r.get('altered') else ''}</td><td>{r['score']}</td><td>{r['area_px']}</td></tr>"
            for k, r in sorted(res.items(), key=lambda kv: ["ABSENT", "UNCERTAIN", "PRESENT", "UNVERIFIABLE"].index(kv[1]["status"]))
            if r["status"] != "UNVERIFIABLE")
        parts.append(f"<h2>{html.escape(name)}</h2><p><b>{n('PRESENT')}</b> present · <b style='color:{COL['ABSENT']}'>{n('ABSENT')}</b> absent · "
                     f"<b>{n('UNCERTAIN')}</b> uncertain · {n('UNVERIFIABLE')} not verifiable</p>"
                     f"<img src='data:image/jpeg;base64,{b64(os.path.join(d, 'result.png'))}' style='max-width:100%'>"
                     f"<details><summary>part table</summary><table><tr><th>part</th><th>status</th><th>score</th><th>area px</th></tr>{rows}</table></details>")
    css = ("body{font-family:system-ui,Arial;background:#161616;color:#eee;max-width:1700px;margin:auto;padding:20px}"
           "table{border-collapse:collapse}td,th{border:1px solid #444;padding:4px 10px}h2{margin-top:40px;border-top:1px solid #444;padding-top:20px}")
    open(a.out, "w", encoding="utf-8").write(f"<!doctype html><meta charset='utf-8'><style>{css}</style>{''.join(parts)}")
    print(f"wrote {a.out} ({len(dirs)} photos)")


if __name__ == "__main__":
    main()

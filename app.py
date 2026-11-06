"""CAD <-> real-photo part inspection - presentation app.

    streamlit run app.py

Modes
  * Analyse   run the checker live on a photo from a folder (or on files you upload)
  * Browse    open results that were saved earlier (works instantly, no models / GPU needed) - ideal for a live demo
"""
import glob
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import run as cli                                             # noqa: E402  (argument parser + component loader)
from partcheck.data import _stencil_entries, discover         # noqa: E402
from partcheck.pipeline import analyze, out_path, pick_golden  # noqa: E402
from partcheck.verifiers import AVAILABLE, crop               # noqa: E402
from partcheck.viz import GREEN, RED, YELLOW                  # noqa: E402

st.set_page_config(page_title="CAD Part Inspector", page_icon="🔩", layout="wide")

BADGE = {"PRESENT": "🟢 PRESENT", "ABSENT": "🔴 ABSENT", "UNCERTAIN": "🟡 UNCERTAIN", "UNVERIFIABLE": "⚪ UNVERIFIABLE"}


# ------------------------------------------------------------------ helpers
def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def table(res, hide_unverifiable=True):
    rows = []
    for n, r in res.items():
        if hide_unverifiable and r["status"] == "UNVERIFIABLE":
            continue
        rows.append({"part": n, "status": BADGE[r["status"]] + (" 🟠 altered" if r.get("altered") else ""),
                     "score": r["score"], "area_px": r["area_px"], "note": r.get("note", "")})
    order = {"🔴": 0, "🟡": 1, "🟢": 2, "⚪": 3}
    df = pd.DataFrame(rows, columns=["part", "status", "score", "area_px", "note"])
    if len(df):
        df = df.sort_values("status", key=lambda s: s.map(lambda x: order[x[0]])).reset_index(drop=True)
    return df


def counts(res):
    n = lambda s: sum(r["status"] == s for r in res.values())
    return n("PRESENT"), n("ABSENT"), n("UNCERTAIN"), n("UNVERIFIABLE"), sum(bool(r.get("altered")) for r in res.values())


def metrics(res):
    p, a, u, v, alt = counts(res)
    c = st.columns(5)
    c[0].metric("🟢 Present", p)
    c[1].metric("🔴 Absent", a)
    c[2].metric("🟡 Uncertain", u)
    c[3].metric("🟠 Altered", alt, help="present, but pixels differ strongly from the golden (complete) photo")
    c[4].metric("⚪ Not verifiable", v, help="hidden / hairline / too small in the CAD view - not guessed")


def read_any(folder, stem):
    """cv2 image `stem`.png or `stem`.jpg from a result folder (demo results are stored as jpg), or None."""
    for ext in (".png", ".jpg"):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return cv2.imread(p)
    return None


def zip_bytes(folder):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(os.path.join(folder, "*")):
            z.write(f, os.path.basename(f))
    return buf.getvalue()


@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_components(argv):
    a = cli.apply_profile(cli.build_parser().parse_args(list(argv)))
    return cli.components(a)


def build_argv(cfg, input_dir, output_dir):
    argv = ["check", "--input_dir", input_dir, "--output_dir", output_dir, "--profile", cfg["profile"],
            "--thr", str(cfg["thr"]), "--uncertain_band", str(cfg["band"]), "--ai_weight", str(cfg["ai_w"]),
            "--max_side", str(cfg["max_side"]), "--method", "ai" if cfg["ai"] else "classical"]
    if cfg["gray"]:
        argv.append("--gray")
    if cfg["sam"] and cfg["ai"]:
        argv.append("--sam")
    if cfg["vlm"]:
        argv += ["--vlm", *cfg["vlm"]]
        if cfg["use_names"]:
            argv.append("--use_names")
        if cfg["vlm_unc"]:
            argv.append("--vlm_uncertain_only")
    if cfg["golden"] != "none":
        argv += ["--golden", cfg["golden"]]
    if cfg.get("head"):
        argv += ["--head", cfg["head"]]
    if cfg.get("bank"):
        argv += ["--bank", cfg["bank"]]
    return argv


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔩 CAD Part Inspector")
mode = st.sidebar.radio("Mode", ["Analyse a photo", "Browse saved results"],
                        help="Browse = instant, shows results saved earlier (no GPU needed).")

cfg = {}
if mode == "Analyse a photo":
    with st.sidebar.expander("⚙️ Engine", expanded=True):
        engine = st.radio("Method", ["AI: DINOv2 + classical cues", "Classical only (no models, CPU)"], index=1,
                          help="The AI method downloads DINOv2 on first use and is much faster with a GPU.")
        cfg["ai"] = engine.startswith("AI")
        cfg["profile"] = st.selectbox("Hardware profile", ["auto", "low", "balanced", "high"], disabled=not cfg["ai"])
        cfg["gray"] = st.checkbox("Grayscale input (reduces CAD-vs-photo colour gap)", value=True, disabled=not cfg["ai"])
        cfg["sam"] = st.checkbox("Add SAM mask cue", value=False, disabled=not cfg["ai"])
        cfg["max_side"] = st.select_slider("Working resolution (px, longest side)", [1024, 1536, 2048, 3072], 2048)
    with st.sidebar.expander("🧠 Optional VLM / grounding verifiers"):
        cfg["vlm"] = st.multiselect("Verifiers (need a GPU for the big ones)", AVAILABLE,
                                    help="InternVL3, Qwen2-VL, SmolVLM, SigLIP, OWLv2, ... see README")
        cfg["use_names"] = st.checkbox("Part names are meaningful words", value=False)
        cfg["vlm_unc"] = st.checkbox("Only run them on borderline parts (faster)", value=True)
    with st.sidebar.expander("🎚️ Decision"):
        cfg["thr"] = st.slider("Present threshold", 0.2, 0.8, 0.5, 0.05)
        cfg["band"] = st.slider("Uncertain band (+/-)", 0.0, 0.3, 0.15, 0.01)
        cfg["ai_w"] = st.slider("AI weight (vs classical cues)", 0.0, 1.0, 0.4, 0.05, disabled=not cfg["ai"])
        cfg["golden"] = st.selectbox("Golden (complete) photo", ["auto", "none"],
                                     help="auto = first image without 'test' in its name. Flags parts whose pixels changed.")
        head = st.text_input("Trained head (.pkl, optional)", "")
        bank = st.text_input("Few-shot bank (.pkl, optional)", "")
        cfg["head"], cfg["bank"] = head.strip() or None, bank.strip() or None

# ------------------------------------------------------------------ Browse mode
if mode == "Browse saved results":
    default = next((d for d in ("data/output", "demo/output") if glob.glob(os.path.join(ROOT, d, "**/report.json"), recursive=True)),
                   "demo/output")
    out_dir = st.sidebar.text_input("Results folder", default)
    out_abs = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    reports = sorted(glob.glob(os.path.join(out_abs, "**", "report.json"), recursive=True))
    st.title("Saved results")
    if not reports:
        st.info(f"No results in `{out_dir}`. Run `python run.py check` first or switch to *Analyse a photo*.")
        st.stop()
    names = [os.path.relpath(os.path.dirname(r), out_abs) for r in reports]
    pick = st.sidebar.selectbox("Photo", names)
    d = os.path.join(out_abs, pick)
    res = json.load(open(os.path.join(d, "report.json")))
    metrics(res)
    t1, t2, t3 = st.tabs(["Result", "Parts", "Panels"])
    with t1:
        st.image(rgb(read_any(d, "result")), width="stretch")
    with t2:
        hide = st.checkbox("Hide not-verifiable parts", True)
        st.dataframe(table(res, hide), width="stretch", hide_index=True)
    with t3:
        c = st.columns(2)
        for i, (f, cap) in enumerate([("original", "Original test image"), ("present_real", "Present parts"),
                                      ("absent_real", "Absent parts (CAD ghost)"), ("absent_cad", "Absent parts on CAD")]):
            im = read_any(d, f)
            if im is not None:
                c[i % 2].image(rgb(im), caption=cap, width="stretch")
    st.download_button("⬇️ Download this result (zip)", zip_bytes(d), file_name=pick.replace("/", "_") + ".zip")
    st.stop()

# ------------------------------------------------------------------ Analyse mode
st.title("CAD ↔ Real part inspection")
st.caption("Which parts of the CAD model are really on the physical assembly? "
           "Green = present · Red = absent · Yellow = uncertain · Orange = present but altered · Grey = cannot be verified from this view")

source = st.sidebar.radio("Data", ["Folder on disk", "Upload files"])
work = st.session_state.setdefault("work", tempfile.mkdtemp(prefix="cadapp_"))
samples, input_dir = [], None

if source == "Folder on disk":
    input_dir = st.sidebar.text_input("Input folder", "data/input")
    input_abs = input_dir if os.path.isabs(input_dir) else os.path.join(ROOT, input_dir)
    try:
        samples = discover(input_abs)
    except Exception as e:                                  # noqa: BLE001
        st.sidebar.error(str(e))
    input_dir = input_abs
else:
    up = st.sidebar.file_uploader("CAD render", ["png", "jpg", "jpeg"])
    mk = st.sidebar.file_uploader("Part-id mask", ["png", "bmp"])
    js = st.sidebar.file_uploader("Part map (json)", ["json"])
    ph = st.sidebar.file_uploader("Real photo(s)", ["png", "jpg", "jpeg"], accept_multiple_files=True)
    if up and mk and js and ph:
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        raw = js.getvalue()
        stencil = _stencil_entries(json.loads(raw))[1] is not None
        mext = os.path.splitext(mk.name)[1]
        if stencil:
            open(f"{work}/FG_upload{os.path.splitext(up.name)[1]}", "wb").write(up.getvalue())
            open(f"{work}/Mask_upload{mext}", "wb").write(mk.getvalue())
            open(f"{work}/parts.json", "wb").write(raw)
            for p in ph:
                open(f"{work}/BG_{p.name}", "wb").write(p.getvalue())
        else:                                               # original assignment layout, one photo
            open(f"{work}/up_CAD{os.path.splitext(up.name)[1]}", "wb").write(up.getvalue())
            open(f"{work}/up_Mask{mext}", "wb").write(mk.getvalue())
            open(f"{work}/up_PartNamesToPixelMap.json", "wb").write(raw)
            open(f"{work}/up_Real{os.path.splitext(ph[0].name)[1]}", "wb").write(ph[0].getvalue())
        input_dir = work
        samples = discover(work)

if not samples:
    st.info("Pick a folder that holds a CAD render, mask, json and photo(s) - or upload them - to start. "
            "Or use **Browse saved results** in the sidebar for an instant demo.")
    st.stop()

cases = sorted({s.case for s in samples})
case = st.sidebar.selectbox("Case", cases)
in_case = [s for s in samples if s.case == case]
sample = st.sidebar.selectbox("Photo to check", in_case, format_func=lambda s: s.stem)

out_dir = os.environ.get("CADAPP_OUTPUT", os.path.join(ROOT, "data", "output"))
argv = build_argv(cfg, input_dir, out_dir)
run_clicked = st.sidebar.button("▶ Run inspection", type="primary", width="stretch")
run_all = st.sidebar.button("Run all photos of this case", width="stretch")


def do_run(smps):
    a = cli.apply_profile(cli.build_parser().parse_args(argv))
    bb, sam, ver, bank = load_components(tuple(argv))
    from partcheck.train import load_bank
    import pickle
    head = pickle.load(open(a.head, "rb")) if a.head else None
    golden = pick_golden(a, samples)
    out = {}
    bar = st.progress(0.0)
    for i, s in enumerate(smps):
        with st.spinner(f"Analysing {s.stem} ..."):
            out[s.stem] = analyze(s, a, bb, sam, ver, bank, head, golden.get(s.name), out_path(out_dir, s))
        bar.progress((i + 1) / len(smps))
    bar.empty()
    return out


if run_clicked or run_all:
    try:
        st.session_state["results"] = do_run(in_case if run_all else [sample])
        st.session_state["last"] = sample.stem
    except Exception as e:                                  # noqa: BLE001
        st.error(f"Run failed: {e}")
        st.exception(e)

results = st.session_state.get("results") or {}
if not results:
    st.info("Press **▶ Run inspection** in the sidebar.")
    st.stop()

shown = st.selectbox("Result for", list(results), index=list(results).index(st.session_state.get("last", list(results)[0]))
                     if st.session_state.get("last") in results else 0) if len(results) > 1 else list(results)[0]
A = results[shown]
metrics(A.res)
tab_res, tab_tab, tab_part, tab_cmp = st.tabs(["Result", "Parts table", "Part inspector", "All photos"])

with tab_res:
    card, detail, panels = A.render()
    st.image(rgb(card), width="stretch")
    with st.expander("Detail: present / absent on CAD and on the photo"):
        st.image(rgb(detail), width="stretch")
    folder = out_path(out_dir, A.sample)
    if os.path.isdir(folder):
        st.download_button("⬇️ Download result (images + report, zip)", zip_bytes(folder), file_name=f"{A.sample.stem}.zip")

with tab_tab:
    hide = st.checkbox("Hide not-verifiable parts", True, key="hide")
    st.dataframe(table(A.res, hide), width="stretch", hide_index=True)
    st.caption("Not verifiable = no visible pixels in the CAD view, only a hairline, or too small to judge - never guessed.")

with tab_part:
    ok = [n for n, r in A.res.items() if r["status"] != "UNVERIFIABLE"]
    ok.sort(key=lambda n: {"ABSENT": 0, "UNCERTAIN": 1, "PRESENT": 2}[A.res[n]["status"]])
    name = st.selectbox("Part", ok, format_func=lambda n: f"{BADGE[A.res[n]['status']]}  {n}")
    from partcheck.data import part_region
    reg = part_region(A.mask, A.parts[name])
    ref = A.cad.copy()
    cnt, _ = cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ref, cnt, -1, (0, 220, 0), 3)
    real_o = A.original.copy()
    cv2.drawContours(real_o, cnt, -1, GREEN if A.res[name]["status"] == "PRESENT" else RED, 3)
    c = st.columns(3)
    c[0].image(rgb(crop(ref, reg, 0.5)), caption="CAD (part outlined)", width="stretch")
    c[1].image(rgb(crop(A.original, reg, 0.5)), caption="Original photo (same place)", width="stretch")
    c[2].image(rgb(crop(real_o, reg, 0.5)), caption="Photo with CAD outline", width="stretch")
    r = A.res[name]
    st.write(f"**{BADGE[r['status']]}**  ·  score `{r['score']}`  ·  visible area `{r['area_px']}` px"
             + ("  ·  🟠 altered vs golden" if r.get("altered") else ""))
    cues = {k: v for k, v in r.items() if k in ("edge", "contrast", "shape", "margin", "local", "bg", "gold", "sam_iou")
            or k.startswith("vlm_") or k.startswith("ref_")}
    cues = {k: v for k, v in cues.items() if isinstance(v, (int, float))}
    if cues:
        st.bar_chart(pd.Series(cues, name="cue value"))
        st.caption("edge/contrast: classical outline & structure · margin/local/bg: DINOv2 patch matching · "
                   "gold: similarity to the complete photo · vlm_*: verifier models")

with tab_cmp:
    if len(results) < 2:
        st.info("Use **Run all photos of this case** to compare every photo side by side.")
    else:
        parts = [n for n in next(iter(results.values())).res]
        mat = pd.DataFrame({stem: {n: BADGE[a.res[n]["status"]].split(" ")[0] + ("🟠" if a.res[n].get("altered") else "")
                                   for n in parts if n in a.res} for stem, a in results.items()})
        mat = mat[~(mat == "⚪").all(axis=1)]
        st.dataframe(mat, width="stretch")
        st.caption("🟢 present · 🔴 absent · 🟡 uncertain · 🟠 altered · ⚪ not verifiable")

#!/usr/bin/env python3
"""Regenerate notebooks/kaggle_run.ipynb and notebooks/colab_run.ipynb (so both stay in sync)."""
import json, os

MD, CODE = "markdown", "code"

COMMON_MODELS = [
(MD, """
## 3. Models (one time, needs Internet)
DINOv2 is the backbone. `--models none` skips it. Verifier models (SigLIP, SmolVLM, Qwen2-VL, InternVL3 ...) download themselves on first use;
pre-download with `--vlm` to keep the later cells quick. Progress bars are switched off to keep the log readable."""),
(CODE, """
!python scripts/download_models.py --models dinov2-base sam-vit-base
# optional, pre-fetch verifiers (each 1-5 GB):
# !python scripts/download_models.py --models none --vlm siglip internvl3-2b qwen2-vl smolvlm
"""),
]

def tail(env):
    return [
(MD, """
## 4. Quick sanity check - classical mode (no model, seconds)
Confirms the data is found and gives a baseline. Statuses: **PRESENT / ABSENT / UNCERTAIN / UNVERIFIABLE** (hidden or hairline parts are never guessed)."""),
(CODE, """
!python run.py check --method classical --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR/classical" --golden auto
"""),
(CODE, """
from partcheck.notebook_utils import show_results, problems
show_results(f"{OUTPUT_DIR}/classical")        # original test image | result | CAD reference + summary
problems(f"{OUTPUT_DIR}/classical")             # only absent / uncertain / altered parts
"""),
(MD, """
## 5. Main run - DINOv2 (zero-shot) + classical cues + golden photo
`--profile auto` sizes the model to your GPU. `--golden auto` uses the first non-"test" photo of each case as the complete reference and
flags parts whose pixels changed (orange)."""),
(CODE, """
!python run.py check --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR/dino" --profile auto --gray --golden auto
"""),
(CODE, """
show_results(f"{OUTPUT_DIR}/dino")
problems(f"{OUTPUT_DIR}/dino")
"""),
(MD, """
## 6. Optional: second-opinion verifiers (VLM / grounding models)
Pick any - they are blended into the score (`--vlm_weight`, default 0.3) and run only on borderline parts with `--vlm_uncertain_only`.
`qwen2-vl` (7B 4-bit loads fine on a T4) and the new **InternVL3** (`internvl3-1b/2b/8b`, Hugging Face `-hf` format) are the interesting ones.
Fixed since the last version: CLIP/SigLIP embedding crash, OWLv2 post-processing crash, unquoted `pip install ...>=` (shell redirect), `torch_dtype` warning."""),
(CODE, """
!pip install -q -U "bitsandbytes>=0.46.1"      # quotes matter: an unquoted >= is a shell redirect
"""),
(CODE, """
# InternVL3 (2B fits a T4 in fp16)
!python run.py check --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR/internvl" --gray --golden auto --vlm internvl3-2b --use_names --vlm_uncertain_only
"""),
(CODE, """
# Qwen2-VL 7B in 4-bit (worked on 2x T4)
# !python run.py check --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR/qwen7b" --gray --golden auto --vlm vlm:Qwen/Qwen2-VL-7B-Instruct --vlm_4bit --use_names --vlm_uncertain_only
# SigLIP / SmolVLM / open-vocabulary detectors (detectors need meaningful part names)
# !python run.py check --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR/siglip" --gray --vlm siglip smolvlm --golden auto
# !python run.py check --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR/det" --gray --vlm grounding-dino owlv2 --use_names --golden auto
"""),
(CODE, """
show_results(f"{OUTPUT_DIR}/internvl")
"""),
(MD, """
## 7. Optional: learn from your photos (few-shot bank, head, fine-tuning)
**Important (bug fixed):** edited/test photos must never be treated as complete. Mark only the *complete* photo with `--ref_glob`
(pattern on `folder/photo`) and give real labels for the rest via `--labels` (see `examples/labels_example.json`).
Everything unlabeled falls back to confident classical predictions."""),
(CODE, """
import os
REF = os.path.basename(INPUT_DIR.rstrip("/")) + "/BG_00"       # the COMPLETE photo of the main case (not TestCase2!)
LABELS = "examples/labels_example.json"
print("reference photo pattern:", REF)
"""),
(CODE, """
# few-shot bank (real-photo prototypes) -> check with it
!python run.py bank  --input_dir "$INPUT_DIR" --gray --ref_glob "$REF" --labels $LABELS --iters 6
!python run.py check --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR/bank" --gray --golden auto --bank models/bank.pkl
"""),
(CODE, """
# few-shot head (small classifier on the cues) - prints leave-one-photo-out accuracy
!python run.py train --input_dir "$INPUT_DIR" --gray --ref_glob "$REF" --labels $LABELS --iters 30
!python run.py check --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR/head" --gray --golden auto --head models/head.pkl
"""),
(CODE, """
# fine-tune the last DINOv2 blocks (GPU, ~10 min on a T4)
!python run.py finetune --input_dir "$INPUT_DIR" --gray --size 518 --steps 300 --unfreeze 2 --ref_glob "$REF" --labels $LABELS
!python run.py check --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR/finetuned" --gray --size 518 --dino models/dinov2-ft --golden auto
"""),
(MD, """
## 8. Which model should I present? - run the benchmark
Scores every configuration against `examples/ground_truth.json` (edit it for your own data) and writes a table you can paste into the slides."""),
(CODE, """
!python scripts/benchmark.py --input_dir "$INPUT_DIR" --truth examples/ground_truth.json --out "$OUTPUT_DIR/benchmark" --configs classical dino dino+siglip dino+smolvlm dino+qwen2-vl dino+internvl3-2b
"""),
(CODE, """
import pandas as pd
pd.read_csv(f"{OUTPUT_DIR}/benchmark/benchmark.csv")
"""),
(MD, """
## 9. Presentation material
**a) Self-contained HTML report** (open in any browser / share the file):"""),
(CODE, """
!python scripts/make_report.py --output_dir "$OUTPUT_DIR/dino" --out "$OUTPUT_DIR/report.html" --title "CAD vs real: part inspection"
"""),
(MD, """
**b) Streamlit app, live** (optional, untested tunnel): starts the app and prints a public link via a free Cloudflare quick tunnel.
In the app choose *Browse saved results* (instant) or *Analyse a photo*."""),
(CODE, """
import re, subprocess, time
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
                 stdout=open("streamlit.log", "w"), stderr=subprocess.STDOUT)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
tun = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8501"], stderr=subprocess.PIPE, text=True)
for line in tun.stderr:
    m = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", line)
    if m:
        print("Open:", m.group(0)); break
"""),
(MD, "## 10. Download everything"),
(CODE, f"""
import shutil
print(shutil.make_archive("{env['zip_to']}", "zip", OUTPUT_DIR))      # download this file from the file browser
"""),
    ]

KAGGLE = [
(MD, """
# CAD Part Checker - Kaggle
**Settings (right panel):** Accelerator **GPU T4 x2 / P100** · Internet **ON** · *Add data*: a dataset containing the project folder `cad_part_checker`
(or `cad_part_checker.zip`) and, if it is not already inside `cad_part_checker/data/input`, a dataset with your files (`FG_*.png`, `BG_*.png`, `Mask_*.bmp`, `*.json`, `TestCase2/`).
The notebook finds both by itself."""),
(CODE, """
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
!nvidia-smi -L
!ls /kaggle/input
"""),
(MD, "## 1. Find the project and the data (automatic)"),
(CODE, """
import glob, json, os, shutil, subprocess

WORK = "/kaggle/working"
proj = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/cad_part_checker/run.py", recursive=True)]
zips = glob.glob("/kaggle/input/**/cad_part_checker.zip", recursive=True)
if proj:
    shutil.copytree(proj[0], f"{WORK}/cad_part_checker", dirs_exist_ok=True)
elif zips:
    subprocess.run(["unzip", "-q", "-o", zips[0], "-d", WORK], check=True)
else:
    raise SystemExit("Add the project as a Kaggle dataset (folder cad_part_checker or cad_part_checker.zip).")
%cd /kaggle/working/cad_part_checker


def has_cases(root):
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f.endswith(".json"):
                try:
                    d = json.load(open(os.path.join(dp, f)))
                except Exception:
                    continue
                if isinstance(d, dict) and any(isinstance(v, list) and v and isinstance(v[0], dict) and "StencilValue" in v[0] for v in d.values()):
                    return True
    return False


INPUT_DIR = None          # <- set a path here to override the automatic search
for cand in ["data/input"] + glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*") + glob.glob("/kaggle/input/*/*/*"):
    if INPUT_DIR is None and os.path.isdir(cand) and has_cases(cand):
        INPUT_DIR = os.path.abspath(cand)
OUTPUT_DIR = "/kaggle/working/output"
print("INPUT_DIR :", INPUT_DIR)
assert INPUT_DIR, "no folder with FG_*/BG_*/Mask_*/*.json found - add your data as a dataset"
"""),
(MD, "## 2. Install and check the environment"),
(CODE, """
!pip install -q -r requirements-colab.txt
!python scripts/check_env.py
"""),
] + COMMON_MODELS + tail({"zip_to": "/kaggle/working/results"})

COLAB = [
(MD, """
# CAD Part Checker - Google Colab
**Runtime -> Change runtime type -> T4 GPU.** Put `cad_part_checker.zip` and your data folder (the unzipped `AI_sample_CAD`: `FG_00.png`, `BG_*.png`, `Mask_*.bmp`, `0000.json`, `TestCase2/`) on Google Drive."""),
(CODE, """
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
!nvidia-smi -L
"""),
(MD, "## 1. Get the project and point to your data"),
(CODE, """
from google.colab import drive
drive.mount("/content/drive")
ZIP        = "/content/drive/MyDrive/cad_part_checker.zip"          # <-- edit
INPUT_DIR  = "/content/drive/MyDrive/cad_data/AI_sample_CAD"        # <-- edit: folder with FG_*/BG_*/Mask_*/*.json
OUTPUT_DIR = "/content/drive/MyDrive/cad_output"                    # results persist in Drive
!unzip -q -o "$ZIP" -d /content
%cd /content/cad_part_checker
"""),
(MD, "## 2. Install and check the environment"),
(CODE, """
!pip install -q -r requirements-colab.txt
!python scripts/check_env.py
"""),
] + COMMON_MODELS + tail({"zip_to": "/content/results"})


def build(cells, path):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == CODE:
            c.update(execution_count=None, outputs=[])
        out.append(c)
    nb = {"cells": out, "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}}
    json.dump(nb, open(path, "w"), indent=1)


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
    build(KAGGLE, os.path.join(root, "kaggle_run.ipynb"))
    build(COLAB, os.path.join(root, "colab_run.ipynb"))
    print("notebooks written")

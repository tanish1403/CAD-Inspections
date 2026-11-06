#!/usr/bin/env python3
"""Download pretrained weights into ./models (or $PARTCHECK_MODELS). Needs internet once; afterwards offline.

  python scripts/download_models.py                          # dinov2-base + sam-vit-base (~0.7 GB)
  python scripts/download_models.py --models dinov2-small    # lighter (CPU / small GPU)
  python scripts/download_models.py --models dinov2-large sam-vit-large
  python scripts/download_models.py --models none --vlm siglip smolvlm owlv2 grounding-dino   # optional verifiers
  python scripts/download_models.py --models none --hf_id Qwen/Qwen2-VL-7B-Instruct          # any HF repo
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from partcheck.models import ALIASES, MODELS_DIR
from partcheck.verifiers import ALIASES as V_ALIASES, AVAILABLE, DEFAULT_ID


def fetch(repo_id, dest):
    from huggingface_hub import snapshot_download
    print(f"Downloading {repo_id} -> {dest}")
    skip = ["*.bin", "*.pt", "*.pth", "*.onnx", "*.h5", "*.msgpack", "*.ot", "*.gguf", "*.tflite"]
    snapshot_download(repo_id=repo_id, local_dir=str(dest), ignore_patterns=skip)        # safetensors first
    if not any(dest.rglob("*.safetensors")):                                             # repo has only .bin
        snapshot_download(repo_id=repo_id, local_dir=str(dest))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dinov2-base", "sam-vit-base"],
                    help=f"core models: {', '.join(ALIASES)} | none")
    ap.add_argument("--vlm", nargs="+", default=[], help=f"optional verifiers: {', '.join(AVAILABLE)}")
    ap.add_argument("--hf_id", nargs="+", default=[], help="any Hugging Face repo id(s)")
    a = ap.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in [m for m in a.models if m != "none"]:
        if name not in ALIASES:
            sys.exit(f"unknown model '{name}'. choose from {list(ALIASES)}")
        fetch(ALIASES[name], MODELS_DIR / name)                 # used as --dino <name> / --sam_model <name>
    for v in a.vlm:
        rid = V_ALIASES[v][1] if v in V_ALIASES else DEFAULT_ID.get(v)
        if rid is None:
            sys.exit(f"unknown verifier '{v}'. choose from {AVAILABLE}")
        fetch(rid, MODELS_DIR / rid)                            # found automatically by --vlm (models/<org>/<name>)
    for rid in a.hf_id:
        fetch(rid, MODELS_DIR / rid)
    print("Done. Local copies in ./models are picked up automatically; no internet needed afterwards.")


if __name__ == "__main__":
    main()

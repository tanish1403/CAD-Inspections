#!/usr/bin/env python3
"""Tell me what hardware I have and which --profile to use."""
import os, platform, sys

def where():
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ: return "Kaggle"
    if "COLAB_GPU" in os.environ or "google.colab" in sys.modules or os.path.exists("/content/sample_data"): return "Google Colab"
    return "local machine"

print(f"Running on : {where()} | Python {platform.python_version()} | {platform.system()}")
try:
    import torch
except Exception as e:
    print("torch      : NOT installed / broken ->", e); sys.exit(0)
print("torch      :", torch.__version__)
rec = "low"
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0); gb = p.total_memory / 1e9
    rec = "high" if gb >= 20 else "balanced" if gb >= 8 else "low"
    print(f"GPU        : {p.name} ({gb:.1f} GB) x{torch.cuda.device_count()}")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("GPU        : Apple MPS")
else:
    print("GPU        : none  -> CPU only. For heavy work use Colab/Kaggle (see docs/COLAB_KAGGLE.md)")
print(f"Recommended: python run.py check --profile {rec}")
if rec == "low":
    print("Tip        : --method classical needs no models at all; --profile low = dinov2-small @518px")

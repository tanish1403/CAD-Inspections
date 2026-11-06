# No GPU? Run on Google Colab or Kaggle

The code picks its hardware automatically (`cuda` -> `mps` -> `cpu`) and `--profile auto` chooses model size from your
GPU memory. Three ways to work, from cheapest to strongest:

| Situation | What to do |
|---|---|
| No GPU, quick try | `python run.py check --method classical` (no models) or `--profile low` (DINOv2-small, 518 px, CPU OK) |
| Local GPU >= 8 GB | `python run.py check --profile auto` (balanced = DINOv2-base, 896 px) |
| No / weak GPU, want full quality, fine-tuning, VLMs | Colab or Kaggle notebooks below (free T4 / P100) |

Check what you have: `python scripts/check_env.py`.

## A) Google Colab (free T4, ~15 GB)
1. Upload `cad_part_checker.zip` and your data folder (the unzipped `AI_sample_CAD`: `FG_00.png`, `BG_*.png`, `Mask_*.bmp`, `0000.json`, `TestCase2/`) to **Google Drive**.
2. Open `notebooks/colab_run.ipynb` in Colab, **Runtime -> T4 GPU**.
3. Edit `ZIP`, `INPUT_DIR`, `OUTPUT_DIR` in the first cell, Run all. Results land in your Drive folder and are shown inline (original | result | CAD).
Sessions reset - models are re-downloaded each time (~1 min).

## B) Kaggle (free 2x T4 or P100, 30 GPU h/week)
1. *Datasets -> New Dataset*: upload the project **folder** `cad_part_checker` (your data may stay inside `cad_part_checker/data/input`) or `cad_part_checker.zip`. Optional second dataset for data.
2. New notebook -> *File -> Import notebook* -> `notebooks/kaggle_run.ipynb`; right panel: **Accelerator = GPU**, **Internet = ON**, *Add data* -> your dataset(s).
3. Run all. The notebook finds the project and the data by itself (`INPUT_DIR` can be overridden). Results: `/kaggle/working/output`, zipped as `results.zip`.
The code uses one GPU. Inputs are read-only (fine).

## C) Mixed workflow (recommended when GPU is scarce)
1. **Locally (CPU):** prepare data, test with `--method classical`, label a few parts in `labels.json`.
2. **On Colab/Kaggle (GPU):** run `bank` / `train` / `finetune` once -> download `models/bank.pkl`, `models/head.pkl` or `models/dinov2-ft/`.
3. **Locally:** `python run.py check --profile low --bank models/bank.pkl --head models/head.pkl` - the small heads and banks are tiny and CPU-friendly
   (use the same `--dino/--size/--gray` as when you built them).

## Memory tips
* Out of memory: `--profile low`, smaller `--size`, `--no_fp16` is *not* needed for memory (fp16 is already on for CUDA), skip `--sam`.
* Big VLMs: `--vlm_4bit` (needs `pip install bitsandbytes`) and `--vlm_uncertain_only` (runs them only on borderline parts).
* Fine-tuning on a T4: `--size 518 --unfreeze 2` (default settings); raise `--size`/`--unfreeze` only if you have >= 24 GB.

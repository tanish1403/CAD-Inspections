# Changelog

## 2.1 (this version) - fixes from the first Kaggle run, InternVL, new visualization, Streamlit app
**Fixed**
- `--vlm clip|siglip`: `'BaseModelOutputWithPooling' object has no attribute 'norm'` (newer transformers return a ModelOutput) -> `as_tensor()`.
- `--vlm owlv2`: `'Owlv2Processor' has no attribute 'post_process_object_detection'` -> version-proof `detections()` (also used for Grounding DINO).
- Notebook: `pip install -U bitsandbytes>=0.46.1` was a shell redirect -> quoted; `torch_dtype` deprecation -> `dtype`; progress-bar spam off.
- `float(loss)` on a tensor that requires grad (fine-tuning warning) -> `.detach()`.
- **Label pollution:** `bank` / `finetune --assume_present` treated the *edited* photos as complete (that made the bank and the fine-tune worse than
  zero-shot). Now only photos matched by `--ref_glob` are references; others use `--labels` or confident classical predictions. `--assume_present` / `--pseudo_labels` removed.
- Notebook paths: project and data are located automatically on Kaggle; one `INPUT_DIR`.

**Added**
- InternVL3 (`internvl3-1b|2b|8b`, Hugging Face `-hf` format) and any chat VLM via `vlm:<hf id>`; forced A/B prompt with more context (was Yes/No).
- Report card shows the **original test image**, the annotated result, the CAD and a status summary (`result.png`); the old 2x2 is `detail_2x2.png`.
- Streamlit app (`app.py`): analyse, browse saved results, part inspector, compare all photos, downloads.
- `scripts/benchmark.py` (choose the model with evidence), `scripts/make_report.py` (self-contained HTML), notebook helpers, `analyze()` API.
- `demo/output`: pre-computed results so the app can be shown without a GPU.

## 2.0
Folder-layout discovery for FG_/BG_/Mask_/json data, UNVERIFIABLE status, golden-photo "altered" flag, classical cues validated on real photos,
few-shot bank / head / fine-tuning, pluggable verifiers, hardware profiles, Colab/Kaggle notebooks.

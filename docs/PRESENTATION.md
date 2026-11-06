# Presentation guide

## 1. The pitch (one sentence)
"We compare the CAD model with a photo of the real assembly, part by part, using foundation models (DINOv2 for visual matching, optional VLMs as
second opinions) and tell you which parts are present, absent, uncertain or simply cannot be judged from this view."

## 2. Which model should I present?

**Evidence from your own Kaggle run (2x T4, `cad-ai-inspection.ipynb`)**

| Configuration | What happened on your photos |
|---|---|
| DINOv2-base zero-shot + classical cues | `TestImage`: both missing parts found. Seconds per photo, no training. `TestCase2` bolts: missed |
| + head (few-shot) | `TestImage` found; but `M8_nuts_3` falsely UNCERTAIN on 3 photos; leave-one-photo-out accuracy 0.80 |
| + bank / + fine-tune (v2.0) | *worse* - caused by the label-pollution bug (edited photos treated as complete). Fixed in v2.1, re-run before judging them |
| Qwen2-VL-7B 4-bit | `TestImage` found; `TestCase2` still missed; one false UNCERTAIN; ~46 s model load; heavy |
| SmolVLM | no change vs DINOv2 |
| CLIP / SigLIP / OWLv2 | crashed (fixed in v2.1, not yet evaluated) |

**Recommendation**

* **Headline model: DINOv2-base (zero-shot) + the classical cues + the golden photo** (`python run.py check --profile auto --gray --golden auto`).
  Why: it is the configuration that is *proven on your data*, runs in seconds on a free T4, needs no training or labels, and every number
  is explainable (edge match, inner structure, patch similarity, pixel change vs golden). Present it as "foundation-model features + transparent geometric checks".
* **Second opinion to show the VLM story: Qwen2-VL** (already proven to load and run on your T4s, 4-bit for 7B) **or InternVL3-2B** (new, small, T4-friendly, likely
  stronger at fine-grained visual reasoning - but *unproven here*). Use `--vlm_uncertain_only` so it only looks at borderline parts - cheap and a good narrative:
  "the fast model decides, the big model double-checks what it is unsure about".
* **Do not lead with** fine-tuning or the head: with 5 photos they cannot be validated (say "future work with more labelled data").
* **Do not claim** the `TestCase2` empty holes are solved - no tested model solves them yet. Present it as the honest limit and the reason for the VLM / few-shot roadmap.

**Pick with evidence in 20 minutes:** `python scripts/benchmark.py --input_dir ... --truth examples/ground_truth.json` (notebook section 8) prints accuracy, recall on absent parts,
uncertain count, and seconds for every configuration. Whichever row has the best *absent recall* without more false alarms is your headline; the numbers go straight on a slide.
Edit `examples/ground_truth.json` first - it contains my reading of your images, check it.

| criterion | DINOv2 + classical | + Qwen2-VL | + InternVL3-2B |
|---|---|---|---|
| proven on your data | yes | runs; no gain yet | not yet |
| speed / GPU memory | seconds / ~3 GB | slow / ~8 GB (4-bit 7B) | medium / ~6 GB |
| explainable | high | medium | medium |
| needs training | no | no | no |
| presentation value | solid, honest | "VLM in the loop" | "newest VLM" |

## 3. Demo script (6-8 minutes)
1. **Problem (1 min):** CAD vs photo, which parts are really there? Show `docs/sample_results/` or the app.
2. **How it works (2 min):** one diagram - CAD+mask -> per-part regions -> DINOv2 patch matching + outline/structure checks -> (optional VLM on borderline parts) -> status. Mention statuses and why UNVERIFIABLE exists.
3. **Live (2 min):** `streamlit run app.py` -> *Browse saved results* (instant) -> `TestImage` (absent, red), `TestImage2/3` (altered, orange). Then *Analyse a photo* if a GPU is available.
4. **Honesty slide (1 min):** TestCase2 empty holes are missed; hidden parts are not guessed; thresholds were tuned on one object.
5. **PS2 + roadmap (1 min):** `docs/architecture_PS2.md` - pose + dimension verification; more labelled data; VLM verifier; fine-tuning.

## 4. Questions you may get
* *Why not just train a detector?* One CAD, 5 photos - foundation models + CAD priors instead of a labelled dataset; training is optional (`train`, `bank`, `finetune`).
* *Why can parts be "unverifiable"?* The CAD mask shows 0 px or a hairline for them (hidden behind others); guessing would create false alarms.
* *What does "altered" mean?* Structure is present but the pixels differ a lot from the complete photo (tampered / damaged). Needs a golden photo from the same setup.
* *How do you know it works?* `scripts/benchmark.py` against labelled photos; current numbers in README. Be explicit about the small sample.
* *What about different viewpoints?* The CAD render and photo are assumed to share the camera pose; otherwise add pose estimation (PS2 document).

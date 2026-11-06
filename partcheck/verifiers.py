"""
Optional second-opinion models ("verifiers"). Enable with  --vlm <name> [<name> ...]

Each verifier answers: "is this CAD part visible in the real photo at the expected place?" -> score in [0, 1].
Scores are blended into the final decision (see --vlm_weight). Nothing here is imported unless you ask for it,
so the base install stays light.

Built-in names (all Hugging Face models, downloaded on first use):
  clip, clip-large, siglip, siglip-large     image/image (+ text with --use_names) similarity, counterfactual
  owlv2, owlv2-large                         open-vocabulary detector, needs --use_names (meaningful part names)
  grounding-dino                             open-vocabulary detector, needs --use_names
  smolvlm, smolvlm-256m, qwen2-vl, qwen2.5-vl, llava-ov, internvl3-1b/2b/8b     chat VLMs asked an A/B question
Any model id:        --vlm vlm:Qwen/Qwen2-VL-7B-Instruct      (kinds: clip, siglip, owlv2, grounding-dino, vlm)
Your own class:      --vlm my_plugin.py:MyVerifier            (subclass Verifier, see README)
Swap checkpoint:     --vlm_model clip=openai/clip-vit-large-patch14
"""
import importlib.util
import os
import re
import cv2
import numpy as np
from .data import bbox
from .models import pick_device, resolve


# ----------------------------------------------------------------- helpers
def crop(img, region, pad_frac=0.15, min_pad=8):
    x0, y0, x1, y1 = [int(v) for v in bbox(region)]
    p = max(min_pad, int(pad_frac * max(x1 - x0, y1 - y0)))
    H, W = img.shape[:2]
    return img[max(y0 - p, 0):min(y1 + p + 1, H), max(x0 - p, 0):min(x1 + p + 1, W)]


def to_pil(bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def erase_region(img, region):
    """Counterfactual 'same photo without the part': fill the region with the border colour."""
    border = np.concatenate([img[:5].reshape(-1, 3), img[-5:].reshape(-1, 3)])
    out = img.copy()
    out[cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = np.median(border, 0)
    return out


def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))


def pretty(name):
    return re.sub(r"[_\-]+", " ", name).strip()


def as_tensor(out):
    """Embedding tensor from whatever `get_*_features` returns (tensor, or a ModelOutput in newer transformers)."""
    if hasattr(out, "shape"):
        return out
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        x = getattr(out, k, None)
        if x is not None:
            return x
    h = getattr(out, "last_hidden_state", None)
    if h is not None:
        return h[:, 0]
    if isinstance(out, (tuple, list)) and out:
        return as_tensor(out[0])
    raise TypeError(f"cannot extract an embedding from {type(out).__name__}")


def detections(processor, outputs, target_sizes, threshold, input_ids=None, text_labels=None):
    """-> [(xyxy box, score)] from a detector output, across transformers versions
    (post_process_object_detection on the processor or its image_processor, or the grounded variant)."""
    ip = getattr(processor, "image_processor", None)
    kw = dict(outputs=outputs, threshold=threshold, target_sizes=target_sizes)
    attempts = [(processor, "post_process_object_detection", kw), (ip, "post_process_object_detection", kw)]
    gk = dict(kw, **({"text_labels": text_labels} if text_labels else {}))
    if input_ids is not None:
        gk = dict(gk, input_ids=input_ids, text_threshold=threshold)
    attempts += [(processor, "post_process_grounded_object_detection", gk),
                 (processor, "post_process_grounded_object_detection",
                  {("box_threshold" if k == "threshold" else k): x for k, x in gk.items()})]
    errs = []
    for obj, name, k in attempts:
        fn = getattr(obj, name, None) if obj is not None else None
        if fn is None:
            continue
        try:
            r = fn(**k)[0]
            return [([float(c) for c in b], float(s)) for b, s in zip(r["boxes"], r["scores"])]
        except TypeError as e:
            errs.append(str(e))
    raise RuntimeError("no compatible detector post-processing found: " + " | ".join(errs[-2:]))


def iou_xyxy(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ----------------------------------------------------------------- base class
class Verifier:
    """Subclass this to plug in your own model.  __init__(model_id, device, **kw); prepare(); score()."""
    name = "verifier"
    needs_names = False
    default_model = None

    def __init__(self, model_id=None, device=None, use_names=False, **kw):
        self.model_id = model_id or self.default_model
        self.device = device
        self.use_names = use_names
        self.kw = kw

    def prepare(self, cad, real_al):
        """Called once per sample with the CAD image and the real image aligned to it (BGR uint8)."""
        self.cad, self.real = cad, real_al

    def score(self, part_name, region):
        """region = boolean HxW mask of the part in the CAD frame. Return float in [0,1]; 1 = present."""
        raise NotImplementedError


# ----------------------------------------------------------------- CLIP / SigLIP
class EmbedVerifier(Verifier):
    """Image/image similarity with a counterfactual: is the real crop closer to the CAD crop than the
    same real crop with the part erased?  With --use_names it also does text zero-shot."""
    default_model = "openai/clip-vit-base-patch32"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import torch
        from transformers import AutoModel, AutoProcessor
        self.t = torch
        self.dev = pick_device(self.device)
        path = resolve(self.model_id)
        self.m = AutoModel.from_pretrained(path).to(self.dev).eval()
        self.p = AutoProcessor.from_pretrained(path)

    def _img(self, crops):
        t = self.t
        inp = self.p(images=[to_pil(c) for c in crops], return_tensors="pt").to(self.dev)
        with t.no_grad():
            f = as_tensor(self.m.get_image_features(pixel_values=inp["pixel_values"]))
            return t.nn.functional.normalize(f.float(), dim=-1)

    def _txt(self, texts):
        t = self.t
        inp = self.p(text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.dev)
        with t.no_grad():
            return t.nn.functional.normalize(as_tensor(self.m.get_text_features(**inp)).float(), dim=-1)

    def score(self, part_name, region):
        e = self._img([crop(self.cad, region), crop(self.real, region), crop(erase_region(self.real, region), region)])
        diff = float((e[0] * e[1]).sum() - (e[0] * e[2]).sum())
        s = sigmoid((diff - 0.02) / 0.03)
        if self.use_names:
            txt = self._txt([f"a photo of a {pretty(part_name)}", "a photo of an empty plain background"])
            if txt.shape[-1] == e.shape[-1]:                 # skip if this checkpoint's text/image spaces differ
                s = 0.5 * (s + float((100 * e[1] @ txt.T).softmax(-1)[0][0]))
        return s


# ----------------------------------------------------------------- open-vocabulary detectors
class Owlv2Verifier(Verifier):
    needs_names = True
    default_model = "google/owlv2-base-patch16-ensemble"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.t = torch
        self.dev = pick_device(self.device)
        path = resolve(self.model_id)
        self.m = Owlv2ForObjectDetection.from_pretrained(path).to(self.dev).eval()
        self.p = Owlv2Processor.from_pretrained(path)

    def prepare(self, cad, real_al):
        super().prepare(cad, real_al)
        self._cache = {}

    def score(self, part_name, region):
        t, (H, W) = self.t, self.real.shape[:2]
        q = pretty(part_name)
        if q not in self._cache:
            inp = self.p(text=[[q]], images=to_pil(self.real), return_tensors="pt").to(self.dev)
            with t.no_grad():
                out = self.m(**inp)
            side = max(H, W)                   # OWLv2 pads the image to a square
            self._cache[q] = detections(self.p, out, t.tensor([[side, side]]), 0.05, text_labels=[[q]])
        box = [float(v) for v in bbox(region)]
        best = max([sc * min(1.0, iou_xyxy(b, box) / 0.5) for b, sc in self._cache[q]] or [0.0])
        return min(1.0, best / 0.3)             # OWLv2 scores are small; 0.3 ~ confident


class GroundingDinoVerifier(Verifier):
    needs_names = True
    default_model = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.t = torch
        self.dev = pick_device(self.device)
        path = resolve(self.model_id)
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(path).to(self.dev).eval()
        self.p = AutoProcessor.from_pretrained(path)

    def score(self, part_name, region):
        t, (H, W) = self.t, self.real.shape[:2]
        img = to_pil(self.real)
        inp = self.p(images=img, text=pretty(part_name).lower() + ".", return_tensors="pt").to(self.dev)
        with t.no_grad():
            out = self.m(**inp)
        dets = detections(self.p, out, [(H, W)], 0.1, input_ids=inp.input_ids, text_labels=[[pretty(part_name)]])
        box = [float(c) for c in bbox(region)]
        return float(max([s * min(1.0, iou_xyxy(b, box) / 0.5) for b, s in dets] or [0.0]))


# ----------------------------------------------------------------- chat VLMs (A/B logit)
class YesNoVLM(Verifier):
    """Shows the model [CAD crop with the part outlined | real crop at the same place] and asks a forced A/B question;
    score = P(A: present) vs P(B: missing).  Works with any chat VLM that has a Hugging Face chat template
    (SmolVLM, Qwen2-VL/2.5-VL, LLaVA-OneVision, InternVL3-hf, ...)."""
    default_model = "HuggingFaceTB/SmolVLM-Instruct"

    def __init__(self, *a, load_4bit=False, trust_remote_code=False, **kw):
        super().__init__(*a, **kw)
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as Auto
        except ImportError:
            from transformers import AutoModelForVision2Seq as Auto
        self.t = torch
        self.dev = pick_device(self.device)
        path = resolve(self.model_id)
        kwargs = dict(trust_remote_code=trust_remote_code)
        dtype = torch.float16 if self.dev == "cuda" else torch.float32
        if load_4bit and self.dev == "cuda":    # needs `pip install -U "bitsandbytes>=0.46.1"`; fits 7-8B VLMs on a 16 GB T4
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            kwargs["device_map"] = "auto"
        try:
            self.m = Auto.from_pretrained(path, dtype=dtype, **kwargs).eval()          # transformers >= 4.56
        except TypeError:
            self.m = Auto.from_pretrained(path, torch_dtype=dtype, **kwargs).eval()    # older versions
        if "device_map" not in kwargs:
            self.m.to(self.dev)
        self.p = AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code)
        tok = self.p.tokenizer
        ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[-1] for w in ws})
        self.A, self.B = ids(["A", " A"]), ids(["B", " B"])

    def _composite(self, region):
        ref = self.cad.copy()
        cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ref, cnts, -1, (0, 220, 0), 3)
        a, b = crop(ref, region, 0.4), crop(self.real, region, 0.4)           # generous context around the part
        h = 384
        a = cv2.resize(a, (max(8, int(a.shape[1] * h / a.shape[0])), h))
        b = cv2.resize(b, (max(8, int(b.shape[1] * h / b.shape[0])), h))
        return to_pil(np.hstack([a, np.full((h, 12, 3), 255, np.uint8), b]))

    def question(self, part_name):
        what = f"the part '{pretty(part_name)}'" if self.use_names else "the part"
        return (f"The left picture is a CAD reference where {what} is outlined in green. The right picture is a photo of "
                f"the real assembly at the same place. Is the outlined part also physically present in the photo, "
                f"i.e. the same kind of object at the same position?\n"
                f"A: yes, it is present.\n"
                f"B: no - the spot is empty (bare surface or an empty hole) or holds a different object.\n"
                f"Answer with a single letter.")

    def score(self, part_name, region):
        t = self.t
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.question(part_name)}]}]
        text = self.p.apply_chat_template(msgs, add_generation_prompt=True)
        img = self._composite(region)
        try:
            inp = self.p(text=[text], images=[img], return_tensors="pt")
        except Exception:                        # some processors (Idefics3/SmolVLM) want nested lists
            inp = self.p(text=[text], images=[[img]], return_tensors="pt")
        inp = inp.to(self.m.device, self.m.dtype)
        with t.no_grad():
            lg = self.m(**inp).logits[0, -1].float()
        return float(t.sigmoid(lg[self.A].max() - lg[self.B].max()))


# ----------------------------------------------------------------- registry
KINDS = {"clip": EmbedVerifier, "siglip": EmbedVerifier, "owlv2": Owlv2Verifier,
         "grounding-dino": GroundingDinoVerifier, "vlm": YesNoVLM}
DEFAULT_ID = {"clip": "openai/clip-vit-base-patch32", "siglip": "google/siglip-base-patch16-224",
              "owlv2": "google/owlv2-base-patch16-ensemble", "grounding-dino": "IDEA-Research/grounding-dino-tiny",
              "vlm": "HuggingFaceTB/SmolVLM-Instruct"}
ALIASES = {  # name -> (kind, hf id)
    "clip-large": ("clip", "openai/clip-vit-large-patch14"),
    "siglip-large": ("siglip", "google/siglip-large-patch16-256"),
    "owlv2-large": ("owlv2", "google/owlv2-large-patch14-ensemble"),
    "smolvlm": ("vlm", "HuggingFaceTB/SmolVLM-Instruct"),
    "smolvlm-256m": ("vlm", "HuggingFaceTB/SmolVLM-256M-Instruct"),
    "qwen2-vl": ("vlm", "Qwen/Qwen2-VL-2B-Instruct"),
    "qwen2.5-vl": ("vlm", "Qwen/Qwen2.5-VL-3B-Instruct"),
    "llava-ov": ("vlm", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"),
    # InternVL in the Hugging Face-native "-hf" format (no remote code needed; needs a recent transformers)
    "internvl3-1b": ("vlm", "OpenGVLab/InternVL3-1B-hf"),
    "internvl3-2b": ("vlm", "OpenGVLab/InternVL3-2B-hf"),
    "internvl3-8b": ("vlm", "OpenGVLab/InternVL3-8B-hf"),
}
AVAILABLE = sorted(set(KINDS) | set(ALIASES))


def _safe(s):
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def load_custom(spec, **kw):
    path, cls = spec.rsplit(":", 1)
    mspec = importlib.util.spec_from_file_location(_safe(os.path.basename(path)), path)
    mod = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mod)
    v = getattr(mod, cls)(**kw)
    if getattr(v, "name", "verifier") == "verifier":
        v.name = _safe(cls)
    return v


def build_verifiers(specs, overrides=None, device=None, use_names=False, load_4bit=False, trust_remote_code=False):
    """specs: list of strings (see module docstring). overrides: {"clip": "hf/id"}."""
    out, overrides = [], overrides or {}
    for spec in specs or []:
        kw = dict(device=device, use_names=use_names)
        if ".py:" in spec:
            v = load_custom(spec, model_id=overrides.get(spec), **kw)
        else:
            if ":" in spec:
                kind, mid = spec.split(":", 1)
            elif spec in ALIASES:
                kind, mid = ALIASES[spec]
            elif spec in KINDS:
                kind, mid = spec, DEFAULT_ID[spec]
            else:
                raise SystemExit(f"unknown --vlm '{spec}'. Available: {', '.join(AVAILABLE)}  "
                                 f"| kind:hf_id | file.py:Class")
            mid = overrides.get(spec, mid)
            extra = dict(load_4bit=load_4bit, trust_remote_code=trust_remote_code) if kind == "vlm" else {}
            v = KINDS[kind](model_id=mid, **kw, **extra)
            v.name = _safe(spec)
        if v.needs_names and not use_names:
            print(f"[warn] verifier '{v.name}' needs meaningful part names - add --use_names")
        out.append(v)
    return out

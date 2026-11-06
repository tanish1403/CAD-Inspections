"""Pretrained backbones: DINOv2 (dense features), SAM (box-prompted masks), plus a test dummy."""
import os
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("PARTCHECK_MODELS", ROOT / "models"))
ALIASES = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "dinov2-giant": "facebook/dinov2-giant",
    "dinov2-reg-small": "facebook/dinov2-with-registers-small",
    "dinov2-reg-base": "facebook/dinov2-with-registers-base",
    "dinov2-reg-large": "facebook/dinov2-with-registers-large",
    "sam-vit-base": "facebook/sam-vit-base",
    "sam-vit-large": "facebook/sam-vit-large",
    "sam-vit-huge": "facebook/sam-vit-huge",
}


PROFILES = {   # hardware presets; explicit CLI flags override them
    "low":      dict(dino="dinov2-small", size=518, sam=False),    # CPU / small GPU / free Colab-Kaggle fallback
    "balanced": dict(dino="dinov2-base", size=896, sam=False),     # 8-16 GB GPU (T4, P100)
    "high":     dict(dino="dinov2-large", size=1204, sam=True),    # >= 20 GB GPU
}


def auto_profile():
    """Pick a profile from the hardware that is actually available."""
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return "high" if gb >= 20 else "balanced" if gb >= 8 else "low"
    except Exception:
        pass
    return "low"


def resolve(name):
    """Existing folder path -> as is; models/<name> if downloaded; else the Hugging Face repo id (auto-cached)."""
    if os.path.isdir(str(name)):
        return str(name)
    local = MODELS_DIR / name
    if local.is_dir() and any(local.iterdir()):
        return str(local)
    return ALIASES.get(name, name)


def pick_device(device=None):
    import torch
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DinoBackbone:
    """DINOv2 -> (g, g, C) L2-normalised patch-feature grid (patch size 14). Works with register variants too."""
    def __init__(self, name="dinov2-base", size=896, device=None, fp16=None):
        import torch
        from transformers import AutoModel
        self.t, self.size = torch, size - size % 14
        self.g = self.size // 14
        self.dev = pick_device(device)
        self.fp16 = (self.dev == "cuda") if fp16 is None else (bool(fp16) and self.dev == "cuda")
        self.m = AutoModel.from_pretrained(resolve(name)).to(self.dev).eval()

    def _input(self, bgr, gray):
        t = self.t
        if gray:  # drops colour -> smaller CAD-render vs photo gap
            bgr = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (self.size, self.size),
                       interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return t.from_numpy(x.transpose(2, 0, 1)[None]).float().to(self.dev)

    def tensor_features(self, bgr, gray=False):
        """Torch (g, g, C) features WITH autograd (used by fine-tuning)."""
        t = self.t
        with t.autocast(device_type="cuda", dtype=t.float16, enabled=self.fp16):
            h = self.m(pixel_values=self._input(bgr, gray)).last_hidden_state[:, -self.g * self.g:, :]
        f = t.nn.functional.normalize(h[0].float(), dim=-1)      # last g*g tokens = patches (skips CLS/registers)
        return f.reshape(self.g, self.g, -1)

    def __call__(self, bgr, gray=False):
        with self.t.no_grad():
            return self.tensor_features(bgr, gray).cpu().numpy()


class DummyBackbone:
    """Raw pixel-patch features. NOT AI - only for tests / running without model weights."""
    def __init__(self, size=448):
        self.size, self.g = size, size // 14

    def __call__(self, bgr, gray=True):
        g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (self.size, self.size),
                       interpolation=cv2.INTER_AREA).astype(np.float32) - 128
        f = g.reshape(self.g, 14, self.g, 14).transpose(0, 2, 1, 3).reshape(self.g, self.g, 196)
        return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-6)


class Sam:
    """Box-prompted SAM; iou(region) = best IoU between SAM's masks and the CAD part region."""
    def __init__(self, name="sam-vit-base", device=None):
        import torch
        from transformers import SamModel, SamProcessor
        self.t = torch
        self.dev = pick_device(device)
        path = resolve(name)
        self.m = SamModel.from_pretrained(path).to(self.dev).eval()
        self.p = SamProcessor.from_pretrained(path)

    def set_image(self, bgr):
        from .data import bbox  # noqa
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.p(self.rgb, return_tensors="pt").to(self.dev)
        with self.t.no_grad():
            self.emb = self.m.get_image_embeddings(inp["pixel_values"])

    def iou(self, region):
        from .data import bbox
        x0, y0, x1, y1 = [int(v) for v in bbox(region)]
        inp = self.p(self.rgb, input_boxes=[[[x0, y0, x1, y1]]], return_tensors="pt").to(self.dev)
        with self.t.no_grad():
            out = self.m(image_embeddings=self.emb, input_boxes=inp["input_boxes"], multimask_output=True)
        ms = self.p.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0][0].numpy()
        return float(max((m & region).sum() / max((m | region).sum(), 1) for m in ms))

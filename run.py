#!/usr/bin/env python3
"""CAD-vs-real part presence checker.

  python run.py check     [options]   run the check (zero-shot by default)
  python run.py train     [options]   few-shot: learn a small head            -> models/head.pkl
  python run.py bank      [options]   few-shot: build real-image prototypes   -> models/bank.pkl
  python run.py finetune  [options]   fine-tune last DINOv2 blocks (GPU)      -> models/dinov2-ft/
Run `python run.py <command> --help` for all options.   Hardware presets: --profile auto|low|balanced|high
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from partcheck.models import PROFILES, DinoBackbone, DummyBackbone, Sam, auto_profile
from partcheck.verifiers import AVAILABLE


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        g = p.add_argument_group("data / hardware")
        g.add_argument("--input_dir", default="data/input")
        g.add_argument("--profile", choices=["auto", *PROFILES], default="auto",
                       help="auto picks from your GPU memory: low (CPU/small GPU) | balanced | high")
        g.add_argument("--device", default=None, help="cuda | mps | cpu (auto by default)")
        g.add_argument("--no_fp16", action="store_true", help="disable half precision on CUDA")
        g.add_argument("--align", choices=["none", "ecc", "silhouette"], default="none",
                       help="none: photos share the CAD camera (default) | ecc: small refinement | silhouette: plain-background photos")
        g.add_argument("--max_side", type=int, default=2048, help="work at this longest side (speed); parts need enough pixels")
        g.add_argument("--min_thickness_px", type=int, default=6, help="thinner visible parts (hairlines) are UNVERIFIABLE")
        g.add_argument("--min_area_px", type=int, default=400,
                       help="parts with fewer visible pixels (at --max_side) are reported UNVERIFIABLE")
        g = p.add_argument_group("input discovery (defaults fit FG_*/BG_*/Mask_*/*.json folders)")
        g.add_argument("--real_glob", help="restrict which images are the real photos, e.g. 'BG_00_Test*'")
        g.add_argument("--cad_glob", help="CAD image pattern (default FG_* | *CAD*)")
        g.add_argument("--mask_glob", help="mask image pattern (default: file named in the json, else *mask*)")
        g.add_argument("--json_glob", help="part-map json pattern")
        g = p.add_argument_group("backbone (override the profile)")
        g.add_argument("--method", choices=["ai", "classical"], default="ai",
                       help="classical = coverage+edge only, no models at all (CPU friendly)")
        g.add_argument("--dino", default=None, help="dinov2-small|base|large|giant|dinov2-reg-*|<folder>|<hf id>")
        g.add_argument("--size", type=int, default=None, help="DINOv2 input size (multiple of 14)")
        g.add_argument("--sam", action="store_true", default=None, help="add SAM box-prompt cue")
        g.add_argument("--sam_model", default="sam-vit-base")
        g.add_argument("--gray", action="store_true", help="grayscale input: reduces CAD-vs-photo colour gap")
        g.add_argument("--backbone", choices=["dino", "dummy"], default="dino",
                       help="'dummy' = weight-free test backbone (not AI)")
        g = p.add_argument_group("zero-shot fusion")
        g.add_argument("--thr", type=float, default=0.5, help="'present' threshold")
        g.add_argument("--m0", type=float, default=0.10, help="DINOv2 margin midpoint")
        g.add_argument("--tau", type=float, default=0.06, help="DINOv2 margin softness")
        g.add_argument("--ai_weight", type=float, default=0.4, help="weight of the DINOv2/SAM cue vs classical cues")

    def learn(p):
        g = p.add_argument_group("learning data")
        g.add_argument("--labels", help='json {"<prefix>": {"<part>": 1|0}} labels for your real photos')
        g.add_argument("--iters", type=int, default=30, help="synthetic variants per sample")
        g.add_argument("--ref_glob", help="photos that are COMPLETE (all verifiable parts present), matched on 'case/image' or "
                                          "'image', e.g. 'input/BG_00' - never point it at edited/test photos")

    c = sub.add_parser("check", help="run presence check on all samples in input_dir")
    common(c)
    c.add_argument("--output_dir", default="data/output")
    c.add_argument("--parts", help="comma-separated part names to check (default all)")
    g = c.add_argument_group("decision")
    g.add_argument("--uncertain_band", type=float, default=0.15, help="|score - thr| below this -> UNCERTAIN")
    g = c.add_argument_group("optional upgrades")
    g.add_argument("--golden", help="a complete real photo of the SAME setup (path) or 'auto' = first non-'test' image "
                                    "of each folder; flags parts that changed (tampered/damaged)")
    g.add_argument("--golden_weight", type=float, default=0.0, help="also blend golden similarity into the score (0 = flag only)")
    g.add_argument("--altered_thr", type=float, default=0.5, help="golden correlation below this = 'altered'")
    g.add_argument("--head", help="trained head from `train`")
    g.add_argument("--bank", help="few-shot prototype bank from `bank`")
    g.add_argument("--ref_weight", type=float, default=0.3, help="weight of the bank cue")
    g = c.add_argument_group("optional VLM / grounding verifiers")
    g.add_argument("--vlm", nargs="+", metavar="NAME", help=f"verifier(s): {', '.join(AVAILABLE)} | kind:hf_id | file.py:Class")
    g.add_argument("--vlm_model", nargs="+", metavar="NAME=HF_ID", help="override a verifier's checkpoint")
    g.add_argument("--vlm_weight", type=float, default=0.3, help="weight of verifier scores in the final score")
    g.add_argument("--vlm_uncertain_only", action="store_true", help="only run verifiers on borderline parts (faster)")
    g.add_argument("--vlm_band", type=float, default=0.2, help="'borderline' = |score - threshold| <= band")
    g.add_argument("--vlm_4bit", action="store_true", help='load chat VLMs in 4-bit (pip install -U "bitsandbytes>=0.46.1")')
    g.add_argument("--trust_remote_code", action="store_true", help="allow custom code from a VLM repo (e.g. native InternVL2.5)")
    g.add_argument("--use_names", action="store_true", help="part names are meaningful words -> use them in prompts")

    t = sub.add_parser("train", help="few-shot: train a small head on synthetic variants of your samples")
    common(t); learn(t)
    t.add_argument("--out_head", default="models/head.pkl")

    b = sub.add_parser("bank", help="few-shot: build per-part real-image prototypes from reference photos")
    common(b); learn(b)
    b.add_argument("--out_bank", default="models/bank.pkl")

    f = sub.add_parser("finetune", help="fine-tune the last DINOv2 blocks on your samples (GPU recommended)")
    common(f); learn(f)
    f.add_argument("--out_model", default="models/dinov2-ft")
    f.add_argument("--steps", type=int, default=300)
    f.add_argument("--lr", type=float, default=1e-5)
    f.add_argument("--unfreeze", type=int, default=2, help="number of last transformer blocks to train")
    return ap


def apply_profile(a):
    name = auto_profile() if a.profile == "auto" else a.profile
    p = PROFILES[name]
    for k, v in p.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    a.profile_used = name
    return a


def components(a):
    """Build the heavy objects once: (backbone, SAM, verifiers, bank). Reused by the CLI, the Streamlit app, notebooks."""
    bb = None
    if a.method == "ai":
        bb = DummyBackbone() if a.backbone == "dummy" else DinoBackbone(a.dino, a.size, a.device, not a.no_fp16)
    sam = Sam(a.sam_model, a.device) if (a.sam and bb is not None and a.backbone != "dummy") else None
    ver, bank = [], None
    if getattr(a, "vlm", None):
        from partcheck.verifiers import build_verifiers
        ov = dict(x.split("=", 1) for x in (a.vlm_model or []))
        ver = build_verifiers(a.vlm, ov, a.device, a.use_names, a.vlm_4bit, a.trust_remote_code)
    if getattr(a, "bank", None):
        from partcheck.train import load_bank
        bank = load_bank(a.bank, a)
        if bb is None:
            sys.exit("--bank needs --method ai")
    return bb, sam, ver, bank


def main(argv=None):
    a = apply_profile(build_parser().parse_args(argv))
    if a.method == "ai" and a.backbone == "dino":
        print(f"[profile={a.profile_used}] dino={a.dino} size={a.size} sam={bool(a.sam)}")
    if a.cmd == "check":
        from partcheck.pipeline import run
        bb, sam, ver, bank = components(a)
        run(a, bb, sam, ver, bank)
        return
    if a.method != "ai":
        sys.exit(f"`{a.cmd}` needs --method ai")
    bb, sam, _, _ = components(a)
    from partcheck import train as T
    {"train": T.train_head, "bank": T.build_bank, "finetune": T.finetune}[a.cmd](a, bb, sam)


if __name__ == "__main__":
    main()

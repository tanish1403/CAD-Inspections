"""
Template for plugging in YOUR OWN model as an optional verifier.

    python run.py check --vlm examples/my_verifier_example.py:MyVerifier --vlm_weight 0.4

Contract
  __init__(model_id, device, use_names, **kw)  -> load your model (model_id comes from --vlm_model if given)
  prepare(cad, real_al)                        -> called once per sample; both are BGR uint8 images in the SAME frame
  score(part_name, region)                     -> float in [0,1]; region = boolean mask of the part (CAD frame)
The returned score is stored as  vlm_<classname>  in report.json and blended into the final decision.
"""
import numpy as np
from partcheck.verifiers import Verifier, crop


class MyVerifier(Verifier):
    name = "my_verifier"
    default_model = None            # e.g. a Hugging Face id or a local path

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # self.model = ...load your model here...

    def score(self, part_name, region):
        real_crop = crop(self.real, region)       # what the photo shows where the part should be
        cad_crop = crop(self.cad, region)         # what the CAD shows there
        # Replace with your model. Toy example: mean absolute colour difference -> similarity
        d = np.abs(real_crop.astype(float).mean() - cad_crop.astype(float).mean()) / 255
        return float(1 - d)

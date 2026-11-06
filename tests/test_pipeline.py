"""Smoke tests on synthetic data (no model weights needed):  python -m pytest -q tests"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_synthetic_sample import make_folder, make_prefix
import run as runner


def report(out, case, stem):
    return json.load(open(os.path.join(out, case, stem, "report.json")))


def status(rep, st):
    return sorted(n for n, r in rep.items() if r["status"] == st)


def test_folder_layout_classical(tmp_path):
    inp, out = str(tmp_path / "case"), str(tmp_path / "out")
    make_folder(inp)
    runner.main(["check", "--method", "classical", "--input_dir", inp, "--output_dir", out, "--golden", "auto"])
    c = "case"
    full = report(out, c, "BG_00")
    assert status(full, "ABSENT") == []
    assert status(full, "UNVERIFIABLE") == ["hidden_nut", "rim"]            # no pixels / hairline -> not guessed
    missing = report(out, c, "BG_00_TestImage")
    assert missing["bolt_b"]["status"] == "ABSENT"
    assert missing["washer_b"]["status"] in ("ABSENT", "UNCERTAIN")
    assert status(missing, "ABSENT") + status(missing, "UNCERTAIN") and "bolt_a" not in status(missing, "ABSENT")
    tampered = report(out, c, "BG_00_TestImage2")                           # scribbled but still there
    assert status(tampered, "ABSENT") == [] and tampered["bolt_c"].get("altered")
    for f in ("result.png", "present_cad.png", "absent_real.png", "report.csv"):
        assert os.path.exists(os.path.join(out, c, "BG_00_TestImage", f))
    assert os.path.exists(os.path.join(out, "summary.csv"))


def test_prefix_layout_classical(tmp_path):
    inp, out = str(tmp_path / "in"), str(tmp_path / "out")
    truth = make_prefix(inp)
    runner.main(["check", "--method", "classical", "--input_dir", inp, "--output_dir", out])
    rep = report(out, "in", "demo")
    assert sorted(set(truth) & set(status(rep, "ABSENT"))) == sorted(truth)


def test_explicit_file_overrides(tmp_path):
    inp, out = str(tmp_path / "case"), str(tmp_path / "out")
    make_folder(inp)
    runner.main(["check", "--method", "classical", "--input_dir", inp, "--output_dir", out,
                 "--real_glob", "BG_00_TestImage.png", "--parts", "bolt_b,bolt_a"])
    rep = report(out, "case", "BG_00_TestImage")
    assert set(rep) == {"bolt_a", "bolt_b"} and rep["bolt_b"]["status"] == "ABSENT"
    assert not os.path.exists(os.path.join(out, "case", "BG_00"))


PLUGIN = '''
from partcheck.verifiers import Verifier
class OracleStub(Verifier):
    """Pretend VLM: everything present except bolt_b."""
    def score(self, part_name, region):
        return 0.0 if part_name == "bolt_b" else 1.0
'''


def test_custom_verifier_plugin(tmp_path):
    inp, out = str(tmp_path / "case"), str(tmp_path / "out")
    make_folder(inp)
    plug = tmp_path / "my_plugin.py"
    plug.write_text(PLUGIN)
    runner.main(["check", "--method", "classical", "--input_dir", inp, "--output_dir", out, "--real_glob",
                 "BG_00.png", "--vlm", f"{plug}:OracleStub", "--vlm_weight", "1.0", "--uncertain_band", "0.0"])
    rep = report(out, "case", "BG_00")                                      # plugin overrides: bolt_b absent
    assert rep["bolt_b"]["status"] == "ABSENT" and rep["bolt_b"]["vlm_oraclestub"] == 0.0


def test_uncertain_only_skips_confident_parts(tmp_path):
    inp, out = str(tmp_path / "case"), str(tmp_path / "out")
    make_folder(inp)
    plug = tmp_path / "my_plugin.py"
    plug.write_text(PLUGIN)
    runner.main(["check", "--method", "classical", "--input_dir", inp, "--output_dir", out, "--real_glob", "BG_00.png",
                 "--vlm", f"{plug}:OracleStub", "--vlm_uncertain_only", "--vlm_band", "0.0"])
    assert all("vlm_oraclestub" not in r for r in report(out, "case", "BG_00").values())


def test_head_and_bank_roundtrip(tmp_path):
    inp, out = str(tmp_path / "case"), str(tmp_path / "out")
    head, bank = str(tmp_path / "head.pkl"), str(tmp_path / "bank.pkl")
    make_folder(inp)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"case/BG_00_TestImage": {"bolt_b": 0, "washer_b": 0}}))
    common = ["--backbone", "dummy", "--input_dir", inp]
    runner.main(["train", *common, "--out_head", head, "--labels", str(labels), "--iters", "6"])
    runner.main(["bank", *common, "--out_bank", bank, "--labels", str(labels), "--iters", "3"])
    runner.main(["check", *common, "--output_dir", out, "--head", head, "--bank", bank, "--real_glob", "BG_00_TestImage.png"])
    rep = report(out, "case", "BG_00_TestImage")
    assert "ref_margin" in rep["bolt_a"] and rep["bolt_b"]["status"] in ("ABSENT", "UNCERTAIN")


def test_profiles_and_registry():
    from partcheck.models import PROFILES
    from partcheck.verifiers import AVAILABLE, build_verifiers
    a = runner.apply_profile(runner.build_parser().parse_args(["check", "--profile", "low", "--size", "700"]))
    assert a.dino == PROFILES["low"]["dino"] and a.size == 700
    assert {"clip", "siglip", "owlv2", "grounding-dino", "smolvlm", "qwen2-vl"} <= set(AVAILABLE)
    try:
        build_verifiers(["no-such-model"])
        assert False
    except SystemExit as e:
        assert "unknown --vlm" in str(e)


# ---------------------------------------------------------------- report card, ref_glob, verifier helpers, app
def test_report_card_contains_original(tmp_path):
    import cv2
    inp, out = str(tmp_path / "case"), str(tmp_path / "out")
    make_folder(inp)
    runner.main(["check", "--method", "classical", "--input_dir", inp, "--output_dir", out, "--real_glob", "BG_00_TestImage.png"])
    d = os.path.join(out, "case", "BG_00_TestImage")
    card, orig = cv2.imread(os.path.join(d, "result.png")), cv2.imread(os.path.join(d, "original.png"))
    assert card is not None and orig is not None and card.shape[0] > card.shape[1] // 4          # panels + summary text
    assert os.path.exists(os.path.join(d, "detail_2x2.png"))
    import numpy as np
    h = 640                                     # the left panel of the card is the untouched photo (below its banner)
    left = card[40:h, 5:150]
    ref = cv2.resize(orig, (int(orig.shape[1] * h / orig.shape[0]), h))[40:h, 5:150]
    assert np.abs(left.astype(int) - ref.astype(int)).mean() < 12


def test_ref_glob_marks_only_selected_photos_complete(tmp_path):
    import argparse
    from partcheck.data import Sample
    from partcheck.train import is_reference
    a = argparse.Namespace(ref_glob="case/BG_00")
    assert is_reference(a, Sample("case", "BG_00", "", "", "", ""))
    assert not is_reference(a, Sample("case", "BG_00_TestImage", "", "", "", ""))
    assert not is_reference(argparse.Namespace(ref_glob=None), Sample("case", "BG_00", "", "", "", ""))


def test_bank_with_reference_only(tmp_path):
    inp, bank = str(tmp_path / "case"), str(tmp_path / "bank.pkl")
    make_folder(inp)
    runner.main(["bank", "--backbone", "dummy", "--input_dir", inp, "--out_bank", bank, "--ref_glob", "BG_00", "--iters", "3"])
    import pickle
    b = pickle.load(open(bank, "rb"))["bank"]
    assert len(b["bolt_b"]["pos"]) >= 1 and len(b["bolt_b"]["neg"]) >= 1      # erased variants supply the negatives


class _Out:                       # stand-in for transformers' ModelOutput
    def __init__(self, **k): self.__dict__.update(k)


def test_embedding_extraction_handles_new_transformers_outputs():
    import numpy as np
    from partcheck.verifiers import as_tensor
    t = np.ones((2, 8))
    assert as_tensor(t) is t
    assert as_tensor(_Out(pooler_output=t, last_hidden_state=np.zeros((2, 3, 8)))) is t
    assert as_tensor(_Out(image_embeds=t, pooler_output=None)) is t
    assert as_tensor(_Out(pooler_output=None, last_hidden_state=np.arange(48).reshape(2, 3, 8)))[0, 0] == 0
    assert as_tensor((t,)) is t


def test_detector_postprocessing_across_versions():
    from partcheck.verifiers import detections

    class NewProc:                                 # only has the grounded variant (the error seen on Kaggle)
        def post_process_grounded_object_detection(self, outputs, threshold, target_sizes, text_labels=None, **k):
            return [{"boxes": [[0, 0, 5, 5]], "scores": [0.9]}]

    class OldProc:
        def post_process_object_detection(self, outputs, threshold, target_sizes):
            return [{"boxes": [[1, 1, 4, 4]], "scores": [0.5]}]

    class WeirdProc:                               # keyword is box_threshold in some versions
        def post_process_grounded_object_detection(self, outputs, input_ids, box_threshold, text_threshold, target_sizes):
            return [{"boxes": [[2, 2, 3, 3]], "scores": [0.7]}]

    assert detections(NewProc(), None, [(1, 1)], 0.1, text_labels=[["x"]])[0][1] == 0.9
    assert detections(OldProc(), None, [(1, 1)], 0.1)[0][0] == [1.0, 1.0, 4.0, 4.0]
    assert detections(WeirdProc(), None, [(1, 1)], 0.1, input_ids=[1])[0][1] == 0.7


def test_internvl_is_registered():
    from partcheck.verifiers import ALIASES, AVAILABLE
    assert {"internvl3-1b", "internvl3-2b", "internvl3-8b"} <= set(AVAILABLE)
    assert ALIASES["internvl3-2b"] == ("vlm", "OpenGVLab/InternVL3-2B-hf")


def test_streamlit_app_runs(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setenv("CADAPP_OUTPUT", str(tmp_path / "app_out"))
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    inp = str(tmp_path / "case")
    make_folder(inp)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()
    assert not at.exception
    [t for t in at.sidebar.text_input if t.label == "Input folder"][0].set_value(inp).run()
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert [m.label for m in at.metric][:2] == ["🟢 Present", "🔴 Absent"]
    at.sidebar.button[1].click().run()                       # run all photos of the case -> comparison tab
    assert not at.exception, at.exception


def test_streamlit_browse_mode_shows_demo_results(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value("Browse saved results")
    at.sidebar.text_input[0].set_value(os.path.join(ROOT, "demo", "output")).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 5 and int(at.metric[0].value) > 0


def test_demo_results_are_complete():
    import glob
    reps = glob.glob(os.path.join(ROOT, "demo", "output", "**", "report.json"), recursive=True)
    assert len(reps) == 5
    assert all(os.path.exists(os.path.join(os.path.dirname(r), "result.jpg")) for r in reps)


def test_golden_auto_ignores_real_glob(tmp_path):
    inp, out = str(tmp_path / "case"), str(tmp_path / "out")
    make_folder(inp)
    runner.main(["check", "--method", "classical", "--input_dir", inp, "--output_dir", out, "--golden", "auto",
                 "--real_glob", "BG_00_TestImage2.png"])
    assert report(out, "case", "BG_00_TestImage2")["bolt_c"].get("altered")

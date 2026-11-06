PY ?= python
install:  ; $(PY) -m pip install -r requirements.txt
env:      ; $(PY) scripts/check_env.py
models:   ; $(PY) scripts/download_models.py
demo:     ; $(PY) scripts/make_synthetic_sample.py --out data/input && $(PY) run.py check --method classical
test:     ; $(PY) -m pytest -q tests

PYTHON ?= python

setup:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -e .[dev]

lint:
	$(PYTHON) -m ruff src tests

format:
	$(PYTHON) -m black src tests

test:
	$(PYTHON) -m pytest -q

train:
	$(PYTHON) -m access_perm.train --config config/default.yaml

report:
	$(PYTHON) -m access_perm.report --config config/default.yaml

pipeline:
	$(PYTHON) -m access_perm.pipeline train --config config/default.yaml

monitor:
	$(PYTHON) -m access_perm.pipeline monitor --config config/default.yaml

serve:
	$(PYTHON) -m uvicorn access_perm.service:app --host 0.0.0.0 --port 8000

.PHONY: setup lint format test train report pipeline monitor serve

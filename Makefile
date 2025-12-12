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

.PHONY: setup lint format test train report

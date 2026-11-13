ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: test data

test:
	$(PY) -m pytest -q

# Rebuild data/prompts.jsonl from Hugging Face. Free: downloads datasets, calls no model.
data:
	$(PY) eval/build_dataset.py

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: test data smoke pilot

# 50 dev prompts into the real caches, then the gap check: make pilot ARGS=--yes
pilot:
	$(PY) eval/run_jev.py --split dev --limit 50 $(ARGS)
	$(PY) eval/run_models.py --split dev --limit 50 $(ARGS)
	$(PY) eval/gap_check.py

# 10 prompts end to end with the real Jev API and local models, capped at $0.50.
# Prints the estimate and stops unless confirmed: make smoke ARGS=--yes
smoke:
	$(PY) eval/smoke.py $(ARGS)

test:
	$(PY) -m pytest -q

# Rebuild data/prompts.jsonl from Hugging Face. Free: downloads datasets, calls no model.
data:
	$(PY) eval/build_dataset.py

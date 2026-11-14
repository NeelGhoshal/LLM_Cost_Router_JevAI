ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: test data smoke pilot eval phrasings checks results

# Full run: make eval ARGS=--yes. Resume-safe: cached calls are skipped.
eval:
	$(PY) eval/run_models.py $(ARGS)
	$(PY) eval/run_jev.py $(ARGS)
	$(PY) eval/run_checks.py $(ARGS)
	$(MAKE) results

# Compare the candidate cascade checks on dev: make checks ARGS=--yes
# Then set cascade_checks in config.yaml before running make eval.
checks:
	$(PY) eval/run_checks.py --candidates $(ARGS)
	$(PY) eval/pick_check.py

# Compare the candidate Jev phrasings on dev: make phrasings ARGS=--yes
# Then set jev_phrasing in config.yaml before running make eval.
phrasings:
	$(PY) eval/run_phrasings.py $(ARGS)
	$(PY) eval/pick_phrasing.py

# Rebuild results/ from the caches. No API calls.
results:
	$(PY) eval/simulate.py
	$(PY) eval/plot.py

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

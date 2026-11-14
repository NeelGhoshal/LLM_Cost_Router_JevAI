"""Compare the candidate Jev phrasings on the dev split (no API calls).

Ranks them by AUROC for predicting "the small model is correct". Put the winner's name in
config.yaml as jev_phrasing, then run eval/run_jev.py. The test split is never read here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.cache import load_jsonl  # noqa: E402
from evalkit.cli import utf8_stdout  # noqa: E402
from router.config import load_config  # noqa: E402
from router.jev import PHRASINGS  # noqa: E402
from evalkit.runs import CACHE_DIR, load_prompts  # noqa: E402
from evalkit.simulate import auroc  # noqa: E402


def main() -> None:
    utf8_stdout()
    cfg = load_config()
    small = {r["id"]: r["correct"] for r in load_jsonl(CACHE_DIR / "llm.jsonl") if r["role"] == "small"}
    jev = {r["id"]: r["probs"] for r in load_jsonl(CACHE_DIR / "jev_phrasings_dev.jsonl")}
    dev = [p for p in load_prompts(split="dev") if p["id"] in small and p["id"] in jev]
    if not dev:
        sys.exit("need cache/llm.jsonl and cache/jev_phrasings_dev.jsonl for the dev split")

    y = np.array([small[p["id"]] for p in dev], bool)
    length = np.array([len(p["prompt"]) for p in dev], float)
    rng = np.random.default_rng(cfg.seed)
    resamples = [rng.integers(0, len(dev), len(dev)) for _ in range(1000)]

    def with_ci(scores: np.ndarray) -> str:
        draws = [auroc(scores[i], y[i]) for i in resamples]
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        return f"{auroc(scores, y):.3f}  [{lo:.3f}, {hi:.3f}]"

    print(f"dev split, {len(dev)} prompts, small-model accuracy {y.mean():.3f}\n")
    print(f"{'phrasing':14s} {'AUROC':>6s}  95% CI            p range")
    best_name, best = None, -1.0
    for name in PHRASINGS:
        scores = np.array([jev[p["id"]][name] for p in dev], float)
        a = auroc(scores, y)
        if a > best:
            best_name, best = name, a
        print(f"{name:14s} {with_ci(scores)}    {scores.min():.2f} to {scores.max():.2f}")
    print(f"{'prompt length':14s} {with_ci(-length)}    (baseline, shorter = easier)")
    print(f"\nhighest dev AUROC: {best_name}. config.yaml currently has jev_phrasing: {cfg.jev_phrasing}")


if __name__ == "__main__":
    main()

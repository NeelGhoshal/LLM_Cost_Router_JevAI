"""Compare the candidate cascade checks on the dev split (no API calls).

Ranks each check, and the maximum over all of them, by AUROC for "the small model's answer is
wrong" among the responses Jev was asked about. Put the winner in config.yaml as cascade_checks,
then run eval/run_checks.py. The test split is never read here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.cache import load_jsonl  # noqa: E402
from evalkit.cli import utf8_stdout  # noqa: E402
from evalkit.runs import CACHE_DIR, load_prompts  # noqa: E402
from evalkit.simulate import auroc  # noqa: E402
from router.config import load_config  # noqa: E402
from router.jev import CHECKS  # noqa: E402


def main() -> None:
    utf8_stdout()
    cfg = load_config()
    small = {r["id"]: r for r in load_jsonl(CACHE_DIR / "llm.jsonl") if r["role"] == "small"}
    checks = {r["id"]: r for r in load_jsonl(CACHE_DIR / "jev_checks_dev.jsonl")}
    dev = [p for p in load_prompts(split="dev") if p["id"] in checks]
    if not dev:
        sys.exit("cache/jev_checks_dev.jsonl is empty. Run: eval/run_checks.py --candidates --yes")

    wrong = np.array([not small[p["id"]]["correct"] for p in dev], bool)
    response_chars = np.array([len(small[p["id"]]["text"]) for p in dev], float)
    rng = np.random.default_rng(cfg.seed)
    resamples = [rng.integers(0, len(dev), len(dev)) for _ in range(1000)]

    def line(name: str, scores: np.ndarray) -> float:
        a = auroc(scores, wrong)
        lo, hi = np.nanpercentile([auroc(scores[i], wrong[i]) for i in resamples], [2.5, 97.5])
        print(f"{name:26s} {a:.3f}  [{lo:.3f}, {hi:.3f}]")
        return a

    n_dev = len(load_prompts(split="dev"))
    print(f"dev split: {len(dev)} of {n_dev} responses checked ({n_dev - len(dev)} had no ANSWER line and escalate anyway)")
    print(f"small model wrong on {wrong.mean():.1%} of the checked responses\n")
    print(f"{'check':26s} AUROC  95% CI   (for 'small answer is wrong')")
    per_check = {name: np.array([checks[p["id"]]["probs"][name] for p in dev], float) for name in CHECKS}
    for name, scores in per_check.items():
        line(name, scores)
    line("max of all checks", np.max(list(per_check.values()), axis=0))
    line("response length (baseline)", response_chars)
    tokens = np.mean([checks[p["id"]]["input_tokens"] for p in dev])
    print(f"\nmean Jev input tokens per call with all {len(CHECKS)} checks: {tokens:.0f}")
    print(f"config.yaml currently has cascade_checks: {list(cfg.cascade_checks)}")


if __name__ == "__main__":
    main()

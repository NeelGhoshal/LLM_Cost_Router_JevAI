"""Gap check from the caches (no API calls): is the small model far enough behind the large one?

If the gap is within about 8 accuracy points, routing is too easy to be interesting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.cache import load_jsonl  # noqa: E402
from evalkit.cli import accuracy, utf8_stdout  # noqa: E402
from router.config import load_config  # noqa: E402
from evalkit.runs import CACHE_DIR  # noqa: E402

MIN_GAP_POINTS = 8.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR / "llm.jsonl")
    args = parser.parse_args()
    utf8_stdout()
    cfg = load_config()

    rows = load_jsonl(args.cache)
    by = {(r["id"], r["role"]): r for r in rows}
    ids = sorted({i for i, role in by if (i, "small") in by and (i, "large") in by})
    if not ids:
        sys.exit("no prompts with both models cached. Run eval/run_models.py first.")
    small = [by[(i, "small")] for i in ids]
    large = [by[(i, "large")] for i in ids]

    print(f"{len(ids)} prompts with both models cached\n")
    print(f"{'source':10s} {'n':>4s} {'small':>7s} {'large':>7s} {'gap':>7s}")
    for source in sorted({r["source"] for r in small}) + ["all"]:
        s = [r for r in small if source in ("all", r["source"])]
        l = [r for r in large if source in ("all", r["source"])]
        print(f"{source:10s} {len(s):4d} {accuracy(s):7.3f} {accuracy(l):7.3f} {100 * (accuracy(l) - accuracy(s)):+6.1f}p")

    print()
    for name, mine in (("small", small), ("large", large)):
        no_answer = sum(r["answer"] is None for r in mine)
        capped = sum(r["output_tokens"] >= cfg.max_output_tokens for r in mine)
        mean_out = sum(r["output_tokens"] for r in mine) / len(mine)
        mean_cost = sum(r["cost"] for r in mine) / len(mine)
        print(
            f"{name} ({mine[0]['model']}): {no_answer} with no ANSWER line, {capped} at the token cap, "
            f"mean output {mean_out:.0f} tokens, mean cost ${mean_cost * 1000:.4f} per 1,000 queries"
        )

    both = sum(s["correct"] and l["correct"] for s, l in zip(small, large))
    only_large = sum(l["correct"] and not s["correct"] for s, l in zip(small, large))
    only_small = sum(s["correct"] and not l["correct"] for s, l in zip(small, large))
    neither = len(ids) - both - only_large - only_small
    print(f"\nboth right {both}, only large {only_large}, only small {only_small}, neither {neither}")
    print(f"oracle accuracy {(both + only_large + only_small) / len(ids):.3f}, routing to large needed on {only_large / len(ids):.1%} of prompts")

    gap = 100 * (accuracy(large) - accuracy(small))
    verdict = "wide enough" if gap > MIN_GAP_POINTS else "TOO NARROW: propose harder sources or a weaker small model"
    print(f"\ngap {gap:+.1f} points against a {MIN_GAP_POINTS:.0f}-point minimum: {verdict}")


if __name__ == "__main__":
    main()

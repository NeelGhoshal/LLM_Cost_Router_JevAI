"""Ask Jev once per prompt whether the small model is enough, and cache the probability."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from evalkit.cli import base_parser, cost_gate, progress, select_prompts  # noqa: E402
from router.config import load_config  # noqa: E402
from router.costs import BudgetGuard  # noqa: E402
from router.jev import PHRASINGS, QUESTION_KEY, TypeSafeJev  # noqa: E402
from evalkit.runs import CACHE_DIR, estimate, run_jev  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR / "jev.jsonl")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config()
    prompts = select_prompts(args)
    max_usd = cfg.max_usd if args.max_usd is None else args.max_usd
    cost_gate(estimate(prompts, cfg, models=False), len(prompts), max_usd, args.yes, cfg)

    guard = BudgetGuard(max_usd)
    question = {QUESTION_KEY: PHRASINGS[cfg.jev_phrasing]}
    made = run_jev(
        prompts, cfg, TypeSafeJev(cfg.jev_model), args.cache, guard, question, progress, phrasing=cfg.jev_phrasing
    )
    print(f"{made} new calls, real spend ${guard.spent:.6f}")


if __name__ == "__main__":
    main()

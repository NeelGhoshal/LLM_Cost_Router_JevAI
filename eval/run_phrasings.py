"""Ask Jev all candidate phrasings (router.jev.PHRASINGS) on the dev split only, one request per prompt.

Questions in one request are evaluated in isolation, so this costs one round trip per prompt.
The test split is never sent here. Compare the results with eval/pick_phrasing.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from evalkit.cli import base_parser, cost_gate, progress, select_prompts  # noqa: E402
from router.config import load_config  # noqa: E402
from router.costs import BudgetGuard  # noqa: E402
from router.jev import PHRASINGS, TypeSafeJev  # noqa: E402
from evalkit.runs import CACHE_DIR, estimate, run_jev  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR / "jev_phrasings_dev.jsonl")
    args = parser.parse_args()
    args.split = "dev"  # phrasings are tuned on dev only

    load_dotenv()
    cfg = load_config()
    prompts = select_prompts(args)
    max_usd = cfg.max_usd if args.max_usd is None else args.max_usd
    est = estimate(prompts, cfg, models=False, jev_questions=len(PHRASINGS))
    cost_gate(est, len(prompts), max_usd, args.yes, cfg)

    guard = BudgetGuard(max_usd)
    made = run_jev(prompts, cfg, TypeSafeJev(cfg.jev_model), args.cache, guard, PHRASINGS, progress)
    print(f"{made} new calls, real spend ${guard.spent:.6f}")


if __name__ == "__main__":
    main()

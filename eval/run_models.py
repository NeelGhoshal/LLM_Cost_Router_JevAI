"""Send every prompt to the small and the large model once and cache the graded results."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from router.cache import load_jsonl  # noqa: E402
from router.cli import accuracy, base_parser, cost_gate, progress, select_prompts  # noqa: E402
from router.config import load_config  # noqa: E402
from router.costs import BudgetGuard  # noqa: E402
from router.llm import make_llm_client  # noqa: E402
from router.runs import CACHE_DIR, estimate, run_models  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR / "llm.jsonl")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config()
    prompts = select_prompts(args)
    max_usd = cfg.max_usd if args.max_usd is None else args.max_usd
    cost_gate(estimate(prompts, cfg, jev_questions=0), len(prompts), max_usd, args.yes, cfg)

    clients = {"small": make_llm_client(cfg.small_model, cfg), "large": make_llm_client(cfg.large_model, cfg)}
    guard = BudgetGuard(max_usd)
    made = run_models(prompts, cfg, clients, args.cache, guard, progress)

    ids = {p["id"] for p in prompts}
    rows = [r for r in load_jsonl(args.cache) if r["id"] in ids]
    print(f"{made} new calls, real spend ${guard.spent:.4f}")
    for role in ("small", "large"):
        mine = [r for r in rows if r["role"] == role]
        print(f"  {role}: accuracy {accuracy(mine):.3f} over {len(mine)}")


if __name__ == "__main__":
    main()

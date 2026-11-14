"""Cascade checks: ask Jev about the small model's cached response to each prompt.

  --candidates   dev split only, every check in router.jev.CHECKS in one request
                 -> cache/jev_checks_dev.jsonl. Compare them with eval/pick_check.py.
  (default)      all prompts, only the checks named in config.yaml (cascade_checks)
                 -> cache/jev_cascade.jsonl. This is what the simulation replays.

Needs cache/llm.jsonl (run eval/run_models.py first). Responses with no ANSWER line are skipped:
the cascade escalates those without calling Jev.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from evalkit.cache import load_jsonl  # noqa: E402
from evalkit.cli import base_parser, cost_gate, progress, select_prompts  # noqa: E402
from evalkit.runs import CACHE_DIR, estimate, run_checks  # noqa: E402
from router.config import load_config  # noqa: E402
from router.costs import BudgetGuard  # noqa: E402
from router.jev import CHECKS, TypeSafeJev  # noqa: E402

RESPONSE_TOKENS = 250  # generous allowance for the response in the state, for the estimate only


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--candidates", action="store_true", help="compare all candidate checks, dev split only")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config()
    if args.candidates:
        args.split = "dev"  # candidates are compared on dev only; test is never sent
        checks, cache = dict(CHECKS), CACHE_DIR / "jev_checks_dev.jsonl"
    else:
        checks, cache = {name: CHECKS[name] for name in cfg.cascade_checks}, CACHE_DIR / "jev_cascade.jsonl"
    tag = "+".join(checks)

    prompts = select_prompts(args)
    small = {r["id"]: r for r in load_jsonl(CACHE_DIR / "llm.jsonl") if r["role"] == "small"}
    missing = [p["id"] for p in prompts if p["id"] not in small]
    if missing:
        sys.exit(f"{len(missing)} prompts have no cached small-model response. Run eval/run_models.py first.")

    max_usd = cfg.max_usd if args.max_usd is None else args.max_usd
    est = estimate(prompts, cfg, models=False, jev_questions=len(checks), response_tokens=RESPONSE_TOKENS)
    print(f"checks: {tag}")
    cost_gate(est, len(prompts), max_usd, args.yes, cfg)

    guard = BudgetGuard(max_usd)
    made = run_checks(prompts, small, cfg, TypeSafeJev(cfg.jev_model), cache, guard, checks, progress, tag=tag)
    skipped = sum(small[p["id"]]["answer"] is None for p in prompts)
    print(f"{made} new calls, {skipped} responses skipped (no ANSWER line), real spend ${guard.spent:.6f}")


if __name__ == "__main__":
    main()

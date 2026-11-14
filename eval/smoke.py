"""Smoke test: 10 dev prompts end to end with the real Jev API and the real LLMs. Hard cap $0.50.

Writes to cache/smoke/, which is separate from the evaluation caches and is deleted at the start of each run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from evalkit.cache import load_jsonl  # noqa: E402
from evalkit.cli import accuracy, cost_gate, progress, utf8_stdout  # noqa: E402
from router.config import load_config  # noqa: E402
from router.costs import BudgetGuard  # noqa: E402
from router.jev import PHRASINGS, QUESTION_KEY, TypeSafeJev  # noqa: E402
from router.llm import make_llm_client  # noqa: E402
from evalkit.runs import CACHE_DIR, estimate, load_prompts, run_jev, run_models, spread  # noqa: E402

N_PROMPTS = 10
MAX_USD = 0.50
SMOKE_DIR = CACHE_DIR / "smoke"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="required to call the real APIs")
    args = parser.parse_args()

    utf8_stdout()
    load_dotenv()
    cfg = load_config()
    prompts = spread(load_prompts(split="dev"), N_PROMPTS)
    if not prompts:
        sys.exit("no prompts found. Run: make data")
    cost_gate(estimate(prompts, cfg), len(prompts), MAX_USD, args.yes, cfg)

    jev = TypeSafeJev(cfg.jev_model)
    available = jev.list_models()  # sends no text
    print(f"Jev key OK. Models: {', '.join(available)}")
    if cfg.jev_model not in available:
        # The list only names aliases, so a pinned version is checked by the first real call.
        print(f"note: {cfg.jev_model} is not in the list; the first Jev call will show whether it is accepted")

    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    guard = BudgetGuard(MAX_USD)
    # Jev first: it is fast, so a bad key or model ID fails before the slow local runs.
    question = {QUESTION_KEY: PHRASINGS[cfg.jev_phrasing]}
    run_jev(prompts, cfg, jev, SMOKE_DIR / "jev.jsonl", guard, question, progress, phrasing=cfg.jev_phrasing)
    clients = {"small": make_llm_client(cfg.small_model, cfg), "large": make_llm_client(cfg.large_model, cfg)}
    run_models(prompts, cfg, clients, SMOKE_DIR / "llm.jsonl", guard, progress)

    llm = load_jsonl(SMOKE_DIR / "llm.jsonl")
    jev_rows = {r["id"]: r for r in load_jsonl(SMOKE_DIR / "jev.jsonl")}
    by = {(r["id"], r["role"]): r for r in llm}

    print(f"\n{'id':16s} {'p_small_ok':>10s} {'small':>6s} {'large':>6s} {'jev_ms':>7s}")
    for p in prompts:
        j, s, l = jev_rows[p["id"]], by[(p["id"], "small")], by[(p["id"], "large")]
        print(f"{p['id']:16s} {j['probs'][QUESTION_KEY]:10.3f} {str(s['correct']):>6s} {str(l['correct']):>6s} {j['latency_ms']:7.0f}")

    for role in ("small", "large"):
        mine = [r for r in llm if r["role"] == role]
        no_answer = sum(r["answer"] is None for r in mine)
        capped = sum(r["output_tokens"] >= cfg.max_output_tokens for r in mine)
        print(
            f"{role} ({mine[0]['model']}): accuracy {accuracy(mine):.2f}, "
            f"{no_answer} with no ANSWER line, {capped} hit the {cfg.max_output_tokens}-token cap"
        )
    print(f"Jev model reported: {sorted({r['model'] for r in jev_rows.values()})}")
    print(f"real spend ${guard.spent:.6f} (cap ${MAX_USD:.2f})")


if __name__ == "__main__":
    main()

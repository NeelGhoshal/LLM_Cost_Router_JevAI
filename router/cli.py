"""Shared pieces of the eval scripts: arguments, the cost gate, and progress output."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from router.config import Config
from router.runs import Estimate, load_prompts, spread


def utf8_stdout() -> None:
    """Model output contains characters the default Windows console encoding cannot print."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def base_parser(description: str) -> argparse.ArgumentParser:
    utf8_stdout()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--split", choices=["dev", "test"], default=None, help="default: both")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts, spread across sources")
    parser.add_argument("--max-usd", type=float, default=None, help="default: max_usd in config.yaml")
    parser.add_argument("--yes", action="store_true", help="required to start a run that calls real APIs")
    return parser


def select_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompts = load_prompts(split=args.split)
    if not prompts:
        sys.exit("no prompts found. Run: make data")
    return spread(prompts, args.limit) if args.limit else prompts


def cost_gate(est: Estimate, n_prompts: int, max_usd: float, yes: bool, cfg: Config) -> None:
    """Print the estimate. Stop unless it fits the cap and --yes was given."""
    print(f"{n_prompts} prompts: {est.describe()}; cap ${max_usd:.2f}")
    if est.real > max_usd:
        sys.exit(
            f"estimate is over the cap. Lower n_prompts (not below 300) or max_output_tokens "
            f"(now {cfg.max_output_tokens}) in config.yaml, or raise --max-usd."
        )
    if not yes:
        sys.exit("not started. Re-run with --yes to proceed.")


def progress(msg: str) -> None:
    print(f"  {msg}", flush=True)


def accuracy(rows: Sequence[dict[str, Any]]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")

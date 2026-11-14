from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from router.config import Config, load_config
from router.console import utf8_stdout
from router.costs import jev_cost, llm_cost
from router.llm import make_llm_client
from router.route import cascade, route


def main() -> None:
    parser = argparse.ArgumentParser(prog="router", description="Answer one prompt through the Jev router.")
    parser.add_argument("prompt")
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="small model answers first, Jev checks the answer, large model only if the check fires",
    )
    parser.add_argument("--threshold", type=float, default=None, help="default: the value in config.yaml")
    args = parser.parse_args()

    load_dotenv()
    utf8_stdout()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    (run_cascade if args.cascade else run_preroute)(cfg, args.prompt, args.threshold)


def run_preroute(cfg: Config, prompt: str, threshold: float | None) -> None:
    d = route(prompt, threshold)
    p = "n/a" if d.p_small_ok is None else f"{d.p_small_ok:.3f}"
    print(f"decision: {d.model}  p_small_ok={p}  jev_latency={d.jev_latency_ms:.0f}ms")
    if d.fallback_reason:
        print(f"fallback: {d.fallback_reason}")

    spec = cfg.small_model if d.model == cfg.small_model.id else cfg.large_model
    r = make_llm_client(spec, cfg).complete(spec.id, prompt)
    print(f"\n{r.text}\n")
    print_cost(cfg, llm_cost(spec, r.input_tokens, r.output_tokens), d.jev_input_tokens)


def run_cascade(cfg: Config, prompt: str, threshold: float | None) -> None:
    r = cascade(prompt, threshold)
    print(f"answered by: {r.model}  ({r.reason})  jev_latency={r.jev_latency_ms:.0f}ms")
    print(f"\n{r.text}\n")
    usd = llm_cost(cfg.small_model, r.small.input_tokens, r.small.output_tokens)
    if r.large:
        usd += llm_cost(cfg.large_model, r.large.input_tokens, r.large.output_tokens)
    seconds = (r.small.latency_ms + r.jev_latency_ms + (r.large.latency_ms if r.large else 0)) / 1000
    print_cost(cfg, usd, r.jev_input_tokens)
    print(f"time: {seconds:.1f}s")


def print_cost(cfg: Config, llm_usd: float, jev_input_tokens: int) -> None:
    jev_usd = jev_cost(jev_input_tokens, cfg.jev_usd_per_m_input)
    kind = "imputed" if cfg.small_model.provider == "ollama" else "billed"
    print(f"cost: llm ${llm_usd:.6f} ({kind}) + jev ${jev_usd:.6f} = ${llm_usd + jev_usd:.6f}")


if __name__ == "__main__":
    main()

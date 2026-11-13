from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from router.config import load_config
from router.costs import jev_cost, llm_cost
from router.llm import make_llm_client
from router.route import route


def main() -> None:
    parser = argparse.ArgumentParser(prog="router", description="Route one prompt with Jev and answer it.")
    parser.add_argument("prompt")
    parser.add_argument("--threshold", type=float, default=None, help="default: threshold in config.yaml")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()

    d = route(args.prompt, args.threshold)
    p = "n/a" if d.p_small_ok is None else f"{d.p_small_ok:.3f}"
    print(f"decision: {d.model}  p_small_ok={p}  jev_latency={d.jev_latency_ms:.0f}ms")
    if d.fallback_reason:
        print(f"fallback: {d.fallback_reason}")

    spec = cfg.small_model if d.model == cfg.small_model.id else cfg.large_model
    r = make_llm_client(spec, cfg).complete(spec.id, args.prompt)
    print(f"\n{r.text}\n")

    llm_usd = llm_cost(spec, r.input_tokens, r.output_tokens)
    jev_usd = jev_cost(d.jev_input_tokens, cfg.jev_usd_per_m_input)
    kind = "imputed" if spec.provider == "ollama" else "billed"
    print(f"cost: llm ${llm_usd:.6f} ({kind}) + jev ${jev_usd:.6f} = ${llm_usd + jev_usd:.6f}")


if __name__ == "__main__":
    main()

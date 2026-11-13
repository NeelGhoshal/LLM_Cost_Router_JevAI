"""Build data/prompts.jsonl from the Hugging Face sources. Free: no model calls."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.config import REPO_ROOT, load_config  # noqa: E402
from router.dataset import load_sources, sample_and_split  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "prompts.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cfg = load_config()
    rows = sample_and_split(load_sources(), cfg.n_prompts, cfg.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter((r["source"], r["split"]) for r in rows)
    print(f"wrote {len(rows)} prompts to {args.out}")
    for (source, split), n in sorted(counts.items()):
        print(f"  {source:9s} {split:4s} {n}")


if __name__ == "__main__":
    main()

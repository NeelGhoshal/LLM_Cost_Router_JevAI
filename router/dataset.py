from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

LETTERS = "ABCDEFGHIJ"

INSTRUCTION = "Reason briefly, then finish with a final line: ANSWER: <letter or number>"

# source name -> Hugging Face (path, config, split)
SOURCES: dict[str, tuple[str, str | None, str]] = {
    "mmlu": ("cais/mmlu", "all", "test"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test"),
}


def _multiple_choice(question: str, options: Sequence[str]) -> str:
    lines = [f"{LETTERS[i]}. {str(opt).strip()}" for i, opt in enumerate(options)]
    return f"{question.strip()}\n\n" + "\n".join(lines) + f"\n\n{INSTRUCTION}"


def format_mmlu(row: Mapping[str, Any]) -> tuple[str, str]:
    return _multiple_choice(row["question"], row["choices"]), LETTERS[row["answer"]]


def format_arc(row: Mapping[str, Any]) -> tuple[str, str]:
    # ARC labels are A-E in most rows and 1-4 in some. Re-letter by position.
    labels = list(row["choices"]["label"])
    gold = LETTERS[labels.index(row["answerKey"])]
    return _multiple_choice(row["question"], row["choices"]["text"]), gold


def format_gsm8k(row: Mapping[str, Any]) -> tuple[str, str]:
    gold = row["answer"].split("####")[-1].strip().replace(",", "")
    return f"{row['question'].strip()}\n\n{INSTRUCTION}", gold


def format_mmlu_pro(row: Mapping[str, Any]) -> tuple[str, str]:
    correct = row["options"][row["answer_index"]]
    options = [o for o in row["options"] if str(o).strip() != "N/A"]
    return _multiple_choice(row["question"], options), LETTERS[options.index(correct)]


FORMATTERS = {
    "mmlu": format_mmlu,
    "arc": format_arc,
    "gsm8k": format_gsm8k,
    "mmlu_pro": format_mmlu_pro,
}


def per_source_counts(n_total: int, sources: Sequence[str]) -> dict[str, int]:
    base, extra = divmod(n_total, len(sources))
    return {s: base + (1 if i < extra else 0) for i, s in enumerate(sources)}


def sample_and_split(rows_by_source: Mapping[str, Sequence], n_total: int, seed: int) -> list[dict[str, Any]]:
    """Seeded sample, roughly equal per source, split 50/50 into dev and test within each source."""
    counts = per_source_counts(n_total, list(rows_by_source))
    out: list[dict[str, Any]] = []
    for source, rows in rows_by_source.items():
        rng = random.Random(f"{seed}-{source}")
        picked = rng.sample(range(len(rows)), counts[source])  # random order, so position decides the split
        n_dev = (len(picked) + 1) // 2
        for pos, idx in enumerate(picked):
            prompt, gold = FORMATTERS[source](rows[idx])
            out.append(
                {
                    "id": f"{source}-{idx}",
                    "source": source,
                    "split": "dev" if pos < n_dev else "test",
                    "prompt": prompt,
                    "gold": gold,
                }
            )
    return out


def load_sources() -> dict[str, Sequence]:
    from datasets import load_dataset

    return {name: load_dataset(path, config, split=split) for name, (path, config, split) in SOURCES.items()}

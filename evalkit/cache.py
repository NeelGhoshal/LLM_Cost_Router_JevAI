from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row and flush, so an interrupted run loses at most the call in flight."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_keys(path: str | Path, fields: Iterable[str]) -> set[tuple]:
    """Keys already cached, for resume. fields is e.g. ("id", "model")."""
    fields = tuple(fields)
    return {tuple(row[f] for f in fields) for row in load_jsonl(path)}

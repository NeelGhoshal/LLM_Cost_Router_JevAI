"""Exact-match graders. Both read only the last `ANSWER:` line of the model output.

An output with no ANSWER line (for example one cut off at the token cap) is graded wrong.
"""

from __future__ import annotations

import re

NUMERIC_SOURCES = {"gsm8k"}

_ANSWER = re.compile(r"ANSWER\s*:\s*(.*)", re.IGNORECASE)
_LETTER = re.compile(r"[\s*_`(\[\"']*([A-Ja-j])(?![A-Za-z])")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def extract_answer(text: str) -> str | None:
    """Whatever follows the last ANSWER: on its line, or None if there is no such line."""
    matches = _ANSWER.findall(text or "")
    return matches[-1].strip() if matches else None


def extract_letter(text: str) -> str | None:
    answer = extract_answer(text)
    if answer is None:
        return None
    m = _LETTER.match(answer)
    return m.group(1).upper() if m else None


def to_number(s: str) -> float | None:
    """First number in s, ignoring thousands separators, currency signs and units."""
    m = _NUMBER.search(s.replace("−", "-"))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def extract_number(text: str) -> float | None:
    answer = extract_answer(text)
    return None if answer is None else to_number(answer)


def grade_letter(text: str, gold: str) -> bool:
    return extract_letter(text) == gold.strip().upper()


def grade_number(text: str, gold: str) -> bool:
    got, want = extract_number(text), to_number(gold)
    return got is not None and want is not None and abs(got - want) < 1e-6


def grade(source: str, text: str, gold: str) -> bool:
    return grade_number(text, gold) if source in NUMERIC_SOURCES else grade_letter(text, gold)

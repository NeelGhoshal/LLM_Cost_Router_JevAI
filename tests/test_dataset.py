from collections import Counter

from router.dataset import (
    INSTRUCTION,
    format_arc,
    format_gsm8k,
    format_mmlu,
    format_mmlu_pro,
    per_source_counts,
    sample_and_split,
)
from router.graders import grade

MMLU_ROW = {"question": "2 + 2 = ?", "subject": "math", "choices": ["3", "4", "5", "6"], "answer": 1}
ARC_ROW = {
    "id": "x",
    "question": "Which is a mammal?",
    "choices": {"text": ["Trout", "Eagle", "Whale", "Frog"], "label": ["A", "B", "C", "D"]},
    "answerKey": "C",
}
ARC_NUMERIC_ROW = {**ARC_ROW, "choices": {**ARC_ROW["choices"], "label": ["1", "2", "3", "4"]}, "answerKey": "3"}
GSM_ROW = {"question": "A farm sells 1,000 eggs twice. How many?", "answer": "1000 * 2 = <<1000*2=2000>>2000\n#### 2,000"}
PRO_ROW = {
    "question": "Pick the last real option.",
    "options": ["a", "N/A", "b", "c", "d", "e", "f", "g", "h", "i"],
    "answer": "J",
    "answer_index": 9,
}


def test_format_mmlu():
    prompt, gold = format_mmlu(MMLU_ROW)
    assert gold == "B"
    assert "A. 3\nB. 4\nC. 5\nD. 6" in prompt
    assert prompt.startswith("2 + 2 = ?")


def test_format_arc_letter_and_numeric_labels():
    assert format_arc(ARC_ROW)[1] == "C"
    prompt, gold = format_arc(ARC_NUMERIC_ROW)
    assert gold == "C"  # label "3" becomes the third letter
    assert "C. Whale" in prompt and "3." not in prompt


def test_format_gsm8k_strips_working_and_commas():
    prompt, gold = format_gsm8k(GSM_ROW)
    assert gold == "2000"
    assert "<<" not in prompt and "####" not in prompt


def test_format_mmlu_pro_drops_na_and_reletters():
    prompt, gold = format_mmlu_pro(PRO_ROW)
    assert "N/A" not in prompt
    assert gold == "I"  # ninth of the nine remaining options
    assert "I. i" in prompt and "J." not in prompt


def test_every_prompt_ends_with_the_same_instruction():
    for fmt, row in [(format_mmlu, MMLU_ROW), (format_arc, ARC_ROW), (format_gsm8k, GSM_ROW), (format_mmlu_pro, PRO_ROW)]:
        assert fmt(row)[0].endswith(INSTRUCTION)


def test_gold_passes_its_own_grader():
    for source, fmt, row in [
        ("mmlu", format_mmlu, MMLU_ROW),
        ("arc", format_arc, ARC_ROW),
        ("gsm8k", format_gsm8k, GSM_ROW),
        ("mmlu_pro", format_mmlu_pro, PRO_ROW),
    ]:
        gold = fmt(row)[1]
        assert grade(source, f"thinking...\nANSWER: {gold}", gold)


def test_per_source_counts_spreads_remainder():
    assert per_source_counts(400, ["a", "b", "c", "d"]) == {"a": 100, "b": 100, "c": 100, "d": 100}
    assert per_source_counts(10, ["a", "b", "c", "d"]) == {"a": 3, "b": 3, "c": 2, "d": 2}


def _fake_sources(n=50):
    return {
        "mmlu": [{**MMLU_ROW, "question": f"q{i}"} for i in range(n)],
        "arc": [{**ARC_ROW, "question": f"q{i}"} for i in range(n)],
        "gsm8k": [{**GSM_ROW, "question": f"q{i}"} for i in range(n)],
        "mmlu_pro": [{**PRO_ROW, "question": f"q{i}"} for i in range(n)],
    }


def test_sample_is_equal_per_source_and_split_is_stratified():
    rows = sample_and_split(_fake_sources(), n_total=40, seed=7)
    assert len(rows) == 40
    assert len({r["id"] for r in rows}) == 40
    assert set(rows[0]) == {"id", "source", "split", "prompt", "gold"}
    counts = Counter((r["source"], r["split"]) for r in rows)
    assert all(counts[(s, split)] == 5 for s in ("mmlu", "arc", "gsm8k", "mmlu_pro") for split in ("dev", "test"))


def test_sample_is_deterministic_and_seed_dependent():
    a = sample_and_split(_fake_sources(), 40, seed=7)
    b = sample_and_split(_fake_sources(), 40, seed=7)
    c = sample_and_split(_fake_sources(), 40, seed=8)
    assert a == b
    assert [r["id"] for r in a] != [r["id"] for r in c]

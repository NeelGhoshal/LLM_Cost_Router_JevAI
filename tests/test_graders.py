import pytest

from evalkit.graders import extract_answer, grade, grade_letter, grade_number


@pytest.mark.parametrize(
    "text",
    [
        "The planet spins faster.\nANSWER: C",
        "reasoning\nANSWER: C.",
        "reasoning\n**ANSWER: C**",
        "reasoning\nANSWER: **C**",
        "reasoning\nANSWER: (C)",
        "reasoning\nAnswer: c",
        "reasoning\nANSWER:C) Planetary days will become shorter.",
        "First I thought ANSWER: A\nbut on reflection\nANSWER: C",  # last one wins
    ],
)
def test_letter_accepts_common_formats(text):
    assert grade_letter(text, "C")


@pytest.mark.parametrize(
    "text",
    [
        "ANSWER: B",
        "The answer is C.",  # no ANSWER line
        "I would pick C but ran out of tok",  # truncated
        "ANSWER: Cannot be determined",  # a word, not a letter
        "ANSWER: 3",
        "",
    ],
)
def test_letter_rejects(text):
    assert not grade_letter(text, "C")


def test_letter_covers_ten_options():
    assert grade_letter("ANSWER: J", "J")
    assert not grade_letter("ANSWER: K", "K")  # outside the A-J range used by any source


@pytest.mark.parametrize(
    "text",
    [
        "16 - 3 - 4 = 9, 9 * 2 = 18\nANSWER: 18",
        "ANSWER: 18.",
        "ANSWER: $18",
        "ANSWER: 18.00",
        "ANSWER: 18 dollars",
        "**ANSWER: 18**",
        "ANSWER: x = 18",
        "ANSWER: 20\nwait, recheck.\nANSWER: 18",
    ],
)
def test_number_accepts_common_formats(text):
    assert grade_number(text, "18")


@pytest.mark.parametrize(
    "text",
    ["ANSWER: 19", "ANSWER: 180", "ANSWER: 1.8", "ANSWER: eighteen", "The total is 18.", ""],
)
def test_number_rejects(text):
    assert not grade_number(text, "18")


def test_number_handles_separators_negatives_and_decimals():
    assert grade_number("ANSWER: 1,234,567", "1234567")
    assert grade_number("ANSWER: 1234567", "1,234,567")
    assert grade_number("ANSWER: -5", "-5")
    assert not grade_number("ANSWER: 5", "-5")
    assert grade_number("ANSWER: 0.5", "0.5")
    assert grade_number("ANSWER: .5", "0.5")


@pytest.mark.parametrize(
    "source,text,gold,expected",
    [
        ("mmlu", "ANSWER: B", "B", True),
        ("mmlu", "ANSWER: A", "B", False),
        ("arc", "ANSWER: D", "D", True),
        ("arc", "ANSWER: 4", "D", False),
        ("mmlu_pro", "ANSWER: I", "I", True),
        ("mmlu_pro", "ANSWER: H", "I", False),
        ("gsm8k", "ANSWER: 42", "42", True),
        ("gsm8k", "ANSWER: 41", "42", False),
        ("gsm8k", "ANSWER: B", "42", False),
    ],
)
def test_grade_dispatches_by_source(source, text, gold, expected):
    assert grade(source, text, gold) is expected


def test_extract_answer():
    assert extract_answer("x\nANSWER:  B  ") == "B"
    assert extract_answer("no final line") is None

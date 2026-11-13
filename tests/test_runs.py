import dataclasses

import pytest

from router.cache import load_jsonl
from router.config import ModelSpec, load_config
from router.costs import BudgetExceeded, BudgetGuard, jev_cost, llm_cost
from router.jev import FakeJev
from router.llm import FakeLLM
from router.runs import estimate, run_jev, run_models, spread

PROMPTS = [
    {"id": "mmlu-1", "source": "mmlu", "split": "dev", "prompt": "Pick one. A. x B. y", "gold": "B"},
    {"id": "gsm8k-1", "source": "gsm8k", "split": "dev", "prompt": "What is 6 * 7?", "gold": "42"},
    {"id": "arc-1", "source": "arc", "split": "test", "prompt": "Pick one. A. p B. q", "gold": "A"},
]


@pytest.fixture
def cfg():
    return load_config()


def _clients(cfg):
    # small always says B / 41, large always answers correctly
    gold = {p["prompt"]: p["gold"] for p in PROMPTS}
    small = FakeLLM(lambda model, prompt: "ANSWER: 41" if "6 * 7" in prompt else "ANSWER: B")
    large = FakeLLM(lambda model, prompt: f"thinking\nANSWER: {gold[prompt]}")
    return {"small": small, "large": large}


def test_run_models_grades_and_costs_every_call(cfg, tmp_path):
    cache = tmp_path / "llm.jsonl"
    clients = _clients(cfg)
    made = run_models(PROMPTS, cfg, clients, cache, BudgetGuard(1.0))

    rows = load_jsonl(cache)
    assert made == len(rows) == 6
    correct = {(r["id"], r["role"]): r["correct"] for r in rows}
    assert correct[("mmlu-1", "small")] and not correct[("gsm8k-1", "small")] and not correct[("arc-1", "small")]
    assert all(correct[(p["id"], "large")] for p in PROMPTS)
    for r in rows:
        spec = cfg.small_model if r["role"] == "small" else cfg.large_model
        assert r["model"] == spec.id
        assert r["cost"] == pytest.approx(llm_cost(spec, r["input_tokens"], r["output_tokens"]))
        assert r["billed"] == 0.0  # ollama is local


def test_run_models_resumes_without_repeating_calls(cfg, tmp_path):
    cache = tmp_path / "llm.jsonl"
    first = _clients(cfg)
    run_models(PROMPTS[:2], cfg, first, cache, BudgetGuard(1.0))

    second = _clients(cfg)
    made = run_models(PROMPTS, cfg, second, cache, BudgetGuard(1.0))

    assert made == 2
    assert [p for _, p in second["small"].calls] == [PROMPTS[2]["prompt"]]
    assert [p for _, p in second["large"].calls] == [PROMPTS[2]["prompt"]]
    assert len(load_jsonl(cache)) == 6

    third = _clients(cfg)
    assert run_models(PROMPTS, cfg, third, cache, BudgetGuard(1.0)) == 0
    assert third["small"].calls == [] and third["large"].calls == []


def test_run_models_counts_billed_providers_against_the_budget(cfg, tmp_path):
    paid = ModelSpec("openrouter", "big", usd_per_m_input=1e6, usd_per_m_output=1e6)  # $1 per token
    cfg = dataclasses.replace(cfg, large_model=paid)
    with pytest.raises(BudgetExceeded):
        run_models(PROMPTS, cfg, _clients(cfg), tmp_path / "llm.jsonl", BudgetGuard(5.0))
    rows = load_jsonl(tmp_path / "llm.jsonl")
    assert any(r["billed"] > 0 for r in rows)  # the call that tripped the guard is still cached


def test_run_jev_caches_resumes_and_costs(cfg, tmp_path):
    cache = tmp_path / "jev.jsonl"
    jev = FakeJev(lambda prompt: 0.2 if "6 * 7" in prompt else 0.8)
    guard = BudgetGuard(1.0)
    assert run_jev(PROMPTS[:2], cfg, jev, cache, guard) == 2

    rows = load_jsonl(cache)
    assert [r["probs"]["small_ok"] for r in rows] == [0.8, 0.2]
    assert rows[0]["cost"] == pytest.approx(jev_cost(rows[0]["input_tokens"], cfg.jev_usd_per_m_input))
    assert guard.spent == pytest.approx(sum(r["cost"] for r in rows))
    assert rows[0]["model"] == "jev-fake"

    again = FakeJev(lambda prompt: 0.5)
    assert run_jev(PROMPTS, cfg, again, cache, BudgetGuard(1.0)) == 1
    assert again.calls == [PROMPTS[2]["prompt"]]


def test_run_jev_stops_at_the_budget(cfg, tmp_path):
    cfg = dataclasses.replace(cfg, jev_usd_per_m_input=1e6)  # $1 per token
    with pytest.raises(BudgetExceeded):
        run_jev(PROMPTS, cfg, FakeJev(lambda p: 0.5), tmp_path / "jev.jsonl", BudgetGuard(30.0))
    assert len(load_jsonl(tmp_path / "jev.jsonl")) < len(PROMPTS)


def test_estimate_separates_real_from_imputed(cfg):
    est = estimate(PROMPTS, cfg)
    assert est.llm_billed == 0.0  # both models are local
    assert est.llm_imputed > 0
    assert est.real == pytest.approx(est.jev) and 0 < est.jev < 0.001
    # never below the fixed per-question overhead Jev was measured to bill
    assert est.jev >= jev_cost(300 * len(PROMPTS), cfg.jev_usd_per_m_input)

    paid = dataclasses.replace(cfg, large_model=ModelSpec("openrouter", "big", 4.0, 20.0))
    worst_large = sum(llm_cost(paid.large_model, len(p["prompt"]) / 3.5, cfg.max_output_tokens) for p in PROMPTS)
    assert estimate(PROMPTS, paid).llm_billed == pytest.approx(worst_large)
    assert estimate(PROMPTS, cfg, jev_questions=0).jev == 0.0
    assert estimate(PROMPTS, cfg, jev_questions=3).jev == pytest.approx(3 * est.jev)


def test_spread_takes_from_every_source():
    many = [{"id": f"{s}-{i}", "source": s} for s in ("a", "b", "c") for i in range(5)]
    picked = spread(many, 5)
    assert [p["id"] for p in picked] == ["a-0", "b-0", "c-0", "a-1", "b-1"]
    assert len(spread(many, 100)) == 15

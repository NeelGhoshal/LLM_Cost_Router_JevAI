import pytest

from evalkit.cache import append_jsonl, done_keys, load_jsonl
from router.config import ModelSpec, load_config
from router.costs import BudgetExceeded, BudgetGuard, is_billed, jev_cost, llm_cost


def test_repo_config_loads():
    cfg = load_config()
    assert cfg.jev_model.startswith("jev-")
    assert cfg.small_model.id != cfg.large_model.id
    assert cfg.small_model.usd_per_m_output < cfg.large_model.usd_per_m_output
    assert cfg.n_prompts >= 300


def test_llm_cost():
    spec = ModelSpec("openrouter", "m", usd_per_m_input=4.0, usd_per_m_output=20.0)
    assert llm_cost(spec, 300, 600) == pytest.approx((300 * 4 + 600 * 20) / 1e6)
    assert llm_cost(spec, 0, 0) == 0


def test_jev_cost_bills_input_only():
    assert jev_cost(1_000_000, 0.042) == pytest.approx(0.042)


def test_is_billed():
    assert not is_billed(ModelSpec("ollama", "m", 1, 1))
    assert is_billed(ModelSpec("openrouter", "m", 1, 1))


def test_budget_guard_aborts_when_exceeded():
    guard = BudgetGuard(max_usd=0.50)
    guard.add(0.30)
    guard.add(0.20)  # exactly at the cap is allowed
    with pytest.raises(BudgetExceeded):
        guard.add(0.01)


def test_cache_resume(tmp_path):
    path = tmp_path / "cache" / "llm.jsonl"
    assert load_jsonl(path) == []
    assert done_keys(path, ("id", "model")) == set()

    append_jsonl(path, {"id": "a", "model": "small", "text": "é"})
    append_jsonl(path, {"id": "a", "model": "large", "text": "x"})

    todo = [("a", "small"), ("a", "large"), ("b", "small")]
    done = done_keys(path, ("id", "model"))
    assert [k for k in todo if k not in done] == [("b", "small")]
    assert load_jsonl(path)[0]["text"] == "é"

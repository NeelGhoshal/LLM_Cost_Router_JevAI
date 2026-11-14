import dataclasses
import logging

import pytest

from router.config import load_config
from router.jev import FakeJev
from router.llm import FakeLLM, LLMResult
from router.route import Cascade, should_escalate


@pytest.fixture
def cfg():
    return dataclasses.replace(load_config(), jev_timeout_s=0.2, cascade_threshold=0.5)


def make(cfg, p_wrong, small_text="small says 4\nANSWER: 4", **jev_kwargs):
    jev = FakeJev(p_wrong, **jev_kwargs)
    small = FakeLLM(lambda model, prompt: small_text)
    large = FakeLLM(lambda model, prompt: "large says 5\nANSWER: 5")
    return Cascade(cfg, jev, small, large), jev, small, large


def test_should_escalate_boundary():
    assert should_escalate(0.5, 0.5)  # at the threshold escalates
    assert not should_escalate(0.49, 0.5)
    assert should_escalate(0.0, 0.0)


def test_passing_check_keeps_the_small_answer_and_never_calls_large(cfg):
    c, jev, small, large = make(cfg, lambda response: 0.1)
    r = c.answer("What is 2+2?")
    assert (r.model, r.escalated, r.p_wrong) == (cfg.small_model.id, False, 0.1)
    assert r.text.endswith("ANSWER: 4") and r.large is None
    assert large.calls == []
    assert "passed" in r.reason


def test_jev_sees_the_prompt_and_the_small_response(cfg):
    c, jev, *_ = make(cfg, lambda response: 0.1)
    c.answer("What is 2+2?")
    assert jev.states == [{"prompt": "What is 2+2?", "response": "small says 4\nANSWER: 4"}]


def test_firing_check_returns_the_large_answer(cfg):
    c, jev, small, large = make(cfg, lambda response: 0.9)
    r = c.answer("What is 2+3?")
    assert (r.model, r.escalated, r.p_wrong) == (cfg.large_model.id, True, 0.9)
    assert r.text.endswith("ANSWER: 5")
    assert len(small.calls) == 1 and len(large.calls) == 1
    assert "fired" in r.reason


def test_threshold_argument_overrides_config(cfg):
    c, *_ = make(cfg, lambda response: 0.3)
    assert not c.answer("x").escalated
    assert c.answer("x", threshold=0.2).escalated


def test_jev_error_escalates_and_logs(cfg, caplog):
    def boom(response):
        raise ConnectionError("down")

    c, jev, small, large = make(cfg, boom)
    with caplog.at_level(logging.WARNING, logger="router"):
        r = c.answer("x")
    assert r.escalated and r.p_wrong is None and r.model == cfg.large_model.id
    assert "ConnectionError" in r.reason and "escalating to large model" in caplog.text


def test_jev_timeout_escalates(cfg):
    c, *_ = make(cfg, lambda response: 0.0, delay_s=1.0)
    r = c.answer("x")
    assert r.escalated and "timeout" in r.reason
    assert r.jev_latency_ms < 900


def test_cut_off_small_response_escalates_without_asking_jev(cfg):
    class Capped:
        def complete(self, model, prompt):
            return LLMResult(model, "still thinking", 10, cfg.max_output_tokens, 1.0)

    jev = FakeJev(lambda response: 0.0)
    large = FakeLLM(lambda model, prompt: "ANSWER: 5")
    r = Cascade(cfg, jev, Capped(), large).answer("x")
    assert r.escalated and jev.calls == [] and r.jev_input_tokens == 0
    assert "cut off" in r.reason

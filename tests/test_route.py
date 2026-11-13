import dataclasses
import logging

import pytest

from router.config import load_config
from router.jev import FakeJev
from router.route import Router, use_large


@pytest.fixture
def cfg():
    return dataclasses.replace(load_config(), jev_timeout_s=0.2)


def test_use_large_threshold_boundary():
    assert use_large(0.49, 0.5)
    assert not use_large(0.5, 0.5)  # at the threshold stays small
    assert not use_large(0.9, 0.5)
    assert not use_large(0.0, 0.0)  # threshold 0 is always-small
    assert use_large(1.0, 1.01)  # threshold above 1 is always-large


def test_routes_small_when_jev_is_confident(cfg):
    d = Router(cfg, FakeJev(lambda p: 0.9)).route("easy", threshold=0.5)
    assert d.model == cfg.small_model.id
    assert d.p_small_ok == 0.9
    assert d.fallback_reason is None
    assert d.jev_input_tokens > 0


def test_routes_large_when_jev_is_doubtful(cfg):
    d = Router(cfg, FakeJev(lambda p: 0.1)).route("hard", threshold=0.5)
    assert d.model == cfg.large_model.id
    assert d.p_small_ok == 0.1


def test_default_threshold_comes_from_config(cfg):
    cfg = dataclasses.replace(cfg, threshold=0.95)
    assert Router(cfg, FakeJev(lambda p: 0.9)).route("x").model == cfg.large_model.id


def test_jev_error_falls_back_to_large_and_logs(cfg, caplog):
    def boom(prompt):
        raise ConnectionError("down")

    with caplog.at_level(logging.WARNING, logger="router"):
        d = Router(cfg, FakeJev(boom)).route("x", threshold=0.5)

    assert d.model == cfg.large_model.id
    assert d.p_small_ok is None
    assert "ConnectionError" in d.fallback_reason
    assert "routing to large model" in caplog.text


def test_jev_timeout_falls_back_to_large(cfg):
    slow = FakeJev(lambda p: 0.99, delay_s=1.0)
    d = Router(cfg, slow).route("x", threshold=0.5)

    assert d.model == cfg.large_model.id
    assert d.p_small_ok is None
    assert "timeout" in d.fallback_reason
    assert d.jev_latency_ms < 900  # returned at the deadline, did not wait for Jev

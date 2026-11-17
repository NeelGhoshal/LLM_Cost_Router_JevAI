import dataclasses

import httpx
import pytest
from fastapi.testclient import TestClient

from router.config import REPO_ROOT, load_config
from router.jev import FakeJev
from router.llm import FakeLLM, LLMResult
from web.app import RateLimiter, Services, UnavailableJev, create_app, load_examples


@pytest.fixture
def cfg():
    return dataclasses.replace(load_config(REPO_ROOT / "config.deploy.yaml"), jev_timeout_s=0.5)


def client_for(cfg, p, *, jev=None, per_minute=100, daily=1000, small_text="ANSWER: B", large_text="ANSWER: C"):
    small = FakeLLM(lambda m, prompt: small_text)
    large = FakeLLM(lambda m, prompt: large_text)
    services = Services(cfg, jev or FakeJev(lambda x: p), small, large, jev_ready=True, llm_ready=True)
    app = create_app(services, RateLimiter(per_minute, daily))
    return TestClient(app), small, large


def test_cascade_keeps_small_answer_when_check_passes(cfg):
    c, small, large = client_for(cfg, 0.1)
    r = c.post("/api/ask", json={"prompt": "2+2?", "mode": "cascade", "threshold": 0.5}).json()
    assert r["answer"] == {"model": cfg.small_model.id, "role": "small", "text": "ANSWER: B"}
    assert r["check"]["p_wrong"] == 0.1 and r["check"]["escalated"] is False
    assert r["large"] is None and not large.calls
    assert r["total"]["usd"] > 0 and r["always_large_usd"] > r["small"]["usd"]


def test_cascade_escalates_when_check_fires(cfg):
    c, small, large = client_for(cfg, 0.9)
    r = c.post("/api/ask", json={"prompt": "hard", "mode": "cascade", "threshold": 0.5}).json()
    assert r["answer"]["role"] == "large" and r["answer"]["text"] == "ANSWER: C"
    assert r["check"]["escalated"] is True and r["large"]["model"] == cfg.large_model.id


def test_cascade_without_jev_falls_back_to_large(cfg):
    c, _, large = client_for(cfg, 0.0, jev=UnavailableJev())
    r = c.post("/api/ask", json={"prompt": "x", "mode": "cascade"}).json()
    assert r["answer"]["role"] == "large"
    assert r["check"]["p_wrong"] is None and "jev error" in r["check"]["fallback"]


def test_preroute_sends_to_small_or_large(cfg):
    c, small, large = client_for(cfg, 0.95)
    r = c.post("/api/ask", json={"prompt": "easy", "mode": "preroute", "threshold": 0.5}).json()
    assert r["route"]["role"] == "small" and r["answered"]["model"] == cfg.small_model.id
    c, small, large = client_for(cfg, 0.2)
    r = c.post("/api/ask", json={"prompt": "hard", "mode": "preroute", "threshold": 0.5}).json()
    assert r["route"]["role"] == "large" and not small.calls


def test_default_threshold_comes_from_config(cfg):
    c, _, _ = client_for(cfg, 0.3)  # cascade_threshold is 0.28, so 0.3 escalates
    r = c.post("/api/ask", json={"prompt": "x"}).json()
    assert r["check"]["threshold"] == cfg.cascade_threshold and r["check"]["escalated"] is True


def test_rejects_empty_and_oversized_prompts(cfg):
    c, _, _ = client_for(cfg, 0.1)
    assert c.post("/api/ask", json={"prompt": ""}).status_code == 422
    assert c.post("/api/ask", json={"prompt": "x" * 5000}).status_code == 422
    assert c.post("/api/ask", json={"prompt": "x", "threshold": 1.5}).status_code == 422


def test_rate_limit_per_visitor(cfg):
    c, _, _ = client_for(cfg, 0.1, per_minute=2)
    codes = [c.post("/api/ask", json={"prompt": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    other = c.post("/api/ask", json={"prompt": "x"}, headers={"x-forwarded-for": "10.0.0.9"})
    assert other.status_code == 200


def test_daily_cap(cfg):
    c, _, _ = client_for(cfg, 0.1, daily=1)
    assert c.post("/api/ask", json={"prompt": "x"}).status_code == 200
    r = c.post("/api/ask", json={"prompt": "x"}, headers={"x-forwarded-for": "10.0.0.9"})
    assert r.status_code == 429 and "daily" in r.json()["detail"]


def test_provider_rate_limit_becomes_429(cfg):
    class Limited:
        def complete(self, model, prompt) -> LLMResult:
            req = httpx.Request("POST", "http://llm.test")
            raise httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req))

    services = Services(cfg, FakeJev(lambda x: 0.1), Limited(), Limited(), True, True)
    c = TestClient(create_app(services, RateLimiter(100, 1000)))
    r = c.post("/api/ask", json={"prompt": "x"})
    assert r.status_code == 429 and "quota" in r.json()["detail"]


def test_missing_model_key_returns_503(cfg):
    services = Services(cfg, FakeJev(lambda x: 0.1), None, None, True, False, "GROQ_API_KEY is not set")
    c = TestClient(create_app(services, RateLimiter(100, 1000)))
    assert c.get("/api/config").json()["llm_ready"] is False
    r = c.post("/api/ask", json={"prompt": "x"})
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]


def test_config_examples_and_static_pages(cfg):
    c, _, _ = client_for(cfg, 0.1)
    conf = c.get("/api/config").json()
    assert conf["small"]["id"] == cfg.small_model.id and conf["jev_ready"] is True
    examples = c.get("/api/examples").json()
    assert examples == load_examples() and {e["source"] for e in examples} == {"arc", "gsm8k", "mmlu", "mmlu_pro"}
    assert "Jev" in c.get("/").text
    assert c.get("/app.js").status_code == 200
    assert c.get("/results/cost_quality.png").status_code == 200

"""Web demo: a small JSON API over the router plus the static frontend in web/static.

Run locally:   uvicorn web.app:app --reload
Environment:   GROQ_API_KEY (models), TYPESAFE_API_KEY (Jev), ROUTER_CONFIG (default config.deploy.yaml),
               RATE_PER_MINUTE (per visitor, default 6), DAILY_REQUEST_CAP (all visitors, default 300).

Without TYPESAFE_API_KEY every request fails safe to the large model, exactly as the CLI router does.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from router.config import REPO_ROOT, Config, ModelSpec, load_config
from router.costs import jev_cost, llm_cost
from router.jev import JevClient, JevResult, TypeSafeJev
from router.llm import HOSTED_PROVIDERS, LLMClient, LLMResult, make_llm_client
from router.route import Cascade, Router

log = logging.getLogger("web")
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_PROMPT_CHARS = 4000
EXAMPLES_PER_SOURCE = 2


class UnavailableJev:
    """Stands in when no Jev key is configured. Every call raises, so the router fails safe."""

    def ask(self, prompt: str, instructions: Any = None) -> JevResult:
        raise RuntimeError("TYPESAFE_API_KEY is not set")

    def ask_state(self, state: Any, instructions: Any) -> JevResult:
        raise RuntimeError("TYPESAFE_API_KEY is not set")


class UnavailableLLM:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def complete(self, model: str, prompt: str) -> LLMResult:
        raise RuntimeError(self.reason)


class RateLimiter:
    """Per-visitor sliding window plus a global daily cap. In memory: one process, resets on restart."""

    def __init__(self, per_minute: int, daily_cap: int) -> None:
        self.per_minute = per_minute
        self.daily_cap = daily_cap
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._day = date.today()
        self._today = 0
        self._lock = threading.Lock()

    def check(self, key: str) -> str | None:
        """Record a request. Returns why it is refused, or None if it may proceed."""
        now = time.monotonic()
        with self._lock:
            if date.today() != self._day:
                self._day, self._today = date.today(), 0
            if self._today >= self.daily_cap:
                return "The demo has reached its daily request limit. It resets at midnight UTC."
            hits = self._hits[key]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return f"Too many requests. Each visitor can send {self.per_minute} per minute."
            hits.append(now)
            self._today += 1
            return None


class AskBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    mode: Literal["cascade", "preroute"] = "cascade"
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


@dataclass
class Services:
    cfg: Config
    jev: JevClient
    small: LLMClient
    large: LLMClient
    jev_ready: bool
    llm_ready: bool
    llm_problem: str | None = None


def llm_view(spec: ModelSpec, r: LLMResult, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "model": spec.id,
        "text": r.text,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "seconds": round(r.latency_ms / 1000, 3),
        "usd": llm_cost(spec, r.input_tokens, r.output_tokens),
    }


def always_large_estimate(cfg: Config, r: LLMResult) -> float:
    """What the large model would have cost for this prompt, from the small model's token counts."""
    return llm_cost(cfg.large_model, r.input_tokens, r.output_tokens)


def run_cascade(s: Services, prompt: str, threshold: float | None) -> dict[str, Any]:
    cfg = s.cfg
    t = cfg.cascade_threshold if threshold is None else threshold
    r = Cascade(cfg, s.jev, s.small, s.large).answer(prompt, t)
    small = llm_view(cfg.small_model, r.small, "small")
    large = llm_view(cfg.large_model, r.large, "large") if r.large else None
    jev_usd = jev_cost(r.jev_input_tokens, cfg.jev_usd_per_m_input)
    checked = r.p_wrong is not None
    usd = small["usd"] + jev_usd + (large["usd"] if large else 0.0)
    seconds = small["seconds"] + r.jev_latency_ms / 1000 + (large["seconds"] if large else 0.0)
    return {
        "mode": "cascade",
        "small": small,
        "check": {
            "p_wrong": r.p_wrong,
            "threshold": t,
            "escalated": r.escalated,
            "reason": r.reason,
            "fallback": None if checked else r.reason,
            "seconds": round(r.jev_latency_ms / 1000, 3),
            "usd": jev_usd,
        },
        "large": large,
        "answer": {"model": r.model, "role": "large" if r.escalated else "small", "text": r.text},
        "total": {"usd": usd, "seconds": round(seconds, 3)},
        "always_large_usd": large["usd"] if large else always_large_estimate(cfg, r.small),
    }


def run_preroute(s: Services, prompt: str, threshold: float | None) -> dict[str, Any]:
    cfg = s.cfg
    t = cfg.threshold if threshold is None else threshold
    d = Router(cfg, s.jev).route(prompt, t)
    role = "small" if d.model == cfg.small_model.id else "large"
    spec = cfg.small_model if role == "small" else cfg.large_model
    r = (s.small if role == "small" else s.large).complete(spec.id, prompt)
    answered = llm_view(spec, r, role)
    jev_usd = jev_cost(d.jev_input_tokens, cfg.jev_usd_per_m_input)
    return {
        "mode": "preroute",
        "route": {
            "p_small_ok": d.p_small_ok,
            "threshold": t,
            "role": role,
            "model": d.model,
            "fallback": d.fallback_reason,
            "seconds": round(d.jev_latency_ms / 1000, 3),
            "usd": jev_usd,
        },
        "answered": answered,
        "answer": {"model": spec.id, "role": role, "text": r.text},
        "total": {"usd": answered["usd"] + jev_usd, "seconds": round(answered["seconds"] + d.jev_latency_ms / 1000, 3)},
        "always_large_usd": answered["usd"] if role == "large" else always_large_estimate(cfg, r),
    }


def load_examples(path: Path = REPO_ROOT / "data" / "prompts.jsonl") -> list[dict[str, str]]:
    """A few held-out test prompts per benchmark, with their gold answers."""
    if not path.exists():
        return []
    picked: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] == "test" and len(picked[row["source"]]) < EXAMPLES_PER_SOURCE:
            picked[row["source"]].append({k: row[k] for k in ("id", "source", "prompt", "gold")})
    return [ex for rows in picked.values() for ex in rows]


def build_services(cfg: Config) -> Services:
    jev_ready = bool(os.environ.get("TYPESAFE_API_KEY"))
    jev: JevClient = UnavailableJev()
    if jev_ready:
        # No SDK retries and a short timeout, so the fail-safe fires instead of the SDK retrying.
        jev = TypeSafeJev(cfg.jev_model, timeout_s=cfg.jev_timeout_s, max_retries=0)
    try:
        small, large = make_llm_client(cfg.small_model, cfg), make_llm_client(cfg.large_model, cfg)
        return Services(cfg, jev, small, large, jev_ready, True)
    except RuntimeError as e:
        problem = str(e)
        return Services(cfg, jev, UnavailableLLM(problem), UnavailableLLM(problem), jev_ready, False, problem)


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(services: Services | None = None, limiter: RateLimiter | None = None) -> FastAPI:
    load_dotenv()
    if services is None:
        services = build_services(load_config(REPO_ROOT / os.environ.get("ROUTER_CONFIG", "config.deploy.yaml")))
    if limiter is None:
        limiter = RateLimiter(int(os.environ.get("RATE_PER_MINUTE", 6)), int(os.environ.get("DAILY_REQUEST_CAP", 300)))
    examples = load_examples()
    metrics_path = REPO_ROOT / "results" / "metrics.json"

    app = FastAPI(title="Jev LLM cost router", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        cfg = services.cfg

        def model(spec: ModelSpec) -> dict[str, Any]:
            return {
                "id": spec.id,
                "provider": spec.provider,
                "usd_per_m_input": spec.usd_per_m_input,
                "usd_per_m_output": spec.usd_per_m_output,
            }

        return {
            "small": model(cfg.small_model),
            "large": model(cfg.large_model),
            "jev_model": cfg.jev_model,
            "jev_ready": services.jev_ready,
            "llm_ready": services.llm_ready,
            "llm_problem": services.llm_problem,
            "prices_are_free_tier": cfg.small_model.provider in HOSTED_PROVIDERS,
            "cascade_threshold": cfg.cascade_threshold,
            "preroute_threshold": cfg.threshold,
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "rate_per_minute": limiter.per_minute,
        }

    @app.get("/api/examples")
    def get_examples() -> list[dict[str, str]]:
        return examples

    @app.get("/api/metrics")
    def metrics() -> Any:
        if not metrics_path.exists():
            raise HTTPException(404, "results/metrics.json is missing")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    @app.post("/api/ask")
    def ask(body: AskBody, request: Request) -> Any:
        if not services.llm_ready:
            raise HTTPException(503, f"The models are not configured on this server: {services.llm_problem}.")
        refused = limiter.check(client_key(request))
        if refused:
            raise HTTPException(429, refused)
        run = run_cascade if body.mode == "cascade" else run_preroute
        try:
            return run(services, body.prompt, body.threshold)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise HTTPException(429, "The free model quota is used up for the moment. Try again in a minute.")
            log.exception("model provider error")
            raise HTTPException(502, f"The model provider returned an error ({e.response.status_code}).")
        except (httpx.TimeoutException, httpx.TransportError):
            log.exception("model provider unreachable")
            raise HTTPException(504, "The model provider did not respond in time. Try again.")

    @app.exception_handler(HTTPException)
    def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    results_dir = REPO_ROOT / "results"
    if results_dir.exists():
        app.mount("/results", StaticFiles(directory=results_dir), name="results")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()

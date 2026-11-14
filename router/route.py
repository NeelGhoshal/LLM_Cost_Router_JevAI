"""Live routing. Two modes:

  Router   pre-routing: ask Jev about the prompt, then call one model.
  Cascade  the small model answers, Jev checks the answer, the large model answers only if
           the check fires.

Both fail safe to the large model when Jev errors or misses its deadline.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

from router.config import Config, load_config
from router.jev import CHECKS, PHRASINGS, QUESTION_KEY, JevClient, JevResult, TypeSafeJev
from router.llm import LLMClient, LLMResult, make_llm_client

log = logging.getLogger("router")


@dataclass(frozen=True)
class Decision:
    model: str
    p_small_ok: float | None
    jev_latency_ms: float
    fallback_reason: str | None = None
    jev_input_tokens: int = 0


@dataclass(frozen=True)
class CascadeResult:
    model: str  # the model whose answer is returned
    text: str
    escalated: bool
    p_wrong: float | None  # Jev's probability that the small answer is wrong; None if not checked
    reason: str  # why the answer was kept or escalated
    jev_latency_ms: float
    jev_input_tokens: int
    small: LLMResult
    large: LLMResult | None


def use_large(p_small_ok: float, threshold: float) -> bool:
    """Route to the large model when Jev's probability that small is enough is below threshold."""
    return p_small_ok < threshold


def should_escalate(p_wrong: float, threshold: float) -> bool:
    """Escalate when Jev's probability that the small answer is wrong reaches the threshold."""
    return p_wrong >= threshold


def call_with_deadline(
    fn: Callable[..., JevResult], timeout_s: float, *args: Any
) -> tuple[JevResult | None, str | None, float]:
    """Run a Jev call under a hard deadline. Returns (result, failure reason, elapsed ms).

    A worker thread enforces the deadline whatever the client does. On timeout the thread is
    left to finish on its own and its result is discarded.
    """
    start = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args)
    try:
        result, reason = future.result(timeout=timeout_s), None
    except FutureTimeout:
        result, reason = None, f"jev timeout after {timeout_s}s"
    except Exception as e:  # any Jev failure must fail safe to the large model
        result, reason = None, f"jev error: {type(e).__name__}"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return result, reason, (time.perf_counter() - start) * 1000


class Router:
    def __init__(self, cfg: Config, jev: JevClient) -> None:
        self.cfg = cfg
        self.jev = jev

    def route(self, prompt: str, threshold: float | None = None) -> Decision:
        threshold = self.cfg.threshold if threshold is None else threshold
        question = {QUESTION_KEY: PHRASINGS[self.cfg.jev_phrasing]}
        result, failure, elapsed_ms = call_with_deadline(self.jev.ask, self.cfg.jev_timeout_s, prompt, question)
        if result is None:
            log.warning("routing to large model: %s", failure)
            return Decision(self.cfg.large_model.id, None, elapsed_ms, fallback_reason=failure)
        p = result.p_small_ok
        spec = self.cfg.large_model if use_large(p, threshold) else self.cfg.small_model
        return Decision(spec.id, p, elapsed_ms, jev_input_tokens=result.input_tokens)


class Cascade:
    def __init__(self, cfg: Config, jev: JevClient, small: LLMClient, large: LLMClient) -> None:
        self.cfg = cfg
        self.jev = jev
        self.small = small
        self.large = large

    def answer(self, prompt: str, threshold: float | None = None) -> CascadeResult:
        threshold = self.cfg.cascade_threshold if threshold is None else threshold
        small = self.small.complete(self.cfg.small_model.id, prompt)

        if small.error or small.output_tokens >= self.cfg.max_output_tokens:
            # A truncated or failed response is not worth checking.
            return self._escalate(prompt, small, None, "small response was cut off or failed", 0.0, 0)

        checks = {name: CHECKS[name] for name in self.cfg.cascade_checks}
        state = {"prompt": prompt, "response": small.text}
        result, failure, elapsed_ms = call_with_deadline(self.jev.ask_state, self.cfg.jev_timeout_s, state, checks)
        if result is None:
            log.warning("escalating to large model: %s", failure)
            return self._escalate(prompt, small, None, failure or "jev failed", elapsed_ms, 0)

        p_wrong = max(result.probs.values())
        if should_escalate(p_wrong, threshold):
            reason = f"check fired (p_wrong {p_wrong:.2f} >= {threshold:.2f})"
            return self._escalate(prompt, small, p_wrong, reason, elapsed_ms, result.input_tokens)
        reason = f"check passed (p_wrong {p_wrong:.2f} < {threshold:.2f})"
        return CascadeResult(
            self.cfg.small_model.id, small.text, False, p_wrong, reason, elapsed_ms, result.input_tokens, small, None
        )

    def _escalate(
        self, prompt: str, small: LLMResult, p_wrong: float | None, reason: str, jev_ms: float, jev_tokens: int
    ) -> CascadeResult:
        large = self.large.complete(self.cfg.large_model.id, prompt)
        return CascadeResult(self.cfg.large_model.id, large.text, True, p_wrong, reason, jev_ms, jev_tokens, small, large)


_router: Router | None = None
_cascade: Cascade | None = None


def _live_jev(cfg: Config) -> TypeSafeJev:
    # No SDK retries and a short timeout, so the fail-safe fires instead of the SDK retrying.
    return TypeSafeJev(cfg.jev_model, timeout_s=cfg.jev_timeout_s, max_retries=0)


def route(prompt: str, threshold: float | None = None) -> Decision:
    """Pre-route with the repo's config.yaml and the real Jev client."""
    global _router
    if _router is None:
        cfg = load_config()
        _router = Router(cfg, _live_jev(cfg))
    return _router.route(prompt, threshold)


def cascade(prompt: str, threshold: float | None = None) -> CascadeResult:
    """Answer through the cascade with the repo's config.yaml, real Jev and real models."""
    global _cascade
    if _cascade is None:
        cfg = load_config()
        small, large = make_llm_client(cfg.small_model, cfg), make_llm_client(cfg.large_model, cfg)
        _cascade = Cascade(cfg, _live_jev(cfg), small, large)
    return _cascade.answer(prompt, threshold)

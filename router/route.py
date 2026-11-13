from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

from router.config import Config, load_config
from router.jev import JevClient, JevResult, TypeSafeJev

log = logging.getLogger("router")


@dataclass(frozen=True)
class Decision:
    model: str
    p_small_ok: float | None
    jev_latency_ms: float
    fallback_reason: str | None = None
    jev_input_tokens: int = 0


def use_large(p_small_ok: float, threshold: float) -> bool:
    """Route to the large model when Jev's probability that small is enough is below threshold."""
    return p_small_ok < threshold


class Router:
    def __init__(self, cfg: Config, jev: JevClient) -> None:
        self.cfg = cfg
        self.jev = jev

    def route(self, prompt: str, threshold: float | None = None) -> Decision:
        threshold = self.cfg.threshold if threshold is None else threshold
        start = time.perf_counter()
        # A worker thread enforces the deadline whatever the client does. On timeout the
        # thread is left to finish on its own; its result is discarded.
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self.jev.ask, prompt)
        try:
            result: JevResult = future.result(timeout=self.cfg.jev_timeout_s)
        except FutureTimeout:
            return self._fallback(start, f"jev timeout after {self.cfg.jev_timeout_s}s")
        except Exception as e:  # any Jev failure must fail safe to the large model
            return self._fallback(start, f"jev error: {type(e).__name__}")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        elapsed_ms = (time.perf_counter() - start) * 1000
        p = result.p_small_ok
        spec = self.cfg.large_model if use_large(p, threshold) else self.cfg.small_model
        return Decision(spec.id, p, elapsed_ms, jev_input_tokens=result.input_tokens)

    def _fallback(self, start: float, reason: str) -> Decision:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.warning("routing to large model: %s", reason)
        return Decision(self.cfg.large_model.id, None, elapsed_ms, fallback_reason=reason)


_default: Router | None = None


def route(prompt: str, threshold: float | None = None) -> Decision:
    """Route with the repo's config.yaml and the real Jev client."""
    global _default
    if _default is None:
        cfg = load_config()
        jev = TypeSafeJev(cfg.jev_model, timeout_s=cfg.jev_timeout_s, max_retries=0)
        _default = Router(cfg, jev)
    return _default.route(prompt, threshold)

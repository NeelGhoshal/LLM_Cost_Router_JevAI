from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import httpx

from router.config import Config, ModelSpec

OPENROUTER_URL = "https://openrouter.ai/api/v1"
REPEAT_ABORT = "token repeat limit reached"  # Ollama's 500 body when a model degenerates


@dataclass(frozen=True)
class LLMResult:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: str | None = None  # set when the model produced no usable output


class LLMClient(Protocol):
    def complete(self, model: str, prompt: str) -> LLMResult: ...


class OllamaClient:
    """Local Ollama via its native chat API, which lets us turn thinking off."""

    def __init__(
        self,
        base_url: str,
        *,
        max_output_tokens: int,
        temperature: float,
        seed: int,
        timeout_s: float = 180.0,
        max_retries: int = 2,
        retry_wait_s: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)
        self._options = {"temperature": temperature, "seed": seed, "num_predict": max_output_tokens}
        self._max_retries = max_retries
        self._retry_wait_s = retry_wait_s

    def complete(self, model: str, prompt: str) -> LLMResult:
        body = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": self._options,
        }
        # The local runner occasionally stalls or returns a 500 (both seen in the M5 run). Retry those.
        for attempt in range(self._max_retries + 1):
            start = time.perf_counter()
            try:
                r = self._http.post("/api/chat", json=body)
                r.raise_for_status()
                break
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
                if isinstance(e, httpx.HTTPStatusError) and REPEAT_ABORT in e.response.text:
                    return self._degenerate(model, prompt, start)
                client_error = isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500
                if client_error or attempt == self._max_retries:
                    raise
                time.sleep(self._retry_wait_s)
        latency_ms = (time.perf_counter() - start) * 1000
        body = r.json()
        return LLMResult(
            model=body.get("model", model),
            text=body["message"]["content"],
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
            latency_ms=latency_ms,
        )


    def _degenerate(self, model: str, prompt: str, start: float) -> LLMResult:
        """The model looped on one token until Ollama aborted it. That is a failed answer, not an
        infrastructure error: return empty text, and charge what a hosted API would have billed
        (it would have run to the output cap). Input tokens are estimated since none are reported."""
        return LLMResult(
            model=model,
            text="",
            input_tokens=round(len(prompt) / 3.5),
            output_tokens=self._options["num_predict"],
            latency_ms=(time.perf_counter() - start) * 1000,
            error=REPEAT_ABORT,
        )


class OpenAICompatClient:
    """Any OpenAI-compatible chat completions endpoint (OpenRouter by default)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        max_output_tokens: int,
        temperature: float,
        seed: int,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._params = {"max_tokens": max_output_tokens, "temperature": temperature, "seed": seed}

    def complete(self, model: str, prompt: str) -> LLMResult:
        start = time.perf_counter()
        r = self._http.post(
            "/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], **self._params},
        )
        r.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000
        body = r.json()
        usage = body.get("usage", {})
        return LLMResult(
            model=body.get("model", model),
            text=body["choices"][0]["message"]["content"] or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )


class FakeLLM:
    """Offline stand-in. answer(model, prompt) returns the text to reply with."""

    def __init__(self, answer: Callable[[str, str], str]) -> None:
        self._answer = answer
        self.calls: list[tuple[str, str]] = []

    def complete(self, model: str, prompt: str) -> LLMResult:
        self.calls.append((model, prompt))
        text = self._answer(model, prompt)
        return LLMResult(
            model=model,
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            latency_ms=1.0,
        )


def make_llm_client(spec: ModelSpec, cfg: Config) -> LLMClient:
    common = dict(max_output_tokens=cfg.max_output_tokens, temperature=cfg.temperature, seed=cfg.seed)
    if spec.provider == "ollama":
        return OllamaClient(cfg.ollama_url, **common)
    if spec.provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return OpenAICompatClient(OPENROUTER_URL, key, **common)
    raise ValueError(f"unknown provider: {spec.provider}")

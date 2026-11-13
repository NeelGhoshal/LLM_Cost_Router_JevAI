from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

QUESTION_KEY = "small_ok"

# Starting phrasing from the spec. Alternatives are compared on dev only (at most 3).
DEFAULT_INSTRUCTIONS = (
    "A small, inexpensive language model will answer the request in `prompt` "
    "correctly on its first attempt."
)


@dataclass(frozen=True)
class JevResult:
    probs: dict[str, float]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str

    @property
    def p_small_ok(self) -> float:
        return self.probs[QUESTION_KEY]


class JevClient(Protocol):
    def ask(self, prompt: str, instructions: Mapping[str, str] | None = None) -> JevResult: ...


class TypeSafeJev:
    """Jev through the official SDK. One request, one noul per entry in instructions.

    max_retries=0 with a short timeout_s is what the live router wants, so a slow or
    failing Jev surfaces quickly instead of the SDK retrying for up to 30 seconds.
    """

    def __init__(
        self,
        model: str,
        *,
        timeout_s: float = 60.0,
        max_retries: int = 5,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: Any = None,
    ) -> None:
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        self._model = model
        self._client = TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            transport=transport,
            timeout=timeout_s,
            retry=RetryPolicy(max_retries=max_retries, timeout=timeout_s),
        )

    def ask(self, prompt: str, instructions: Mapping[str, str] | None = None) -> JevResult:
        from typesafe_sdk import Noul

        instructions = instructions or {QUESTION_KEY: DEFAULT_INSTRUCTIONS}
        start = time.perf_counter()
        r = self._client.system_one(
            state={"prompt": prompt},
            questions={name: Noul(instructions=text) for name, text in instructions.items()},
            model=self._model,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return JevResult(
            probs={name: r.nouls[name].noul for name in instructions},
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            latency_ms=latency_ms,
            model=r.model,
        )

    def close(self) -> None:
        self._client.close()


class FakeJev:
    """Offline stand-in. p(prompt) returns the probability, or raises to simulate an error."""

    def __init__(self, p: Callable[[str], float], *, delay_s: float = 0.0, model: str = "jev-fake") -> None:
        self._p = p
        self._delay_s = delay_s
        self._model = model
        self.calls: list[str] = []

    def ask(self, prompt: str, instructions: Mapping[str, str] | None = None) -> JevResult:
        self.calls.append(prompt)
        if self._delay_s:
            time.sleep(self._delay_s)
        value = self._p(prompt)
        names = list(instructions) if instructions else [QUESTION_KEY]
        return JevResult(
            probs={name: value for name in names},
            input_tokens=len(prompt.split()) + 20,
            output_tokens=1,
            latency_ms=self._delay_s * 1000,
            model=self._model,
        )

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
class Phrasing:
    """One way of asking the noul. true/false are optional criteria for the yes/no boundary."""

    instructions: str
    true: str | None = None
    false: str | None = None


# The three phrasings compared on dev (the spec allows at most 3). The winner is named in config.yaml.
PHRASINGS: dict[str, Phrasing] = {
    "statement": Phrasing(DEFAULT_INSTRUCTIONS),
    "question": Phrasing(
        "Would a small, inexpensive language model answer the request in the prompt field "
        "correctly on its first attempt?"
    ),
    "criteria": Phrasing(
        "Is the request in the prompt field easy enough that a small, inexpensive language model "
        "would answer it correctly on its first attempt?",
        true="A routine request: common knowledge, a single well-known fact, or arithmetic with a few simple steps.",
        false="Needs specialist knowledge, several reasoning steps, careful calculation, "
        "or a choice between closely similar options.",
    ),
}

# Cascade checks: Jev reads the request AND the small model's response, and each check returns
# the probability that something is wrong with it. Candidates are compared on dev only (at most 3);
# the ones in use are named in config.yaml under cascade_checks.
_BROKEN = "The response has this problem."
_FINE = "The response does not have this problem."
CHECKS: dict[str, Phrasing] = {
    "wrong_answer": Phrasing(
        "Is the final answer given in `response` a wrong answer to the request in `prompt`?",
        true="The final answer is incorrect.",
        false="The final answer is correct.",
    ),
    "flawed_reasoning": Phrasing(
        "Does the reasoning in `response` contain a mistake: a false statement, a calculation error, "
        "a step that does not follow, or a final answer that does not match the reasoning?",
        true=_BROKEN,
        false=_FINE,
    ),
    "guess": Phrasing(
        "Does `response` reach its final answer without real justification, for example by guessing, "
        "by hesitating between alternatives, or by skipping the work the request in `prompt` needs?",
        true=_BROKEN,
        false=_FINE,
    ),
}

Instructions = Mapping[str, "str | Phrasing"]


def as_phrasing(value: "str | Phrasing") -> Phrasing:
    return value if isinstance(value, Phrasing) else Phrasing(value)


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
    def ask(self, prompt: str, instructions: Instructions | None = None) -> JevResult: ...

    def ask_state(self, state: Mapping[str, Any], instructions: Instructions) -> JevResult: ...


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

    def ask(self, prompt: str, instructions: Instructions | None = None) -> JevResult:
        return self.ask_state({"prompt": prompt}, instructions or {QUESTION_KEY: DEFAULT_INSTRUCTIONS})

    def ask_state(self, state: Mapping[str, Any], instructions: Instructions) -> JevResult:
        from typesafe_sdk import Noul, NoulCriteria

        questions = {}
        for name, value in instructions.items():
            ph = as_phrasing(value)
            criteria = NoulCriteria(true=ph.true, false=ph.false) if ph.true and ph.false else None
            questions[name] = Noul(instructions=ph.instructions, criteria=criteria)
        start = time.perf_counter()
        r = self._client.system_one(state=dict(state), questions=questions, model=self._model)
        latency_ms = (time.perf_counter() - start) * 1000
        return JevResult(
            probs={name: r.nouls[name].noul for name in instructions},
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            latency_ms=latency_ms,
            model=r.model,
        )

    def list_models(self) -> list[str]:
        """Key check. Sends no text."""
        return [m.name for m in self._client.models.list().models]

    def close(self) -> None:
        self._client.close()


class FakeJev:
    """Offline stand-in. p(prompt) returns the probability, or raises to simulate an error."""

    def __init__(self, p: Callable[[str], float], *, delay_s: float = 0.0, model: str = "jev-fake") -> None:
        self._p = p
        self._delay_s = delay_s
        self._model = model
        self.calls: list[str] = []
        self.states: list[dict[str, Any]] = []

    def ask(self, prompt: str, instructions: Instructions | None = None) -> JevResult:
        return self.ask_state({"prompt": prompt}, instructions or {QUESTION_KEY: DEFAULT_INSTRUCTIONS})

    def ask_state(self, state: Mapping[str, Any], instructions: Instructions) -> JevResult:
        """p receives the prompt for routing questions, and the response when the state has one."""
        prompt = state["prompt"]
        self.calls.append(prompt)
        self.states.append(dict(state))
        if self._delay_s:
            time.sleep(self._delay_s)
        value = self._p(state.get("response", prompt))
        names = list(instructions)
        return JevResult(
            probs={name: value for name in names},
            input_tokens=sum(len(str(v).split()) for v in state.values()) + 20,
            output_tokens=1,
            latency_ms=self._delay_s * 1000,
            model=self._model,
        )

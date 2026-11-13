from __future__ import annotations

from router.config import ModelSpec


def llm_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    """USD for one LLM call at the spec's per-million-token prices."""
    return (input_tokens * spec.usd_per_m_input + output_tokens * spec.usd_per_m_output) / 1e6


def jev_cost(input_tokens: int, usd_per_m_input: float) -> float:
    """USD for one Jev call. Jev bills input tokens only."""
    return input_tokens * usd_per_m_input / 1e6


def is_billed(spec: ModelSpec) -> bool:
    """Local providers cost nothing real; their prices are imputed for the cost axis only."""
    return spec.provider != "ollama"


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    """Tracks real spend and aborts the run once it passes max_usd."""

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent = 0.0

    def add(self, usd: float) -> None:
        self.spent += usd
        if self.spent > self.max_usd:
            raise BudgetExceeded(f"spent ${self.spent:.4f}, over the ${self.max_usd:.2f} cap")

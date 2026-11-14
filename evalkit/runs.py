"""Paid-once runs that fill the caches. Everything downstream replays these files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evalkit.cache import append_jsonl, done_keys, load_jsonl
from router.config import REPO_ROOT, Config, ModelSpec
from router.costs import BudgetGuard, is_billed, jev_cost, llm_cost
from evalkit.graders import extract_answer, grade
from router.jev import DEFAULT_INSTRUCTIONS, QUESTION_KEY, JevClient
from router.llm import LLMClient

PROMPTS_PATH = REPO_ROOT / "data" / "prompts.jsonl"
CACHE_DIR = REPO_ROOT / "cache"

Prompt = Mapping[str, Any]
Progress = Callable[[str], None]

CHARS_PER_TOKEN = 3.5  # deliberately low, so estimates run high
# Jev bills a fixed overhead per question on top of the state. The M3 smoke run measured
# about 300 tokens for our one-sentence noul; 350 keeps the estimate on the high side.
JEV_QUESTION_TOKENS = 350


def load_prompts(path: str | Path = PROMPTS_PATH, split: str | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    return [r for r in rows if split is None or r["split"] == split]


def spread(prompts: Sequence[Prompt], n: int) -> list[Prompt]:
    """First n prompts taken round-robin across sources, so a small run still covers every source."""
    by_source: dict[str, list[Prompt]] = {}
    for p in prompts:
        by_source.setdefault(p["source"], []).append(p)
    out: list[Prompt] = []
    i = 0
    while len(out) < min(n, len(prompts)):
        for rows in by_source.values():
            if i < len(rows) and len(out) < n:
                out.append(rows[i])
        i += 1
    return out


@dataclass(frozen=True)
class Estimate:
    llm_billed: float
    llm_imputed: float
    jev: float

    @property
    def real(self) -> float:
        return self.llm_billed + self.jev

    def describe(self) -> str:
        return (
            f"estimated real spend ${self.real:.4f} "
            f"(LLMs ${self.llm_billed:.4f}, Jev ${self.jev:.4f}); "
            f"imputed LLM cost for the chart ${self.llm_imputed:.4f}"
        )


def estimate(
    prompts: Sequence[Prompt], cfg: Config, *, models: bool = True, jev_questions: int = 1, response_tokens: int = 0
) -> Estimate:
    """Worst case: every answer uses the full output cap, and Jev bills the state once per question."""
    in_tokens = [len(p["prompt"]) / CHARS_PER_TOKEN for p in prompts]
    billed = imputed = 0.0
    if models:
        for spec in (cfg.small_model, cfg.large_model):
            cost = sum(llm_cost(spec, t, cfg.max_output_tokens) for t in in_tokens)
            imputed += cost
            billed += cost if is_billed(spec) else 0.0
    # response_tokens: extra state per call when Jev also reads a model response (cascade checks)
    jev = sum(
        jev_cost((t + response_tokens + JEV_QUESTION_TOKENS) * jev_questions, cfg.jev_usd_per_m_input)
        for t in in_tokens
    )
    return Estimate(billed, imputed, jev if jev_questions else 0.0)


def run_models(
    prompts: Sequence[Prompt],
    cfg: Config,
    clients: Mapping[str, LLMClient],
    cache_path: str | Path,
    guard: BudgetGuard,
    progress: Progress = lambda msg: None,
) -> int:
    """Send every prompt to both models once. Returns the number of new calls made."""
    done = done_keys(cache_path, ("id", "model"))
    made = 0
    # One model at a time, so a local server is not swapping models on every call.
    for role, spec in (("small", cfg.small_model), ("large", cfg.large_model)):
        todo = [p for p in prompts if (p["id"], spec.id) not in done]
        for i, p in enumerate(todo, 1):
            r = clients[role].complete(spec.id, p["prompt"])
            cost = llm_cost(spec, r.input_tokens, r.output_tokens)
            row = _model_row(p, role, spec, r.text, r.input_tokens, r.output_tokens, cost, r.latency_ms)
            if r.error:
                row["error"] = r.error
            append_jsonl(cache_path, row)
            made += 1
            progress(f"{role} {i}/{len(todo)} {p['id']} correct={row['correct']} {r.latency_ms / 1000:.1f}s")
            guard.add(row["billed"])
    return made


def _model_row(p: Prompt, role: str, spec: ModelSpec, text: str, in_tok: int, out_tok: int, cost: float, latency_ms: float) -> dict[str, Any]:
    return {
        "id": p["id"],
        "source": p["source"],
        "split": p["split"],
        "role": role,
        "model": spec.id,
        "answer": extract_answer(text),
        "correct": grade(p["source"], text, p["gold"]),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost": cost,  # at config prices; imputed when the provider is local
        "billed": cost if is_billed(spec) else 0.0,  # real spend
        "latency_ms": round(latency_ms, 1),
        "text": text,
    }


def run_checks(
    prompts: Sequence[Prompt],
    small_rows: Mapping[str, Mapping[str, Any]],
    cfg: Config,
    jev: JevClient,
    cache_path: str | Path,
    guard: BudgetGuard,
    checks: Mapping[str, Any],
    progress: Progress = lambda msg: None,
    tag: str | None = None,
) -> int:
    """Cascade checks: one Jev call per prompt on the request plus the small model's response.

    A response with no ANSWER line breaks the output contract and is escalated without asking
    Jev, so no call is made (and no fee paid) for it. Rows are tagged with the set of checks
    asked; only rows with the same tag count as done.
    """
    done = {r["id"] for r in load_jsonl(cache_path) if r.get("checks") == tag}
    todo = [p for p in prompts if p["id"] not in done and p["id"] in small_rows and small_rows[p["id"]]["answer"] is not None]
    for i, p in enumerate(todo, 1):
        r = jev.ask_state({"prompt": p["prompt"], "response": small_rows[p["id"]]["text"]}, checks)
        cost = jev_cost(r.input_tokens, cfg.jev_usd_per_m_input)
        append_jsonl(
            cache_path,
            {
                "id": p["id"],
                "source": p["source"],
                "split": p["split"],
                "checks": tag,
                "probs": r.probs,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost": cost,
                "latency_ms": round(r.latency_ms, 1),
                "model": r.model,
            },
        )
        progress(f"check {i}/{len(todo)} {p['id']} " + " ".join(f"{k}={v:.2f}" for k, v in r.probs.items()))
        guard.add(cost)
    return len(todo)


def run_jev(
    prompts: Sequence[Prompt],
    cfg: Config,
    jev: JevClient,
    cache_path: str | Path,
    guard: BudgetGuard,
    instructions: Mapping[str, Any] | None = None,
    progress: Progress = lambda msg: None,
    phrasing: str | None = None,
) -> int:
    """One Jev call per prompt. Returns the number of new calls made.

    phrasing names the wording behind the single routing question. Rows are tagged with it and
    only rows with the same tag count as done, so changing the phrasing never reuses old answers.
    """
    instructions = dict(instructions or {QUESTION_KEY: DEFAULT_INSTRUCTIONS})
    done = {r["id"] for r in load_jsonl(cache_path) if r.get("phrasing") == phrasing}
    todo = [p for p in prompts if p["id"] not in done]
    for i, p in enumerate(todo, 1):
        r = jev.ask(p["prompt"], instructions)
        cost = jev_cost(r.input_tokens, cfg.jev_usd_per_m_input)
        append_jsonl(
            cache_path,
            {
                "id": p["id"],
                "source": p["source"],
                "split": p["split"],
                "phrasing": phrasing,
                "probs": r.probs,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost": cost,
                "latency_ms": round(r.latency_ms, 1),
                "model": r.model,
            },
        )
        progress(f"jev {i}/{len(todo)} {p['id']} " + " ".join(f"{k}={v:.3f}" for k, v in r.probs.items()))
        guard.add(cost)
    return len(todo)

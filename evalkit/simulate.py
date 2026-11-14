"""Replay routing policies from the caches. No API calls anywhere in this module.

Two families of policy:

  pre-routing  decide from the prompt alone, then call one model.
  cascade      the small model always answers; a check on its response decides whether the
               large model answers too. Cost = small + check fee + large when escalated.

Costs are mean USD per query at the config prices. Jev policies include Jev's own cost.
The oracle and the length baselines pay nothing for their decision.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Table:
    """One row per prompt, as parallel arrays."""

    ids: np.ndarray
    source: np.ndarray
    split: np.ndarray
    length: np.ndarray  # prompt length in characters
    p: np.ndarray  # Jev's probability that the small model is enough
    small_correct: np.ndarray
    large_correct: np.ndarray
    small_cost: np.ndarray
    large_cost: np.ndarray
    jev_cost: np.ndarray
    jev_latency_ms: np.ndarray
    # cascade inputs
    small_answered: np.ndarray  # False when the small response has no ANSWER line
    response_chars: np.ndarray  # length of the small model's response
    flag: np.ndarray  # Jev's probability that the small answer is wrong; NaN when not checked
    check_cost: np.ndarray  # Jev fee for the check; 0 when not checked
    check_latency_ms: np.ndarray  # NaN when not checked
    small_seconds: np.ndarray  # measured wall-clock time of each model call
    large_seconds: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, index: np.ndarray) -> "Table":
        """Rows selected by a boolean mask or an index array (repeats allowed, for the bootstrap)."""
        return Table(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def where_split(self, split: str) -> "Table":
        return self.take(self.split == split)


def build_table(
    prompts: Sequence[Mapping[str, Any]],
    llm_rows: Sequence[Mapping[str, Any]],
    jev_rows: Sequence[Mapping[str, Any]],
    question: str,
    check_rows: Sequence[Mapping[str, Any]] = (),
    checks: Sequence[str] = (),
) -> Table:
    """Join prompts with the caches. Prompts missing from the model or routing cache are left out.

    With several checks the flag is the highest of them, so one strong flag is enough.
    A response with no ANSWER line is never checked: the cascade escalates it for free.
    """
    llm = {(r["id"], r["role"]): r for r in llm_rows}
    jev = {r["id"]: r for r in jev_rows if question in r["probs"]}
    rows = [p for p in prompts if (p["id"], "small") in llm and (p["id"], "large") in llm and p["id"] in jev]
    chk = {r["id"]: r for r in check_rows if all(c in r["probs"] for c in checks)} if checks else {}

    def checked(p: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return chk.get(p["id"]) if llm[(p["id"], "small")]["answer"] is not None else None

    def flag(p: Mapping[str, Any]) -> float:
        r = checked(p)
        return max(r["probs"][c] for c in checks) if r else float("nan")

    def col(fn: Callable[[Mapping[str, Any]], Any], dtype: Any) -> np.ndarray:
        return np.array([fn(p) for p in rows], dtype=dtype)

    return Table(
        ids=col(lambda p: p["id"], object),
        source=col(lambda p: p["source"], object),
        split=col(lambda p: p["split"], object),
        length=col(lambda p: len(p["prompt"]), float),
        p=col(lambda p: jev[p["id"]]["probs"][question], float),
        small_correct=col(lambda p: llm[(p["id"], "small")]["correct"], bool),
        large_correct=col(lambda p: llm[(p["id"], "large")]["correct"], bool),
        small_cost=col(lambda p: llm[(p["id"], "small")]["cost"], float),
        large_cost=col(lambda p: llm[(p["id"], "large")]["cost"], float),
        jev_cost=col(lambda p: jev[p["id"]]["cost"], float),
        jev_latency_ms=col(lambda p: jev[p["id"]]["latency_ms"], float),
        small_answered=col(lambda p: llm[(p["id"], "small")]["answer"] is not None, bool),
        response_chars=col(lambda p: len(llm[(p["id"], "small")].get("text", "")), float),
        flag=col(flag, float),
        check_cost=col(lambda p: checked(p)["cost"] if checked(p) else 0.0, float),
        check_latency_ms=col(lambda p: checked(p)["latency_ms"] if checked(p) else float("nan"), float),
        small_seconds=col(lambda p: llm[(p["id"], "small")]["latency_ms"] / 1000, float),
        large_seconds=col(lambda p: llm[(p["id"], "large")]["latency_ms"] / 1000, float),
    )


def time_view(t: Table) -> Table:
    """The same table with every cost replaced by measured seconds, so each policy can be
    replayed with wall-clock time as the cost. For local models this is the real cost."""
    return replace(
        t,
        small_cost=t.small_seconds,
        large_cost=t.large_seconds,
        jev_cost=t.jev_latency_ms / 1000,
        check_cost=np.nan_to_num(t.check_latency_ms, nan=0.0) / 1000,
    )


# ---------------------------------------------------------------- policies


def jev_use_large(t: Table, threshold: float) -> np.ndarray:
    """Same rule as router.route.use_large: large when p_small_ok is below the threshold."""
    return t.p < threshold


def length_use_large(t: Table, max_chars: float) -> np.ndarray:
    """Baseline: a long prompt goes to the large model."""
    return t.length > max_chars


def oracle_use_large(t: Table) -> np.ndarray:
    """Large only when small is wrong and large is right."""
    return ~t.small_correct & t.large_correct


def evaluate(t: Table, use_large: np.ndarray, pays_jev: bool = False) -> dict[str, float]:
    correct = np.where(use_large, t.large_correct, t.small_correct)
    cost = np.where(use_large, t.large_cost, t.small_cost) + (t.jev_cost if pays_jev else 0.0)
    return {"accuracy": float(correct.mean()), "cost": float(cost.mean()), "frac_large": float(use_large.mean())}


# ---------------------------------------------------------------- cascade


def cascade_escalate(t: Table, threshold: float) -> np.ndarray:
    """Escalate when the response breaks the output contract or Jev's flag reaches the threshold.
    An answered response that was never checked has no flag and is treated as flagged."""
    return ~t.small_answered | (np.nan_to_num(t.flag, nan=1.0) >= threshold)


def response_length_escalate(t: Table, max_chars: float) -> np.ndarray:
    """Baseline cascade with no Jev: escalate contract failures and long (laboured) responses."""
    return ~t.small_answered | (t.response_chars > max_chars)


def evaluate_cascade(t: Table, escalate: np.ndarray, pays_check: bool) -> dict[str, float]:
    """The small model is always paid for. The large model's answer replaces it when escalated."""
    correct = np.where(escalate, t.large_correct, t.small_correct)
    cost = t.small_cost + np.where(escalate, t.large_cost, 0.0) + (t.check_cost if pays_check else 0.0)
    return {"accuracy": float(correct.mean()), "cost": float(cost.mean()), "frac_large": float(escalate.mean())}


def sweep_cascade(t: Table) -> list[dict[str, float]]:
    """From contract failures only (threshold above every flag) to escalating everything."""
    flags = t.flag[~np.isnan(t.flag)]
    thresholds = [float("inf")] + sorted(set(flags.tolist()), reverse=True) + [0.0]
    return [{"threshold": th, **evaluate_cascade(t, cascade_escalate(t, th), pays_check=True)} for th in thresholds]


def sweep_response_length(t: Table) -> list[dict[str, float]]:
    cutoffs = [float("inf")] + sorted(set(t.response_chars.tolist()), reverse=True) + [-1.0]
    return [
        {"threshold": c, **evaluate_cascade(t, response_length_escalate(t, c), pays_check=False)} for c in cutoffs
    ]


def always_small(t: Table) -> dict[str, float]:
    return evaluate(t, np.zeros(len(t), bool))


def always_large(t: Table) -> dict[str, float]:
    return evaluate(t, np.ones(len(t), bool))


def oracle(t: Table) -> dict[str, float]:
    return evaluate(t, oracle_use_large(t))


def random_accuracy_at_cost(t: Table, cost: float) -> float:
    """Expected accuracy of random routing that spends `cost` per query (the straight line
    between always-small and always-large, extended if cost falls outside it)."""
    small, large = always_small(t), always_large(t)
    frac = (cost - small["cost"]) / (large["cost"] - small["cost"])
    return small["accuracy"] + frac * (large["accuracy"] - small["accuracy"])


def sweep_jev(t: Table) -> list[dict[str, float]]:
    """Every distinct operating point, from always-small (threshold 0) to always-large."""
    thresholds = [0.0] + sorted(set(np.nextafter(t.p, np.inf).tolist()))
    return [{"threshold": th, **evaluate(t, jev_use_large(t, th), pays_jev=True)} for th in thresholds]


def sweep_length(t: Table) -> list[dict[str, float]]:
    """From always-small (cutoff at the longest prompt) to always-large (cutoff below the shortest)."""
    cutoffs = sorted(set(t.length.tolist()), reverse=True) + [float(t.length.min()) - 1]
    return [{"threshold": c, **evaluate(t, length_use_large(t, c))} for c in cutoffs]


def cheapest_within(sweep: Sequence[Mapping[str, float]], min_accuracy: float) -> Mapping[str, float]:
    """Cheapest point that reaches min_accuracy. If none does, the most accurate point."""
    ok = [pt for pt in sweep if pt["accuracy"] >= min_accuracy - 1e-12]
    if ok:
        return min(ok, key=lambda pt: (pt["cost"], -pt["accuracy"]))
    return max(sweep, key=lambda pt: (pt["accuracy"], -pt["cost"]))


# ---------------------------------------------------------------- metrics


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a positive outranks a negative, ties counting half. NaN if one class is empty."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    i = 0
    while i < len(scores):  # average ranks over ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration(p: np.ndarray, correct: np.ndarray, bins: int = 5) -> list[dict[str, float]]:
    """Equal-count bins of Jev probability against observed small-model accuracy."""
    order = np.argsort(p, kind="mergesort")
    out = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            out.append(
                {
                    "p_min": float(p[chunk].min()),
                    "p_max": float(p[chunk].max()),
                    "n": int(len(chunk)),
                    "mean_p": float(p[chunk].mean()),
                    "small_accuracy": float(correct[chunk].mean()),
                }
            )
    return out


def calibration_table(t: Table, bins: int = 5) -> list[dict[str, float]]:
    """Calibration bins, plus what the large model does in each bin: its accuracy and its cost.
    Those columns show whether sending a bin to the large model is worth it."""
    order = np.argsort(t.p, kind="mergesort")
    out = calibration(t.p, t.small_correct, bins)
    for row, chunk in zip(out, [c for c in np.array_split(order, bins) if len(c)]):
        row["large_accuracy"] = float(t.large_correct[chunk].mean())
        row["mean_small_cost"] = float(t.small_cost[chunk].mean())
        row["mean_large_cost"] = float(t.large_cost[chunk].mean())
    return out


def half_share_thresholds(t: Table) -> tuple[float, float]:
    """Jev threshold and length cutoff that each send half of t to the large model."""
    return float(np.median(t.p)), float(np.median(t.length))


def headline_stats(
    t: Table,
    jev_threshold: float,
    length_cutoff: float,
    jev_half_threshold: float | None = None,
    length_half_cutoff: float | None = None,
    cascade_threshold: float | None = None,
    response_cutoff: float | None = None,
) -> dict[str, float]:
    """Every scalar reported with a confidence interval, for one sample of prompts.

    The *_half_* arguments are a second, price-free operating point: thresholds that sent half
    of the dev prompts to the large model. Random routing at the same share is the comparison.
    """
    small, large, orc = always_small(t), always_large(t), oracle(t)
    jev = evaluate(t, jev_use_large(t, jev_threshold), pays_jev=True)
    length = evaluate(t, length_use_large(t, length_cutoff))
    auroc_jev = auroc(t.p, t.small_correct)
    auroc_length = auroc(-t.length, t.small_correct)  # shorter prompt = predicted easier
    large_needed = oracle_use_large(t)
    out = _headline(small, large, orc, jev, length, auroc_jev, auroc_length)
    # Can the score find the prompts where the large model actually helps?
    out["auroc_large_needed_jev"] = auroc(-t.p, large_needed)
    out["auroc_large_needed_length"] = auroc(t.length, large_needed)
    half_points = (
        ("jev_half", jev_half_threshold, jev_use_large, True),
        ("length_half", length_half_cutoff, length_use_large, False),
    )
    for name, threshold, rule, pays in half_points:
        if threshold is None:
            continue
        r = evaluate(t, rule(t, threshold), pays_jev=pays)
        random_same_share = small["accuracy"] + r["frac_large"] * (large["accuracy"] - small["accuracy"])
        out[f"{name}_accuracy"] = r["accuracy"]
        out[f"{name}_cost"] = r["cost"]
        out[f"{name}_frac_large"] = r["frac_large"]
        out[f"{name}_accuracy_minus_random_same_share"] = r["accuracy"] - random_same_share

    cascades = (
        ("cascade", cascade_threshold, cascade_escalate, True),
        ("response_length", response_cutoff, response_length_escalate, False),
    )
    for name, threshold, rule, pays in cascades:
        if threshold is None:
            continue
        r = evaluate_cascade(t, rule(t, threshold), pays_check=pays)
        out[f"{name}_accuracy"] = r["accuracy"]
        out[f"{name}_cost"] = r["cost"]
        out[f"{name}_frac_large"] = r["frac_large"]
        out[f"{name}_accuracy_minus_large"] = r["accuracy"] - large["accuracy"]
        out[f"{name}_cost_saved_pct"] = 100 * (1 - r["cost"] / large["cost"])
        out[f"{name}_accuracy_minus_random_at_cost"] = r["accuracy"] - random_accuracy_at_cost(t, r["cost"])
    if cascade_threshold is not None:
        seen = t.small_answered & ~np.isnan(t.flag)
        wrong = ~t.small_correct
        out["auroc_check"] = auroc(t.flag[seen], wrong[seen])
        out["auroc_response_length"] = auroc(t.response_chars[seen], wrong[seen])
        out["auroc_check_minus_response_length"] = out["auroc_check"] - out["auroc_response_length"]
        out["mean_check_cost"] = float(t.check_cost.mean())
    return out


def _headline(small, large, orc, jev, length, auroc_jev, auroc_length) -> dict[str, float]:
    return {
        "always_small_accuracy": small["accuracy"],
        "always_small_cost": small["cost"],
        "always_large_accuracy": large["accuracy"],
        "always_large_cost": large["cost"],
        "oracle_accuracy": orc["accuracy"],
        "oracle_cost": orc["cost"],
        "jev_accuracy": jev["accuracy"],
        "jev_cost": jev["cost"],
        "jev_frac_large": jev["frac_large"],
        "jev_accuracy_minus_large": jev["accuracy"] - large["accuracy"],
        "jev_cost_saved_pct": 100 * (1 - jev["cost"] / large["cost"]),
        "length_accuracy": length["accuracy"],
        "length_cost": length["cost"],
        "length_frac_large": length["frac_large"],
        "length_accuracy_minus_large": length["accuracy"] - large["accuracy"],
        "length_cost_saved_pct": 100 * (1 - length["cost"] / large["cost"]),
        "auroc_jev": auroc_jev,
        "auroc_length": auroc_length,
        "auroc_jev_minus_length": auroc_jev - auroc_length,
    }


def bootstrap_ci(
    t: Table, stat: Callable[[Table], Mapping[str, float]], n_resamples: int = 1000, seed: int = 7
) -> dict[str, dict[str, float]]:
    """Point estimate with a 95% percentile interval, resampling prompts with replacement."""
    rng = np.random.default_rng(seed)
    point = stat(t)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_resamples):
        sample = stat(t.take(rng.integers(0, len(t), len(t))))
        for k, v in sample.items():
            draws[k].append(v)
    out = {}
    for k, v in point.items():
        lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
        out[k] = {"value": float(v), "ci_low": float(lo), "ci_high": float(hi)}
    return out


def latency(ms: np.ndarray) -> dict[str, float]:
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)), "n": int(len(ms))}


# ---------------------------------------------------------------- curve-level comparison with random


def gain_over_random(t: Table, point: Mapping[str, float]) -> float:
    """Accuracy of an operating point minus random routing's accuracy at the same cost."""
    return point["accuracy"] - random_accuracy_at_cost(t, point["cost"])


def best_gain_point(t: Table, sweep: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    """The operating point furthest above the random line, among those no dearer than always-large.
    Used on dev to pick a second threshold that does not depend on matching always-large."""
    ceiling = always_large(t)["cost"]
    candidates = [pt for pt in sweep if pt["cost"] <= ceiling] or list(sweep)
    return max(candidates, key=lambda pt: (gain_over_random(t, pt), -pt["cost"]))


def mean_gain_over_random(t: Table, sweep: Sequence[Mapping[str, float]], grid: int = 101) -> float:
    """Average accuracy advantage over random routing across every budget from always-small to
    always-large. No threshold is chosen, so nothing here can be tuned to the split.

    The accuracy of the policy at a budget is read off its sweep by linear interpolation. Below
    its cheapest point the fallback is always-small, which is always available.
    """
    small, large = always_small(t), always_large(t)
    pts = sorted(sweep, key=lambda pt: pt["cost"])
    costs = np.array([pt["cost"] for pt in pts])
    accs = np.array([pt["accuracy"] for pt in pts])
    budgets = np.linspace(small["cost"], large["cost"], grid)
    policy = np.where(budgets < costs[0], small["accuracy"], np.interp(budgets, costs, accs))
    frac = (budgets - small["cost"]) / (large["cost"] - small["cost"])
    random = small["accuracy"] + frac * (large["accuracy"] - small["accuracy"])
    return float((policy - random).mean())

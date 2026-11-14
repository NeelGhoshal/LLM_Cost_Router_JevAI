"""Replay every routing policy from the caches and write results/metrics.json. No API calls.

Thresholds are chosen on dev. Every reported number comes from test.

Each policy is replayed under two cost views:
  usd      config prices per token (imputed when the models run locally) plus Jev's real fee
  seconds  measured wall-clock time of each call, including the Jev round trip
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.cache import load_jsonl  # noqa: E402
from evalkit.cli import utf8_stdout  # noqa: E402
from evalkit.runs import CACHE_DIR, load_prompts  # noqa: E402
from evalkit.simulate import (  # noqa: E402
    Table,
    always_large,
    always_small,
    best_gain_point,
    bootstrap_ci,
    build_table,
    calibration_table,
    cascade_escalate,
    cheapest_within,
    evaluate_cascade,
    gain_over_random,
    half_share_thresholds,
    headline_stats,
    latency,
    mean_gain_over_random,
    oracle,
    sweep_cascade,
    sweep_jev,
    sweep_length,
    sweep_response_length,
    time_view,
)
from router.config import REPO_ROOT, load_config  # noqa: E402
from router.jev import CHECKS, PHRASINGS, QUESTION_KEY  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
WITHIN_POINTS = 1.0  # headline: stay within this many accuracy points of always-large
N_RESAMPLES = 1000

# name, sweep function, key of its threshold in headline_stats
POLICIES = [
    ("jev", sweep_jev, "jev_threshold"),
    ("length", sweep_length, "length_cutoff"),
    ("cascade", sweep_cascade, "cascade_threshold"),
    ("response_length", sweep_response_length, "response_cutoff"),
]


def view_block(dev: Table, test: Table, seed: int, has_cascade: bool) -> dict:
    """Tune on dev, report on test, for one cost view."""
    policies = [p for p in POLICIES if has_cascade or p[0] in ("jev", "length")]
    target = always_large(dev)["accuracy"] - WITHIN_POINTS / 100
    picks = {name: cheapest_within(sweep(dev), target) for name, sweep, _ in policies}
    jev_half, length_half = half_share_thresholds(dev)
    thresholds = {key: picks[name]["threshold"] for name, _, key in policies}

    # Second cascade operating point: the dev threshold furthest above the random line.
    knee = best_gain_point(dev, sweep_cascade(dev)) if has_cascade else None

    def stat(t: Table) -> dict[str, float]:
        out = headline_stats(t, jev_half_threshold=jev_half, length_half_cutoff=length_half, **thresholds)
        for name, sweep, _ in policies:
            out[f"{name}_mean_gain_over_random"] = mean_gain_over_random(t, sweep(t))
        if knee is not None:
            large = always_large(t)
            r = evaluate_cascade(t, cascade_escalate(t, knee["threshold"]), pays_check=True)
            out["cascade_knee_accuracy"] = r["accuracy"]
            out["cascade_knee_cost"] = r["cost"]
            out["cascade_knee_frac_large"] = r["frac_large"]
            out["cascade_knee_accuracy_minus_large"] = r["accuracy"] - large["accuracy"]
            out["cascade_knee_cost_saved_pct"] = 100 * (1 - r["cost"] / large["cost"])
            out["cascade_knee_accuracy_minus_random_at_cost"] = gain_over_random(t, r)
        return out

    stats = bootstrap_ci(test, stat, n_resamples=N_RESAMPLES, seed=seed)
    return {
        "dev": {
            "always_small": always_small(dev),
            "always_large": always_large(dev),
            "target_accuracy": target,
            "operating_points": {name: dict(pick) for name, pick in picks.items()},
            "cascade_knee": dict(knee) if knee else None,
            "half_share_jev_threshold": jev_half,
            "half_share_length_cutoff": length_half,
        },
        "test": {
            "stats": stats,
            "policies": {"always_small": always_small(test), "always_large": always_large(test), "oracle": oracle(test)},
            "curves": {name: sweep(test) for name, sweep, _ in policies},
        },
    }


def main() -> None:
    utf8_stdout()
    cfg = load_config()
    prompts = load_prompts()
    jev_rows = [r for r in load_jsonl(CACHE_DIR / "jev.jsonl") if r.get("phrasing") == cfg.jev_phrasing]
    check_tag = "+".join(cfg.cascade_checks)
    check_rows = [r for r in load_jsonl(CACHE_DIR / "jev_cascade.jsonl") if r.get("checks") == check_tag]
    table = build_table(
        prompts, load_jsonl(CACHE_DIR / "llm.jsonl"), jev_rows, QUESTION_KEY, check_rows, cfg.cascade_checks
    )
    dev, test = table.where_split("dev"), table.where_split("test")
    if len(table) < len(prompts):
        print(f"warning: only {len(table)} of {len(prompts)} prompts are in both caches")
    if not len(dev) or not len(test):
        sys.exit("caches are empty for dev or test. Run eval/run_models.py and eval/run_jev.py first.")
    has_cascade = bool(check_rows)
    if not has_cascade:
        print("note: cache/jev_cascade.jsonl is empty, so the cascade is left out. Run eval/run_checks.py.")

    by_source = {}
    for source in sorted(set(test.source.tolist())):
        sub = test.take(test.source == source)
        by_source[source] = {
            "n": len(sub),
            "always_small_accuracy": always_small(sub)["accuracy"],
            "always_large_accuracy": always_large(sub)["accuracy"],
            "mean_p_small_ok": float(sub.p.mean()),
        }
    checked = test.take(test.small_answered)
    passed = {}
    if has_cascade:
        # How trustworthy is a small answer that Jev lets through, at a few thresholds?
        for th in (0.2, 0.3, 0.5, 0.7):
            keep = test.take(test.small_answered & (test.flag < th))
            passed[str(th)] = {
                "share_kept": len(keep) / len(test),
                "small_accuracy_when_kept": float(keep.small_correct.mean()) if len(keep) else float("nan"),
            }

    metrics = {
        "meta": {
            "jev_model_configured": cfg.jev_model,
            "jev_models_reported": sorted({r["model"] for r in jev_rows + check_rows}),
            "jev_phrasing": cfg.jev_phrasing,
            "jev_instructions": PHRASINGS[cfg.jev_phrasing].instructions,
            "cascade_checks": {name: CHECKS[name].instructions for name in cfg.cascade_checks},
            "small_model": cfg.small_model.id,
            "large_model": cfg.large_model.id,
            "prices_usd_per_m": {
                "small": [cfg.small_model.usd_per_m_input, cfg.small_model.usd_per_m_output],
                "large": [cfg.large_model.usd_per_m_input, cfg.large_model.usd_per_m_output],
                "jev_input": cfg.jev_usd_per_m_input,
            },
            "prices_are_imputed": cfg.small_model.provider == "ollama",
            "n_dev": len(dev),
            "n_test": len(test),
            "bootstrap_resamples": N_RESAMPLES,
            "seed": cfg.seed,
            "within_points": WITHIN_POINTS,
        },
        "views": {
            "usd": view_block(dev, test, cfg.seed, has_cascade),
            "seconds": view_block(time_view(dev), time_view(test), cfg.seed, has_cascade),
        },
        "test": {
            "calibration": calibration_table(test, bins=5),
            "by_source": by_source,
            "mean_jev_cost": float(test.jev_cost.mean()),
            "small_unanswered_share": float((~test.small_answered).mean()),
            "small_accuracy_when_answered": float(checked.small_correct.mean()),
            "passed_by_check": passed,
        },
        "jev_latency": {
            "routing_question": latency(table.jev_latency_ms),
            "cascade_check": latency(table.check_latency_ms[table.small_answered]) if has_cascade else None,
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    print_summary(metrics, has_cascade)
    print(f"\nwrote {out}")


ROWS = [
    ("always-small", "always_small"),
    ("always-large", "always_large"),
    ("oracle (pre-routing)", "oracle"),
    ("prompt length", "length"),
    ("Jev pre-routing", "jev"),
    ("response-length cascade", "response_length"),
    ("Jev cascade", "cascade"),
    ("Jev cascade, best-gain", "cascade_knee"),
]


def print_summary(m: dict, has_cascade: bool) -> None:
    meta = m["meta"]
    print(f"dev {meta['n_dev']} prompts, test {meta['n_test']} prompts, Jev {meta['jev_models_reported']}")
    print(f"pre-routing phrasing: {meta['jev_phrasing']}; cascade checks: {list(meta['cascade_checks'])}")

    for view, unit, scale, fmt in (("usd", "USD per 1,000 queries", 1000, ".4f"), ("seconds", "seconds per query", 1, ".2f")):
        block = m["views"][view]
        s = block["test"]["stats"]

        def ci(key: str, k: float = 1.0, f: str = ".3f") -> str:
            v = s[key]
            return f"{v['value'] * k:{f}} [{v['ci_low'] * k:{f}}, {v['ci_high'] * k:{f}}]"

        print(f"\n=== cost view: {unit} (test split, 95% bootstrap CI) ===")
        picks = block["dev"]["operating_points"]
        print("dev-chosen thresholds: " + ", ".join(f"{k} {v['threshold']:.3g}" for k, v in picks.items()))
        print(f"{'policy':24s} {'accuracy':24s} {'cost':26s} {'to large':>9s} {'saved vs large':>22s}")
        for name, key in ROWS:
            if f"{key}_accuracy" not in s:
                continue
            share = f"{s[key + '_frac_large']['value']:.0%}" if f"{key}_frac_large" in s else ""
            saved = ci(key + "_cost_saved_pct", 1, ".1f") + "%" if f"{key}_cost_saved_pct" in s else ""
            print(f"{name:24s} {ci(key + '_accuracy'):24s} {ci(key + '_cost', scale, fmt):26s} {share:>9s} {saved:>22s}")
        if has_cascade:
            knee = block["dev"]["cascade_knee"]
            print(f"Jev cascade vs always-large: accuracy {ci('cascade_accuracy_minus_large', 100, '+.1f')} points")
            print(
                f"Jev cascade at the dev best-gain threshold ({knee['threshold']:.2f}): accuracy vs always-large "
                f"{ci('cascade_knee_accuracy_minus_large', 100, '+.1f')} points, "
                f"vs random at equal cost {ci('cascade_knee_accuracy_minus_random_at_cost', 100, '+.1f')} points"
            )
        print("mean accuracy advantage over random routing across all budgets (points):")
        for name, key in ROWS:
            if f"{key}_mean_gain_over_random" in s:
                print(f"  {name:24s} {ci(key + '_mean_gain_over_random', 100, '+.1f')}")

    s = m["views"]["usd"]["test"]["stats"]

    def ci(key: str, f: str = ".3f") -> str:
        v = s[key]
        return f"{v['value']:{f}} [{v['ci_low']:{f}}, {v['ci_high']:{f}}]"

    print("\n=== signal quality (test split) ===")
    print(f"pre-routing, AUROC for 'small is correct': Jev {ci('auroc_jev')}, prompt length {ci('auroc_length')}")
    print(f"pre-routing, AUROC for 'large is needed': Jev {ci('auroc_large_needed_jev')}, prompt length {ci('auroc_large_needed_length')}")
    if has_cascade:
        print(
            f"cascade check, AUROC for 'small answer is wrong': Jev {ci('auroc_check')}, "
            f"response length {ci('auroc_response_length')}, difference {ci('auroc_check_minus_response_length', '+.3f')}"
        )
        t = m["test"]
        print(f"small model: {t['small_unanswered_share']:.1%} of responses have no ANSWER line; accuracy on the rest {t['small_accuracy_when_answered']:.3f}")
        print("small answers that pass the check (flag below threshold):")
        for th, row in t["passed_by_check"].items():
            print(f"  threshold {th}: keeps {row['share_kept']:.0%} of prompts, small accuracy on those {row['small_accuracy_when_kept']:.3f}")
    lat = m["jev_latency"]
    print(f"\nJev latency: routing question p50 {lat['routing_question']['p50_ms']:.0f} ms, p95 {lat['routing_question']['p95_ms']:.0f} ms", end="")
    if lat["cascade_check"]:
        print(f"; cascade check p50 {lat['cascade_check']['p50_ms']:.0f} ms, p95 {lat['cascade_check']['p95_ms']:.0f} ms")
    else:
        print()


if __name__ == "__main__":
    main()

import numpy as np
import pytest

from router.route import use_large
from evalkit.simulate import (
    always_large,
    always_small,
    auroc,
    best_gain_point,
    bootstrap_ci,
    build_table,
    calibration,
    calibration_table,
    cascade_escalate,
    evaluate_cascade,
    gain_over_random,
    mean_gain_over_random,
    response_length_escalate,
    sweep_cascade,
    sweep_response_length,
    time_view,
    cheapest_within,
    evaluate,
    half_share_thresholds,
    headline_stats,
    jev_use_large,
    latency,
    length_use_large,
    oracle,
    random_accuracy_at_cost,
    sweep_jev,
    sweep_length,
)

# id, prompt length, p_small_ok, small correct, large correct
SPEC = [
    ("a", 10, 0.9, True, True),
    ("b", 20, 0.8, True, True),
    ("c", 30, 0.3, False, True),
    ("d", 40, 0.2, False, False),
    ("e", 50, 0.6, True, False),
    ("f", 60, 0.1, False, True),
]
SMALL_COST, LARGE_COST, JEV_COST = 1.0, 10.0, 0.5


def make_table(spec=SPEC):
    prompts = [{"id": i, "source": "s", "split": "test", "prompt": "x" * n} for i, n, *_ in spec]
    llm = []
    for i, n, _, s, l in spec:
        llm.append({"id": i, "role": "small", "correct": s, "cost": SMALL_COST, "answer": "A", "text": "r" * n, "latency_ms": 1000.0})
        llm.append({"id": i, "role": "large", "correct": l, "cost": LARGE_COST, "answer": "A", "text": "", "latency_ms": 4000.0})
    jev = [{"id": i, "probs": {"small_ok": p}, "cost": JEV_COST, "latency_ms": 50.0 + k} for k, (i, _, p, _, _) in enumerate(spec)]
    return build_table(prompts, llm, jev, "small_ok")


def test_build_table_skips_prompts_missing_from_a_cache():
    prompts = [{"id": i, "source": "s", "split": "dev", "prompt": "x"} for i in ("a", "b", "c")]
    row = {"correct": True, "cost": 1.0, "answer": "A", "text": "", "latency_ms": 1.0}
    llm = [{"id": "a", "role": r, **row} for r in ("small", "large")]
    llm += [{"id": "b", "role": "small", **row}]  # b has no large row
    jev = [{"id": i, "probs": {"small_ok": 0.5}, "cost": 0.1, "latency_ms": 1.0} for i in ("a", "b")]
    t = build_table(prompts, llm, jev, "small_ok")
    assert list(t.ids) == ["a"]
    assert len(build_table(prompts, llm, jev, "other_question")) == 0


def test_baselines():
    t = make_table()
    assert always_small(t) == {"accuracy": pytest.approx(3 / 6), "cost": 1.0, "frac_large": 0.0}
    assert always_large(t) == {"accuracy": pytest.approx(4 / 6), "cost": 10.0, "frac_large": 1.0}


def test_oracle_uses_large_only_when_it_helps():
    t = make_table()
    o = oracle(t)
    assert o["frac_large"] == pytest.approx(2 / 6)  # c and f
    assert o["accuracy"] == pytest.approx(5 / 6)  # everything except d
    assert o["cost"] == pytest.approx((4 * 1 + 2 * 10) / 6)


def test_jev_policy_includes_jev_cost_and_matches_the_live_rule():
    t = make_table()
    mask = jev_use_large(t, 0.5)
    assert list(mask) == [use_large(p, 0.5) for p in t.p]
    r = evaluate(t, mask, pays_jev=True)
    # large for c, d, f -> right on a, b, c, e, f
    assert r["accuracy"] == pytest.approx(5 / 6)
    assert r["cost"] == pytest.approx((3 * 1 + 3 * 10) / 6 + JEV_COST)
    assert evaluate(t, mask)["cost"] == pytest.approx((3 * 1 + 3 * 10) / 6)


def test_sweep_jev_runs_from_always_small_to_always_large():
    t = make_table()
    sweep = sweep_jev(t)
    assert len(sweep) == 7  # six distinct probabilities plus threshold 0
    assert sweep[0]["frac_large"] == 0 and sweep[-1]["frac_large"] == 1
    assert sweep[0]["cost"] == pytest.approx(SMALL_COST + JEV_COST)
    assert sweep[-1]["cost"] == pytest.approx(LARGE_COST + JEV_COST)
    assert [pt["frac_large"] for pt in sweep] == sorted(pt["frac_large"] for pt in sweep)


def test_sweep_length_sends_long_prompts_to_large():
    t = make_table()
    sweep = sweep_length(t)
    assert sweep[0]["frac_large"] == 0 and sweep[-1]["frac_large"] == 1
    assert sweep[0]["cost"] == SMALL_COST and sweep[-1]["cost"] == LARGE_COST  # no routing fee
    assert list(length_use_large(t, 45)) == [False, False, False, False, True, True]


def test_cheapest_within_picks_lowest_cost_that_meets_the_target():
    sweep = [
        {"threshold": 0.0, "accuracy": 0.50, "cost": 1.0},
        {"threshold": 0.4, "accuracy": 0.79, "cost": 4.0},
        {"threshold": 0.6, "accuracy": 0.80, "cost": 6.0},
        {"threshold": 0.9, "accuracy": 0.80, "cost": 9.0},
    ]
    assert cheapest_within(sweep, 0.79)["threshold"] == 0.4
    assert cheapest_within(sweep, 0.80)["threshold"] == 0.6
    assert cheapest_within(sweep, 0.95)["threshold"] == 0.6  # unreachable: most accurate, then cheapest


def test_random_routing_is_the_straight_line():
    t = make_table()
    assert random_accuracy_at_cost(t, 1.0) == pytest.approx(3 / 6)
    assert random_accuracy_at_cost(t, 10.0) == pytest.approx(4 / 6)
    assert random_accuracy_at_cost(t, 5.5) == pytest.approx((3 / 6 + 4 / 6) / 2)


def test_auroc():
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) == 0.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 0, 1, 0])) == 0.5
    assert auroc(np.array([0.9, 0.5, 0.5, 0.1]), np.array([1, 1, 0, 0])) == pytest.approx(0.875)  # one tie
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))
    t = make_table()
    assert auroc(t.p, t.small_correct) == 1.0  # the three right answers have the three highest p


def test_calibration_bins_cover_everything_once():
    p = np.linspace(0.05, 0.95, 10)
    correct = p > 0.5
    bins = calibration(p, correct, bins=5)
    assert [b["n"] for b in bins] == [2, 2, 2, 2, 2]
    assert bins[0]["small_accuracy"] == 0.0 and bins[-1]["small_accuracy"] == 1.0
    assert bins[0]["p_max"] < bins[1]["p_min"]
    assert bins[0]["mean_p"] == pytest.approx(0.1)


def test_headline_stats_savings_and_deltas():
    t = make_table()
    h = headline_stats(t, jev_threshold=0.5, length_cutoff=25)
    assert h["jev_accuracy_minus_large"] == pytest.approx(5 / 6 - 4 / 6)
    assert h["jev_cost_saved_pct"] == pytest.approx(100 * (1 - 6.0 / 10.0))
    # c and f are where large is needed; they hold 2 of the 3 lowest probabilities
    assert h["auroc_large_needed_jev"] == pytest.approx(7 / 8)
    assert "jev_half_accuracy" not in h
    assert h["length_frac_large"] == pytest.approx(4 / 6)
    assert h["auroc_jev"] == 1.0
    assert h["auroc_jev_minus_length"] == pytest.approx(1.0 - h["auroc_length"])


def test_half_share_point_compares_with_random_at_the_same_share():
    t = make_table()
    jev_t, len_c = half_share_thresholds(t)
    assert (jev_t, len_c) == (pytest.approx(0.45), 35.0)
    h = headline_stats(t, 0.5, 25, jev_t, len_c)
    assert h["jev_half_frac_large"] == pytest.approx(0.5) and h["length_half_frac_large"] == pytest.approx(0.5)
    random_half = (3 / 6 + 4 / 6) / 2
    assert h["jev_half_accuracy_minus_random_same_share"] == pytest.approx(5 / 6 - random_half)
    # length sends d, e, f to large: right on a, b, f
    assert h["length_half_accuracy_minus_random_same_share"] == pytest.approx(3 / 6 - random_half)
    assert h["jev_half_cost"] == pytest.approx(5.5 + JEV_COST) and h["length_half_cost"] == pytest.approx(5.5)


def test_calibration_table_adds_large_model_columns():
    rows = calibration_table(make_table(), bins=3)
    assert [r["n"] for r in rows] == [2, 2, 2]
    assert rows[0]["small_accuracy"] == 0.0 and rows[0]["large_accuracy"] == 0.5  # d and f
    assert rows[0]["mean_large_cost"] == LARGE_COST and rows[0]["mean_small_cost"] == SMALL_COST


def test_bootstrap_ci_brackets_the_point_estimate_and_is_seeded():
    t = make_table(SPEC * 20)
    stat = lambda s: headline_stats(s, 0.5, 25)  # noqa: E731
    a = bootstrap_ci(t, stat, n_resamples=200, seed=7)
    b = bootstrap_ci(t, stat, n_resamples=200, seed=7)
    assert a == b
    acc = a["always_small_accuracy"]
    assert acc["value"] == pytest.approx(0.5)
    assert acc["ci_low"] < acc["value"] < acc["ci_high"]
    assert acc["ci_high"] - acc["ci_low"] < 0.3
    assert a["always_small_cost"] == {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_latency_percentiles():
    out = latency(np.arange(1, 101, dtype=float))
    assert out["p50_ms"] == pytest.approx(50.5) and out["p95_ms"] == pytest.approx(95.05) and out["n"] == 100


# ---------------------------------------------------------------- cascade

CHECK_FEE = 0.25
# id, small answered, Jev flag (P small answer is wrong), small correct, large correct, response chars
CASCADE_SPEC = [
    ("a", True, 0.05, True, True, 100),
    ("b", True, 0.10, True, True, 120),
    ("c", True, 0.90, False, True, 300),
    ("d", True, 0.80, False, False, 500),
    ("e", True, 0.60, True, False, 90),
    ("f", False, None, False, True, 400),  # no ANSWER line: never checked, always escalated
]


def make_cascade_table():
    prompts = [{"id": i, "source": "s", "split": "test", "prompt": "x"} for i, *_ in CASCADE_SPEC]
    llm, checks = [], []
    for i, answered, flag, s, l, chars in CASCADE_SPEC:
        llm.append({"id": i, "role": "small", "correct": s, "cost": SMALL_COST, "answer": "A" if answered else None, "text": "r" * chars, "latency_ms": 1000.0})
        llm.append({"id": i, "role": "large", "correct": l, "cost": LARGE_COST, "answer": "A", "text": "", "latency_ms": 4000.0})
        if flag is not None:
            checks.append({"id": i, "probs": {"wrong_answer": flag, "other": 1 - flag}, "cost": CHECK_FEE, "latency_ms": 80.0})
    jev = [{"id": i, "probs": {"small_ok": 0.5}, "cost": JEV_COST, "latency_ms": 50.0} for i, *_ in CASCADE_SPEC]
    return build_table(prompts, llm, jev, "small_ok", checks, ("wrong_answer",))


def test_cascade_table_marks_unanswered_as_unchecked():
    t = make_cascade_table()
    assert list(t.small_answered) == [True] * 5 + [False]
    assert np.isnan(t.flag[-1]) and t.check_cost[-1] == 0.0
    assert list(t.flag[:5]) == [0.05, 0.10, 0.90, 0.80, 0.60]
    assert list(t.response_chars) == [100, 120, 300, 500, 90, 400]


def test_cascade_flag_is_the_highest_of_several_checks():
    prompts = [{"id": "a", "source": "s", "split": "test", "prompt": "x"}]
    row = {"correct": True, "cost": 1.0, "answer": "A", "text": "", "latency_ms": 1.0}
    llm = [{"id": "a", "role": r, **row} for r in ("small", "large")]
    jev = [{"id": "a", "probs": {"small_ok": 0.5}, "cost": 0.1, "latency_ms": 1.0}]
    checks = [{"id": "a", "probs": {"x": 0.2, "y": 0.7}, "cost": 0.1, "latency_ms": 1.0}]
    assert build_table(prompts, llm, jev, "small_ok", checks, ("x", "y")).flag[0] == 0.7
    assert build_table(prompts, llm, jev, "small_ok", checks, ("x",)).flag[0] == 0.2


def test_cascade_escalates_flagged_and_contract_failures():
    t = make_cascade_table()
    assert list(cascade_escalate(t, 0.7)) == [False, False, True, True, False, True]
    assert list(cascade_escalate(t, float("inf"))) == [False] * 5 + [True]
    assert cascade_escalate(t, 0.0).all()


def test_cascade_cost_always_pays_small_and_the_check_fee_only_when_checked():
    t = make_cascade_table()
    r = evaluate_cascade(t, cascade_escalate(t, 0.7), pays_check=True)
    # small on all six, the check on five, large on c, d and f
    assert r["cost"] == pytest.approx((6 * SMALL_COST + 5 * CHECK_FEE + 3 * LARGE_COST) / 6)
    # a, b, e keep the small answer (all right); c and f are fixed by large; d stays wrong
    assert r["accuracy"] == pytest.approx(5 / 6)
    assert r["frac_large"] == pytest.approx(3 / 6)
    assert evaluate_cascade(t, cascade_escalate(t, 0.7), pays_check=False)["cost"] == pytest.approx(
        (6 * SMALL_COST + 3 * LARGE_COST) / 6
    )


def test_sweep_cascade_runs_from_contract_only_to_everything():
    t = make_cascade_table()
    sweep = sweep_cascade(t)
    assert sweep[0]["frac_large"] == pytest.approx(1 / 6) and sweep[-1]["frac_large"] == 1.0
    assert [pt["frac_large"] for pt in sweep] == sorted(pt["frac_large"] for pt in sweep)
    assert sweep[-1]["cost"] == pytest.approx(SMALL_COST + LARGE_COST + 5 * CHECK_FEE / 6)


def test_response_length_cascade_has_no_fee():
    t = make_cascade_table()
    assert list(response_length_escalate(t, 250)) == [False, False, True, True, False, True]
    sweep = sweep_response_length(t)
    assert sweep[0]["cost"] == pytest.approx((6 * SMALL_COST + LARGE_COST) / 6)
    assert sweep[-1]["frac_large"] == 1.0


def test_headline_stats_cascade_block():
    t = make_cascade_table()
    h = headline_stats(t, 0.5, 25, cascade_threshold=0.7, response_cutoff=250)
    large = always_large(t)
    assert h["cascade_accuracy_minus_large"] == pytest.approx(5 / 6 - large["accuracy"])
    assert h["cascade_cost_saved_pct"] == pytest.approx(100 * (1 - h["cascade_cost"] / LARGE_COST))
    assert h["cascade_accuracy_minus_random_at_cost"] == pytest.approx(
        h["cascade_accuracy"] - random_accuracy_at_cost(t, h["cascade_cost"])
    )
    # among the five checked responses, the two wrong ones (c, d) have the two highest flags
    assert h["auroc_check"] == 1.0
    assert h["response_length_cost"] == pytest.approx((6 * SMALL_COST + 3 * LARGE_COST) / 6)
    assert "cascade_accuracy" not in headline_stats(t, 0.5, 25)


def test_time_view_swaps_costs_for_measured_seconds():
    t = time_view(make_cascade_table())
    assert always_small(t)["cost"] == 1.0 and always_large(t)["cost"] == 4.0
    r = evaluate_cascade(t, cascade_escalate(t, 0.7), pays_check=True)
    assert r["cost"] == pytest.approx((6 * 1.0 + 5 * 0.08 + 3 * 4.0) / 6)
    assert evaluate(t, jev_use_large(t, 0.6), pays_jev=True)["cost"] == pytest.approx(4.0 + 0.05)


def test_mean_gain_over_random_is_zero_for_the_random_line_and_positive_above_it():
    t = make_table()
    small, large = always_small(t), always_large(t)
    on_the_line = [dict(small), dict(large)]
    assert mean_gain_over_random(t, on_the_line) == pytest.approx(0.0, abs=1e-9)
    # a policy that reaches always-large accuracy at the midpoint cost sits above the line
    mid = {"cost": (small["cost"] + large["cost"]) / 2, "accuracy": large["accuracy"]}
    better = [dict(small), mid, dict(large)]
    assert mean_gain_over_random(t, better) == pytest.approx((large["accuracy"] - small["accuracy"]) / 4, rel=0.02)
    # a policy whose cheapest point costs more than always-small falls back to always-small below it
    late = [mid, dict(large)]
    assert mean_gain_over_random(t, late) < mean_gain_over_random(t, better)


def test_best_gain_point_ignores_points_dearer_than_always_large():
    t = make_table()
    small = always_small(t)
    sweep = [
        {"threshold": 1, "cost": small["cost"], "accuracy": small["accuracy"]},
        {"threshold": 2, "cost": 4.0, "accuracy": 0.65},
        {"threshold": 3, "cost": 20.0, "accuracy": 0.99},
    ]
    assert best_gain_point(t, sweep)["threshold"] == 2
    assert gain_over_random(t, sweep[1]) == pytest.approx(0.65 - random_accuracy_at_cost(t, 4.0))

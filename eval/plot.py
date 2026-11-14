"""Draw results/cost_quality.png from results/metrics.json. No API calls, no cache reads.

Two panels, one per cost view: imputed dollars and measured seconds. Colour says where the
routing signal comes from (Jev or a length heuristic); line style says when it is used
(solid = cascade, after the small model has answered; dotted = pre-routing, from the prompt alone).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Two series colours (validated as a pair for colour-blind separation) plus neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
JEV = "#2a78d6"
HEURISTIC = "#eb6834"

DOTTED = (0, (1.2, 1.6))
# curve key -> (label, colour, line style, width)
CURVES = {
    "cascade": ("Jev cascade (checks the small answer)", JEV, "-", 2.2),
    "jev": ("Jev pre-routing (reads the prompt only)", JEV, DOTTED, 1.8),
    "response_length": ("Response-length cascade", HEURISTIC, "-", 2.2),
    "length": ("Prompt-length pre-routing", HEURISTIC, DOTTED, 1.8),
}
VIEWS = [("usd", "Imputed cost per 1,000 queries (USD)", 1000), ("seconds", "Measured time per query (seconds)", 1)]


def draw(ax, block: dict, xlabel: str, scale: float, legend: bool) -> None:
    test, stats = block["test"], block["test"]["stats"]
    ax.set_facecolor(SURFACE)

    def xy(points):
        pts = sorted(points, key=lambda p: (p["cost"], p["accuracy"]))
        return [p["cost"] * scale for p in pts], [p["accuracy"] * 100 for p in pts]

    small, large, orc = (test["policies"][k] for k in ("always_small", "always_large", "oracle"))
    ax.plot(
        [small["cost"] * scale, large["cost"] * scale],
        [small["accuracy"] * 100, large["accuracy"] * 100],
        color=MUTED, linewidth=1.4, linestyle=(0, (5, 3)), label="Random routing", zorder=2,
    )
    for key, (label, color, style, width) in CURVES.items():
        if key in test["curves"]:
            ax.plot(*xy(test["curves"][key]), color=color, linewidth=width, linestyle=style, label=label, zorder=3)

    def point(policy: dict, color: str, marker: str, label: str, dx: float, dy: float, ha: str = "left") -> None:
        x, y = policy["cost"] * scale, policy["accuracy"] * 100
        ax.scatter([x], [y], s=62, color=color, marker=marker, edgecolor=SURFACE, linewidth=2, zorder=6)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    color=INK, fontsize=9, zorder=7, linespacing=1.35,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.88, "pad": 1.5})

    point(small, INK, "o", "Always small", 8, -10)
    point(large, INK, "o", "Always large", -8, 11, ha="right")
    point(orc, INK, "D", "Oracle", 8, 0)
    # Two dev-chosen cascade thresholds: closest to always-large, and furthest above random.
    for key, name, dx, dy, ha in (
        ("cascade", "Cascade, match-large threshold", 14, -26, "left"),
        ("cascade_knee", "Cascade, best-gain threshold", -10, 22, "right"),
    ):
        if f"{key}_accuracy" not in stats:
            continue
        saved = stats[f"{key}_cost_saved_pct"]["value"]
        delta = stats[f"{key}_accuracy_minus_large"]["value"] * 100
        cost = f"{saved:.0f}% cheaper" if saved >= 0.5 else f"{-saved:.0f}% dearer" if saved <= -0.5 else "same cost"
        at = {"cost": stats[f"{key}_cost"]["value"], "accuracy": stats[f"{key}_accuracy"]["value"]}
        point(at, JEV, "o", f"{name}\n{cost}, {delta:+.1f} pts", dx, dy, ha)

    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, length=3)
    ax.margins(x=0.05, y=0.08)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.12 * (x1 - x0))
    if legend:
        leg = ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9,
                        fontsize=8.5, handlelength=2.6)
        for text in leg.get_texts():
            text.set_color(INK)


def main() -> None:
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        sys.exit("results/metrics.json not found. Run eval/simulate.py first.")
    m = json.loads(path.read_text(encoding="utf-8"))
    meta = m["meta"]

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for i, (ax, (view, xlabel, scale)) in enumerate(zip(axes, VIEWS)):
        draw(ax, m["views"][view], xlabel, scale, legend=(i == 1))
    axes[0].set_ylabel("Accuracy (%)", color=INK_SECONDARY)

    fig.suptitle(
        f"Cost versus accuracy, test split ({meta['n_test']} prompts)",
        x=0.05, y=0.975, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    priced = "imputed list prices (the models run locally)" if meta["prices_are_imputed"] else "provider list prices"
    fig.text(
        0.05, 0.922,
        f"Small {meta['small_model']}, large {meta['large_model']}, router {', '.join(meta['jev_models_reported'])}. "
        f"Left: {priced} plus Jev's fee. Right: wall-clock time including the Jev call.",
        ha="left", fontsize=9.5, color=INK_SECONDARY,
    )
    s = m["views"]["usd"]["test"]["stats"]
    notes = []
    if "auroc_check" in s:
        a = s["auroc_check"]
        notes.append(
            f"Cascade check, AUROC for spotting a wrong small answer: {a['value']:.2f} [{a['ci_low']:.2f}, {a['ci_high']:.2f}]."
        )
    a = s["auroc_jev"]
    notes.append(
        f"Pre-routing, AUROC for predicting a correct small answer: {a['value']:.2f} [{a['ci_low']:.2f}, {a['ci_high']:.2f}]."
    )
    fig.text(
        0.05, 0.018,
        " ".join(notes) + " 95% bootstrap intervals.\n"
        "Thresholds were chosen on the dev split; point labels compare with always-large. Solid lines are cascades (the small model always runs first); dotted lines route from the prompt alone.",
        ha="left", va="bottom", fontsize=8.5, color=MUTED, linespacing=1.5,
    )
    fig.subplots_adjust(left=0.05, right=0.985, top=0.885, bottom=0.16, wspace=0.06)

    out = RESULTS_DIR / "cost_quality.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

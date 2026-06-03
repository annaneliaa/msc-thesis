"""
Per-scenario feature analysis for run_compare experiments.

Loads baseline-vs-symbolic compare results (fixed-window grouping) for one or
more scenarios from a run_compare run and produces:

  - Feature pipeline funnel: n_mined → n_after_filter → n_final → n_nonzero_coeff
  - Top-K feature Jaccard overlap heatmap across scenarios
  - FP analysis: baseline vs symbolic FP / precision / recall / F1 per scenario
  - Mining type composition of learned features (itemset / sequence / OR)
  - Source label breakdown (attack-mined vs benign-mined survivors)
  - Generalization gap (train AUC vs test AUC)
  - Coefficient vs permutation importance agreement
  - Signed coefficients diverging bar (top-k by |coeff|, coloured by sign)
  - Sign-split breakdown (mining type & source label split by coeff sign)

Output (under artifacts/experiments/run_scenario_features/scenario_features_<ts>/):
    analysis_<ts>.txt
    feature_funnel.png
    feature_overlap.png
    fp_analysis.png
    feature_type_breakdown.png
    coeff_vs_perm_agreement.png
    signed_coefficients.png
    sign_split_breakdown.png

Usage:
    # all scenarios from the latest run
    python src/thesis/scripts/run_scenario_feature_analysis.py --all

    # all scenarios from a specific run (name or prefix)
    python src/thesis/scripts/run_scenario_feature_analysis.py --all --run compare_20260603_185353

    # specific scenarios
    python src/thesis/scripts/run_scenario_feature_analysis.py fox harrison

    # list available runs
    python src/thesis/scripts/run_scenario_feature_analysis.py --list-runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_COMPARE_BASE = _REPO / "artifacts" / "experiments" / "run_compare"
_ANALYSIS_BASE = _REPO / "artifacts" / "experiments" / "run_scenario_features"

_SCENARIO_COLORS = {
    "fox": "#4C72B0",
    "harrison": "#DD8452",
    "russellmitchell": "#55A868",
    "santos": "#C44E52",
    "shaw": "#8172B3",
    "wardbeck": "#937860",
    "wheeler": "#DA8BC3",
    "wilson": "#8C8C8C",
}
_DEFAULT_COLOR = "#AAAAAA"


def _color(scenario: str) -> str:
    return _SCENARIO_COLORS.get(scenario, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _latest_run_dir() -> Path:
    candidates = sorted(
        p for p in _COMPARE_BASE.iterdir() if p.is_dir() and (p / "scenario").is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {_COMPARE_BASE}")
    return candidates[-1]


def _scenarios_in_run(run_dir: Path) -> list[str]:
    scenario_base = run_dir / "scenario"
    if not scenario_base.is_dir():
        return []
    return sorted(p.name for p in scenario_base.iterdir() if p.is_dir())


def load_all(
    run_dir: Path, scenarios: list[str]
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Returns (compare_by_scenario, symbolic_by_scenario).
    compare_by_scenario[s] is the compare JSON (has "baseline", "symbolic", "filtered", ...).
    symbolic_by_scenario[s] is the full symbolic results JSON.
    """
    compare_by_scenario: dict[str, dict] = {}
    symbolic_by_scenario: dict[str, dict] = {}

    for scenario in scenarios:
        scenario_dir = run_dir / "scenario" / scenario
        candidates = sorted(scenario_dir.glob("compare_*.json"))
        if not candidates:
            print(
                f"  [warn] No compare JSON for scenario '{scenario}' in {scenario_dir}"
            )
            continue
        compare_path = candidates[-1]
        print(f"  Loading [{scenario}]: {compare_path.name}")
        with compare_path.open() as f:
            cmp = json.load(f)
        compare_by_scenario[scenario] = cmp

        results_file = cmp.get("symbolic", {}).get("results_file")
        if results_file and Path(results_file).exists():
            with open(results_file) as f:
                symbolic_by_scenario[scenario] = json.load(f)
        else:
            print(f"  [warn] No symbolic results_file for '{scenario}'")

    return compare_by_scenario, symbolic_by_scenario


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _i(v: Any) -> int:
    return int(v) if v is not None else 0


def feature_funnel(sym: dict) -> dict:
    m = sym.get("mining", {})
    met = sym.get("metrics", {})
    top_coeff = met.get("top_feature_importances", {}).get("by_coefficient", {})
    top_perm = met.get("top_feature_importances", {}).get("by_permutation", {})

    n_nonzero_coeff = sum(1 for v in top_coeff.values() if v["importance"] != 0)
    n_nonzero_perm = sum(1 for v in top_perm.values() if v["importance"] > 0)

    n_mined = (
        _i(m.get("n_itemsets_mined"))
        + _i(m.get("n_sequences_mined"))
        + _i(m.get("n_or_mined"))
    )
    n_abs_parts = (
        _i(m.get("n_itemsets_after_abstraction"))
        + _i(m.get("n_sequences_after_abstraction"))
        + _i(m.get("n_or_after_abstraction"))
    )
    n_after_abstraction = n_abs_parts if n_abs_parts > 0 else n_mined
    n_after_filter = _i(m.get("n_candidate_features"))

    return {
        "n_mined": n_mined,
        "n_after_abstraction": n_after_abstraction,
        "n_after_filter": n_after_filter,
        "n_final": _i(m.get("n_features_final")),
        "n_nonzero_coeff": n_nonzero_coeff,
        "n_nonzero_perm": n_nonzero_perm,
        "n_symbolic_used": _i(met.get("n_symbolic_features_used")),
        "filter_config": m.get("filter_config"),
    }


def top_feature_names(sym: dict, k: int, by: str = "by_coefficient") -> set[str]:
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    ranked = sorted(top.items(), key=lambda kv: abs(kv[1]["importance"]), reverse=True)
    nonzero = [(name, info) for name, info in ranked if info["importance"] != 0]
    return {name for name, _ in nonzero[:k]}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def mining_type_breakdown(
    sym: dict, by: str = "by_coefficient", sign: str = "positive"
) -> dict[str, int]:
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    counts: dict[str, int] = {}
    for info_dict in top.values():
        imp = info_dict["importance"]
        if imp == 0:
            continue
        if sign == "positive" and imp < 0:
            continue
        if sign == "negative" and imp > 0:
            continue
        mtype = info_dict.get("feature_info", {}).get("mining_type", "base")
        counts[mtype] = counts.get(mtype, 0) + 1
    return counts


def source_label_breakdown(
    sym: dict, by: str = "by_coefficient", sign: str = "positive"
) -> dict[str, int]:
    top = sym.get("metrics", {}).get("top_feature_importances", {}).get(by, {})
    counts: dict[str, int] = {}
    for info_dict in top.values():
        imp = info_dict["importance"]
        if imp == 0:
            continue
        if sign == "positive" and imp < 0:
            continue
        if sign == "negative" and imp > 0:
            continue
        src = info_dict.get("feature_info", {}).get("source_label", "base")
        counts[src] = counts.get(src, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Console tables
# ---------------------------------------------------------------------------


def print_funnel_table(scenarios: list[str], funnels: dict[str, dict]) -> None:
    stages = [
        ("n_mined", "Mined total"),
        ("n_after_abstraction", "After abstraction"),
        ("n_after_filter", "After filter (+OR)"),
        ("n_final", "Final (dedup)"),
        ("n_nonzero_coeff", "Nonzero coeff"),
        ("n_nonzero_perm", "Nonzero perm"),
    ]
    col_w = 16
    print("\n" + "═" * (26 + col_w * len(scenarios)))
    print("  FEATURE PIPELINE FUNNEL")
    print("─" * (26 + col_w * len(scenarios)))
    header = f"  {'Stage':<24}" + "".join(f"{s:>{col_w}}" for s in scenarios)
    print(header)
    print("─" * (26 + col_w * len(scenarios)))
    for key, label in stages:
        row = f"  {label:<24}" + "".join(
            f"{funnels[s].get(key, 0):>{col_w},}" for s in scenarios
        )
        print(row)
    print("═" * (26 + col_w * len(scenarios)) + "\n")


def print_fp_table(
    scenarios: list[str], compare: dict[str, dict], sym_data: dict[str, dict]
) -> None:
    col_w = 13
    print("═" * (26 + col_w * len(scenarios)))
    print("  FALSE POSITIVE ANALYSIS")
    print("─" * (26 + col_w * len(scenarios)))
    header = f"  {'Metric':<24}" + "".join(f"{s:>{col_w}}" for s in scenarios)
    print(header)
    print("─" * (26 + col_w * len(scenarios)))

    for label, source, key in [
        ("Baseline FP", "baseline", "fp"),
        ("Symbolic FP", "symbolic", "fp"),
        ("Baseline precision", "baseline", "precision"),
        ("Symbolic precision", "symbolic", "precision"),
        ("Baseline recall", "baseline", "recall"),
        ("Symbolic recall", "symbolic", "recall"),
        ("Baseline F1", "baseline", "f1"),
        ("Symbolic F1", "symbolic", "f1"),
    ]:
        vals = []
        for s in scenarios:
            met = compare.get(s, {}).get(source, {}).get("metrics", {})
            v = met.get(key, float("nan"))
            if key in ("fp", "tp", "tn", "fn"):
                vals.append(f"{int(v):>{col_w},}" if v == v else f"{'?':>{col_w}}")
            else:
                vals.append(f"{v:>{col_w}.3f}" if v == v else f"{'?':>{col_w}}")
        print(f"  {label:<24}" + "".join(vals))

    print("─" * (26 + col_w * len(scenarios)))
    row_delta = f"  {'FP delta (sym-base)':<24}"
    row_pct = f"  {'FP reduction %':<24}"
    for s in scenarios:
        base_fp = (
            compare.get(s, {})
            .get("baseline", {})
            .get("metrics", {})
            .get("fp", float("nan"))
        )
        sym_fp = (
            compare.get(s, {})
            .get("symbolic", {})
            .get("metrics", {})
            .get("fp", float("nan"))
        )
        if base_fp == base_fp and sym_fp == sym_fp:
            delta = int(sym_fp) - int(base_fp)
            pct = (base_fp - sym_fp) / base_fp * 100 if base_fp > 0 else 0.0
            row_delta += f"{delta:>+{col_w},}"
            row_pct += f"{pct:>{col_w}.1f}%"
        else:
            row_delta += f"{'?':>{col_w}}"
            row_pct += f"{'?':>{col_w}}"
    print(row_delta)
    print(row_pct)
    print("═" * (26 + col_w * len(scenarios)) + "\n")


def print_generalization_table(scenarios: list[str], sym_data: dict[str, dict]) -> None:
    col_w = 13
    print("═" * (26 + col_w * len(scenarios)))
    print("  GENERALIZATION GAP (train AUC - test AUC)")
    print("─" * (26 + col_w * len(scenarios)))
    header = f"  {'Metric':<24}" + "".join(f"{s:>{col_w}}" for s in scenarios)
    print(header)
    print("─" * (26 + col_w * len(scenarios)))
    for label, key in [
        ("Test AUC", "auc"),
        ("Train AUC", "train_auc"),
        ("Gap (train-test)", "performance_gap_train_vs_test"),
        ("Feature sparsity", "feature_sparsity"),
    ]:
        row = f"  {label:<24}"
        for s in scenarios:
            met = sym_data.get(s, {}).get("metrics", {})
            v = met.get(key, float("nan"))
            row += f"{v:>{col_w}.4f}" if v == v else f"{'?':>{col_w}}"
        print(row)
    print("═" * (26 + col_w * len(scenarios)) + "\n")


def print_type_breakdown(scenarios: list[str], sym_data: dict[str, dict]) -> None:
    all_types: set[str] = set()
    breakdowns = {}
    for s in scenarios:
        bd = mining_type_breakdown(sym_data.get(s, {}))
        breakdowns[s] = bd
        all_types |= set(bd.keys())

    col_w = 10
    print("═" * (26 + col_w * len(scenarios)))
    print("  NONZERO-COEFF FEATURE TYPE BREAKDOWN")
    print("─" * (26 + col_w * len(scenarios)))
    header = f"  {'Type':<24}" + "".join(f"{s:>{col_w}}" for s in scenarios)
    print(header)
    print("─" * (26 + col_w * len(scenarios)))
    for mtype in sorted(all_types):
        row = f"  {mtype:<24}" + "".join(
            f"{breakdowns[s].get(mtype, 0):>{col_w},}" for s in scenarios
        )
        print(row)

    print("─" * (26 + col_w * len(scenarios)))
    print("  SOURCE LABEL BREAKDOWN")
    print("─" * (26 + col_w * len(scenarios)))
    all_src: set[str] = set()
    src_breakdowns = {}
    for s in scenarios:
        bd = source_label_breakdown(sym_data.get(s, {}))
        src_breakdowns[s] = bd
        all_src |= set(bd.keys())
    for src in sorted(all_src):
        row = f"  {src:<24}" + "".join(
            f"{src_breakdowns[s].get(src, 0):>{col_w},}" for s in scenarios
        )
        print(row)
    print("═" * (26 + col_w * len(scenarios)) + "\n")


def print_overlap_table(
    scenarios: list[str], sym_data: dict[str, dict], k: int
) -> None:
    feature_sets = {s: top_feature_names(sym_data.get(s, {}), k) for s in scenarios}
    short = [s[:6] for s in scenarios]
    print(f"  TOP-{k} FEATURE JACCARD OVERLAP (by_coefficient, nonzero only)")
    print("─" * (10 + 9 * len(scenarios)))
    header = f"  {'':8}" + "".join(f"{lbl:>9}" for lbl in short)
    print(header)
    for i, sa in enumerate(scenarios):
        row = f"  {short[i]:<8}"
        for sb in scenarios:
            j = jaccard(feature_sets[sa], feature_sets[sb])
            row += f"{j:>9.3f}"
        print(row)
    print()

    if len(scenarios) > 1:
        shared = set.intersection(*feature_sets.values())
        print(f"  Shared by ALL scenarios ({len(shared)} features):")
        for f in sorted(shared):
            print(f"    {f}")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _filtered_label(filtered: bool) -> str:
    return "data: filtered" if filtered else "data: raw"


def plot_funnel(
    scenarios: list[str], funnels: dict[str, dict], filtered: bool, out_dir: Path
) -> None:
    stages = [
        ("n_mined", "Mined"),
        ("n_after_abstraction", "After\nabstraction"),
        ("n_after_filter", "After filter\n(+OR pass-through)"),
        ("n_final", "Final\n(dedup)"),
        ("n_nonzero_coeff", "Learned\n(nonzero coeff)"),
    ]
    n_stages = len(stages)
    n_sc = len(scenarios)
    x = np.arange(n_stages)
    w = 0.15
    offsets = [w * (i - (n_sc - 1) / 2) for i in range(n_sc)]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, sc in enumerate(scenarios):
        vals = [funnels[sc].get(key, 0) for key, _ in stages]
        bars = ax.bar(x + offsets[i], vals, w, label=sc, color=_color(sc))
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:,}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in stages])
    ax.set_ylabel("Feature count (log scale)")
    ax.set_title("Feature pipeline funnel by scenario (fixed-window grouping)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        _filtered_label(filtered),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "feature_funnel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_overlap_heatmap(
    scenarios: list[str],
    sym_data: dict[str, dict],
    k: int,
    filtered: bool,
    out_dir: Path,
) -> None:
    feature_sets = {s: top_feature_names(sym_data.get(s, {}), k) for s in scenarios}
    n = len(scenarios)
    matrix = np.zeros((n, n))
    for i, sa in enumerate(scenarios):
        for j, sb in enumerate(scenarios):
            matrix[i, j] = jaccard(feature_sets[sa], feature_sets[sb])

    short = [s[:8] for s in scenarios]
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Jaccard similarity")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticklabels(short)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black" if matrix[i, j] < 0.7 else "white",
            )
    ax.set_title(f"Top-{k} feature Jaccard overlap across scenarios (nonzero coeff)")
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        _filtered_label(filtered),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "feature_overlap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_fp_analysis(
    scenarios: list[str], compare: dict[str, dict], filtered: bool, out_dir: Path
) -> None:
    metrics_to_plot = [
        ("fp", "False Positives", True),
        ("precision", "Precision", False),
        ("recall", "Recall", False),
        ("f1", "F1", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    x = np.arange(len(scenarios))
    w = 0.35

    for ax, (key, title, is_count) in zip(axes, metrics_to_plot):
        base_vals = [
            compare.get(s, {})
            .get("baseline", {})
            .get("metrics", {})
            .get(key, float("nan"))
            for s in scenarios
        ]
        sym_vals = [
            compare.get(s, {})
            .get("symbolic", {})
            .get("metrics", {})
            .get(key, float("nan"))
            for s in scenarios
        ]
        b1 = ax.bar(
            x - w / 2, base_vals, w, label="Baseline", color="#5B9BD5", alpha=0.85
        )
        b2 = ax.bar(
            x + w / 2, sym_vals, w, label="Symbolic", color="#ED7D31", alpha=0.85
        )
        for bar, val in zip(list(b1) + list(b2), base_vals + sym_vals):
            if val == val and val > 0:
                fmt = f"{int(val)}" if is_count else f"{val:.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([s[:8] for s in scenarios], rotation=20, ha="right")
        if not is_count:
            ax.set_ylim(0, 1.15)
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Baseline vs Symbolic detection metrics across scenarios (fixed-window)"
    )
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        _filtered_label(filtered),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "fp_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_type_breakdown(
    scenarios: list[str], sym_data: dict[str, dict], filtered: bool, out_dir: Path
) -> None:
    all_types_set: set[str] = set()
    breakdowns = {}
    for s in scenarios:
        bd = mining_type_breakdown(sym_data.get(s, {}))
        breakdowns[s] = bd
        all_types_set |= set(bd.keys())
    all_types = sorted(all_types_set)

    type_colors = {
        "itemset": "#4C72B0",
        "item_sequence": "#55A868",
        "or_itemset": "#DD8452",
        "base": "#8C8C8C",
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(scenarios))
    bottoms = np.zeros(len(scenarios))
    for mtype in all_types:
        vals = np.array([breakdowns[s].get(mtype, 0) for s in scenarios], dtype=float)
        bars = ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=mtype,
            color=type_colors.get(mtype, "#BBBBBB"),
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(val)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel("Count of nonzero-coeff features")
    ax.set_title("Feature type breakdown per scenario (nonzero model coefficients)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        _filtered_label(filtered),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "feature_type_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_coeff_vs_perm(
    scenarios: list[str],
    sym_data: dict[str, dict],
    k: int,
    filtered: bool,
    out_dir: Path,
) -> None:
    coeff_sets = {
        s: top_feature_names(sym_data.get(s, {}), k, "by_coefficient")
        for s in scenarios
    }
    perm_sets = {
        s: top_feature_names(sym_data.get(s, {}), k, "by_permutation")
        for s in scenarios
    }
    agreement = [jaccard(coeff_sets[s], perm_sets[s]) for s in scenarios]

    fig, ax = plt.subplots(figsize=(max(6, len(scenarios)), 3))
    bars = ax.bar(
        [s[:8] for s in scenarios],
        agreement,
        color=[_color(s) for s in scenarios],
        alpha=0.85,
    )
    for bar, val in zip(bars, agreement):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Jaccard similarity")
    ax.set_title(f"Coeff vs permutation importance agreement (top-{k}) per scenario")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        _filtered_label(filtered),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "coeff_vs_perm_agreement.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_signed_coefficients(
    scenarios: list[str],
    sym_data: dict[str, dict],
    k: int,
    filtered: bool,
    out_dir: Path,
) -> None:
    from matplotlib.patches import Patch

    present = [s for s in scenarios if s in sym_data]
    if not present:
        return

    n = len(present)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    fig_h = max(k * 0.28 + 1.5, 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, fig_h * n_rows))
    axes_flat: list = np.array(axes).flatten().tolist() if n > 1 else [axes]

    for ax_idx, sc in enumerate(present):
        ax = axes_flat[ax_idx]
        top = (
            sym_data[sc]
            .get("metrics", {})
            .get("top_feature_importances", {})
            .get("by_coefficient", {})
        )
        ranked = sorted(
            top.items(), key=lambda kv: abs(kv[1]["importance"]), reverse=True
        )
        ranked = [
            (name, info["importance"])
            for name, info in ranked
            if info["importance"] != 0
        ][:k]
        ranked = ranked[::-1]

        names = [r[0][:50] for r in ranked]
        values = [r[1] for r in ranked]
        colors = ["#4C72B0" if v > 0 else "#C94040" for v in values]

        y = np.arange(len(names))
        ax.barh(y, values, color=colors, alpha=0.85, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(sc, fontsize=9)
        ax.set_xlabel("Coefficient value", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    for ax in axes_flat[len(present) :]:
        ax.set_visible(False)

    legend_elements = [
        Patch(facecolor="#4C72B0", alpha=0.85, label="Positive → attack"),
        Patch(facecolor="#C94040", alpha=0.85, label="Negative → benign"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"Top-{k} features by |coefficient| per scenario — logistic regression",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.text(
        0.99,
        0.01,
        _filtered_label(filtered),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "signed_coefficients.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_sign_split_breakdown(
    scenarios: list[str], sym_data: dict[str, dict], filtered: bool, out_dir: Path
) -> None:
    from matplotlib.patches import Patch

    type_colors = {
        "itemset": "#4C72B0",
        "item_sequence": "#55A868",
        "or_itemset": "#DD8452",
        "base": "#8C8C8C",
    }
    src_colors = {
        "attack": "#C94040",
        "benign": "#4C72B0",
        "unknown": "#AAAAAA",
    }

    fig, (ax_type, ax_src) = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(scenarios))

    for ax, breakdown_fn, colors, title in [
        (
            ax_type,
            mining_type_breakdown,
            type_colors,
            "Mining type by coefficient sign",
        ),
        (
            ax_src,
            source_label_breakdown,
            src_colors,
            "Source label by coefficient sign",
        ),
    ]:
        all_cats: set[str] = set()
        pos_bds: dict[str, dict[str, int]] = {}
        neg_bds: dict[str, dict[str, int]] = {}
        for s in scenarios:
            sym = sym_data.get(s, {})
            pos_bds[s] = breakdown_fn(sym, sign="positive")
            neg_bds[s] = breakdown_fn(sym, sign="negative")
            all_cats |= set(pos_bds[s]) | set(neg_bds[s])

        pos_bottoms = np.zeros(len(scenarios))
        neg_bottoms = np.zeros(len(scenarios))
        legend_handles: list = []

        for cat in sorted(all_cats):
            color = colors.get(cat, "#BBBBBB")
            pos_vals = np.array(
                [pos_bds[s].get(cat, 0) for s in scenarios], dtype=float
            )
            neg_vals = np.array(
                [-neg_bds[s].get(cat, 0) for s in scenarios], dtype=float
            )

            bars_pos = ax.bar(x, pos_vals, bottom=pos_bottoms, color=color, alpha=0.85)
            bars_neg = ax.bar(
                x, neg_vals, bottom=neg_bottoms, color=color, alpha=0.45, hatch="//"
            )

            for bar, val in zip(bars_pos, pos_vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(val)),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                        fontweight="bold",
                    )
            for bar, val in zip(bars_neg, neg_vals):
                if val < 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(-val)),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                        fontweight="bold",
                    )

            pos_bottoms += pos_vals
            neg_bottoms += neg_vals
            legend_handles.append(Patch(facecolor=color, alpha=0.85, label=cat))

        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([s[:8] for s in scenarios], rotation=20, ha="right")
        ax.set_ylabel("Feature count")
        ax.set_title(title)
        ax.legend(handles=legend_handles, fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.02,
            0.98,
            "↑ positive coeff (attack)",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            color="#333333",
        )
        ax.text(
            0.02,
            0.02,
            "↓ negative coeff (benign)",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            color="#333333",
        )

    fig.suptitle("Feature breakdown by coefficient sign across scenarios", fontsize=11)
    fig.tight_layout()
    fig.text(
        0.99,
        0.01,
        _filtered_label(filtered),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
        transform=fig.transFigure,
    )
    out = out_dir / "sign_split_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


class _Tee:
    def __init__(self, log_path: Path) -> None:
        self._file = log_path.open("w", encoding="utf-8", buffering=1)
        self._stdout = sys.__stdout__

    def write(self, s: str) -> int:
        self._stdout.write(s)
        self._file.write(s)
        return len(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def fileno(self) -> int:
        return self._stdout.fileno()

    def close(self) -> None:
        self._file.close()


def main() -> None:
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="Per-scenario feature analysis for run_compare experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_scenario_feature_analysis.py --all
  python run_scenario_feature_analysis.py --all --run compare_20260603_185353
  python run_scenario_feature_analysis.py fox harrison --run compare_20260603_185353
  python run_scenario_feature_analysis.py --list-runs
""",
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        help="Scenario names to analyse. Omit when using --all.",
    )
    parser.add_argument(
        "--run",
        default=None,
        metavar="RUN_NAME",
        help="Run directory name (or prefix) under run_compare/. Default: latest.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all scenarios found in the selected run.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available run directories and their scenarios, then exit.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top features for Jaccard overlap and signed-coeff plot (default: 25).",
    )
    args = parser.parse_args()

    if args.list_runs:
        runs = sorted(
            p
            for p in _COMPARE_BASE.iterdir()
            if p.is_dir() and (p / "scenario").is_dir()
        )
        if not runs:
            print(f"No runs found under {_COMPARE_BASE}")
        else:
            print(f"Available runs under {_COMPARE_BASE}:")
            for r in runs:
                sc_list = _scenarios_in_run(r)
                print(f"  {r.name}  [{', '.join(sc_list)}]")
        return

    # Resolve run-level directory
    if args.run is not None:
        candidate = _COMPARE_BASE / args.run
        if candidate.is_dir():
            run_dir = candidate
        else:
            matches = sorted(
                p
                for p in _COMPARE_BASE.iterdir()
                if p.is_dir() and p.name.startswith(args.run)
            )
            if not matches:
                print(f"[error] No run matching '{args.run}' under {_COMPARE_BASE}")
                sys.exit(1)
            run_dir = matches[-1]
            print(f"  Matched run: {run_dir.name}")
    else:
        run_dir = _latest_run_dir()
        print(f"  Using latest run: {run_dir.name}")

    # Determine scenarios
    if args.all:
        scenarios = _scenarios_in_run(run_dir)
        if not scenarios:
            print(f"[error] No scenarios found under {run_dir / 'scenario'}")
            sys.exit(1)
    elif args.scenarios:
        scenarios = args.scenarios
    else:
        print("[error] Provide scenario names or use --all.")
        sys.exit(1)

    print(f"  Scenarios: {scenarios}\n")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = _ANALYSIS_BASE / f"scenario_features_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"analysis_{ts}.txt"
    tee = _Tee(log_path)
    sys.stdout = tee

    try:
        compare, sym_data = load_all(run_dir, scenarios)

        present = [s for s in scenarios if s in compare]
        if not present:
            print("[error] No scenarios loaded successfully.")
            sys.exit(1)
        if len(present) < len(scenarios):
            missing = set(scenarios) - set(present)
            print(f"  [warn] Skipped (no data): {sorted(missing)}")

        # Any scenario's filtered flag (they should all agree)
        filtered = any(compare[s].get("filtered", False) for s in present)

        funnels = {s: feature_funnel(sym_data[s]) for s in present if s in sym_data}

        print_funnel_table(present, funnels)
        print_fp_table(present, compare, sym_data)
        print_generalization_table(present, sym_data)
        print_type_breakdown(present, sym_data)
        print_overlap_table(present, sym_data, args.top_k)

        print(f"\n[plots] Writing to {out_dir}")
        if funnels:
            plot_funnel(present, funnels, filtered, out_dir)
        if len(present) > 1:
            plot_overlap_heatmap(present, sym_data, args.top_k, filtered, out_dir)
        plot_fp_analysis(present, compare, filtered, out_dir)
        if sym_data:
            plot_type_breakdown(present, sym_data, filtered, out_dir)
            plot_coeff_vs_perm(present, sym_data, args.top_k, filtered, out_dir)
            plot_signed_coefficients(present, sym_data, args.top_k, filtered, out_dir)
            plot_sign_split_breakdown(present, sym_data, filtered, out_dir)

        print(f"\nAnalysis written to {out_dir}")
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()

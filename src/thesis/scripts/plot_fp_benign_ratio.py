"""
Plot FP rate vs benign traffic ratio and n_features, comparing raw vs filtered runs.

Loads the latest raw and filtered compare runs from artifacts/experiments/run_compare/,
computes FP rate and benign ratio per scenario x model, and generates:
  - scatter_fp_vs_benign.png   -- FP rate vs fraction of benign traffic in test set
  - scatter_fp_vs_features.png -- FP rate vs number of features (log x)
  - bar_fp_rate.png            -- FP rate per scenario, grouped by condition

Usage:
    python src/thesis/scripts/plot_fp_benign_ratio.py [--out-dir DIR]
    python src/thesis/scripts/plot_fp_benign_ratio.py \\
        --raw-run compare_20260603_185353 \\
        --filtered-run compare_20260603_192756
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_COMPARE_DIR = _REPO / "artifacts" / "experiments" / "run_compare"
_DEFAULT_OUT = _REPO / "artifacts" / "experiments" / "fp_benign_ratio"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_COLORS = {
    "raw": "#E05C2A",
    "filtered": "#2A7BE0",
}
_MARKERS = {
    "baseline": "o",
    "symbolic": "s",
}
_LABELS = {
    ("raw", "baseline"): "raw · baseline",
    ("raw", "symbolic"): "raw · symbolic",
    ("filtered", "baseline"): "filtered · baseline",
    ("filtered", "symbolic"): "filtered · symbolic",
}


def _find_latest_run(filtered: bool) -> Path | None:
    """Return the most recent compare_* dir that matches the filtered flag."""
    for run_dir in sorted(_COMPARE_DIR.glob("compare_*"), reverse=True):
        for scenario_dir in (
            (run_dir / "scenario").iterdir() if (run_dir / "scenario").exists() else []
        ):
            for compare_json in sorted(scenario_dir.glob("compare_*.json")):
                d = json.loads(compare_json.read_text())
                if d.get("filtered", False) == filtered:
                    return run_dir
    return None


def _load_run(run_dir: Path) -> list[dict]:
    """Load all per-scenario compare JSONs from a run directory."""
    records = []
    scenario_root = run_dir / "scenario"
    if not scenario_root.exists():
        return records

    for scenario_dir in sorted(scenario_root.iterdir()):
        compare_files = sorted(scenario_dir.glob("compare_*.json"))
        if not compare_files:
            continue
        data = json.loads(compare_files[-1].read_text())
        filtered_flag = data.get("filtered", False)
        condition = "filtered" if filtered_flag else "raw"
        scenario = data["scenario"]

        for model_type in ("baseline", "symbolic"):
            m = data[model_type]["metrics"]
            tp = m.get("tp")
            fp = m.get("fp")
            tn = m.get("tn")
            fn = m.get("fn")

            # skip scenarios where the experiment failed (no confusion matrix)
            if any(v is None for v in (tp, fp, tn, fn)):
                continue
            total = tp + fp + tn + fn
            if total == 0:
                continue

            benign = fp + tn
            fp_rate = fp / benign if benign > 0 else float("nan")
            benign_ratio = benign / total

            records.append(
                {
                    "scenario": scenario,
                    "condition": condition,
                    "model_type": model_type,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "fp_rate": fp_rate,
                    "benign_ratio": benign_ratio,
                    "n_features": data[model_type]["n_features"],
                    "n_transactions": data[model_type]["n_transactions"],
                }
            )
    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _legend_handles() -> list[mpatches.Patch]:
    handles = []
    for (cond, mtype), label in _LABELS.items():
        handles.append(
            mpatches.Patch(
                facecolor=_COLORS[cond],
                edgecolor="white" if mtype == "baseline" else _COLORS[cond],
                linewidth=1.5,
                label=label,
                hatch="" if mtype == "baseline" else "///",
            )
        )
    return handles


def _scatter_handles() -> list:
    handles = []
    for cond, color in _COLORS.items():
        for mtype, marker in _MARKERS.items():
            h = plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=9,
                label=_LABELS[(cond, mtype)],
            )
            handles.append(h)
    return handles


def plot_fp_vs_benign(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: FP rate (y) vs benign traffic ratio in test set (x)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond in ("raw", "filtered"):
        for mtype in ("baseline", "symbolic"):
            sub = df[(df["condition"] == cond) & (df["model_type"] == mtype)]
            if sub.empty:
                continue
            ax.scatter(
                sub["benign_ratio"],
                sub["fp_rate"],
                color=_COLORS[cond],
                marker=_MARKERS[mtype],
                s=90,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.6,
                label=_LABELS[(cond, mtype)],
                zorder=3,
            )
            # annotate each point with the scenario name
            for _, row in sub.iterrows():
                if not (math.isnan(row["fp_rate"]) or math.isnan(row["benign_ratio"])):
                    ax.annotate(
                        row["scenario"],
                        (row["benign_ratio"], row["fp_rate"]),
                        textcoords="offset points",
                        xytext=(5, 4),
                        fontsize=7,
                        color=_COLORS[cond],
                        alpha=0.9,
                    )

    ax.set_xlabel("Benign traffic ratio (test set)")
    ax.set_ylabel("FP rate  [FP / (FP + TN)]")
    ax.set_title(
        "FP rate vs fraction of benign traffic in test set\n(raw vs. detector-filtered data)"
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(handles=_scatter_handles(), fontsize=8, loc="upper left")
    fig.tight_layout()
    out = out_dir / "scatter_fp_vs_benign.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_fp_vs_features(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: FP rate (y) vs n_features (x, log scale)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond in ("raw", "filtered"):
        for mtype in ("baseline", "symbolic"):
            sub = df[(df["condition"] == cond) & (df["model_type"] == mtype)]
            if sub.empty:
                continue
            ax.scatter(
                sub["n_features"],
                sub["fp_rate"],
                color=_COLORS[cond],
                marker=_MARKERS[mtype],
                s=90,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.6,
                label=_LABELS[(cond, mtype)],
                zorder=3,
            )
            for _, row in sub.iterrows():
                if not math.isnan(row["fp_rate"]):
                    ax.annotate(
                        row["scenario"],
                        (row["n_features"], row["fp_rate"]),
                        textcoords="offset points",
                        xytext=(5, 4),
                        fontsize=7,
                        color=_COLORS[cond],
                        alpha=0.9,
                    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of features (log scale)")
    ax.set_ylabel("FP rate  [FP / (FP + TN)]")
    ax.set_title("FP rate vs model feature count\n(raw vs. detector-filtered data)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3, which="both")
    ax.legend(handles=_scatter_handles(), fontsize=8, loc="upper left")
    fig.tight_layout()
    out = out_dir / "scatter_fp_vs_features.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_bar_fp_rate(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: FP rate per scenario, raw vs filtered x baseline vs symbolic."""
    scenarios = sorted(df["scenario"].unique())
    n = len(scenarios)
    x = np.arange(n)
    w = 0.2

    offsets = {
        ("raw", "baseline"): -1.5,
        ("raw", "symbolic"): -0.5,
        ("filtered", "baseline"): 0.5,
        ("filtered", "symbolic"): 1.5,
    }

    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 5))

    for (cond, mtype), offset in offsets.items():
        vals = []
        for sc in scenarios:
            row = df[
                (df["scenario"] == sc)
                & (df["condition"] == cond)
                & (df["model_type"] == mtype)
            ]
            vals.append(row["fp_rate"].values[0] if not row.empty else float("nan"))

        bars = ax.bar(
            x + offset * w,
            vals,
            w,
            color=_COLORS[cond],
            hatch="" if mtype == "baseline" else "///",
            edgecolor="white",
            linewidth=0.5,
            label=_LABELS[(cond, mtype)],
            alpha=0.85,
        )
        for bar, v in zip(bars, vals):
            if not math.isnan(v) and v > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("FP rate  [FP / (FP + TN)]")
    ax.set_title("FP rate per scenario — raw vs. filtered data")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "bar_fp_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot FP rate vs benign ratio and n_features for raw vs filtered runs."
    )
    parser.add_argument(
        "--raw-run",
        default=None,
        help="Name of the raw compare run directory (e.g. compare_20260603_185353). "
        "Auto-detected if omitted.",
    )
    parser.add_argument(
        "--filtered-run",
        default=None,
        help="Name of the filtered compare run directory. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory for plots (default: {_DEFAULT_OUT})",
    )
    args = parser.parse_args()

    raw_dir = (
        _COMPARE_DIR / args.raw_run
        if args.raw_run
        else _find_latest_run(filtered=False)
    )
    filtered_dir = (
        _COMPARE_DIR / args.filtered_run
        if args.filtered_run
        else _find_latest_run(filtered=True)
    )

    if raw_dir is None:
        raise RuntimeError("Could not find a raw (filtered=False) compare run.")
    if filtered_dir is None:
        raise RuntimeError("Could not find a filtered (filtered=True) compare run.")

    print(f"Raw run:      {raw_dir.name}")
    print(f"Filtered run: {filtered_dir.name}")

    records = _load_run(raw_dir) + _load_run(filtered_dir)
    if not records:
        raise RuntimeError("No data loaded — check run directories.")

    df = pd.DataFrame(records)
    print(f"\nLoaded {len(df)} records across {df['scenario'].nunique()} scenarios.")
    print(
        df[
            [
                "scenario",
                "condition",
                "model_type",
                "fp_rate",
                "benign_ratio",
                "n_features",
            ]
        ].to_string(index=False)
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[plots] Saving to {out_dir}")
    plot_fp_vs_benign(df, out_dir)
    plot_fp_vs_features(df, out_dir)
    plot_bar_fp_rate(df, out_dir)

    csv_out = out_dir / "fp_benign_ratio_data.csv"
    df.to_csv(csv_out, index=False)
    print(f"  Saved → {csv_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()

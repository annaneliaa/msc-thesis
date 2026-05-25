"""
One-factor-at-a-time hyperparameter sweep for the symbolic experiment.

Sweeps each parameter independently while keeping the others at their default
values (strict-filter baseline). Tracks the number of symbolic features that
survive post-mining filtering and the model metrics (AUC, F1, precision, recall)
for each configuration.

Parameters swept:
  min_support            Mining-level frequency threshold (ECLAT / PrefixSpan).
  max_overlap            Max ratio of minority-class support to majority-class
                         support; controls how much a pattern may appear in the
                         "wrong" class before being filtered out.
  max_confidence_attack  Upper bound on attack confidence; drops patterns that
                         appear too frequently in attacks, keeping only patterns
                         that are genuinely rare or absent in attack traffic.
  min_confidence_benign  Minimum fraction of benign transactions containing the
                         pattern; ensures sufficient benign coverage.

Each run creates a new timestamped directory. Re-running adds new sweep points.
Use --no-run to plot from the most recent run's results without running new points.

Usage:
    python src/thesis/scripts/run_sweep.py fox
    python src/thesis/scripts/run_sweep.py fox --params min_support max_overlap
    python src/thesis/scripts/run_sweep.py fox --no-run   # plot from most recent run

Output (all under artifacts/experiments/run_sweep/sweep_<run_ts>/):
    scenario/<scenario>/results.csv             all sweep results for this run
    plots/<scenario>_sweep_<param>.png          dual-axis plot per parameter
    plots/<scenario>_sweep_overview.png         2×2 summary grid
    plots/sweep_<run_ts>.log
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from thesis.experiments.symbolic import (
    SymbolicExperimentConfig,
    run_symbolic_experiment,
)
from thesis.paths import ABSTRACTION_MAP_PATH


class _Tee:
    """Mirror stdout to both terminal and a log file."""

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


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from {here}")


_REPO = _find_repo_root()
sys.path.insert(0, str(_REPO / "src"))


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

# Strict-filter defaults — the baseline for all unvaried parameters.
DEFAULTS: dict = {
    "min_support": 0.05,
    "min_support_count": 50,
    "min_abs_support_diff": 0.20,
    "min_confidence_attack": 0.0,
    "max_confidence_attack": None,  # null = no upper bound (strict-filter default)
    "min_confidence_benign": 0.0,
    "max_overlap": 0.3,
    "min_lift": 2.0,
}

SWEEP_GRID: dict[str, list] = {
    "min_support": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
    "max_overlap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    "max_confidence_attack": [1.0, 0.5, 0.3, 0.2, 0.1, 0.05],
    "min_confidence_benign": [0.0, 0.05, 0.10, 0.20, 0.30],
}

ALL_PARAMS = list(SWEEP_GRID.keys())

# ---------------------------------------------------------------------------
# Filter YAML writing
# ---------------------------------------------------------------------------


def _write_filter_yaml(tmp_dir: Path, params: dict[str, float]) -> Path:
    """Write a filter config YAML with the given parameter values."""

    def _f(v):
        return float(v) if v is not None else None

    cfg = {
        "itemsets": {
            "min_k": 2,
            "max_k": None,
            "min_support_count": int(params["min_support_count"]),
            "min_abs_support_diff": float(params["min_abs_support_diff"]),
            "min_confidence_attack": float(params["min_confidence_attack"]),
            "max_confidence_attack": _f(params["max_confidence_attack"]),
            "min_confidence_benign": float(params["min_confidence_benign"]),
            "max_overlap": _f(params["max_overlap"]),
            "remove_subsumed": True,
        },
        "item_sequences": {
            "min_k": 3,
            "min_support_count": int(params["min_support_count"]),
            "min_abs_support_diff": float(params["min_abs_support_diff"]),
            "min_confidence_attack": float(params["min_confidence_attack"]),
            "max_confidence_attack": _f(params["max_confidence_attack"]),
            "min_confidence_benign": float(params["min_confidence_benign"]),
            "min_lift": float(params["min_lift"]),
            "max_overlap": _f(params["max_overlap"]),
            "remove_subsumed": True,
        },
        "feature_selection": {
            "top_k": None,
            "min_utility_score": None,
        },
    }
    path = tmp_dir / "filter.yaml"
    with path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return path


# ---------------------------------------------------------------------------
# Running a single sweep point
# ---------------------------------------------------------------------------


def _run_point(scenario: str, sweep_param: str, value: float, tmp_dir: Path) -> dict:
    """Run one sweep point and return a flat metrics dict."""
    params = {**DEFAULTS, sweep_param: value}

    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        filter_yaml = _write_filter_yaml(Path(td), params)

        cfg = SymbolicExperimentConfig(
            scenario=scenario,
            min_support=params["min_support"],
            filter_config=filter_yaml,
            abstraction_map_path=ABSTRACTION_MAP_PATH,
            model_name="logreg_sweep",
            model_version="0.1.0",
        )
        result = run_symbolic_experiment(cfg)

    n_sym = result.metrics.get("n_symbolic_features_used", result.n_features - 8)

    return {
        "scenario": scenario,
        "sweep_param": sweep_param,
        "value": value,
        "n_features_total": result.n_features,
        "n_symbolic_features": n_sym,
        "auc": result.auc,
        "balanced_accuracy": result.metrics.get("balanced_accuracy", float("nan")),
        "precision": result.metrics.get("precision", float("nan")),
        "recall": result.metrics.get("recall", float("nan")),
        "f1": result.metrics.get("f1", float("nan")),
        "tp": result.metrics.get("tp", float("nan")),
        "fp": result.metrics.get("fp", float("nan")),
        "tn": result.metrics.get("tn", float("nan")),
        "fn": result.metrics.get("fn", float("nan")),
    }


# ---------------------------------------------------------------------------
# Loading / saving results
# ---------------------------------------------------------------------------


def _find_latest_csv(scenario: str) -> Path | None:
    """Find the most recent results.csv across all sweep run dirs for this scenario."""
    candidates = sorted(
        (_REPO / "artifacts" / "experiments" / "run_sweep").glob(
            f"*/scenario/{scenario}/results.csv"
        )
    )
    return candidates[-1] if candidates else None


def _load_cached(csv_path: Path, scenario: str | None = None) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if scenario is not None:
        latest = _find_latest_csv(scenario)
        if latest is not None:
            print(f"  [cache] Loading sweep results from previous run: {latest}")
            return pd.read_csv(latest)
    return pd.DataFrame()


def _is_cached(cached: pd.DataFrame, sweep_param: str, value: float) -> bool:
    if cached.empty:
        return False
    mask = (cached["sweep_param"] == sweep_param) & (
        cached["value"].round(6) == round(value, 6)
    )
    return bool(mask.any())


def _append_and_save(cached: pd.DataFrame, row: dict, csv_path: Path) -> pd.DataFrame:
    new_row = pd.DataFrame([row])
    updated = pd.concat([cached, new_row], ignore_index=True)
    updated.to_csv(csv_path, index=False)
    return updated


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PARAM_LABELS = {
    "min_support": "min_support (mining threshold)",
    "max_overlap": "max_overlap (minority/majority ratio)",
    "min_confidence_attack": "min_confidence_attack",
    "min_confidence_benign": "min_confidence_benign",
}


def _plot_sweep_param(
    df: pd.DataFrame, sweep_param: str, scenario: str, out_dir: Path, ax=None
) -> None:
    sub = df[df["sweep_param"] == sweep_param].sort_values("value")
    if sub.empty:
        return

    standalone = ax is None
    if standalone:
        fig, ax1 = plt.subplots(figsize=(8, 5))
    else:
        ax1 = ax
        fig = ax.get_figure()

    color_feat = "#2196F3"
    color_auc = "#E91E63"
    color_f1 = "#4CAF50"

    ax1.bar(
        range(len(sub)),
        sub["n_symbolic_features"],
        color=color_feat,
        alpha=0.65,
        label="symbolic features",
    )
    ax1.set_ylabel("# symbolic features (post-filter)", color=color_feat)
    ax1.tick_params(axis="y", labelcolor=color_feat)
    ax1.set_xticks(range(len(sub)))
    ax1.set_xticklabels([f"{v:.3g}" for v in sub["value"]], rotation=30, ha="right")
    ax1.set_xlabel(PARAM_LABELS.get(sweep_param, sweep_param))

    ax2 = ax1.twinx()
    ax2.plot(
        range(len(sub)),
        sub["auc"],
        marker="o",
        color=color_auc,
        linewidth=2,
        label="AUC",
    )
    ax2.plot(
        range(len(sub)),
        sub["f1"],
        marker="s",
        color=color_f1,
        linewidth=2,
        linestyle="--",
        label="F1",
    )
    ax2.set_ylabel("Score", color="black")
    ax2.set_ylim(0, 1.05)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    title = f"{scenario} — sweep: {sweep_param}"
    if standalone:
        ax1.set_title(title)
        fig.tight_layout()
        out_path = out_dir / f"{scenario}_sweep_{sweep_param}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {out_path}")
    else:
        ax1.set_title(title, fontsize=10)


def plot_all_sweeps(
    df: pd.DataFrame, scenario: str, out_dir: Path, params: list[str]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Individual plots
    for param in params:
        _plot_sweep_param(df, param, scenario, out_dir)

    # Overview grid (2×2 or 1×N depending on how many params)
    n = len(params)
    if n < 2:
        return

    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, param in enumerate(params):
        _plot_sweep_param(df, param, scenario, out_dir, ax=axes_flat[i])

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{scenario} — hyperparameter sweep overview", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"{scenario}_sweep_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OFAT hyperparameter sweep for the symbolic experiment."
    )
    parser.add_argument("scenario", help="Scenario name (e.g. fox)")
    parser.add_argument(
        "--params",
        nargs="+",
        choices=ALL_PARAMS,
        default=ALL_PARAMS,
        help="Which parameters to sweep (default: all four).",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running experiments; plot from cached results only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a cached result exists for the sweep point.",
    )
    args = parser.parse_args()

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = _REPO / "artifacts" / "experiments" / "run_sweep" / f"sweep_{run_ts}"
    scenario_dir = run_dir / "scenario" / args.scenario
    plots_dir = run_dir / "plots"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = scenario_dir / "results.csv"
    tmp_dir = run_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    log_path = plots_dir / f"sweep_{run_ts}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Logging to {log_path}")

    try:
        _sweep_body(args, plots_dir, csv_path, tmp_dir)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"Log saved → {log_path}")


def _sweep_body(
    args: object,
    plots_dir: Path,
    csv_path: Path,
    tmp_dir: Path,
) -> None:
    cached = _load_cached(csv_path, scenario=args.scenario if args.no_run else None)

    if not args.no_run:
        for param in args.params:
            values = SWEEP_GRID[param]
            print(f"\n{'='*60}")
            print(f" Sweep: {param}  ({len(values)} values)")
            print(f"{'='*60}")
            for value in values:
                if not args.force and _is_cached(cached, param, value):
                    print(f"  [{param}={value:.4g}] cached — skipping.")
                    continue
                print(f"\n  [{param}={value:.4g}] running...")
                row = _run_point(args.scenario, param, value, tmp_dir)
                cached = _append_and_save(cached, row, csv_path)
                print(
                    f"  [{param}={value:.4g}] AUC={row['auc']:.4f}  sym_features={row['n_symbolic_features']}"
                )

    if cached.empty:
        print("No results available. Exiting.")
        return

    print(f"\n[results] {len(cached)} sweep points in {csv_path}")

    params_with_data = [p for p in args.params if p in cached["sweep_param"].values]
    print(f"[plots] Generating plots for: {params_with_data}")
    plot_all_sweeps(cached, args.scenario, plots_dir, params_with_data)

    print("\nDone.")


if __name__ == "__main__":
    main()

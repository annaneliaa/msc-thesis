"""
Compare baseline vs symbolic across one or more scenarios.

For each scenario: skips if a compare result already exists, otherwise runs.
Use --force to re-run even when results exist.

Usage:
    python src/thesis/scripts/run_compare_scenarios.py fox wheeler harrison wilson santos
    python src/thesis/scripts/run_compare_scenarios.py fox --force
    python src/thesis/scripts/run_compare_scenarios.py fox wheeler --no-run

Output (all under artifacts/experiments/run_compare/):
    <scenarios>/compare_<ts>.json        per-scenario result
    plots/<scenarios>/compare_table.csv
    plots/<scenarios>/compare_delta_table.csv
    plots/<scenarios>/compare_table.txt
    plots/<scenarios>/compare_auc.png
    plots/<scenarios>/compare_metrics.png
    plots/<scenarios>/compare_features.png
    plots/<scenarios>/compare_fp.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from thesis.experiments.baseline import (
    BaselineExperimentConfig,
    run_baseline_experiment,
)
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
    raise RuntimeError(f"Could not find repo root (no pyproject.toml) from {here}")


_REPO = _find_repo_root()
sys.path.insert(0, str(_REPO / "src"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILTER_CONFIG = _REPO / "src/thesis/configs/mining_filters_strict.yaml"
EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_compare"


# ---------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------


def _latest_json(directory: Path, prefix: str) -> Path | None:
    candidates = sorted(directory.glob(f"{prefix}_*.json"))
    return candidates[-1] if candidates else None


def _load_compare_result(scenario: str) -> dict | None:
    """Load the most recent compare_*.json for a scenario, if it exists."""
    scenario_dir = EXPERIMENTS_DIR / scenario
    if not scenario_dir.exists():
        return None
    path = _latest_json(scenario_dir, "compare")
    if path is None:
        return None
    with path.open() as f:
        data = json.load(f)
    return {
        "scenario": scenario,
        "baseline": {
            **data["baseline"]["metrics"],
            "n_features": data["baseline"]["n_features"],
            "n_transactions": data["baseline"]["n_transactions"],
        },
        "symbolic": {
            **data["symbolic"]["metrics"],
            "n_features": data["symbolic"]["n_features"],
            "n_transactions": data["symbolic"]["n_transactions"],
        },
        "filter_config": data.get("filter_config"),
        "source": path.name,
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _run_compare(
    scenario: str,
    filter_config: Path = FILTER_CONFIG,
    model_name: str = "logreg",
) -> dict:
    print(f"\n{'='*60}")
    print(f" Running compare: {scenario}")
    print(f"{'='*60}")

    print("\n--- Phase 1/2: baseline ---")
    baseline = run_baseline_experiment(
        BaselineExperimentConfig(scenario=scenario, model_name=model_name)
    )

    print("\n--- Phase 2/2: symbolic ---")
    symbolic = run_symbolic_experiment(
        SymbolicExperimentConfig(
            scenario=scenario,
            filter_config=filter_config,
            abstraction_map_path=ABSTRACTION_MAP_PATH,
            model_name=model_name,
        )
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = EXPERIMENTS_DIR / scenario
    results_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "experiment": "compare",
        "scenario": scenario,
        "timestamp": timestamp,
        "model_name": model_name,
        "filter_config": str(filter_config),
        "baseline": {
            "schema_name": baseline.schema_name,
            "schema_version": baseline.schema_version,
            "auc": baseline.auc,
            "n_features": baseline.n_features,
            "n_transactions": baseline.n_transactions,
            "metrics": baseline.metrics,
            "results_file": str(baseline.results_file),
        },
        "symbolic": {
            "schema_name": symbolic.schema_name,
            "schema_version": symbolic.schema_version,
            "auc": symbolic.auc,
            "n_features": symbolic.n_features,
            "n_transactions": symbolic.n_transactions,
            "metrics": symbolic.metrics,
            "results_file": str(symbolic.results_file),
        },
        "delta": {
            "auc": round(symbolic.auc - baseline.auc, 6),
            "n_features": symbolic.n_features - baseline.n_features,
        },
    }

    out_path = results_dir / f"compare_{timestamp}.json"
    with out_path.open("w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Saved → {out_path}")

    return {
        "scenario": scenario,
        "model_name": model_name,
        "baseline": {
            **baseline.metrics,
            "n_features": baseline.n_features,
            "n_transactions": baseline.n_transactions,
        },
        "symbolic": {
            **symbolic.metrics,
            "n_features": symbolic.n_features,
            "n_transactions": symbolic.n_transactions,
        },
        "filter_config": str(filter_config),
        "source": out_path.name,
    }


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

TABLE_METRICS = [
    "auc",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "n_features",
    "n_transactions",
]


def _build_long_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for kind in ("baseline", "symbolic"):
            m = r[kind]
            row = {"scenario": r["scenario"], "model": kind}
            for metric in TABLE_METRICS:
                row[metric] = m.get(metric, float("nan"))
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in ["auc", "balanced_accuracy", "precision", "recall", "f1"]:
        df[col] = df[col].round(4)
    return df


def _build_delta_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        b, s = r["baseline"], r["symbolic"]
        rows.append(
            {
                "scenario": r["scenario"],
                "baseline_auc": round(b.get("auc", float("nan")), 4),
                "symbolic_auc": round(s.get("auc", float("nan")), 4),
                "delta_auc": round(
                    s.get("auc", float("nan")) - b.get("auc", float("nan")), 4
                ),
                "baseline_f1": round(b.get("f1", float("nan")), 4),
                "symbolic_f1": round(s.get("f1", float("nan")), 4),
                "delta_f1": round(
                    s.get("f1", float("nan")) - b.get("f1", float("nan")), 4
                ),
                "baseline_precision": round(b.get("precision", float("nan")), 4),
                "symbolic_precision": round(s.get("precision", float("nan")), 4),
                "baseline_recall": round(b.get("recall", float("nan")), 4),
                "symbolic_recall": round(s.get("recall", float("nan")), 4),
                "base_features": int(b.get("n_features", 0)),
                "sym_features": int(s.get("n_features", 0)),
                "n_transactions": int(b.get("n_transactions", 0)),
                "baseline_fp": int(b.get("fp", 0)),
                "symbolic_fp": int(s.get("fp", 0)),
                "delta_fp": int(s.get("fp", 0)) - int(b.get("fp", 0)),
            }
        )
    return pd.DataFrame(rows)


def _format_text_table(delta_df: pd.DataFrame) -> str:
    sep = "─" * 104
    lines = [
        sep,
        f"  {'scenario':<14} {'base AUC':>9} {'sym AUC':>9} {'ΔAUC':>7}  "
        f"{'base F1':>8} {'sym F1':>8} {'ΔF1':>7}  "
        f"{'base FP':>7} {'sym FP':>7} {'ΔFP':>6}  "
        f"{'base feat':>9} {'sym feat':>9}  {'n_tx':>7}",
        sep,
    ]
    for _, r in delta_df.iterrows():
        lines.append(
            f"  {r['scenario']:<14} {r['baseline_auc']:>9.4f} {r['symbolic_auc']:>9.4f} {r['delta_auc']:>+7.4f}  "
            f"{r['baseline_f1']:>8.4f} {r['symbolic_f1']:>8.4f} {r['delta_f1']:>+7.4f}  "
            f"{r['baseline_fp']:>7d} {r['symbolic_fp']:>7d} {r['delta_fp']:>+6d}  "
            f"{r['base_features']:>9d} {r['sym_features']:>9d}  {r['n_transactions']:>7d}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _sort_key(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    key = {s: i for i, s in enumerate(order)}
    return df.sort_values(
        "scenario", key=lambda s: s.map(lambda x: key.get(x, len(order)))
    )


def plot_auc_comparison(delta_df: pd.DataFrame, out_dir: Path) -> None:
    df = _sort_key(delta_df, delta_df["scenario"].tolist())
    scenarios = df["scenario"].tolist()
    x = np.arange(len(scenarios))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(scenarios) * 1.6), 5))
    b1 = ax.bar(x - w / 2, df["baseline_auc"], w, label="Baseline (base features)")
    b2 = ax.bar(x + w / 2, df["symbolic_auc"], w, label="Symbolic (base + mined)")
    for bar in (*b1, *b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("AUC")
    ax.set_title("Baseline vs Symbolic — AUC per scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_auc.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_dir / 'compare_auc.png'}")


def plot_metrics_breakdown(delta_df: pd.DataFrame, out_dir: Path) -> None:
    df = _sort_key(delta_df, delta_df["scenario"].tolist())
    scenarios = df["scenario"].tolist()
    x = np.arange(len(scenarios))
    w = 0.18

    fig, axes = plt.subplots(
        1, 2, figsize=(max(12, len(scenarios) * 2.8), 5), sharey=True
    )
    for ax, col_prefix, title in [
        (axes[0], "baseline", "Baseline"),
        (axes[1], "symbolic", "Symbolic"),
    ]:
        for label, col, offset in [
            ("AUC", f"{col_prefix}_auc", -1.5),
            ("Precision", f"{col_prefix}_precision", -0.5),
            ("Recall", f"{col_prefix}_recall", 0.5),
            ("F1", f"{col_prefix}_f1", 1.5),
        ]:
            ax.bar(x + offset * w, df[col], w, label=label)
        ax.set_title(f"{title} — metrics per scenario")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_dir / 'compare_metrics.png'}")


def plot_feature_counts(delta_df: pd.DataFrame, out_dir: Path) -> None:
    df = _sort_key(delta_df, delta_df["scenario"].tolist())
    scenarios = df["scenario"].tolist()
    x = np.arange(len(scenarios))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(scenarios) * 1.6), 5))
    b1 = ax.bar(x - w / 2, df["base_features"], w, label="Base features")
    b2 = ax.bar(x + w / 2, df["sym_features"], w, label="Symbolic features (total)")
    for bar in (*b1, *b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Number of features")
    ax.set_title("Feature count — base vs symbolic schema")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_features.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_dir / 'compare_features.png'}")


def plot_fp_comparison(delta_df: pd.DataFrame, out_dir: Path) -> None:
    df = _sort_key(delta_df, delta_df["scenario"].tolist())
    scenarios = df["scenario"].tolist()
    x = np.arange(len(scenarios))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(scenarios) * 1.6), 5))
    b1 = ax.bar(x - w / 2, df["baseline_fp"], w, label="Baseline FP")
    b2 = ax.bar(x + w / 2, df["symbolic_fp"], w, label="Symbolic FP")
    for bar in (*b1, *b2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("False Positives (test set)")
    ax.set_title("False positive count — baseline vs symbolic")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_fp.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_dir / 'compare_fp.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline vs symbolic comparison for one or more scenarios."
    )
    parser.add_argument(
        "scenarios", nargs="+", help="Scenario names (e.g. fox wheeler harrison)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if compare results already exist.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running; only load existing results and (re-)plot.",
    )
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=FILTER_CONFIG,
        help=f"Path to the mining filter YAML (default: {FILTER_CONFIG})",
    )
    parser.add_argument(
        "--model-name",
        default="logreg",
        help="Model to use: logreg, logreg_l1, random_forest, lstm (default: logreg)",
    )
    args = parser.parse_args()

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    subdir_name = "_".join(args.scenarios)
    log_dir = EXPERIMENTS_DIR / "plots" / subdir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"compare_{run_ts}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Logging to {log_path}")

    try:
        _run_main(args, run_ts, subdir_name)
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"Log saved → {log_path}")


def _run_main(args: object, run_ts: str, subdir_name: str) -> None:
    all_results: list[dict] = []

    for scenario in args.scenarios:
        existing = _load_compare_result(scenario)

        if args.no_run:
            if existing is not None:
                print(f"[{scenario}] Loaded existing results ({existing['source']})")
                all_results.append(existing)
            else:
                print(f"[{scenario}] No existing results — skipping (--no-run is set).")
            continue

        if existing is not None and not args.force:
            print(
                f"[{scenario}] Skipping — compare result already exists ({existing['source']}). Use --force to re-run."
            )
            all_results.append(existing)
            continue

        r = _run_compare(
            scenario, filter_config=args.filter_config, model_name=args.model_name
        )
        all_results.append(r)

    if not all_results:
        print("No results to process. Exiting.")
        return

    long_df = _build_long_table(all_results)
    delta_df = _build_delta_table(all_results)

    out_dir = EXPERIMENTS_DIR / "plots" / subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "compare_table.csv"
    long_df.to_csv(csv_path, index=False)
    print(f"\n[table] Saved long-form CSV → {csv_path}")

    delta_csv = out_dir / "compare_delta_table.csv"
    delta_df.to_csv(delta_csv, index=False)
    print(f"[table] Saved delta CSV → {delta_csv}")

    txt_path = out_dir / "compare_table.txt"
    text_table = _format_text_table(delta_df)
    txt_path.write_text(text_table, encoding="utf-8")
    print(f"[table] Saved formatted table → {txt_path}")

    print("\n" + text_table)

    print(f"\n[plots] Saving to {out_dir}")
    plot_auc_comparison(delta_df, out_dir)
    plot_metrics_breakdown(delta_df, out_dir)
    plot_feature_counts(delta_df, out_dir)
    plot_fp_comparison(delta_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

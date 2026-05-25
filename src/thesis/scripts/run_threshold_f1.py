"""
run_threshold_f1.py

For a given scenario, runs all four experiment configurations:
  - baseline  × fixed_2s
  - symbolic  × fixed_2s
  - baseline  × alertbert
  - symbolic  × alertbert

Then scans F1 (+ precision/recall) across decision thresholds and plots
the results.  The heavy lifting (mining, encoding, training) reuses the
same cache dirs as run_grouping_compare.py so repeated runs are cheap.

Usage:
    python src/thesis/scripts/run_threshold_f1.py <scenario> \\
        --alertbert-model-id mlm_1l_1h_16d_original_1_60k \\
        [--alertbert-models-path external/AlertBERT/saved_models] \\
        [--filter-config src/thesis/configs/mining_filters_strict.yaml] \\
        [--thresholds 101]

Output (under artifacts/experiments/run_threshold_f1/threshold_f1_<run_ts>/):
    scenario/<scenario>/threshold_f1_<ts>.csv    -- F1/P/R at each threshold
    plots/threshold_f1_<ts>.png                  -- F1 curve plot
    plots/threshold_f1_<ts>.log
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from thesis.config import AlertBERTConfig, GroupingConfig
from thesis.experiments.baseline import run_baseline_experiment
from thesis.experiments.symbolic import run_symbolic_experiment
from thesis.paths import ABSTRACTION_MAP_PATH, CACHE_DIR
from thesis.registry.models import resolve_model_paths
from thesis.schemas.experiments import (
    BaselineExperimentConfig,
    SymbolicExperimentConfig,
)

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_threshold_f1"


_STYLE = {
    "baseline_fixed_2s": {
        "color": "#2166ac",
        "ls": "-",
        "label": "baseline / fixed_2s",
    },
    "symbolic_fixed_2s": {
        "color": "#4dac26",
        "ls": "-",
        "label": "symbolic / fixed_2s",
    },
    "baseline_alertbert": {
        "color": "#d6604d",
        "ls": "--",
        "label": "baseline / alertbert",
    },
    "symbolic_alertbert": {
        "color": "#f1a340",
        "ls": "--",
        "label": "symbolic / alertbert",
    },
}


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def _fixed_grouping() -> GroupingConfig:
    return GroupingConfig(mode="fixed_2s")


def _alertbert_grouping(model_id: str, models_path: str) -> GroupingConfig:
    return GroupingConfig(
        mode="alertbert",
        alertbert=AlertBERTConfig(
            model_id=model_id,
            models_path=models_path,
            delta=2.0,
            theta=6.0,
            dim_reduction=2,
            device="cpu",
        ),
    )


# ---------------------------------------------------------------------------
# Probability extraction
# ---------------------------------------------------------------------------


def _extract_probas(
    cache_dir: Path,
    scenario: str,
    schema_name: str,
    grouping_mode: str,
    model_name: str = "logreg",
    test_frac: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reload the encoded-transaction parquet and saved model to reconstruct
    the exact test-set probabilities used during training.

    The split logic mirrors train_model_for_schema / make_holdout_split:
    no timestamp column is present in the parquet, so rows are used in
    natural (parquet) order and split at int((1 - test_frac) * n).
    """
    safe_name = schema_name.replace("+", "_").replace("/", "_")
    parquet_path = (
        cache_dir / scenario / "transactions" / f"transactions_{safe_name}.parquet"
    )
    grouping_tag = grouping_mode.replace("-", "_")
    model_version = f"0.1.0_{safe_name}_{grouping_tag}"
    model_path, metadata_path, _ = resolve_model_paths(
        scenario, model_name, model_version
    )

    if not parquet_path.exists():
        raise FileNotFoundError(f"Encoded transactions not found: {parquet_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    with metadata_path.open() as f:
        metadata = json.load(f)
    feature_names: list[str] = metadata["features"]

    df = pd.read_parquet(parquet_path)
    y = df["tx_label"].map({"benign": 0, "attack": 1})
    X = df[feature_names].fillna(0)
    mask = y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    n = len(X)
    split = int((1 - test_frac) * n)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:].to_numpy()

    model = joblib.load(model_path)
    proba_test = model.predict_proba(X_test)[:, 1]

    return y_test, proba_test


# ---------------------------------------------------------------------------
# Threshold scanning
# ---------------------------------------------------------------------------


def _scan_thresholds(
    y_test: np.ndarray,
    proba_test: np.ndarray,
    n_thresholds: int = 101,
) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows = []
    for th in thresholds:
        y_pred = (proba_test >= th).astype(int)
        rows.append(
            {
                "threshold": float(th),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_curves(
    curves: dict[str, pd.DataFrame],
    scenario: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(f"Threshold analysis — scenario: {scenario}", fontsize=13)

    metrics = ["f1", "precision", "recall"]
    ylabels = ["F1", "Precision", "Recall"]

    # Draw symbolic first so baseline (drawn on top) stays visible when curves overlap.
    draw_order = [k for k in _STYLE if k in curves and k.startswith("symbolic")] + [
        k for k in _STYLE if k in curves and k.startswith("baseline")
    ]

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        for key in draw_order:
            df = curves[key]
            style = _STYLE[key]
            # find best F1 threshold for annotation
            best_idx = df["f1"].idxmax()
            best_th = df.loc[best_idx, "threshold"]
            best_val = df.loc[best_idx, metric]

            ax.plot(
                df["threshold"],
                df[metric],
                color=style["color"],
                ls=style["ls"],
                lw=1.8,
                label=style["label"],
            )
            if metric == "f1":
                ax.scatter(
                    [best_th],
                    [best_val],
                    color=style["color"],
                    marker="o",
                    s=50,
                    zorder=5,
                )

        ax.axvline(0.5, color="grey", lw=0.8, ls=":", alpha=0.7)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if metric == "f1":
            ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F1-vs-threshold curves for baseline/symbolic × fixed_2s/alertbert"
    )
    parser.add_argument("scenario", help="Scenario name (e.g. fox)")
    parser.add_argument(
        "--alertbert-model-id",
        default="mlm_1l_1h_16d_original_1_60k",
    )
    parser.add_argument(
        "--alertbert-models-path",
        default=str(_REPO / "external" / "AlertBERT" / "saved_models"),
    )
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        default=101,
        help="Number of threshold steps between 0 and 1 (default: 101)",
    )
    parser.add_argument(
        "--clear-parquets",
        action="store_true",
        help="Delete cached parquets before running (forces re-encoding)",
    )
    args = parser.parse_args()

    scenario: str = args.scenario
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    run_dir = EXPERIMENTS_DIR / f"threshold_f1_{run_ts}"
    scenario_dir = run_dir / "scenario" / scenario
    plots_dir = run_dir / "plots"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_path = plots_dir / f"threshold_f1_{run_ts}.log"
    _tee = _Tee(log_path)
    sys.stdout = _tee
    print(f"Log saved → {log_path}")

    try:
        _main_body(args, scenario, scenario_dir, plots_dir, run_ts)
    finally:
        sys.stdout = sys.__stdout__
        _tee.close()


def _main_body(
    args, scenario: str, scenario_dir: Path, plots_dir: Path, run_ts: str
) -> None:
    cache_dir_fw = CACHE_DIR / "grouping_compare" / scenario / "fixed_2s"
    cache_dir_ab = CACHE_DIR / "grouping_compare" / scenario / "alertbert"
    alertbert_groups_dir = (
        CACHE_DIR / "alertbert_groups" / scenario / args.alertbert_model_id
    )
    for d in [cache_dir_fw, cache_dir_ab, alertbert_groups_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.clear_parquets:
        print("--- Clearing cached parquets ---")
        for cache_dir in [cache_dir_fw, cache_dir_ab]:
            tx_dir = cache_dir / scenario / "transactions"
            if tx_dir.exists():
                for pq in tx_dir.glob("*.parquet"):
                    print(f"  Deleting {pq.name}")
                    pq.unlink()

    grouping_fw = _fixed_grouping()
    grouping_ab = _alertbert_grouping(
        args.alertbert_model_id, args.alertbert_models_path
    )

    # ------------------------------------------------------------------
    # 1. Run experiments (mining + training; encoding is cached per-run)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" Threshold F1 experiment: {scenario}")
    print(f"{'='*60}")

    # Model paths are scoped by both schema name and grouping mode, so
    # fixed_2s and alertbert models do not collide.  Probabilities are still
    # extracted immediately after each experiment so the parquet and model
    # are always in sync.

    curves: dict[str, pd.DataFrame] = {}
    all_rows: list[dict] = []

    def _run_and_record(
        label: str, cache_dir: Path, schema_name: str, grouping_mode: str
    ) -> None:
        print(f"  {label} ...", end=" ", flush=True)
        try:
            y_test, proba_test = _extract_probas(
                cache_dir=cache_dir,
                scenario=scenario,
                schema_name=schema_name,
                grouping_mode=grouping_mode,
            )
            df_curve = _scan_thresholds(
                y_test, proba_test, n_thresholds=args.thresholds
            )
            df_curve.insert(0, "config", label)
            curves[label] = df_curve
            all_rows.append(df_curve)
            best = df_curve.loc[df_curve["f1"].idxmax()]
            print(
                f"best F1={best['f1']:.4f} @ threshold={best['threshold']:.2f} "
                f"(P={best['precision']:.4f}, R={best['recall']:.4f})"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n--- [1/4] baseline × fixed_2s ---")
    run_baseline_experiment(
        BaselineExperimentConfig(
            scenario=scenario,
            cache_dir=cache_dir_fw,
            grouping=grouping_fw,
        )
    )
    print("\n--- Extracting probabilities ---")
    _run_and_record("baseline_fixed_2s", cache_dir_fw, "base", grouping_fw.mode)

    print("\n--- [2/4] symbolic × fixed_2s ---")
    run_symbolic_experiment(
        SymbolicExperimentConfig(
            scenario=scenario,
            cache_dir=cache_dir_fw,
            grouping=grouping_fw,
            filter_config=args.filter_config,
            abstraction_map_path=ABSTRACTION_MAP_PATH,
        )
    )
    print("\n--- Extracting probabilities ---")
    _run_and_record(
        "symbolic_fixed_2s", cache_dir_fw, "base+symbolic", grouping_fw.mode
    )

    print("\n--- [3/4] baseline × alertbert ---")
    run_baseline_experiment(
        BaselineExperimentConfig(
            scenario=scenario,
            cache_dir=cache_dir_ab,
            grouping=grouping_ab,
            grouping_cache_dir=alertbert_groups_dir,
        )
    )
    print("\n--- Extracting probabilities ---")
    _run_and_record("baseline_alertbert", cache_dir_ab, "base", grouping_ab.mode)

    print("\n--- [4/4] symbolic × alertbert ---")
    run_symbolic_experiment(
        SymbolicExperimentConfig(
            scenario=scenario,
            cache_dir=cache_dir_ab,
            grouping=grouping_ab,
            filter_config=args.filter_config,
            abstraction_map_path=ABSTRACTION_MAP_PATH,
            grouping_cache_dir=alertbert_groups_dir,
        )
    )
    print("\n--- Extracting probabilities ---")
    _run_and_record(
        "symbolic_alertbert", cache_dir_ab, "base+symbolic", grouping_ab.mode
    )

    # ------------------------------------------------------------------
    # 3. Save CSV
    # ------------------------------------------------------------------
    csv_path = scenario_dir / f"threshold_f1_{run_ts}.csv"
    if not all_rows:
        print("ERROR: No curves were extracted! Check errors above.")
        return

    if len(curves) < 4:
        print(
            f"\nWARNING: Only {len(curves)}/4 curves extracted:"
            f" {list(curves.keys())}"
        )
        print("Run with --clear-parquets to force fresh encoding.")

    pd.concat(all_rows, ignore_index=True).to_csv(csv_path, index=False)
    print(f"\nCSV saved → {csv_path}")

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    plot_path = plots_dir / f"threshold_f1_{run_ts}.png"
    _plot_curves(curves, scenario, plot_path)

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    print(f"\n{'─'*64}")
    print(f"{'config':<25}  {'best F1':>8}  {'threshold':>9}  {'P':>6}  {'R':>6}")
    print(f"{'─'*64}")
    for key, df_curve in curves.items():
        best = df_curve.loc[df_curve["f1"].idxmax()]
        print(
            f"{key:<25}  {best['f1']:>8.4f}  {best['threshold']:>9.3f}"
            f"  {best['precision']:>6.4f}  {best['recall']:>6.4f}"
        )
    print(f"{'─'*64}")

    # also show F1 at the standard 0.5 threshold
    print("\n--- F1 at threshold = 0.50 ---")
    for key, df_curve in curves.items():
        row = df_curve.loc[(df_curve["threshold"] - 0.5).abs().idxmin()]
        print(
            f"  {key:<25}  F1={row['f1']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}"
        )


# ---------------------------------------------------------------------------
# Tee (stdout → terminal + log file)
# ---------------------------------------------------------------------------


class _Tee:
    def __init__(self, path: Path) -> None:
        self._file = path.open("w", encoding="utf-8")
        self._stdout = sys.__stdout__

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


if __name__ == "__main__":
    main()

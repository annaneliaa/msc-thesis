"""
Visualisation companion for run_mining_window_sweep.py.

Usage:
    # Explicit run directory
    python src/thesis/scripts/run_mining_window_sweep_plots.py \
        --run-dir artifacts/experiments/mining_window_sweep/<dataset>/<timestamp>/

    # Or auto-pick the most recent run for a dataset
    python src/thesis/scripts/run_mining_window_sweep_plots.py --dataset cscas
    python src/thesis/scripts/run_mining_window_sweep_plots.py --dataset ait-ads

Reads the 7 analysis CSVs produced by the sweep script and saves PNG plots
under <run_dir>/plots/. The plot implementations live in
thesis.visualization.mining.window_sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thesis.visualization.mining import window_sweep as plots

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
_EXPERIMENTS_DIR = _REPO / "artifacts" / "experiments" / "run_mining_window_sweep"


def _latest_run_dir(dataset: str) -> Path:
    """Most recent timestamped run directory under mining_window_sweep/<dataset>/.

    Run directories are named %Y%m%d_%H%M%S, so lexicographic sort == chronological.
    """
    dataset_dir = _EXPERIMENTS_DIR / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"No runs found for dataset '{dataset}' at {dataset_dir}\n"
            "Run run_mining_window_sweep.py for this dataset first."
        )
    run_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {dataset_dir}")
    return run_dirs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mining window sweep results")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a specific mining_window_sweep run directory. "
        "Mutually exclusive with --dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. 'ait-ads', 'cscas') — auto-selects the most "
        "recent run under artifacts/experiments/mining_window_sweep/<dataset>/. "
        "Mutually exclusive with --run-dir.",
    )
    parser.add_argument(
        "--gran",
        type=float,
        nargs="*",
        default=None,
        help="Restrict to specific granularity values (e.g. --gran 0.1 0.2)",
    )
    args = parser.parse_args()

    if args.run_dir is None and args.dataset is None:
        parser.error("Provide --run-dir or --dataset.")
    if args.run_dir is not None and args.dataset is not None:
        parser.error("Provide only one of --run-dir or --dataset, not both.")

    if args.dataset is not None:
        run_dir = _latest_run_dir(args.dataset)
        print(f"[dataset={args.dataset}] Using latest run: {run_dir}")
    else:
        run_dir = args.run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    def _load(name: str) -> pd.DataFrame | None:
        p = run_dir / name
        if not p.exists():
            print(f"  [skip] {name} not found")
            return None
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            print(f"  [skip] {name} is empty (no rows to diagnose)")
            return None
        if args.gran is not None and "gran" in df.columns:
            df = df[df["gran"].isin(args.gran)]
        return df

    print(f"Reading CSVs from {run_dir}")
    t1 = _load("table1_stability.csv")
    t2 = _load("table2_sharing.csv")
    t3 = _load("table3_convergence.csv")
    t4 = _load("table4_dropped_diagnosis.csv")
    t5 = _load("table5_benign_vs_mixed.csv")
    t6 = _load("table6_persistence.csv")

    print(f"\nGenerating plots → {plots_dir}")

    if t3 is not None and not t3.empty:
        plots.plot_jaccard_stability(t3, plots_dir)

    if t1 is not None and not t1.empty:
        plots.plot_feature_count(t1, plots_dir)
        plots.plot_churn(t1, plots_dir)

    if t6 is not None and not t6.empty:
        plots.plot_persistence(t6, plots_dir)

    if t5 is not None and not t5.empty and t1 is not None and not t1.empty:
        plots.plot_attack_contamination(t5, t1, plots_dir)

    if t2 is not None and not t2.empty and t6 is not None and not t6.empty:
        plots.plot_cross_scenario_sharing(t2, t6, plots_dir)

    if t4 is not None and not t4.empty:
        plots.plot_dropped_diagnosis(t4, plots_dir)

    plots.plot_feature_lifecycle(run_dir, plots_dir, gran_filter=args.gran)

    print("\nDone.")


if __name__ == "__main__":
    main()

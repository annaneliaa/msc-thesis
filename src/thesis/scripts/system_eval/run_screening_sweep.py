"""
In-Window Baseline (Screening Sweep) CLI.

Runs experiments.screening_sweep.run_screening_sweep_experiment for one or
more scenarios: for a grid of (mining_setting x granularity), mines an
attribute schema inside every chronological window (on that window's train
split only) and evaluates cheap screening models (default: LogReg)
purely within that window, alongside a no-symbolic baseline pass. See
experiments/screening_sweep.py's module docstring for the full experiment
design.

Usage:
  # CSCAS (pre-grouped Suricata scenario) -- run once first:
  #   python src/thesis/scripts/run_ingest_cscas.py
  # Default mining settings (configs/screening_mining_settings.yaml: the
  # min_growth_rate x max_depth grid from shell-scripts/sweep_attribute_schema.sh)
  # and default granularities (0.1 0.2 0.33 0.5 1.0, that same script's MINE_FRACS):
  python src/thesis/scripts/system_eval/run_screening_sweep.py cscas

  # Fewer windows per granularity (cheaper pass)
  python src/thesis/scripts/system_eval/run_screening_sweep.py cscas \\
      --granularities 0.2 0.5 --windows-per-gran 3

  # AIT-ADS scenario, filtered alerts
  python src/thesis/scripts/system_eval/run_screening_sweep.py fox --filtered naive50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thesis.config import GroupingConfig
from thesis.configs import dataset_for_scenario
from thesis.system_eval.screening_sweep import run_screening_sweep_experiment
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.schemas.experiments import ScreeningSweepConfig
from thesis.scripts.system_eval._common import cache_dir_for
from thesis.training.model_factory import MODEL_FACTORIES
from thesis.visualization.eda import SCENARIOS as ALL_SCENARIOS

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

_DEFAULT_MINING_SETTINGS = (
    _REPO / "src" / "thesis" / "configs" / "screening_mining_settings.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="In-window screening sweep: (mining_setting x granularity), evaluated within each window.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scenarios", nargs="*", help="Scenario names (e.g. cscas, fox)."
    )
    parser.add_argument(
        "--all",
        dest="all_scenarios",
        action="store_true",
        help=f"Run all scenarios: {', '.join(ALL_SCENARIOS)}.",
    )
    parser.add_argument(
        "--granularities",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.33, 0.5, 1.0],
        metavar="FRAC",
        help="Window size as fraction of total alert_groups. Default: 0.1 0.2 0.33 0.5 1.0",
    )
    parser.add_argument(
        "--mining-settings",
        type=Path,
        default=_DEFAULT_MINING_SETTINGS,
        dest="mining_settings",
        metavar="YAML",
        help=f"Named mining-setting grid axis. Default: {_DEFAULT_MINING_SETTINGS}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_FACTORIES),
        default=["logreg"],
        metavar="MODEL",
        help="Screening models for the symbolic pass. Default: logreg",
    )
    parser.add_argument(
        "--baseline-models",
        nargs="+",
        choices=list(MODEL_FACTORIES),
        default=["logreg"],
        dest="baseline_models",
        metavar="MODEL",
        help="Screening models for the no-symbolic baseline pass. Default: logreg",
    )
    parser.add_argument(
        "--train-frac-within-window",
        type=float,
        default=0.7,
        dest="train_frac_within_window",
        metavar="FRAC",
        help="Fraction of each window (chronologically first) used for train. Default: 0.7",
    )
    parser.add_argument(
        "--windows-per-gran",
        type=int,
        default=None,
        dest="windows_per_gran",
        metavar="N",
        help=(
            "Evaluate only N evenly-spaced windows per granularity instead of "
            "all of them. Default: unset (evaluate every window -- the full "
            "sweep is the default per the experiment design; only subsample "
            "if that's computationally infeasible)."
        ),
    )
    parser.add_argument(
        "--filtered",
        nargs="?",
        const="",
        default=None,
        metavar="METHOD",
        help="AIT-ADS only: use filtered alerts. Optionally a balancing method (e.g. naive50).",
    )
    parser.add_argument(
        "--window-size", type=int, default=2, metavar="W", dest="window_size"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any cached mined schema and re-mine.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        dest="n_jobs",
        metavar="N",
        help="(granularity, window) tasks to run concurrently (thread pool). Default: 4",
    )
    parser.add_argument("--output-dir", type=Path, default=None, dest="output_dir")
    args = parser.parse_args()

    if args.all_scenarios:
        args.scenarios = list(ALL_SCENARIOS)
    elif not args.scenarios:
        parser.error("Specify at least one scenario name or use --all.")

    method = args.filtered if args.filtered else None
    filtered = args.filtered is not None

    for scenario in args.scenarios:
        is_cscas = dataset_for_scenario(scenario) == "cscas"
        grouping = (
            GroupingConfig(mode=CSCAS_PREGROUPED_METHOD)
            if is_cscas
            else GroupingConfig(window_size=args.window_size)
        )
        alerts_filename = (
            f"alerts_filtered_{method}.json" if method else "alerts_filtered.json"
        )
        alerts_path = (
            _REPO / "artifacts" / "processed-data" / scenario / alerts_filename
            if filtered and not is_cscas
            else None
        )
        cache_dir = cache_dir_for(scenario, filtered, method, args.window_size)
        results_dir = (
            args.output_dir / scenario if args.output_dir is not None else None
        )

        config = ScreeningSweepConfig(
            scenario=scenario,
            granularities=args.granularities,
            mining_settings_path=args.mining_settings,
            models=args.models,
            baseline_models=args.baseline_models,
            train_frac_within_window=args.train_frac_within_window,
            windows_per_gran=args.windows_per_gran,
            cache_dir=cache_dir,
            grouping=grouping,
            alerts_json_path=alerts_path,
            results_dir=results_dir,
            force_remine=args.force,
            n_jobs=args.n_jobs,
        )
        out_dir = run_screening_sweep_experiment(config)
        print(f"\n[{scenario}] Screening sweep results → {out_dir}")


if __name__ == "__main__":
    main()

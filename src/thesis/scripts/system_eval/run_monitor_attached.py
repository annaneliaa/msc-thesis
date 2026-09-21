"""
Monitor Attached CLI.

Runs system_eval.monitor_attached.run_monitor_attached_experiment for one
scenario: for each shortlisted symbolic config, mines a schema + Vk and
fits a model once on window 0's train split (identical setup to
run_monitor_drift.py), then walks the schema/model/Vk forward one window
at a time -- but here the monitor's decision is actually acted on: a
sustained covariate-drift signal retrains the current model in place, a
sustained calibration-drift signal remines a new schema+Vk and retrains
against it. Every horizon is timed (fast-route encode+predict cost,
workload-funnel escalation/suppression) and every retrain/remine event is
timed and diffed against the schema it replaced -- see
system_eval/monitor_attached.py's module docstring for the full design.

--psi-threshold/--cal-threshold should be the values selected by Drift
Signal EDA's threshold sweep (03_monitor_signal_drift.ipynb, Analysis 3),
not the untuned module defaults run_monitor_drift.py's logged `elevated`
column used -- pass them explicitly for a real run.

Usage:
  # The parameter grid (configs/screening_mining_settings.yaml) x granularities,
  # with tuned thresholds from the Drift Signal EDA sweep
  python src/thesis/scripts/system_eval/run_monitor_attached.py cscas \\
      --granularities 0.1 0.25 0.5 \\
      --psi-threshold 0.15 --cal-threshold 0.12

  # Override with a hand-built shortlist CSV
  python src/thesis/scripts/system_eval/run_monitor_attached.py cscas \\
      --shortlist my_shortlist.csv --psi-threshold 0.15 --cal-threshold 0.12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thesis.config import GroupingConfig
from thesis.configs import dataset_for_scenario
from thesis.system_eval.monitor_attached import run_monitor_attached_experiment
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.schemas.experiments import MonitorAttachedConfig
from thesis.scripts.system_eval._common import (
    build_shortlist_from_mining_grid,
    cache_dir_for,
)

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

_DEFAULT_MINING_SETTINGS = (
    _REPO / "src" / "thesis" / "configs" / "screening_mining_settings.yaml"
)

# CSCAS baseline's own train/test boundary (baselines/cscas_base.py's
# split_time) -- mirrors run_temporal_decay.py / run_monitor_drift.py's own
# constant of the same name/value.
_CSCAS_BASELINE_SPLIT_TIME = "2022-01-26 06:23:21+02:00"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor Attached: freeze a schema+model+Vk mined/trained on window "
        "0's train split, then walk forward acting on the monitor's decision at every "
        "horizon -- retraining or remining in place instead of only observing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scenario", help="Scenario name (e.g. cscas, fox).")
    parser.add_argument(
        "--shortlist",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Pre-built shortlist CSV (feature_set,mining_setting,granularity,model). "
            "Optional -- default is every entry in --mining-settings x --granularities "
            "x --models. Only feature_set='symbolic' rows ever produce a DynamicSchema "
            "-- every other row is skipped (a monitor needs a Vk to attach to)."
        ),
    )
    parser.add_argument(
        "--granularities",
        nargs="+",
        type=float,
        default=None,
        metavar="FRAC",
        help="Granularities to cross with the mining-settings grid. Required unless --shortlist is given.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg"],
        metavar="MODEL",
        help="Models to cross with the mining-settings grid (ignored with --shortlist). Default: logreg",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        dest="train_frac",
        metavar="FRAC",
        help="W_src (window 0) internal train/test split fraction. Default: 0.7",
    )
    parser.add_argument(
        "--source-split-mode",
        choices=["window0", "baseline_split"],
        default="window0",
        dest="source_split_mode",
        help=(
            "What the source window W_src is -- identical semantics to "
            "run_monitor_drift.py's flag of the same name. 'baseline_split' lines "
            "this run's horizons up with a monitor_drift.py/temporal_decay.py run "
            "using the same mode. CSCAS only."
        ),
    )
    parser.add_argument(
        "--source-split-time",
        default=None,
        dest="source_split_time",
        metavar="ISO8601",
        help=(
            "Train/test boundary instant for --source-split-mode baseline_split "
            f"(e.g. '{_CSCAS_BASELINE_SPLIT_TIME}'). Defaults to the CSCAS "
            "baseline's split_time when the scenario is cscas."
        ),
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "calibrated_recall"],
        default="fixed",
        dest="threshold_mode",
        help="How every decision threshold (initial fit and every retrain/remine) is chosen. Default: fixed (0.5)",
    )
    parser.add_argument(
        "--calibrated-recall-target",
        type=float,
        default=0.90,
        dest="calibrated_recall_target",
        metavar="RECALL",
        help="Recall target for --threshold-mode calibrated_recall. Default: 0.90",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.1,
        dest="psi_threshold",
        metavar="PSI",
        help="Signal-1 elevation cutoff -- pass the value Drift Signal EDA's sweep selected. Default: 0.1 (untuned)",
    )
    parser.add_argument(
        "--cal-threshold",
        type=float,
        default=0.10,
        dest="cal_threshold",
        metavar="DRIFT",
        help="Signal-2 elevation cutoff -- pass the value Drift Signal EDA's sweep selected. Default: 0.10 (untuned)",
    )
    parser.add_argument(
        "--monitor-consecutive-windows",
        type=int,
        default=3,
        dest="monitor_consecutive_windows",
        metavar="N",
        help="Consecutive elevated horizons before a soft-alerted signal hard-triggers. Default: 3",
    )
    parser.add_argument(
        "--monitor-min-samples-signal-2",
        type=int,
        default=30,
        dest="monitor_min_samples_signal_2",
        metavar="N",
        help="Minimum labeled matching rows before a compound rule's calibration drift is evaluated. Default: 30",
    )
    parser.add_argument(
        "--latency-sample-n",
        type=int,
        default=200,
        dest="latency_sample_n",
        metavar="N",
        help="Total individually-timed single-alert-group calls per config, spread "
        "across horizons, for a real p50/p95/p99 latency distribution. 0 disables it. Default: 200",
    )
    parser.add_argument(
        "--mining-settings",
        type=Path,
        default=_DEFAULT_MINING_SETTINGS,
        dest="mining_settings",
        metavar="YAML",
        help=f"Named mining-setting grid axis (same file Experiment 1 used). Default: {_DEFAULT_MINING_SETTINGS}",
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
        "--n-jobs",
        type=int,
        default=4,
        dest="n_jobs",
        metavar="N",
        help="Shortlisted configs to run concurrently (thread pool). Default: 4",
    )
    parser.add_argument("--output-dir", type=Path, default=None, dest="output_dir")
    args = parser.parse_args()

    if args.shortlist is not None:
        shortlist_path = args.shortlist
    else:
        if not args.granularities:
            parser.error("--granularities is required unless --shortlist is given")
        derived = build_shortlist_from_mining_grid(
            args.mining_settings,
            args.granularities,
            args.models,
            include_baseline=False,  # a baseline row has no DynamicSchema -- always skipped here
        )
        derived = derived[derived["feature_set"] == "symbolic"].reset_index(drop=True)
        derived_dir = (
            _REPO / "artifacts" / "experiments" / "monitor_attached" / args.scenario
        )
        derived_dir.mkdir(parents=True, exist_ok=True)
        shortlist_path = derived_dir / "_derived_shortlist.csv"
        derived.to_csv(shortlist_path, index=False)
        print(
            f"  Shortlist ({len(derived)} symbolic configs) = {args.mining_settings.name} "
            f"× granularities={args.granularities} × models={args.models} → {shortlist_path}"
        )

    scenario = args.scenario
    is_cscas = dataset_for_scenario(scenario) == "cscas"
    source_split_time = args.source_split_time
    if args.source_split_mode == "baseline_split":
        if source_split_time is None and is_cscas:
            source_split_time = _CSCAS_BASELINE_SPLIT_TIME
        if source_split_time is None:
            parser.error(
                "--source-split-mode baseline_split requires --source-split-time "
                "for a non-CSCAS scenario"
            )
    grouping = (
        GroupingConfig(mode=CSCAS_PREGROUPED_METHOD)
        if is_cscas
        else GroupingConfig(window_size=args.window_size)
    )
    method = args.filtered if args.filtered else None
    filtered = args.filtered is not None
    alerts_filename = (
        f"alerts_filtered_{method}.json" if method else "alerts_filtered.json"
    )
    alerts_path = (
        _REPO / "artifacts" / "processed-data" / scenario / alerts_filename
        if filtered and not is_cscas
        else None
    )
    cache_dir = cache_dir_for(scenario, filtered, method, args.window_size)
    results_dir = args.output_dir / scenario if args.output_dir is not None else None

    config = MonitorAttachedConfig(
        scenario=scenario,
        shortlist_path=shortlist_path,
        train_frac_within_window=args.train_frac,
        source_split_mode=args.source_split_mode,
        source_split_time=source_split_time,
        mining_settings_path=args.mining_settings,
        threshold_mode=args.threshold_mode,
        calibrated_recall_target=args.calibrated_recall_target,
        cache_dir=cache_dir,
        grouping=grouping,
        alerts_json_path=alerts_path,
        results_dir=results_dir,
        n_jobs=args.n_jobs,
        monitor_consecutive_windows=args.monitor_consecutive_windows,
        monitor_min_samples_signal_2=args.monitor_min_samples_signal_2,
        psi_threshold=args.psi_threshold,
        cal_threshold=args.cal_threshold,
        latency_sample_n=args.latency_sample_n,
    )
    out_dir = run_monitor_attached_experiment(config)
    print(f"\n[{scenario}] Monitor Attached results → {out_dir}")


if __name__ == "__main__":
    main()

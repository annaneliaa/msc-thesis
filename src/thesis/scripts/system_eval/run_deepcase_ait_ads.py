"""
Ours vs DeepCASE end to end on AIT-ADS, frozen, same label budget -- see
thesis.system_eval.deepcase_ait_ads.

DeepCASE trains once per seed on the training alerts of every pool scenario
(the within-scenario 70/30 split of the mining comparison); our system
(contrast tree + RF, and the no-mining arm, no hour_of_day) trains per
scenario on DeepCASE's per-scenario label budget. Both are scored on the same
test alerts (alert level) and test groups (group level).

Writes config.json, split.csv, tuning.csv, deepcase_budget.csv,
alert_metrics.csv, group_metrics.csv, workload.csv, ours_step_rows.csv,
bootstrap.csv, summary.txt and seed<k>/deepcase_*.parquet under
artifacts/experiments/deepcase_ait_ads/<ts>/.

Usage:
    python src/thesis/scripts/system_eval/run_deepcase_ait_ads.py
    # paper defaults for DeepCASE, no tuning:
    python src/thesis/scripts/system_eval/run_deepcase_ait_ads.py --no-tune
    # smoke test (fox only, 1 seed, 2 epochs, 5 query iterations, no tuning):
    python src/thesis/scripts/system_eval/run_deepcase_ait_ads.py --smoke
"""

from __future__ import annotations

import argparse
import atexit
import resource
import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.deepcase_ait_ads import (  # noqa: E402
    BUDGET_VARIANTS,
    EVENT_FIELDS,
    OUR_ARMS,
    AitAdsH2HConfig,
    run_deepcase_ait_ads,
)
from thesis.system_eval.deepcase_comparison import (  # noqa: E402
    TUNE_OBJECTIVES,
    DeepCaseConfig,
)
from thesis.system_eval.mining_comparison import (  # noqa: E402
    AIT_ADS_SCENARIOS,
    COMPARISON_K,
)


def _report_peak_memory() -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kb = rss // 1024 if sys.platform == "darwin" else rss
    print(f"\tMaximum resident set size (kbytes): {kb}", flush=True)


def main() -> None:
    atexit.register(_report_peak_memory)
    d = DeepCaseConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=list(AIT_ADS_SCENARIOS),
        help="DeepCASE training pool",
    )
    p.add_argument(
        "--eval-scenarios", nargs="+", default=None, help="default: --scenarios"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(d.seeds))
    p.add_argument("--event-field", choices=EVENT_FIELDS, default="signature")
    p.add_argument("--grouping", default="cscas_grouping")
    dc = p.add_argument_group("DeepCASE")
    dc.add_argument("--epochs", type=int, default=d.epochs)
    dc.add_argument("--batch-size", type=int, default=d.batch_size)
    dc.add_argument("--query-iterations", type=int, default=d.query_iterations)
    dc.add_argument("--samples-per-cluster", type=int, default=d.samples_per_cluster)
    dc.add_argument("--no-tune", action="store_true")
    dc.add_argument("--context-length", type=int, default=d.context_length)
    dc.add_argument("--threshold", type=float, default=d.threshold)
    dc.add_argument("--eps", type=float, default=d.eps)
    dc.add_argument(
        "--tune-lengths", type=int, nargs="+", default=list(d.tune_context_lengths)
    )
    dc.add_argument("--tune-epochs", type=int, default=None)
    dc.add_argument(
        "--tune-objective", choices=TUNE_OBJECTIVES, default=d.tune_objective
    )
    dc.add_argument("--device", default=None)
    ours = p.add_argument_group("ours")
    ours.add_argument("--arms", nargs="+", choices=OUR_ARMS, default=list(OUR_ARMS))
    ours.add_argument(
        "--budget-variants",
        nargs="+",
        choices=BUDGET_VARIANTS,
        default=list(BUDGET_VARIANTS),
    )
    ours.add_argument("--k", type=int, default=COMPARISON_K)
    ours.add_argument("--settings-variant", default="default")
    ours.add_argument(
        "--with-hour",
        action="store_true",
        help="keep hour_of_day in our base features (shortcut; off by default)",
    )
    ours.add_argument("--mining-cache-dir", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="fox only, 1 seed, 2 epochs, 5 query iterations, no tuning, 200 boots",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    a = p.parse_args()

    dcfg = DeepCaseConfig(
        seeds=tuple(a.seeds),
        epochs=a.epochs,
        batch_size=a.batch_size,
        query_iterations=a.query_iterations,
        samples_per_cluster=a.samples_per_cluster,
        tune=not a.no_tune,
        context_length=a.context_length,
        threshold=a.threshold,
        eps=a.eps,
        tune_context_lengths=tuple(a.tune_lengths),
        tune_epochs=a.tune_epochs,
        tune_objective=a.tune_objective,
        device=a.device,
    )
    cfg = AitAdsH2HConfig(
        scenarios=tuple(a.scenarios),
        eval_scenarios=tuple(a.eval_scenarios) if a.eval_scenarios else None,
        seeds=tuple(a.seeds),
        grouping=a.grouping,
        event_field=a.event_field,
        deepcase=dcfg,
        arms=tuple(a.arms),
        budget_variants=tuple(a.budget_variants),
        k=a.k,
        settings_variant=a.settings_variant,
        no_hour=not a.with_hour,
        mining_cache_dir=a.mining_cache_dir,
        n_boot=a.n_boot,
        out_dir=a.out_dir,
    )
    if a.smoke:
        cfg = replace(
            cfg,
            scenarios=("fox",),
            eval_scenarios=None,
            seeds=(0,),
            n_boot=200,
            deepcase=replace(dcfg, epochs=2, query_iterations=5, tune=False),
        )
    run_deepcase_ait_ads(cfg)


if __name__ == "__main__":
    main()

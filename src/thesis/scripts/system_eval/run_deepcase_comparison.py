"""
DeepCASE on CSCAS, frozen (DeepCASE comparison plan, step 5.2) -- see
thesis.system_eval.deepcase_comparison.

Tunes context length x tau_confidence x eps on W0 (paper App. D), trains the
ContextBuilder on W0 per seed, labels clusters with the simulated analyst
(10 samples per cluster + every outlier = B0) and scores the 54-day test
period per horizon window.

Writes config.json, tuning.csv, windows.csv, horizon_metrics.csv, budget.csv,
summary.txt and seed<k>/{test_predictions,analyst_items}.parquet +
train_clusters.csv under artifacts/experiments/deepcase_comparison/cscas/<ts>/.

Usage:
    python src/thesis/scripts/system_eval/run_deepcase_comparison.py
    # paper defaults, no tuning:
    python src/thesis/scripts/system_eval/run_deepcase_comparison.py --no-tune
    # smoke test:
    python src/thesis/scripts/system_eval/run_deepcase_comparison.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.deepcase_comparison import (  # noqa: E402
    TUNE_OBJECTIVES,
    DeepCaseConfig,
    run_deepcase_comparison,
)


def main() -> None:
    d = DeepCaseConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(d.seeds))
    p.add_argument("--gran", type=float, default=d.granularity)
    p.add_argument("--missing-host", choices=["none", "extip"], default=d.missing_host)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--query-iterations", type=int, default=d.query_iterations)
    p.add_argument("--samples-per-cluster", type=int, default=d.samples_per_cluster)
    p.add_argument(
        "--no-tune",
        action="store_true",
        help="skip W0 tuning; use --context-length/--threshold/--eps",
    )
    p.add_argument("--context-length", type=int, default=d.context_length)
    p.add_argument("--threshold", type=float, default=d.threshold)
    p.add_argument("--eps", type=float, default=d.eps)
    p.add_argument(
        "--tune-lengths", type=int, nargs="+", default=list(d.tune_context_lengths)
    )
    p.add_argument("--tune-epochs", type=int, default=None)
    p.add_argument(
        "--tune-objective", choices=TUNE_OBJECTIVES, default=d.tune_objective
    )
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-test-rows", type=int, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="20k/20k rows, 2 epochs, 5 query iterations, 1 seed, small grid",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    cfg = DeepCaseConfig(
        seeds=tuple(a.seeds),
        granularity=a.gran,
        missing_host=a.missing_host,
        epochs=a.epochs,
        query_iterations=a.query_iterations,
        samples_per_cluster=a.samples_per_cluster,
        tune=not a.no_tune,
        context_length=a.context_length,
        threshold=a.threshold,
        eps=a.eps,
        tune_context_lengths=tuple(a.tune_lengths),
        tune_epochs=a.tune_epochs,
        tune_objective=a.tune_objective,
        max_train_rows=a.max_train_rows,
        max_test_rows=a.max_test_rows,
        device=a.device,
        out_dir=a.out_dir,
    )
    if a.smoke:
        cfg.seeds = (0,)
        cfg.epochs = 2
        cfg.query_iterations = 5
        cfg.max_train_rows = a.max_train_rows or 20_000
        cfg.max_test_rows = a.max_test_rows or 20_000
        cfg.tune_context_lengths = (5, 10)
        cfg.tune_thresholds = (0.2, 0.5)
        cfg.tune_eps = (0.1, 0.5)
    run_deepcase_comparison(cfg)


if __name__ == "__main__":
    main()

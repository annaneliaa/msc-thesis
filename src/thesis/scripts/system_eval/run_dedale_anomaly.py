"""
Anomaly-detection arm on DEDALE Sigma (thesis.system_eval.dedale_anomaly), on
the dataset's recommended split: weeks 1-2 (no attack) train, weeks 3-4 test.
IsolationForest (per seed) and OneClassSVM on base / base + items /
base + items + novelty-count features; threshold at a quantile of the
training scores (no labels needed).

Writes per_run_results.csv, summary.csv/.txt, windows.csv, config.json,
test_scores.csv.gz, bootstrap.csv/.txt and test_attacks.csv under
artifacts/experiments/anomaly_dedale[_no_host][_no_hour]/<timestamp>/.

Usage:
    python src/thesis/scripts/system_eval/run_dedale_anomaly.py
    python src/thesis/scripts/system_eval/run_dedale_anomaly.py --no-host --no-hour
    python src/thesis/scripts/system_eval/run_dedale_anomaly.py \
        --grouping time_delta --grouping-seconds 60 --quantile 0.995
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.dedale_anomaly import (  # noqa: E402
    FEATURE_SETS,
    MODELS,
    AnomalyConfig,
    run_dedale_anomaly,
)
from thesis.system_eval.dedale_comparison import (  # noqa: E402
    DEDALE_SIGMA_PATH,
    GROUPINGS,
    DedaleConfig,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=FEATURE_SETS
    )
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="threshold = this quantile of the training scores",
    )
    p.add_argument("--grouping", choices=sorted(GROUPINGS), default="cscas_grouping")
    p.add_argument("--grouping-seconds", type=float, default=None)
    p.add_argument(
        "--cut",
        default="weeks",
        help="weeks (06/01, the recommended split; default) or another "
        "cut accepted by dedale_comparison.split_groups",
    )
    p.add_argument("--no-host", action="store_true")
    p.add_argument("--no-hour", action="store_true")
    p.add_argument("--mixed-as-attack", action="store_true")
    p.add_argument("--data", type=Path, default=DEDALE_SIGMA_PATH)
    p.add_argument("--output-dir", type=Path, default=None)
    a = p.parse_args()

    run_dedale_anomaly(
        AnomalyConfig(
            seeds=tuple(a.seeds),
            feature_sets=tuple(a.feature_sets),
            models=tuple(a.models),
            quantile=a.quantile,
            no_host=a.no_host,
            no_hour=a.no_hour,
            mixed_as_attack=a.mixed_as_attack,
            out_dir=a.output_dir,
            dedale=DedaleConfig(
                grouping=a.grouping,
                grouping_seconds=a.grouping_seconds,
                cut=a.cut,
                path=a.data,
            ),
        )
    )


if __name__ == "__main__":
    main()

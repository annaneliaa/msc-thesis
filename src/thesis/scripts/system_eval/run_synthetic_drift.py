"""
Synthetic drift test CLI (CSCAS): positive control for the monitor signals.
See thesis.system_eval.synthetic_drift for the injections.

Usage:
  python src/thesis/scripts/system_eval/run_synthetic_drift.py
  python src/thesis/scripts/system_eval/run_synthetic_drift.py --models xgboost --kinds new_values --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.config import GroupingConfig  # noqa: E402
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD  # noqa: E402
from thesis.scripts.system_eval._common import cache_dir_for  # noqa: E402
from thesis.system_eval.synthetic_drift import (  # noqa: E402
    DEFAULT_FRACTIONS,
    Injection,
    SyntheticDriftConfig,
    default_injections,
    run_synthetic_drift_experiment,
)

_CSCAS_BASELINE_SPLIT_TIME = "2022-01-26 06:23:21+02:00"
_DEFAULT_MINING_SETTINGS = (
    _REPO / "src" / "thesis" / "configs" / "monitor_eda_mining_setting.yaml"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mining-settings", type=Path, default=_DEFAULT_MINING_SETTINGS)
    p.add_argument("--mining-setting", default="gr3_md1_mda4_rounds2")
    p.add_argument("--models", nargs="+", default=["xgboost", "rf"])
    p.add_argument("--granularity", type=float, default=0.05)
    p.add_argument("--inject-from-fraction", type=float, default=0.5)
    p.add_argument(
        "--kinds",
        nargs="+",
        default=None,
        choices=list(DEFAULT_FRACTIONS),
        help="Injection kinds to run (default: all). The clean run is always included.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Only the strongest fraction per kind (quick check).",
    )
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    injections = default_injections()
    if args.kinds:
        injections = [i for i in injections if i.kind == "none" or i.kind in args.kinds]
    if args.smoke:
        injections = [Injection("none", 0.0)] + [
            Injection(k, max(DEFAULT_FRACTIONS[k]))
            for k in DEFAULT_FRACTIONS
            if not args.kinds or k in args.kinds
        ]

    config = SyntheticDriftConfig(
        scenario="cscas",
        mining_setting=args.mining_setting,
        mining_settings_path=args.mining_settings,
        models=tuple(args.models),
        granularity=args.granularity,
        inject_from_fraction=args.inject_from_fraction,
        injections=injections,
        source_split_mode="baseline_split",
        source_split_time=_CSCAS_BASELINE_SPLIT_TIME,
        cache_dir=cache_dir_for("cscas", False, None, 2),
        grouping=GroupingConfig(mode=CSCAS_PREGROUPED_METHOD),
        results_dir=args.output_dir,
        random_seed=args.seed,
        n_jobs=args.n_jobs,
    )
    print(f"  {len(injections)} injections × {len(args.models)} models")
    out_dir = run_synthetic_drift_experiment(config)
    print(f"\n[cscas] Synthetic drift results → {out_dir}")


if __name__ == "__main__":
    main()

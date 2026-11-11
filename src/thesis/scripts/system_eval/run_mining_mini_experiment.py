"""
Mining comparison Phase 4: mini experiment per method on CSCAS and AIT-ADS.

Mines each method on one window, applies the precision gate (g = 2) and the
budget K, encodes the next window and checks the features are usable (see
thesis.system_eval.mining_mini_experiment). Writes summary.csv (one row per
method x dataset), features.csv (every budgeted feature with train/test
firing, attack rates, single-signature flag) and config.json under
artifacts/experiments/mining_mini/<timestamp>/.
Analysis: notebooks/paper/04_mining_mini_experiment.ipynb.

Usage:
    python src/thesis/scripts/system_eval/run_mining_mini_experiment.py
    python src/thesis/scripts/system_eval/run_mining_mini_experiment.py --datasets ait_ads --scenario fox
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.mining_mini_experiment import (  # noqa: E402
    COMPARISON_TOKEN_ARM,
    METHODS,
    TOKEN_ARMS,
    MiniConfig,
    ait_ads_window,
    cscas_window,
    default_out_dir,
    run_mini,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datasets", nargs="+", default=["cscas", "ait_ads"])
    p.add_argument("--scenario", default="fox", help="AIT-ADS scenario")
    p.add_argument(
        "--token-arm",
        default=COMPARISON_TOKEN_ARM,
        choices=TOKEN_ARMS,
        help="AIT-ADS token arm, for every method",
    )
    p.add_argument("--train-frac", type=float, default=0.7, help="AIT-ADS split")
    p.add_argument("--gran", type=float, default=0.05, help="CSCAS window size")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--k", type=int, default=MiniConfig.k)
    p.add_argument("--pooled-budget", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    a = p.parse_args()

    config = MiniConfig(
        k=a.k,
        per_class_budget=not a.pooled_budget,
        seed=a.seed,
        methods=tuple(a.methods),
    )
    windows = []
    if "cscas" in a.datasets:
        windows.append(cscas_window(a.gran))
    if "ait_ads" in a.datasets:
        windows.append(
            ait_ads_window(
                a.scenario,
                train_frac=a.train_frac,
                lookback_minutes=config.lookback_minutes,
                token_arm=a.token_arm,
            )
        )
    out = a.output_dir or default_out_dir()
    summary, _ = run_mini(windows, config, out)
    cols = [
        c
        for c in (
            "dataset",
            "method",
            "status",
            "n_raw",
            "n_gated",
            "n_budget",
            "live_share_test",
            "single_signature_share",
            "test_attack_coverage",
            "mine_seconds",
        )
        if c in summary.columns
    ]
    print(summary[cols].to_string(index=False))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

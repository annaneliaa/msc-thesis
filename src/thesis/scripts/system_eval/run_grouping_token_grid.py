"""
Mining comparison Phase 4b: grouping x token granularity on AIT-ADS.

For one scenario, runs the Phase 4 mini experiment (mine on the first
train_frac of groups, precision gate with benign <= pi, budget K, encode the
rest) for every grouping x token arm:

  groupings: cscas_grouping (reference), fixed_window, time_delta, alertbert
  token arms: T0 full codes | T1 + short_src, short_cat | T2 + short_family,
              sig_family | T3 coarse only (no short / sig)

Atemporal methods (eclat, contrast_tree, ripper) run on every arm; temporal
methods (prefixspan, mctsextent) mine look-back sequences of full codes and
run on T0 only. All groupings are cut at the same moment (the
cscas_grouping 70% point). Eclat's count floor is relative to the mining window's
attack count. Reports per cell: generalised restatement share (short code
as signature identity), unseen-code attack coverage, live share, coverage,
base rate. Writes summary.csv / features.csv / config.json under
artifacts/experiments/mining_grouping_tokens/<timestamp>/.
Analysis: notebooks/paper/05_grouping_token_granularity.ipynb.

Usage:
    python src/thesis/scripts/system_eval/run_grouping_token_grid.py
    python src/thesis/scripts/system_eval/run_grouping_token_grid.py --groupings fixed_window --arms T0 T3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.mining.common.attribute_features import (  # noqa: E402
    default_leaky_attribute_fields,
)
from thesis.paths import ROOT  # noqa: E402
from thesis.system_eval.mining_mini_experiment import (  # noqa: E402
    METHODS,
    TOKEN_ARMS,
    MiniConfig,
    ait_ads_window,
    lookback_attacher,
    reference_split_ts,
    run_mini,
)

GROUPINGS = ("cscas_grouping", "fixed_window", "time_delta", "alertbert")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scenario", default="fox")
    p.add_argument("--groupings", nargs="+", default=list(GROUPINGS))
    p.add_argument("--arms", nargs="+", default=list(TOKEN_ARMS), choices=TOKEN_ARMS)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--k", type=int, default=MiniConfig.k)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    a = p.parse_args()

    config = MiniConfig(
        k=a.k,
        seed=a.seed,
        methods=tuple(a.methods),
        eclat_min_support_count="relative",
    )
    # One split moment for every grouping: the cscas_grouping 70% point.
    split_ts = reference_split_ts(a.scenario, train_frac=a.train_frac)
    print(f"Common split: ts {split_ts}")
    windows = []
    for grouping in a.groupings:
        # Load + look-back once per grouping; arms only change the item fields.
        base = ait_ads_window(
            a.scenario,
            method=grouping,
            train_frac=a.train_frac,
            lookback_minutes=config.lookback_minutes,
            split_ts=split_ts,
        )
        for arm in a.arms:
            include, exclude = TOKEN_ARMS[arm]
            windows.append(
                replace(
                    base,
                    name=base.name.replace(" T0 ", f" {arm} "),
                    token_arm=arm,
                    include_fields=set(include),
                    exclude_fields=default_leaky_attribute_fields(a.scenario) | exclude,
                    attach_steps=lookback_attacher(
                        a.scenario, arm, config.lookback_minutes
                    ),
                )
            )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = (
        a.output_dir
        or ROOT / "artifacts" / "experiments" / "mining_grouping_tokens" / ts
    )
    summary, _ = run_mini(windows, config, out)
    cols = [
        "grouping",
        "token_arm",
        "method",
        "status",
        "n_train_attack",
        "n_budget",
        "live_share_test",
        "single_signature_share",
        "test_attack_coverage",
        "unseen_attack_coverage",
    ]
    print(summary[[c for c in cols if c in summary]].to_string(index=False))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

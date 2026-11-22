"""
Mining comparison on DEDALE Sigma (thesis.system_eval.dedale_comparison):
Base(5) alone vs. Base(5) + each method's features, under the same item
table, gate, dedupes, budget K and class-weighted RF as the AIT-ADS
comparison. One step: a chronological cut (default 2025-01-07, infection
stage in training, lateral movement + collection in test).

Data: data/dedale/dedale_sigma.jsonl (CATS repo, datasets/dedale_sigma.zip).

Writes per_step_results.csv, features.csv, windows.csv, config.json,
summary.csv, helps_hurts.csv, test_scores.csv.gz and bootstrap.csv/.txt under
artifacts/experiments/mining_comparison_dedale/<timestamp>/.

Usage:
    python src/thesis/scripts/system_eval/run_dedale_comparison.py --seeds 0
    python src/thesis/scripts/system_eval/run_dedale_comparison.py \
        --grouping time_delta --grouping-seconds 60
    python src/thesis/scripts/system_eval/run_dedale_comparison.py --groups-only
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd  # noqa: E402

from thesis.system_eval.dedale_comparison import (  # noqa: E402
    DEDALE_SIGMA_PATH,
    GROUPINGS,
    DedaleConfig,
    build_dedale_groups,
    flag_all_f1,
    run_dedale_comparison,
    split_groups,
)
from thesis.system_eval.mining_comparison import (  # noqa: E402
    AIT_ADS_METHODS,
    COMPARISON_K,
    ComparisonConfig,
)
from thesis.system_eval.mining_mini_experiment import (  # noqa: E402
    COMPARISON_TOKEN_ARM,
    TOKEN_ARMS,
    MiniConfig,
    labelled,
)


def groups_report(dconfig: DedaleConfig) -> None:
    groups = build_dedale_groups(dconfig)
    counts = pd.Series([g.group_label for g in groups]).value_counts()
    print(
        f"{dconfig.grouping} ({dconfig.grouping_seconds or 'default'} s): "
        f"{len(groups)} groups, {counts.to_dict()}"
    )
    tr, te, desc = split_groups(labelled(groups), dconfig.cut)
    n = lambda gs: sum(g.group_label == "attack" for g in gs)  # noqa: E731
    print(
        f"cut {desc}: train {len(tr)} ({n(tr)} attack), "
        f"test {len(te)} ({n(te)} attack), flag-all test F1 {flag_all_f1(te):.4f}"
    )
    for g in tr + te:
        if g.group_label == "attack":
            side = "train" if g in tr else "test"
            print(
                f"  {side:5} {pd.Timestamp(g.start_ts, unit='s')} n={g.n_alerts:2d} "
                f"{sorted(i for i in g.raw_items if not i.startswith('sig:'))}"
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--methods", nargs="+", default=list(AIT_ADS_METHODS), choices=AIT_ADS_METHODS
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--k", type=int, default=COMPARISON_K, help="feature budget K")
    p.add_argument("--grouping", choices=sorted(GROUPINGS), default="cscas_grouping")
    p.add_argument(
        "--grouping-seconds",
        type=float,
        default=None,
        help="time_delta gap / fixed_window size (default: repo default)",
    )
    p.add_argument(
        "--cut",
        default="attack_week",
        help="attack_week (07/01), weeks (06/01), 70_30, or an ISO time",
    )
    p.add_argument(
        "--token-arm", default=COMPARISON_TOKEN_ARM, choices=sorted(TOKEN_ARMS)
    )
    p.add_argument(
        "--eclat-min-count",
        default="relative",
        help="Eclat's absolute count floor: an int, or 'relative' "
        "(max(3, 5%% of training attacks); the AIT-ADS default 10 "
        "is above DEDALE's training attack-group count)",
    )
    p.add_argument(
        "--no-host",
        action="store_true",
        help="leave the host field out of the item table",
    )
    p.add_argument(
        "--no-hour",
        action="store_true",
        help="leave hour_of_day out of the base schema",
    )
    p.add_argument("--mixed-as-attack", action="store_true")
    p.add_argument(
        "--mine-once",
        action="store_true",
        help="mine RIPPER / MCTSExtent once instead of once per seed",
    )
    p.add_argument("--data", type=Path, default=DEDALE_SIGMA_PATH)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--groups-only",
        action="store_true",
        help="print the group counts and the split, run nothing",
    )
    a = p.parse_args()

    dconfig = DedaleConfig(
        grouping=a.grouping, grouping_seconds=a.grouping_seconds, cut=a.cut, path=a.data
    )
    if a.groups_only:
        groups_report(dconfig)
        return
    eclat_count = (
        a.eclat_min_count if a.eclat_min_count == "relative" else int(a.eclat_min_count)
    )
    experiment = "mining_comparison_dedale"
    if a.no_host:
        experiment += "_no_host"
    if a.no_hour:
        experiment += "_no_hour"
    config = ComparisonConfig(
        dataset="dedale",
        seeds=tuple(a.seeds),
        methods=tuple(a.methods),
        mining=replace(MiniConfig(), k=a.k, eclat_min_support_count=eclat_count),
        reseed_stochastic_miners=not a.mine_once,
        extra_exclude_fields=frozenset({"host"}) if a.no_host else frozenset(),
        base_exclude_features=frozenset({"hour_of_day"}) if a.no_hour else frozenset(),
        out_dir=a.output_dir,
        experiment=experiment,
        grouping=a.grouping,
        token_arm=a.token_arm,
        save_test_scores=True,
        mixed_as_attack=a.mixed_as_attack,
    )
    run_dedale_comparison(config, dconfig)


if __name__ == "__main__":
    main()

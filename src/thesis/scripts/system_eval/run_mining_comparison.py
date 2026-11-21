"""
Mining comparison (paper plan 7.6): Base(5) alone vs. Base(5) + each
method's features, every method under the same item table, precision gate,
near-duplicate dedupe, budget K and class-weighted RF
(see thesis.system_eval.mining_comparison).

--dataset cscas (default): walk-forward over CSCAS, atemporal methods.
--dataset ait_ads: leave-one-scenario-out over the AIT-ADS scenarios
(cscas_grouping, token arm T2), all five methods by default. --ait-protocol
phase runs the within-scenario attack-phase walk instead (reference only).

Writes per_step_results.csv, features.csv, windows.csv, config.json,
summary.csv, helps_hurts.csv, stability.csv and stats.json (Friedman +
Nemenyi) under artifacts/experiments/mining_comparison/<timestamp>/.

Usage:
    python src/thesis/scripts/system_eval/run_mining_comparison.py
    python src/thesis/scripts/system_eval/run_mining_comparison.py --max-steps 2 --seeds 0
    python src/thesis/scripts/system_eval/run_mining_comparison.py --dataset ait_ads
    python src/thesis/scripts/system_eval/run_mining_comparison.py --dataset ait_ads \
        --test-scenarios fox --seeds 0
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

from thesis.mining.common.attribute_features import (  # noqa: E402
    CSCAS_SIGNATURE_DERIVED_FIELDS,
)
from thesis.system_eval.mining_comparison import (  # noqa: E402
    AIT_ADS_METHODS,
    AIT_ADS_SCENARIOS,
    ATEMPORAL_METHODS,
    COMPARISON_K,
    PHASE_GAP_SECONDS,
    ComparisonConfig,
    run_comparison,
)
from thesis.system_eval.mining_settings_sweep import (  # noqa: E402
    VARIANTS,
    apply_variant,
)
from thesis.system_eval.mining_mini_experiment import (  # noqa: E402
    COMPARISON_TOKEN_ARM,
    TOKEN_ARMS,
    MiniConfig,
)


def _report_peak_memory() -> None:
    """Print this process's peak RSS in GNU time's format, so the launcher's
    peak_memory.txt works where /usr/bin/time -v isn't installed (the DGX
    container). ru_maxrss is kB on Linux, bytes on macOS."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kb = rss // 1024 if sys.platform == "darwin" else rss
    print(f"\tMaximum resident set size (kbytes): {kb}", flush=True)


def main() -> None:
    atexit.register(_report_peak_memory)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", choices=("cscas", "ait_ads"), default="cscas")
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=AIT_ADS_METHODS,
        help="default: eclat contrast_tree ripper on CSCAS, all five on AIT-ADS",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--k", type=int, default=COMPARISON_K, help="feature budget K")
    p.add_argument(
        "--pooled-budget", action="store_true", help="one pool instead of K/2 per class"
    )
    p.add_argument("--gran", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--no-signature-items",
        action="store_true",
        help="leave the signature-derived fields out of the item table (Check 3)",
    )
    p.add_argument(
        "--mine-once",
        action="store_true",
        help="mine RIPPER once per step instead of once per seed",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--stage",
        choices=("all", "mine", "train", "prepare"),
        default="all",
        help="all: mine what isn't cached + train; mine: only mine and cache; "
        "train: only train from the cache; prepare: build the AIT-ADS "
        "look-back cache",
    )
    p.add_argument(
        "--mining-cache-dir",
        type=Path,
        default=None,
        help="cache of mined candidates (needed for --stage mine/train); "
        f"e.g. {_REPO / 'artifacts/cache/mining_comparison'}",
    )
    p.add_argument(
        "--settings-variant",
        choices=VARIANTS,
        default="default",
        help="mining settings grid point (settings-sensitivity check): every "
        "method's main knob at its low / default / high value",
    )
    p.add_argument(
        "--mixed-as-attack",
        action="store_true",
        help="AIT-ADS: label groups mixing benign and attack alerts as attack "
        "instead of dropping them (label-purity runs)",
    )
    p.add_argument(
        "--no-lookback-cache",
        action="store_true",
        help="AIT-ADS: build look-back steps from the raw alerts every time",
    )
    ait = p.add_argument_group("AIT-ADS")
    ait.add_argument(
        "--ait-protocol",
        choices=("split", "loso", "phase"),
        default="split",
        help="within-scenario chronological 70/30 split (default, the thesis "
        "setup), leave-one-scenario-out, or the within-scenario phase walk",
    )
    ait.add_argument(
        "--scenarios",
        nargs="+",
        default=list(AIT_ADS_SCENARIOS),
        metavar="S",
        help="scenario pool (LOSO trains on all but the held-out one)",
    )
    ait.add_argument(
        "--test-scenarios",
        nargs="+",
        default=None,
        metavar="S",
        help="split: only run these scenarios; LOSO: only the folds holding "
        "out these (default: all)",
    )
    ait.add_argument("--grouping", default="cscas_grouping")
    ait.add_argument(
        "--token-arm", default=COMPARISON_TOKEN_ARM, choices=sorted(TOKEN_ARMS)
    )
    ait.add_argument("--phase-gap-seconds", type=int, default=PHASE_GAP_SECONDS)
    a = p.parse_args()

    methods = a.methods or list(
        ATEMPORAL_METHODS if a.dataset == "cscas" else AIT_ADS_METHODS
    )
    if a.dataset == "cscas" and set(methods) - set(ATEMPORAL_METHODS):
        p.error(f"CSCAS runs {list(ATEMPORAL_METHODS)} only")
    if a.dataset == "ait_ads" and a.no_signature_items:
        p.error("--no-signature-items is the CSCAS Check 3 condition only")
    experiment = "mining_comparison"
    if a.dataset == "ait_ads":
        # LOSO keeps the plain name its earlier runs used.
        experiment += (
            "_ait_ads"
            + {"split": "_split", "loso": "", "phase": "_phase"}[a.ait_protocol]
        )
    if a.no_signature_items:
        experiment += "_no_sig"
    if a.settings_variant != "default":
        experiment += f"_settings_{a.settings_variant}"

    config = ComparisonConfig(
        dataset=a.dataset,
        gran=a.gran,
        seeds=tuple(a.seeds),
        methods=tuple(methods),
        mining=apply_variant(
            replace(MiniConfig(), k=a.k, per_class_budget=not a.pooled_budget),
            a.settings_variant,
        ),
        reseed_stochastic_miners=not a.mine_once,
        extra_exclude_fields=(
            frozenset(CSCAS_SIGNATURE_DERIVED_FIELDS)
            if a.no_signature_items
            else frozenset()
        ),
        max_steps=a.max_steps,
        out_dir=a.output_dir,
        experiment=experiment,
        scenarios=tuple(a.scenarios),
        grouping=a.grouping,
        token_arm=a.token_arm,
        phase_gap_seconds=a.phase_gap_seconds,
        ait_ads_protocol=a.ait_protocol,
        # AIT-ADS has 5 steps: test with a bootstrap over test groups.
        save_test_scores=a.dataset == "ait_ads",
        stage=a.stage,
        mining_cache_dir=a.mining_cache_dir,
        use_lookback_cache=not a.no_lookback_cache,
        mixed_as_attack=a.mixed_as_attack,
        test_scenarios=tuple(a.test_scenarios) if a.test_scenarios else None,
    )
    run_comparison(config)


if __name__ == "__main__":
    main()

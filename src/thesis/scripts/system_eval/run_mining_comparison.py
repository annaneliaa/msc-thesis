"""
Mining comparison (paper plan 7.6): walk-forward over CSCAS, Base(5) alone
vs. Base(5) + each method's features, every method under the same item
table, precision gate, near-duplicate dedupe, budget K and class-weighted RF
(see thesis.system_eval.mining_comparison).

Writes per_step_results.csv, features.csv, windows.csv, config.json,
summary.csv, helps_hurts.csv, stability.csv and stats.json (Friedman +
Nemenyi) under artifacts/experiments/mining_comparison/<timestamp>/.

Usage:
    python src/thesis/scripts/system_eval/run_mining_comparison.py
    python src/thesis/scripts/system_eval/run_mining_comparison.py --max-steps 2 --seeds 0
"""

from __future__ import annotations

import argparse
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
    ATEMPORAL_METHODS,
    ComparisonConfig,
    run_comparison,
)
from thesis.system_eval.mining_mini_experiment import MiniConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(ATEMPORAL_METHODS),
        choices=ATEMPORAL_METHODS,
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--k", type=int, default=MiniConfig.k, help="feature budget K")
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
    a = p.parse_args()

    config = ComparisonConfig(
        gran=a.gran,
        seeds=tuple(a.seeds),
        methods=tuple(a.methods),
        mining=replace(MiniConfig(), k=a.k, per_class_budget=not a.pooled_budget),
        reseed_stochastic_miners=not a.mine_once,
        extra_exclude_fields=(
            frozenset(CSCAS_SIGNATURE_DERIVED_FIELDS)
            if a.no_signature_items
            else frozenset()
        ),
        max_steps=a.max_steps,
        out_dir=a.output_dir,
        experiment=(
            "mining_comparison_no_sig" if a.no_signature_items else "mining_comparison"
        ),
    )
    run_comparison(config)


if __name__ == "__main__":
    main()

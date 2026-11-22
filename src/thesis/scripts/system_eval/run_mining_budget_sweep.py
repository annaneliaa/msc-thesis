"""
Budget robustness sweep for the mining comparison: train every method at
K in {10, 25, 50, 60} (per class) and K = 60 pooled, from the same cached
mining output, and check whether the ranking of methods holds
(see thesis.system_eval.mining_budget_sweep).

CSCAS: mines once into the mining cache (the main CSCAS run predates the
cache), then trains each variant. AIT-ADS: the full comparison run already
cached every fold's candidates, so the mine stage is a quick no-op.

Usage:
    python src/thesis/scripts/system_eval/run_mining_budget_sweep.py --dataset cscas
    python src/thesis/scripts/system_eval/run_mining_budget_sweep.py --dataset ait_ads
    # AIT-ADS main protocol (without hour_of_day in the base features)
    python src/thesis/scripts/system_eval/run_mining_budget_sweep.py --dataset ait_ads --no-hour
    # smoke test
    python src/thesis/scripts/system_eval/run_mining_budget_sweep.py --dataset cscas \\
        --max-steps 2 --seeds 0 --ks 10 60 --pooled-at
    # re-summarise finished variant runs
    python src/thesis/scripts/system_eval/run_mining_budget_sweep.py \\
        --summarize-only artifacts/experiments/mining_budget_sweep/cscas/<ts>
"""

from __future__ import annotations

import argparse
import atexit
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.mining_budget_sweep import (  # noqa: E402
    DEFAULT_KS,
    budget_variants,
    default_out_root,
    run_budget_sweep,
    write_budget_summary,
)
from thesis.system_eval.mining_comparison import (  # noqa: E402
    AIT_ADS_METHODS,
    AIT_ADS_SCENARIOS,
    ATEMPORAL_METHODS,
    COMPARISON_K,
    ComparisonConfig,
)
from thesis.system_eval.mining_mini_experiment import MiniConfig  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from run_mining_comparison import _report_peak_memory  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", choices=("cscas", "ait_ads"), default="cscas")
    p.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    p.add_argument(
        "--pooled-at",
        nargs="*",
        type=int,
        default=[COMPARISON_K],
        help="K values to also run with one pooled budget (none: no pooled run)",
    )
    p.add_argument("--reference", default=f"k{COMPARISON_K}")
    p.add_argument("--methods", nargs="+", default=None, choices=AIT_ADS_METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--scenarios", nargs="+", default=list(AIT_ADS_SCENARIOS))
    p.add_argument(
        "--mining-cache-dir",
        type=Path,
        default=_REPO / "artifacts" / "cache" / "mining_comparison",
    )
    p.add_argument(
        "--skip-mine",
        action="store_true",
        help="don't run the mine stage (everything must already be cached)",
    )
    p.add_argument(
        "--no-hour",
        action="store_true",
        help="AIT-ADS: drop hour_of_day from the base features (the main protocol)",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--summarize-only",
        type=Path,
        default=None,
        metavar="DIR",
        help="only rebuild the budget summary from DIR's finished variant runs",
    )
    a = p.parse_args()
    atexit.register(_report_peak_memory)

    if a.summarize_only:
        dirs = {
            d.name: d
            for d in sorted(a.summarize_only.iterdir())
            if (d / "per_step_results.csv").exists()
        }
        write_budget_summary(a.summarize_only, dirs, a.reference)
        return

    methods = a.methods or list(
        ATEMPORAL_METHODS if a.dataset == "cscas" else AIT_ADS_METHODS
    )
    if a.dataset == "cscas" and set(methods) - set(ATEMPORAL_METHODS):
        p.error(f"CSCAS runs {list(ATEMPORAL_METHODS)} only")
    if a.dataset == "cscas" and a.no_hour:
        p.error("--no-hour is an AIT-ADS option")
    tag = a.dataset + ("_no_hour" if a.no_hour else "")
    variants = budget_variants(tuple(a.ks), tuple(a.pooled_at))
    if a.reference not in {v.label for v in variants}:
        p.error(f"reference {a.reference} is not one of the variants")
    base = ComparisonConfig(
        dataset=a.dataset,
        seeds=tuple(a.seeds),
        methods=tuple(methods),
        mining=MiniConfig(k=COMPARISON_K),
        max_steps=a.max_steps,
        scenarios=tuple(a.scenarios),
        mining_cache_dir=a.mining_cache_dir,
        experiment=f"mining_budget_sweep_{tag}",
        base_exclude_features=(
            frozenset({"hour_of_day"}) if a.no_hour else frozenset()
        ),
        # AIT-ADS has 5 steps: each variant gets the test-group bootstrap.
        save_test_scores=a.dataset == "ait_ads",
    )
    out_root = a.output_dir or default_out_root(tag)
    print(f"Output → {out_root} | variants: {[v.label for v in variants]}")
    run_budget_sweep(
        base, variants, out_root, mine=not a.skip_mine, reference=a.reference
    )
    print(f"Saved → {out_root}")


if __name__ == "__main__":
    main()

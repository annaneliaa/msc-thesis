"""
Merge mining-comparison runs (e.g. the per-fold AIT-ADS LOSO runs that
run_mining_comparison_ait_ads.sh starts in parallel) into one run folder,
then recompute summary.csv, helps_hurts.csv, stability.csv and stats.json on
the whole (see thesis.system_eval.mining_comparison.merge_runs).

Usage:
    python src/thesis/scripts/system_eval/merge_mining_comparison.py \\
        --runs <run>/folds/fox <run>/folds/harrison ... --output-dir <run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

from thesis.system_eval.mining_comparison import merge_runs  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = merge_runs(a.runs, a.output_dir)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

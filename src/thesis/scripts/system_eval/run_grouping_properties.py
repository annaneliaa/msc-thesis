"""
Grouping × mining, step 1 (GM_PLAN.md): the descriptive table of group
properties per AIT-ADS scenario × grouping, from the cached groups (no
mining). See thesis.system_eval.grouping_properties.

Writes properties.csv (one row per scenario × grouping) and by_grouping.csv
(mean over scenarios) under artifacts/experiments/grouping_properties/<ts>/.

Usage:
    python src/thesis/scripts/system_eval/run_grouping_properties.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd  # noqa: E402

from thesis.paths import CACHE_DIR  # noqa: E402
from thesis.system_eval.grouping_properties import (  # noqa: E402
    load_groups_by,
    properties_table,
    summarize_by_grouping,
)
from thesis.system_eval.mining_comparison import AIT_ADS_SCENARIOS  # noqa: E402

GROUPINGS = ("cscas_grouping", "fixed_window", "time_delta", "alertbert", "deepcase")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scenarios", nargs="+", default=list(AIT_ADS_SCENARIOS))
    p.add_argument("--groupings", nargs="+", default=list(GROUPINGS))
    p.add_argument("--output-dir", type=Path, default=None)
    a = p.parse_args()

    groups_by = load_groups_by(a.scenarios, a.groupings, CACHE_DIR)
    table = properties_table(groups_by)
    by_grouping = summarize_by_grouping(table)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = a.output_dir or _REPO / "artifacts/experiments/grouping_properties" / ts
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "properties.csv", index=False)
    by_grouping.to_csv(out / "by_grouping.csv")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(by_grouping.round(3).T.to_string())
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

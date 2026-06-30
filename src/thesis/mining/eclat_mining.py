from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from thesis.mining.load_mining_alert_groups import build_tidsets


def eclat_recursive(
    prefix: tuple[str, ...],
    items_with_tidsets: list[tuple[str, set[int]]],
    min_count: int,
    results: list[dict],
    max_len: int | None = None,
) -> None:
    """
    Simple Eclat recursion.
    """
    for i, (item, tidset) in enumerate(items_with_tidsets):
        new_itemset = prefix + (item,)
        support_count = len(tidset)

        if support_count < min_count:
            continue

        results.append(
            {
                "itemset": new_itemset,
                "k": len(new_itemset),
                "support_count": support_count,
            }
        )

        if max_len is not None and len(new_itemset) >= max_len:
            continue

        suffix: list[tuple[str, set[int]]] = []
        for j in range(i + 1, len(items_with_tidsets)):
            other_item, other_tidset = items_with_tidsets[j]
            new_tidset = tidset & other_tidset
            if len(new_tidset) >= min_count:
                suffix.append((other_item, new_tidset))

        if suffix:
            eclat_recursive(
                prefix=new_itemset,
                items_with_tidsets=suffix,
                min_count=min_count,
                results=results,
                max_len=max_len,
            )


def run_eclat(
    alert_groups: list[frozenset[str]],
    min_support: float = 0.05,
    max_len: int | None = 3,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run Eclat on a list of alert_groups.

    Returns one row per frequent itemset.
    """
    n_tx = len(alert_groups)
    if n_tx == 0:
        return pd.DataFrame(
            columns=["itemset", "itemset_str", "k", "support_count", "support"]
        )

    min_count = max(1, int(min_support * n_tx))
    tidsets = build_tidsets(alert_groups, run_dir=run_dir)

    items_with_tidsets = sorted(tidsets.items(), key=lambda x: (x[0], len(x[1])))

    with open(run_dir / "items_with_tidsets.json", "w") as f:
        json.dump(
            [
                {
                    "item": item,
                    "support_count": len(tidset),
                    "tids": sorted(tidset),
                }
                for item, tidset in items_with_tidsets
            ],
            f,
            indent=2,
        )

    results: list[dict] = []
    eclat_recursive(
        prefix=(),
        items_with_tidsets=items_with_tidsets,
        min_count=min_count,
        results=results,
        max_len=max_len,
    )

    if not results:
        return pd.DataFrame(
            columns=["itemset", "itemset_str", "k", "support_count", "support"]
        )

    out = pd.DataFrame(results).drop_duplicates(subset=["itemset"]).copy()
    out["itemset_str"] = out["itemset"].apply(lambda x: " | ".join(x))
    out["support"] = out["support_count"] / n_tx
    out["n_alert_groups"] = n_tx
    out["min_support"] = min_support
    out["min_count"] = min_count

    return out.sort_values(
        by=["k", "support_count", "itemset_str"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

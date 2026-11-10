from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from thesis.schemas.mining import MiningAlertGroup


def load_mining_alert_groups_from_cache(
    path: str | Path = "artifacts/cache/alert_groups/alert_groups.json",
) -> list[MiningAlertGroup]:
    """
    Load MiningAlertGroup objects from cached JSON file.

    Expected format:
    - list of dicts
    - items: list[str] -> converted to set[str]
    - alert_labels: list[str] | None -> set[str] | None
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"AlertGroups file not found: {path}")

    with open(path, "r") as f:
        raw = json.load(f)

    alert_groups: list[MiningAlertGroup] = []

    for row in raw:
        tx = MiningAlertGroup(
            alert_group_id=row["alert_group_id"],
            window_start=row.get("window_start"),
            window_end=row.get("window_end"),
            n_alerts=row.get("n_alerts"),
            items=set(row.get("raw_items", [])),
            sorted_items=[set(itemset) for itemset in (row.get("sorted_items") or [])],
            group_label=row.get("group_label"),
            alert_labels=(
                set(row["alert_labels"])
                if row.get("alert_labels") is not None
                else None
            ),
            weight=row.get("weight", 1.0),
        )
        alert_groups.append(tx)

    return alert_groups


def prepare_alert_groups(
    alert_groups: Sequence[MiningAlertGroup],
    run_dir: Path | None = None,
) -> list[MiningAlertGroup]:
    """
    Clean mining alert_groups before itemset mining.

    Keeps only alert_groups that:
    - have a non-empty items set
    - have a non-null group_label
    """
    prepared: list[MiningAlertGroup] = []

    for tx in alert_groups:
        items = {str(x).strip() for x in tx.items if str(x).strip()}
        if not items:
            continue
        if tx.group_label is None:
            continue

        prepared.append(
            MiningAlertGroup(
                alert_group_id=tx.alert_group_id,
                window_start=tx.window_start,
                window_end=tx.window_end,
                n_alerts=tx.n_alerts,
                items=items,
                sorted_items=tx.sorted_items,
                group_label=tx.group_label,
                alert_labels=(
                    set(tx.alert_labels) if tx.alert_labels is not None else None
                ),
                weight=tx.weight,
            )
        )

    if run_dir is not None:
        prepared_df = pd.DataFrame(
            [
                {
                    "alert_group_id": tx.alert_group_id,
                    "window_start": tx.window_start,
                    "window_end": tx.window_end,
                    "n_alerts": tx.n_alerts,
                    "items": sorted(tx.items),
                    "basket_size": len(tx.items),
                    "group_label": tx.group_label,
                    "alert_labels": (
                        sorted(tx.alert_labels) if tx.alert_labels is not None else None
                    ),
                    "weight": tx.weight,
                }
                for tx in prepared
            ]
        )
        prepared_df.to_csv(run_dir / "prepared_alert_groups.csv", index=False)

    return prepared


def load_and_prepare_mining_alert_groups(
    path: str | Path = "artifacts/cache/alert_groups/alert_groups.json",
    run_dir: Path | None = None,
) -> list[MiningAlertGroup]:
    """
    Load cached MiningAlertGroup records and prepare them for mining.
    """
    alert_groups = load_mining_alert_groups_from_cache(path)
    prepared_alert_groups = prepare_alert_groups(alert_groups, run_dir=run_dir)
    print(
        f"Loaded {len(alert_groups)} alert_groups, prepared {len(prepared_alert_groups)} for mining."
    )
    return prepared_alert_groups


def build_tidsets(
    alert_groups: Iterable[frozenset[str]],
    run_dir: Path,
) -> dict[str, set[int]]:
    """
    Build vertical tidsets for Eclat.
    """
    tidsets: dict[str, set[int]] = {}

    for tid, basket in enumerate(alert_groups):
        for item in basket:
            tidsets.setdefault(item, set()).add(tid)

    with open(run_dir / "tidsets.json", "w") as f:
        json.dump(
            {item: sorted(tids) for item, tids in tidsets.items()},
            f,
            indent=2,
        )

    return tidsets

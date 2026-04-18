from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from thesis.schemas.mining import MiningTransaction


def prepare_transactions(
    transactions: Sequence[MiningTransaction],
    run_dir: Path | None = None,
) -> list[MiningTransaction]:
    """
    Clean mining transactions before itemset mining.

    Keeps only transactions that:
    - have a non-empty items set
    - have a non-null tx_label
    """
    prepared: list[MiningTransaction] = []

    for tx in transactions:
        items = {str(x).strip() for x in tx.items if str(x).strip()}
        if not items:
            continue
        if tx.tx_label is None:
            continue

        prepared.append(
            MiningTransaction(
                transaction_id=tx.transaction_id,
                window_start=tx.window_start,
                window_end=tx.window_end,
                n_alerts=tx.n_alerts,
                items=items,
                tx_label=tx.tx_label,
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
                    "transaction_id": tx.transaction_id,
                    "window_start": tx.window_start,
                    "window_end": tx.window_end,
                    "n_alerts": tx.n_alerts,
                    "items": sorted(tx.items),
                    "basket_size": len(tx.items),
                    "tx_label": tx.tx_label,
                    "alert_labels": (
                        sorted(tx.alert_labels) if tx.alert_labels is not None else None
                    ),
                    "weight": tx.weight,
                }
                for tx in prepared
            ]
        )
        prepared_df.to_csv(run_dir / "prepared_transactions.csv", index=False)

    return prepared


def build_tidsets(
    transactions: Iterable[frozenset[str]],
    run_dir: Path,
) -> dict[str, set[int]]:
    """
    Build vertical tidsets for Eclat.
    """
    tidsets: dict[str, set[int]] = {}

    for tid, basket in enumerate(transactions):
        for item in basket:
            tidsets.setdefault(item, set()).add(tid)

    with open(run_dir / "tidsets.json", "w") as f:
        json.dump(
            {item: sorted(tids) for item, tids in tidsets.items()},
            f,
            indent=2,
        )

    return tidsets

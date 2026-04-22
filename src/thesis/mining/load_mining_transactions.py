from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from thesis.schemas.mining import MiningTransaction


def load_mining_transactions_from_cache(
    path: str | Path = "artifacts/cache/transactions/transactions.json",
) -> list[MiningTransaction]:
    """
    Load MiningTransaction objects from cached JSON file.

    Expected format:
    - list of dicts
    - items: list[str] -> converted to set[str]
    - alert_labels: list[str] | None -> set[str] | None
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Transactions file not found: {path}")

    with open(path, "r") as f:
        raw = json.load(f)

    transactions: list[MiningTransaction] = []

    for row in raw:
        tx = MiningTransaction(
            transaction_id=row["transaction_id"],
            window_start=row.get("window_start"),
            window_end=row.get("window_end"),
            n_alerts=row.get("n_alerts"),
            items=set(row.get("abs_items", [])),
            tx_label=row.get("tx_label"),
            alert_labels=(
                set(row["alert_labels"])
                if row.get("alert_labels") is not None
                else None
            ),
            weight=row.get("weight", 1.0),
        )
        transactions.append(tx)

    return transactions


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


def load_and_prepare_mining_transactions(
    path: str | Path = "artifacts/cache/transactions/transactions.json",
    run_dir: Path | None = None,
) -> list[MiningTransaction]:
    """
    Load cached MiningTransaction records and prepare them for mining.
    """
    transactions = load_mining_transactions_from_cache(path)
    prepared_transactions = prepare_transactions(transactions, run_dir=run_dir)
    print(
        f"Loaded {len(transactions)} transactions, prepared {len(prepared_transactions)} for mining."
    )
    return prepared_transactions


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

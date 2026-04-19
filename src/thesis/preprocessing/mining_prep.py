from __future__ import annotations

from typing import List

from thesis.schemas.preprocessing import GroupSnapshot, Transaction


def build_transaction(snapshot: GroupSnapshot) -> Transaction:
    """
    Convert a GroupSnapshot into a Transaction.
    No abstraction is applied yet.
    """

    return Transaction(
        transaction_id=f"{snapshot.method}:{snapshot.group_id}",
        group_id=snapshot.group_id,
        group_method=snapshot.method,
        start_ts=snapshot.start_ts,
        end_ts=snapshot.end_ts,
        n_alerts=snapshot.n_alerts,
        items=set(snapshot.items),  # direct pass-through (no abstraction)
        raw_items=set(snapshot.items),  # keep raw copy for later use
        alert_ids=list(snapshot.alert_ids),
        tx_label=snapshot.tx_label,
        alert_labels=(
            set(snapshot.alert_labels) if snapshot.alert_labels is not None else None
        ),
        weight=1.0,  # no decay yet
    )


def build_transactions(snapshots: List[GroupSnapshot]) -> List[Transaction]:
    """
    Convert multiple GroupSnapshots into Transactions.
    """
    return [build_transaction(s) for s in snapshots]

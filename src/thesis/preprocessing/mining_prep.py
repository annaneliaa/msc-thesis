from __future__ import annotations

from typing import List

from thesis.schemas.preprocessing import GroupSnapshot, Transaction


def build_transaction(snapshot: GroupSnapshot) -> Transaction:
    """
    Convert a GroupSnapshot into a Transaction.
    Only mail host abstraction is applied now.
    Baseline features are computed and stored in the Transaction for later use.
    """

    return Transaction(
        transaction_id=snapshot.group_id,
        group_id=snapshot.group_id,
        method=snapshot.method,
        start_ts=snapshot.start_ts,
        end_ts=snapshot.end_ts,
        n_alerts=snapshot.n_alerts,
        abs_items=set(snapshot.items),  # use abstracted items for mining
        raw_items=set(snapshot.items),  # keep raw copy for later use
        sorted_items=snapshot.sorted_items,
        alert_ids=list(snapshot.alert_ids),
        alert_ips=set(snapshot.alert_ips),  # include alert IPs in transaction
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

from __future__ import annotations

from typing import List

from thesis.schemas.preprocessing import GroupSnapshot, Transaction


def abstract_mail_hosts(snapshot: GroupSnapshot) -> list[str]:
    """
    Replace host:<name>_mail patterns with a generic host:mail_host token.
    """
    abstracted_items = set()

    for item in snapshot.items:
        if item.startswith("host:"):
            host = item.split("host:", 1)[1]

            # detect mail hosts like: taylorcruz_mail
            if host.endswith("_mail"):
                abstracted_items.add("host:mail_host")
                continue

        abstracted_items.add(item)

    return abstracted_items


def build_transaction(snapshot: GroupSnapshot) -> Transaction:
    """
    Convert a GroupSnapshot into a Transaction.
    Only mail host abstraction is applied now.
    Baseline features are computed and stored in the Transaction for later use.
    """

    abstracted_items = abstract_mail_hosts(snapshot)

    return Transaction(
        transaction_id=snapshot.group_id,
        group_id=snapshot.group_id,
        method=snapshot.method,
        start_ts=snapshot.start_ts,
        end_ts=snapshot.end_ts,
        n_alerts=snapshot.n_alerts,
        abs_items=set(abstracted_items),  # use abstracted items for mining
        raw_items=set(snapshot.items),  # keep raw copy for later use
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

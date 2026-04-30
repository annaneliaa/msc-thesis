from pathlib import Path
from typing import Iterable, Dict

from thesis.schemas.cache import AlertCacheEntry, GroupCacheEntry
from thesis.schemas.preprocessing import GroupingRecord
from thesis.preprocessing.cache import TokenCache
from thesis.schemas.preprocessing import TokenizedAlert


def label_window_from_alert_labels(
    alert_labels: set[str],
    benign_label: str = "false_positive",
) -> str:
    labels = {str(lbl) for lbl in alert_labels if lbl is not None}

    has_benign = benign_label in labels
    has_attack = any(lbl != benign_label for lbl in labels)

    if has_attack and has_benign:
        return "mixed"
    if has_attack:
        return "attack"
    return "benign"


class CacheIngestor:
    """
    Writes tokenized alerts into the alert store of the cache.

    Group ingestion is intentionally left out for now.
    """

    def __init__(self, cache: TokenCache) -> None:
        self.cache = cache

    # -------------------------
    # alert ingestion
    # -------------------------

    @staticmethod
    def _to_alert_cache_entry(alert: TokenizedAlert) -> AlertCacheEntry:
        """
        Convert a TokenizedAlert into an AlertCacheEntry.
        """
        return AlertCacheEntry(
            alert_id=alert.alert_id,
            ts=alert.ts,
            ip=alert.ip,
            host=alert.host,
            short=alert.short,
            event_label=alert.event_label,
            time_label=alert.time_label,
        )

    def ingest_alert(self, alert: TokenizedAlert) -> None:
        """
        Ingest a single tokenized alert into the alert store.
        """
        entry = self._to_alert_cache_entry(alert)
        self.cache.write_alert_entry(entry)

    def ingest_alert_batch(
        self,
        alerts: Iterable[TokenizedAlert],
        batch_name: str,
    ) -> Path:
        """
        Ingest a batch of tokenized alerts into the alert store.
        Returns the written batch file path.
        """
        entries = [self._to_alert_cache_entry(alert) for alert in alerts]
        return self.cache.write_alert_batch(entries, batch_name=batch_name)

    # -------------------------
    # group ingestion
    # -------------------------
    def ingest_groups(
        self,
        alerts: list[TokenizedAlert],
        grouping_records: list[GroupingRecord],
        benign_label: str = "false_positive",
    ) -> None:
        """
        Build GroupCacheEntry objects from alerts + grouping records
        and write them to the cache.
        """

        # index alerts by id
        alerts_by_id: Dict[str, TokenizedAlert] = {a.alert_id: a for a in alerts}

        groups: Dict[str, GroupCacheEntry] = {}

        # sort grouping_records by alert timestamp so that items from earlier alerts are appended first, to preserve order
        for record in sorted(
            grouping_records,
            key=lambda r: alerts_by_id[r.alert_id].ts
            if r.alert_id in alerts_by_id
            else 0,
        ):
            alert = alerts_by_id.get(record.alert_id)
            if alert is None:
                continue

            if record.group_id not in groups:
                groups[record.group_id] = GroupCacheEntry(
                    group_id=record.group_id,
                    method=record.method,
                    status="open",
                    start_ts=alert.ts,
                    end_ts=alert.ts,
                    last_update_ts=alert.ts,
                    alert_ids=[],
                    n_alerts=0,
                    items=set(),
                    sorted_items=[],
                    alert_ips=set(),
                    alert_labels=None,
                    tx_label=None,
                    version=1,
                )

            group = groups[record.group_id]

            # update group
            group.alert_ids.append(alert.alert_id)
            group.n_alerts += 1

            group.sorted_items.extend(sorted(alert.tokens))
            group.items |= set(alert.tokens)
            group.alert_ips |= {alert.ip} if alert.ip else set()

            group.start_ts = min(group.start_ts, alert.ts)
            group.end_ts = max(group.end_ts, alert.ts)
            group.last_update_ts = max(group.last_update_ts, alert.ts)

            # labels
            if alert.time_label is not None:
                if group.alert_labels is None:
                    group.alert_labels = set()
                group.alert_labels.add(str(alert.time_label))

                group.tx_label = label_window_from_alert_labels(
                    group.alert_labels,
                    benign_label=benign_label,
                )

        # mark all as closed (since fixed windows are complete)
        for group in groups.values():
            group.status = "closed"

        # write to cache
        for group in groups.values():
            self.cache.write_group_entry(group)

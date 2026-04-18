from thesis.schemas.cache import AlertCacheEntry, WindowCacheEntry
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


def ingest_tokenized_alert(
    cache: TokenCache,
    alert: TokenizedAlert,
    benign_label: str = "false_positive",
) -> None:
    alert_entry = AlertCacheEntry(
        alert_id=alert.alert_id,
        ts=alert.ts,
        window_id=alert.window_id,
        repr_tokens=set(alert.repr_tokens),
        mining_tokens=set(alert.mining_tokens),
        ip=alert.ip,
        host=alert.host,
        short=alert.short,
        time_label=alert.time_label,
        event_label=alert.event_label,
    )
    cache.write_alert_entry(alert_entry)

    window = cache.read_window_entry(alert.window_id)
    if window is None:
        window = WindowCacheEntry(
            window_id=alert.window_id,
            start_ts=alert.window_id * 2,
            end_ts=alert.window_id * 2 + 1,
        )

    if alert.alert_id not in window.alert_ids:
        window.alert_ids.append(alert.alert_id)

    window.items |= set(alert.mining_tokens)

    if alert.host:
        window.hosts.add(alert.host)
    if alert.short:
        window.signatures.add(alert.short)

    if alert.time_label is not None:
        if window.alert_labels is None:
            window.alert_labels = set()
        window.alert_labels.add(str(alert.time_label))
        window.tx_label = label_window_from_alert_labels(
            window.alert_labels,
            benign_label=benign_label,
        )

    cache.write_window_entry(window)


def ingest_tokenized_alert_batch(
    cache: TokenCache,
    alerts: list[TokenizedAlert],
    alert_batch_name: str,
    benign_label: str = "false_positive",
) -> None:
    alert_entries_by_id: dict[str, AlertCacheEntry] = {}
    windows_by_id: dict[int, WindowCacheEntry] = {}

    for alert in alerts:
        alert_entry = AlertCacheEntry(
            alert_id=alert.alert_id,
            ts=alert.ts,
            window_id=alert.window_id,
            repr_tokens=set(alert.repr_tokens),
            mining_tokens=set(alert.mining_tokens),
            ip=alert.ip,
            host=alert.host,
            short=alert.short,
            time_label=alert.time_label,
            event_label=alert.event_label,
        )
        alert_entries_by_id[alert.alert_id] = alert_entry

        if alert.window_id not in windows_by_id:
            window = cache.read_window_entry(alert.window_id)
            if window is None:
                window = WindowCacheEntry(
                    window_id=alert.window_id,
                    start_ts=alert.window_id * 2,
                    end_ts=alert.window_id * 2 + 1,
                )
            windows_by_id[alert.window_id] = window

        window = windows_by_id[alert.window_id]

        if alert.alert_id not in window.alert_ids:
            window.alert_ids.append(alert.alert_id)

        window.items |= set(alert.mining_tokens)

        if alert.host:
            window.hosts.add(alert.host)
        if alert.short:
            window.signatures.add(alert.short)

        if alert.time_label is not None:
            if window.alert_labels is None:
                window.alert_labels = set()
            window.alert_labels.add(str(alert.time_label))
            window.tx_label = label_window_from_alert_labels(
                window.alert_labels,
                benign_label=benign_label,
            )

    cache.write_alert_batch(
        list(alert_entries_by_id.values()), batch_name=alert_batch_name
    )

    for window in windows_by_id.values():
        cache.write_window_entry(window)

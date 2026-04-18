from thesis.schemas.cache import AlertCacheEntry, WindowCacheEntry
from thesis.preprocessing.cache import TokenCache
from thesis.schemas.preprocessing import TokenizedAlert


def ingest_tokenized_alert(cache: TokenCache, alert: TokenizedAlert) -> None:
    alert_entry = AlertCacheEntry(
        alert_id=alert.alert_id,
        ts=alert.ts,
        window_id=alert.window_id,
        repr_tokens=set(alert.repr_tokens),
        mining_tokens=set(alert.mining_tokens),
        ip=alert.ip,
        host=alert.host,
        short=alert.short,
    )
    cache.write_alert_entry(alert_entry)

    window = cache.read_window_entry(alert.window_id)
    if window is None:
        window = WindowCacheEntry(
            window_id=alert.window_id,
            start_ts=alert.window_id * 2,
            end_ts=alert.window_id * 2 + 1,
        )

    window.alert_ids.append(alert.alert_id)
    window.items |= set(alert.mining_tokens)

    if alert.host:
        window.hosts.add(alert.host)
    if alert.short:
        window.signatures.add(alert.short)

    cache.write_window_entry(window)

from thesis.schemas.cache import AlertCacheEntry, WindowCacheEntry
from thesis.preprocessing.cache import TokenCache
from thesis.schemas.preprocessing import TokenizedAlert


def ingest_tokenized_alert(cache: TokenCache, alert: TokenizedAlert) -> None:
    cache.alert_store[alert.alert_id] = AlertCacheEntry(
        alert_id=alert.alert_id,
        ts=alert.ts,
        window_id=alert.window_id,
        repr_tokens=set(alert.repr_tokens),
        mining_tokens=set(alert.mining_tokens),
        ip=alert.ip,
        host=alert.host,
        short=alert.short,
    )

    if alert.window_id not in cache.window_store:
        cache.window_store[alert.window_id] = WindowCacheEntry(
            window_id=alert.window_id,
            start_ts=alert.window_id * 2,
            end_ts=alert.window_id * 2 + 1,
        )

    window = cache.window_store[alert.window_id]
    window.alert_ids.append(alert.alert_id)
    window.items |= set(alert.mining_tokens)

    if alert.host:
        window.hosts.add(alert.host)
    if alert.short:
        window.signatures.add(alert.short)

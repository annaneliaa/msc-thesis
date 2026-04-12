import pandas as pd

from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.cache_ingestor import ingest_tokenized_alert
from thesis.schemas.preprocessing import TokenizedAlert


def make_tokenized_alert() -> TokenizedAlert:
    return TokenizedAlert(
        alert_id="abc123",
        ts=1642213952,
        time_norm=pd.Timestamp("2022-01-15 09:12:32+00:00"),
        window_id=821106976,
        name="Wazuh: ClamAV database update",
        ip="172.17.131.81",
        host="mail",
        short="W-Sys-Cav",
        repr_tokens={
            "short:W-Sys-Cav",
            "host:mail",
            "name:wazuh clamav database update",
            "ip:172.17.131.81",
        },
        mining_tokens={
            "short:W-Sys-Cav",
            "host:mail",
            "database",
            "update",
        },
        raw={
            "time": 1642213952,
            "name": "Wazuh: ClamAV database update",
            "ip": "172.17.131.81",
            "host": "mail",
            "short": "W-Sys-Cav",
            "time_label": "false_positive",
            "event_label": "-",
        },
    )


def test_single_alert_ingestion_adds_alert_to_alert_store():
    cache = TokenCache()
    alert = make_tokenized_alert()

    ingest_tokenized_alert(cache, alert)

    assert alert.alert_id in cache.alert_store

    entry = cache.alert_store[alert.alert_id]
    assert entry.alert_id == alert.alert_id
    assert entry.ts == alert.ts
    assert entry.window_id == alert.window_id
    assert entry.repr_tokens == alert.repr_tokens
    assert entry.mining_tokens == alert.mining_tokens
    assert entry.ip == alert.ip
    assert entry.host == alert.host
    assert entry.short == alert.short


def test_single_alert_ingestion_creates_window_entry():
    cache = TokenCache()
    alert = make_tokenized_alert()

    ingest_tokenized_alert(cache, alert)

    assert alert.window_id in cache.window_store

    window = cache.window_store[alert.window_id]
    assert window.window_id == alert.window_id
    assert window.alert_ids == [alert.alert_id]
    assert window.items == alert.mining_tokens
    assert window.hosts == {"mail"}
    assert window.signatures == {"W-Sys-Cav"}
    assert window.closed is False


def test_single_alert_ingestion_sets_window_bounds():
    cache = TokenCache()
    alert = make_tokenized_alert()

    ingest_tokenized_alert(cache, alert)

    window = cache.window_store[alert.window_id]
    assert window.start_ts == 1642213952
    assert window.end_ts == 1642213953


def test_single_alert_ingestion_copies_token_sets():
    cache = TokenCache()
    alert = make_tokenized_alert()

    ingest_tokenized_alert(cache, alert)

    entry = cache.alert_store[alert.alert_id]
    window = cache.window_store[alert.window_id]

    assert entry.repr_tokens == alert.repr_tokens
    assert entry.repr_tokens is not alert.repr_tokens

    assert entry.mining_tokens == alert.mining_tokens
    assert entry.mining_tokens is not alert.mining_tokens

    assert window.items == alert.mining_tokens
    assert window.items is not alert.mining_tokens

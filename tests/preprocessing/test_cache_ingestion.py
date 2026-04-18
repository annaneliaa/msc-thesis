import json
import pandas as pd

from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.cache_ingestor import ingest_tokenized_alert
from thesis.schemas.preprocessing import TokenizedAlert
from thesis.schemas.cache import AlertCacheEntry, WindowCacheEntry


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


def test_single_alert_ingestion_writes_cache_files(tmp_path):
    cache = TokenCache(cache_dir=tmp_path)
    alert = make_tokenized_alert()

    ingest_tokenized_alert(cache, alert)

    alert_path = tmp_path / "alerts" / f"{alert.alert_id}.json"
    window_path = tmp_path / "windows" / f"{alert.window_id}.json"

    assert alert_path.exists()
    assert window_path.exists()


def test_single_alert_ingestion_writes_expected_alert_json(tmp_path):
    cache = TokenCache(cache_dir=tmp_path)
    alert = make_tokenized_alert()

    ingest_tokenized_alert(cache, alert)

    alert_path = tmp_path / "alerts" / f"{alert.alert_id}.json"

    with alert_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["alert_id"] == alert.alert_id
    assert payload["ts"] == alert.ts
    assert payload["window_id"] == alert.window_id
    assert set(payload["repr_tokens"]) == alert.repr_tokens
    assert set(payload["mining_tokens"]) == alert.mining_tokens
    assert payload["ip"] == alert.ip
    assert payload["host"] == alert.host
    assert payload["short"] == alert.short


def test_single_alert_ingestion_writes_expected_window_json(tmp_path):
    cache = TokenCache(cache_dir=tmp_path)
    alert = make_tokenized_alert()

    ingest_tokenized_alert(cache, alert)

    window_path = tmp_path / "windows" / f"{alert.window_id}.json"

    with window_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["window_id"] == alert.window_id
    assert payload["start_ts"] == 1642213952
    assert payload["end_ts"] == 1642213953
    assert payload["alert_ids"] == [alert.alert_id]
    assert set(payload["items"]) == alert.mining_tokens
    assert set(payload["hosts"]) == {"mail"}
    assert set(payload["signatures"]) == {"W-Sys-Cav"}
    assert payload["closed"] is False


def test_read_alert_entry_reconstructs_alert_cache_entry(tmp_path):
    cache = TokenCache(cache_dir=tmp_path)
    alert = make_tokenized_alert()

    ingest_tokenized_alert(cache, alert)

    entry = cache.read_alert_entry(alert.alert_id)

    assert isinstance(entry, AlertCacheEntry)
    assert entry is not None
    assert entry.alert_id == alert.alert_id
    assert entry.ts == alert.ts
    assert entry.window_id == alert.window_id
    assert entry.repr_tokens == alert.repr_tokens
    assert entry.mining_tokens == alert.mining_tokens
    assert entry.ip == alert.ip
    assert entry.host == alert.host
    assert entry.short == alert.short


def test_read_window_entry_reconstructs_window_cache_entry(tmp_path):
    cache = TokenCache(cache_dir=tmp_path)
    alert = make_tokenized_alert()

    ingest_tokenized_alert(cache, alert)

    entry = cache.read_window_entry(alert.window_id)

    assert isinstance(entry, WindowCacheEntry)
    assert entry is not None
    assert entry.window_id == alert.window_id
    assert entry.start_ts == 1642213952
    assert entry.end_ts == 1642213953
    assert entry.alert_ids == [alert.alert_id]
    assert entry.items == alert.mining_tokens
    assert entry.hosts == {"mail"}
    assert entry.signatures == {"W-Sys-Cav"}
    assert entry.closed is False


def test_read_alert_entry_returns_none_for_missing_file(tmp_path):
    cache = TokenCache(cache_dir=tmp_path)

    entry = cache.read_alert_entry("does_not_exist")

    assert entry is None


def test_read_window_entry_returns_none_for_missing_file(tmp_path):
    cache = TokenCache(cache_dir=tmp_path)

    entry = cache.read_window_entry(999999)

    assert entry is None

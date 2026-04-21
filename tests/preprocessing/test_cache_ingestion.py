import json
import pandas as pd

from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.cache_ingestor import CacheIngestor
from thesis.schemas.preprocessing import TokenizedAlert, GroupingRecord
from thesis.schemas.cache import AlertCacheEntry, GroupCacheEntry

SCENARIO = "test_scenario"


def make_tokenized_alert() -> TokenizedAlert:
    return TokenizedAlert(
        alert_id="abc123",
        ts=1642213952,
        time_norm=pd.Timestamp("2022-01-15 09:12:32+00:00"),
        name="Wazuh: ClamAV database update",
        ip="172.17.131.81",
        host="mail",
        short="W-Sys-Cav",
        tokens={
            "short:W-Sys-Cav",
            "host:mail",
            "name:wazuh clamav database update",
            "ip:172.17.131.81",
            "database",
            "update",
        },
        time_label="false_positive",
        event_label="-",
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


def make_grouping_record() -> GroupingRecord:
    return GroupingRecord(
        alert_id="abc123",
        group_id="group_1",
        method="fixed_window",
    )


def test_single_alert_ingestion_writes_cache_files(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)
    ingestor = CacheIngestor(cache=cache)
    alert = make_tokenized_alert()

    ingestor.ingest_alert_batch(
        alerts=[alert],
        batch_name=SCENARIO,
    )

    alert_path = tmp_path / SCENARIO / "alerts" / f"{SCENARIO}.json"

    assert alert_path.exists()


def test_single_alert_ingestion_writes_expected_alert_json(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)
    ingestor = CacheIngestor(cache=cache)
    alert = make_tokenized_alert()

    ingestor.ingest_alert_batch(
        alerts=[alert],
        batch_name=SCENARIO,
    )

    alert_path = tmp_path / SCENARIO / "alerts" / f"{SCENARIO}.json"

    with alert_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert isinstance(payload, list)
    assert len(payload) == 1

    entry = payload[0]
    assert entry["alert_id"] == alert.alert_id
    assert entry["ts"] == alert.ts
    assert entry["ip"] == alert.ip
    assert entry["host"] == alert.host
    assert entry["short"] == alert.short
    assert entry["time_label"] == alert.time_label
    assert entry["event_label"] == alert.event_label


def test_group_ingestion_writes_expected_group_json(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)
    ingestor = CacheIngestor(cache=cache)
    alert = make_tokenized_alert()
    record = make_grouping_record()

    ingestor.ingest_groups(
        alerts=[alert],
        grouping_records=[record],
    )

    group_path = tmp_path / SCENARIO / "groups" / f"{record.group_id}.json"

    assert group_path.exists()

    with group_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["group_id"] == record.group_id
    assert payload["method"] == record.method
    assert payload["status"] == "closed"
    assert payload["start_ts"] == alert.ts
    assert payload["end_ts"] == alert.ts
    assert payload["last_update_ts"] == alert.ts
    assert payload["alert_ids"] == [alert.alert_id]
    assert payload["n_alerts"] == 1
    assert set(payload["items"]) == alert.tokens
    assert set(payload["alert_labels"]) == {"false_positive"}
    assert payload["tx_label"] == "benign"
    assert payload["version"] == 1


def test_read_alert_batch_reconstructs_alert_cache_entries(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)
    ingestor = CacheIngestor(cache=cache)
    alert = make_tokenized_alert()

    ingestor.ingest_alert_batch(
        alerts=[alert],
        batch_name=SCENARIO,
    )

    entries = cache.read_alert_batch(SCENARIO)

    assert isinstance(entries, list)
    assert len(entries) == 1

    entry = entries[0]
    assert isinstance(entry, AlertCacheEntry)
    assert entry.alert_id == alert.alert_id
    assert entry.ts == alert.ts
    assert entry.ip == alert.ip
    assert entry.host == alert.host
    assert entry.short == alert.short
    assert entry.time_label == alert.time_label
    assert entry.event_label == alert.event_label


def test_read_group_entry_reconstructs_group_cache_entry(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)
    ingestor = CacheIngestor(cache=cache)
    alert = make_tokenized_alert()
    record = make_grouping_record()

    ingestor.ingest_groups(
        alerts=[alert],
        grouping_records=[record],
    )

    entry = cache.read_group_entry(record.group_id)

    assert isinstance(entry, GroupCacheEntry)
    assert entry is not None
    assert entry.group_id == record.group_id
    assert entry.method == record.method
    assert entry.status == "closed"
    assert entry.start_ts == alert.ts
    assert entry.end_ts == alert.ts
    assert entry.last_update_ts == alert.ts
    assert entry.alert_ids == [alert.alert_id]
    assert entry.n_alerts == 1
    assert entry.items == alert.tokens
    assert entry.alert_labels == {"false_positive"}
    assert entry.tx_label == "benign"
    assert entry.version == 1


def test_read_alert_batch_returns_empty_list_for_missing_file(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)

    entries = cache.read_alert_batch("does_not_exist")

    assert entries == []


def test_read_group_entry_returns_none_for_missing_file(tmp_path):
    cache = TokenCache(cache_dir=tmp_path, scenario=SCENARIO)

    entry = cache.read_group_entry("does_not_exist")

    assert entry is None

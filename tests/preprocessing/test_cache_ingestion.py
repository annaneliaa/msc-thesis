import json
import pandas as pd

from thesis.caching.cache import TokenCache
from thesis.caching.ingestor import CacheIngestor
from thesis.schemas.preprocessing import TokenizedAlert
from thesis.schemas.groups import GroupingRecord
from thesis.schemas.cache import GroupCacheEntry

SCENARIO = "test_scenario"


def make_tokenized_alert() -> TokenizedAlert:
    return TokenizedAlert(
        alert_id="abc123",
        ts=1642213952,
        time_norm=pd.Timestamp("2022-01-15 09:12:32+00:00"),
        signature="Wazuh: ClamAV database update",
        ip="172.17.131.81",
        host="mail",
        short="W-Sys-Cav",
        tokens={
            "short:W-Sys-Cav",
            "host:mail",
            "sig:database",
            "sig:update",
        },
        label="false_positive",
        event_label="-",
        raw={
            "time": 1642213952,
            "signature": "Wazuh: ClamAV database update",
            "ip": "172.17.131.81",
            "host": "mail",
            "short": "W-Sys-Cav",
            "label": "false_positive",
            "event_label": "-",
        },
    )


def make_grouping_record() -> GroupingRecord:
    return GroupingRecord(
        alert_id="abc123",
        group_id="group_1",
        method="fixed_window",
    )


def test_group_ingestion_writes_expected_group_json(tmp_path):
    cache = TokenCache(cache_dir=tmp_path / SCENARIO)
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
    assert payload["group_label"] == "benign"
    assert payload["version"] == 1


def test_read_group_entry_reconstructs_group_cache_entry(tmp_path):
    cache = TokenCache(cache_dir=tmp_path / SCENARIO)
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
    assert entry.group_label == "benign"
    assert entry.version == 1


def test_read_group_entry_returns_none_for_missing_file(tmp_path):
    cache = TokenCache(cache_dir=tmp_path / SCENARIO)

    entry = cache.read_group_entry("does_not_exist")

    assert entry is None

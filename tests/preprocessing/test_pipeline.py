import json

from typer.testing import CliRunner

from thesis.cli import app
from thesis.preprocessing.cache import TokenCache
from thesis.preprocessing.group_selector import select_group_snapshots

runner = CliRunner()
SCENARIO = "fox"


def write_cli_input_file(tmp_path, scenario, payload):
    alerts_dir = tmp_path / "artifacts" / "processed-data" / scenario
    alerts_dir.mkdir(parents=True, exist_ok=True)

    alerts_file = alerts_dir / "alerts.json"
    with alerts_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    return alerts_file


def test_preprocess_alert_batch_cli_runs_full_pipeline(tmp_path, monkeypatch):
    alerts_payload = [
        {
            "time": 1642213952,
            "name": "Wazuh: ClamAV database update",
            "ip": "172.17.131.81",
            "host": "mail",
            "short": "W-Sys-Cav",
            "time_label": "false_positive",
            "event_label": "-",
        },
        {
            "time": 1642213953,
            "name": "Suricata: TLS invalid handshake",
            "ip": "172.17.131.90",
            "host": "web",
            "short": "A-Network-Tls",
            "time_label": "true_positive",
            "event_label": "attack_1",
        },
    ]

    write_cli_input_file(tmp_path, SCENARIO, alerts_payload)
    cache_dir = tmp_path / "cache"

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "preprocess-alert-batch",
            "--scenario",
            SCENARIO,
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Processed 2 alerts." in result.stdout

    cache = TokenCache(cache_dir=cache_dir, scenario=SCENARIO)

    alert_files = list((cache_dir / SCENARIO / "alerts").glob("*.json"))
    group_files = list((cache_dir / SCENARIO / "groups").glob("*.json"))

    assert len(alert_files) == 1
    assert alert_files[0].name == f"{SCENARIO}.json"
    assert len(group_files) >= 1

    alert_entries = cache.read_alert_batch(SCENARIO)
    assert len(alert_entries) == 2

    snapshots = select_group_snapshots(cache=cache, require_closed=True)
    assert len(snapshots) >= 1

    total_alert_ids = sum(len(s.alert_ids) for s in snapshots)
    assert total_alert_ids == 2

    all_items = set().union(*(s.items for s in snapshots))
    assert "short:W-Sys-Cav" in all_items
    assert "host:mail" in all_items
    assert "short:A-Network-Tls" in all_items
    assert "host:web" in all_items
    assert any("database" in item for item in all_items)
    assert any("update" in item for item in all_items)
    assert any("tls" in item for item in all_items)
    assert any("invalid" in item for item in all_items)
    assert any("handshake" in item for item in all_items)

    all_labels = set()
    for s in snapshots:
        if s.alert_labels:
            all_labels |= s.alert_labels

    assert "false_positive" in all_labels
    assert "true_positive" in all_labels


def test_duplicate_alert_ingestion_does_not_duplicate_group_entries(
    tmp_path, monkeypatch
):
    alert_payload = {
        "time": 1642213952,
        "name": "Wazuh: ClamAV database update",
        "ip": "172.17.131.81",
        "host": "mail",
        "short": "W-Sys-Cav",
        "time_label": "false_positive",
        "event_label": "-",
    }

    alerts_payload = [alert_payload, alert_payload]

    write_cli_input_file(tmp_path, SCENARIO, alerts_payload)
    cache_dir = tmp_path / "cache"

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "preprocess-alert-batch",
            "--scenario",
            SCENARIO,
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Processed 2 alerts." in result.stdout

    cache = TokenCache(cache_dir=cache_dir, scenario=SCENARIO)

    alert_files = list((cache_dir / SCENARIO / "alerts").glob("*.json"))
    group_files = list((cache_dir / SCENARIO / "groups").glob("*.json"))

    assert len(alert_files) == 1
    assert len(group_files) == 1

    # Current pipeline behavior: duplicates are preserved in alert batch storage
    alert_entries = cache.read_alert_batch(SCENARIO)
    assert len(alert_entries) == 2
    assert len({entry.alert_id for entry in alert_entries}) == 1

    snapshots = select_group_snapshots(cache=cache, require_closed=True)
    assert len(snapshots) == 1

    snap = snapshots[0]
    assert snap.n_alerts == 2
    assert len(snap.alert_ids) == 2
    assert len(set(snap.alert_ids)) == 1

    assert "short:W-Sys-Cav" in snap.items
    assert "host:mail" in snap.items
    assert any("database" in item for item in snap.items)
    assert any("update" in item for item in snap.items)

    assert snap.alert_labels == {"false_positive"}
    assert snap.tx_label == "benign"


def test_empty_alerts_are_not_added_to_cache(tmp_path, monkeypatch):
    alerts_payload = [
        {
            "time": None,
            "name": None,
            "ip": None,
            "host": None,
            "short": None,
            "time_label": None,
            "event_label": None,
        }
    ]

    write_cli_input_file(tmp_path, SCENARIO, alerts_payload)
    cache_dir = tmp_path / "cache"

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "preprocess-alert-batch",
            "--scenario",
            SCENARIO,
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout

    alert_dir = cache_dir / SCENARIO / "alerts"
    group_dir = cache_dir / SCENARIO / "groups"

    alert_files = list(alert_dir.glob("*.json")) if alert_dir.exists() else []
    group_files = list(group_dir.glob("*.json")) if group_dir.exists() else []

    assert len(alert_files) == 0
    assert len(group_files) == 0


def test_alerts_json_to_group_snapshot_selection_full_pipeline(tmp_path, monkeypatch):
    alerts_payload = [
        {
            "time": 1642213952,
            "name": "Wazuh: ClamAV database update",
            "ip": "172.17.131.81",
            "host": "mail",
            "short": "W-Sys-Cav",
            "time_label": "false_positive",
            "event_label": "-",
        },
        {
            "time": 1642213953,
            "name": "Suricata: TLS invalid handshake",
            "ip": "172.17.131.90",
            "host": "web",
            "short": "A-Network-Tls",
            "time_label": "true_positive",
            "event_label": "attack_1",
        },
    ]

    write_cli_input_file(tmp_path, SCENARIO, alerts_payload)
    cache_dir = tmp_path / "cache"

    monkeypatch.chdir(tmp_path)

    preprocess_result = runner.invoke(
        app,
        [
            "preprocess-alert-batch",
            "--scenario",
            SCENARIO,
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert preprocess_result.exit_code == 0, preprocess_result.stdout
    assert "Processed 2 alerts." in preprocess_result.stdout

    cache = TokenCache(cache_dir=cache_dir, scenario=SCENARIO)
    snapshots = select_group_snapshots(cache=cache, require_closed=True)

    assert len(snapshots) >= 1

    total_alert_ids = sum(len(s.alert_ids) for s in snapshots)
    assert total_alert_ids == 2

    all_items = set().union(*(s.items for s in snapshots))
    assert "short:W-Sys-Cav" in all_items
    assert "short:A-Network-Tls" in all_items
    assert any("database" in item for item in all_items)
    assert any("update" in item for item in all_items)
    assert any("tls" in item for item in all_items)
    assert any("invalid" in item for item in all_items)
    assert any("handshake" in item for item in all_items)

    all_labels = set()
    tx_labels = set()

    for s in snapshots:
        if s.alert_labels:
            all_labels |= s.alert_labels
        if s.tx_label:
            tx_labels.add(s.tx_label)

    assert "false_positive" in all_labels
    assert "true_positive" in all_labels
    assert any(lbl in {"mixed", "attack", "benign"} for lbl in tx_labels)

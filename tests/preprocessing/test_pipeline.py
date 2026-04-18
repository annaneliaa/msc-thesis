import json

from typer.testing import CliRunner

from thesis.cli import app
from thesis.preprocessing.cache import TokenCache


runner = CliRunner()
SCENARIO = "fox"


def test_preprocess_alert_batch_cli_runs_full_pipeline(tmp_path):
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

    alerts_file = tmp_path / "alerts.json"
    cache_dir = tmp_path / "cache"

    with alerts_file.open("w", encoding="utf-8") as f:
        json.dump(alerts_payload, f)

    result = runner.invoke(
        app,
        [
            "preprocess-alert-batch",
            str(alerts_file),
            "--scenario",
            SCENARIO,
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Processed 2 alerts." in result.stdout

    cache = TokenCache(cache_dir=cache_dir, scenario=SCENARIO)

    alert_files = list((cache_dir / SCENARIO / "alerts").glob("*.json"))
    window_files = list((cache_dir / SCENARIO / "windows").glob("*.json"))

    # batch alert storage -> one batch file
    assert len(alert_files) == 1
    assert alert_files[0].name == f"{SCENARIO}.json"

    # both alerts fall into same 2-second window
    assert len(window_files) == 1

    alert_entries = cache.read_alert_batch(SCENARIO)
    assert len(alert_entries) == 2

    window_entry = cache.read_window_entry(821106976)
    assert window_entry is not None
    assert window_entry.window_id == 821106976
    assert window_entry.start_ts == 1642213952
    assert window_entry.end_ts == 1642213953
    assert len(window_entry.alert_ids) == 2
    assert window_entry.hosts == {"mail", "web"}
    assert window_entry.signatures == {"W-Sys-Cav", "A-Network-Tls"}
    assert window_entry.alert_labels == {"false_positive", "true_positive"}
    assert window_entry.tx_label == "mixed"
    assert window_entry.items == {
        "short:W-Sys-Cav",
        "host:mail",
        "sig:database",
        "sig:update",
        "short:A-Network-Tls",
        "host:web",
        "sig:tls",
        "sig:invalid",
        "sig:handshake",
    }


def test_duplicate_alert_ingestion_does_not_duplicate_window_entries(tmp_path):
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

    alerts_file = tmp_path / "alerts.json"
    cache_dir = tmp_path / "cache"

    with alerts_file.open("w", encoding="utf-8") as f:
        json.dump(alerts_payload, f)

    result = runner.invoke(
        app,
        [
            "preprocess-alert-batch",
            str(alerts_file),
            "--scenario",
            SCENARIO,
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Processed 2 alerts." in result.stdout

    cache = TokenCache(cache_dir=cache_dir, scenario=SCENARIO)

    alert_files = list((cache_dir / SCENARIO / "alerts").glob("*.json"))
    window_files = list((cache_dir / SCENARIO / "windows").glob("*.json"))

    # batch storage -> one alert batch file
    assert len(alert_files) == 1
    assert len(window_files) == 1

    # assumes your batch ingestor deduplicates by alert_id before writing batch
    alert_entries = cache.read_alert_batch(SCENARIO)
    assert len(alert_entries) == 1

    window_entry = cache.read_window_entry(821106976)
    assert window_entry is not None

    assert len(window_entry.alert_ids) == 1
    assert len(set(window_entry.alert_ids)) == 1

    assert window_entry.items == {
        "short:W-Sys-Cav",
        "host:mail",
        "sig:database",
        "sig:update",
    }

    assert window_entry.hosts == {"mail"}
    assert window_entry.signatures == {"W-Sys-Cav"}
    assert window_entry.alert_labels == {"false_positive"}
    assert window_entry.tx_label == "benign"


def test_empty_alerts_are_not_added_to_cache(tmp_path):
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

    alerts_file = tmp_path / "alerts.json"
    cache_dir = tmp_path / "cache"

    with alerts_file.open("w", encoding="utf-8") as f:
        json.dump(alerts_payload, f)

    result = runner.invoke(
        app,
        [
            "preprocess-alert-batch",
            str(alerts_file),
            "--scenario",
            SCENARIO,
            "--cache-dir",
            str(cache_dir),
        ],
    )

    # requires process_alert_batch() to skip bad rows instead of crashing
    assert result.exit_code == 0

    alert_files = list((cache_dir / SCENARIO / "alerts").glob("*.json"))
    window_files = list((cache_dir / SCENARIO / "windows").glob("*.json"))

    assert len(alert_files) == 0
    assert len(window_files) == 0


def test_alerts_json_to_transaction_selection_full_pipeline(tmp_path):
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

    alerts_file = tmp_path / "alerts.json"
    cache_dir = tmp_path / "cache"

    with alerts_file.open("w", encoding="utf-8") as f:
        json.dump(alerts_payload, f)

    preprocess_result = runner.invoke(
        app,
        [
            "preprocess-alert-batch",
            str(alerts_file),
            "--scenario",
            SCENARIO,
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert preprocess_result.exit_code == 0
    assert "Processed 2 alerts." in preprocess_result.stdout

    select_result = runner.invoke(
        app,
        [
            "select-transactions",
            "--cache-dir",
            str(cache_dir),
            "--scenario",
            SCENARIO,
        ],
    )

    assert select_result.exit_code == 0
    assert "window_id=821106976" in select_result.stdout
    assert "n_alerts=2" in select_result.stdout
    assert "sig:database" in select_result.stdout
    assert "sig:update" in select_result.stdout
    assert "sig:tls" in select_result.stdout
    assert "sig:invalid" in select_result.stdout
    assert "sig:handshake" in select_result.stdout

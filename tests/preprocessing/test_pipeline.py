import json

from typer.testing import CliRunner

from thesis.cli import app
from thesis.preprocessing.cache import TokenCache


runner = CliRunner()


def test_preprocess_single_alert_cli_runs_full_pipeline(tmp_path):
    alert_payload = {
        "time": 1642213952,
        "name": "Wazuh: ClamAV database update",
        "ip": "172.17.131.81",
        "host": "mail",
        "short": "W-Sys-Cav",
        "time_label": "false_positive",
        "event_label": "-",
    }

    alert_file = tmp_path / "alert.json"
    cache_dir = tmp_path / "cache"

    with alert_file.open("w", encoding="utf-8") as f:
        json.dump(alert_payload, f)

    result = runner.invoke(
        app,
        [
            "preprocess-single-alert",
            str(alert_file),
            "--scenario",
            "fox",
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Processed alert_id=" in result.stdout
    assert "Window ID=821106976" in result.stdout

    cache = TokenCache(cache_dir=cache_dir)

    alert_files = list((cache_dir / "alerts").glob("*.json"))
    window_files = list((cache_dir / "windows").glob("*.json"))

    assert len(alert_files) == 1
    assert len(window_files) == 1

    alert_entry = cache.read_alert_entry(alert_files[0].stem)
    assert alert_entry is not None
    assert alert_entry.ts == 1642213952
    assert alert_entry.window_id == 821106976
    assert alert_entry.host == "mail"
    assert alert_entry.short == "W-Sys-Cav"
    assert alert_entry.ip == "172.17.131.81"

    window_entry = cache.read_window_entry(821106976)
    assert window_entry is not None
    assert window_entry.window_id == 821106976
    assert window_entry.start_ts == 1642213952
    assert window_entry.end_ts == 1642213953
    assert len(window_entry.alert_ids) == 1
    assert window_entry.hosts == {"mail"}
    assert window_entry.signatures == {"W-Sys-Cav"}

from thesis.preprocessing.parsing import ParsedAlert, parse_alert_row


def test_parse_alert_row_returns_parsed_alert_with_expected_values():
    row = {
        "time": 1642213952,
        "name": "Wazuh: ClamAV database update",
        "ip": "172.17.131.81",
        "host": "mail",
        "short": "W-Sys-Cav",
    }

    parsed = parse_alert_row(
        row=row,
        scenario="fox",
        time_col="time",
        window_size_seconds=2,
        keep_raw=True,
    )

    assert isinstance(parsed, ParsedAlert)

    assert parsed.ts == 1642213952
    assert parsed.window_id == 821106976
    assert str(parsed.time_norm) == "2022-01-15 09:12:32+00:00"

    assert parsed.name == "Wazuh: ClamAV database update"
    assert parsed.ip == "172.17.131.81"
    assert parsed.host == "mail"
    assert parsed.short == "W-Sys-Cav"

    assert parsed.raw == row
    assert isinstance(parsed.alert_id, str)
    assert len(parsed.alert_id) == 40


def test_parse_alert_row_normalizes_missing_values():
    row = {
        "time": 1642213952,
        "name": "",
        "ip": None,
        "host": "mail",
        "short": "W-Sys-Cav",
    }

    parsed = parse_alert_row(row=row, scenario="fox")

    assert parsed.name is None
    assert parsed.ip is None
    assert parsed.host == "mail"
    assert parsed.short == "W-Sys-Cav"

import pytest

from thesis.preprocessing.parsing import parse_incoming_alert
from thesis.schemas.preprocessing import IncomingAlert, ParsedAlert


def test_parse_alert_row_returns_parsed_alert_with_expected_values():
    alert = IncomingAlert(
        time=1642213952,
        name="Wazuh: ClamAV database update",
        ip="172.17.131.81",
        host="mail",
        short="W-Sys-Cav",
        time_label="false_positive",
        event_label="-",
    )

    parsed = parse_incoming_alert(
        alert=alert,
        scenario="fox",
        window_size_seconds=2,
        keep_raw=True,
    )

    assert isinstance(parsed, ParsedAlert)

    assert parsed.ts == 1642213952
    assert parsed.window_id == 821106976

    assert parsed.name == "Wazuh: ClamAV database update"
    assert parsed.ip == "172.17.131.81"
    assert parsed.host == "mail"
    assert parsed.short == "W-Sys-Cav"

    assert parsed.raw == {
        "time": 1642213952,
        "name": "Wazuh: ClamAV database update",
        "ip": "172.17.131.81",
        "host": "mail",
        "short": "W-Sys-Cav",
        "time_label": "false_positive",
        "event_label": "-",
    }

    assert isinstance(parsed.alert_id, str)
    assert len(parsed.alert_id) == 40


def test_parse_alert_row_normalizes_missing_values():
    alert = IncomingAlert(
        time=1642213952,
        name="",
        ip=None,
        host="mail",
        short="W-Sys-Cav",
        time_label="false_positive",
        event_label="-",
    )

    parsed = parse_incoming_alert(alert=alert, scenario="fox")

    assert parsed.name is None
    assert parsed.ip is None
    assert parsed.host == "mail"
    assert parsed.short == "W-Sys-Cav"


def test_parse_alert_row_raises_on_invalid_timestamp():
    alert = IncomingAlert(
        time="not_a_timestamp",
        name="Wazuh: ClamAV database update",
        ip="172.17.131.81",
        host="mail",
        short="W-Sys-Cav",
        time_label="false_positive",
        event_label="-",
    )

    with pytest.raises(ValueError):
        parse_incoming_alert(alert=alert, scenario="fox")


def test_incoming_alert_missing_time_raises_type_error():
    with pytest.raises(TypeError):
        IncomingAlert(
            name="Wazuh: ClamAV database update",
            ip="172.17.131.81",
            host="mail",
            short="W-Sys-Cav",
            time_label="false_positive",
            event_label="-",
        )

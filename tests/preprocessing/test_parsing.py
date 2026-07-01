import pytest

from thesis.preprocessing.parsing import (
    normalize_missing_value,
    parse_incoming_alert,
)
from thesis.schemas.preprocessing import IncomingAlert, ParsedAlert


def test_parse_alert_row_returns_parsed_alert_with_expected_values():
    alert = IncomingAlert(
        time=1642213952,
        signature="Wazuh: ClamAV database update",
        ip="172.17.131.81",
        host="mail",
        short="W-Sys-Cav",
        label="false_positive",
        event_label="-",
    )

    parsed = parse_incoming_alert(
        alert=alert,
        scenario="fox",
        keep_raw=True,
    )

    assert isinstance(parsed, ParsedAlert)

    assert parsed.ts == 1642213952

    assert parsed.signature == "Wazuh: ClamAV database update"
    assert parsed.ip == "172.17.131.81"
    assert parsed.host == "mail"
    assert parsed.short == "W-Sys-Cav"

    assert parsed.raw == {
        "time": 1642213952,
        "signature": "Wazuh: ClamAV database update",
        "ip": "172.17.131.81",
        "host": "mail",
        "short": "W-Sys-Cav",
        "label": "false_positive",
        "event_label": "-",
    }

    assert isinstance(parsed.alert_id, str)
    assert len(parsed.alert_id) == 40


def test_parse_alert_row_normalizes_missing_values():
    alert = IncomingAlert(
        time=1642213952,
        signature="",
        ip=None,
        host="mail",
        short="W-Sys-Cav",
        label="false_positive",
        event_label="-",
    )

    parsed = parse_incoming_alert(alert=alert, scenario="fox")

    assert parsed.signature is None
    assert parsed.ip is None
    assert parsed.host == "mail"
    assert parsed.short == "W-Sys-Cav"


def test_parse_alert_row_raises_on_invalid_timestamp():
    alert = IncomingAlert(
        time="not_a_timestamp",
        signature="Wazuh: ClamAV database update",
        ip="172.17.131.81",
        host="mail",
        short="W-Sys-Cav",
        label="false_positive",
        event_label="-",
    )

    with pytest.raises(ValueError):
        parse_incoming_alert(alert=alert, scenario="fox")


def test_incoming_alert_missing_time_raises_type_error():
    with pytest.raises(TypeError):
        IncomingAlert(
            signature="Wazuh: ClamAV database update",
            ip="172.17.131.81",
            host="mail",
            short="W-Sys-Cav",
            label="false_positive",
            event_label="-",
        )


# ---------------------------------------------------------------------------
# normalize_missing_value
# ---------------------------------------------------------------------------


def test_normalize_missing_value_returns_none_for_none():
    assert normalize_missing_value(None) is None


def test_normalize_missing_value_returns_none_for_empty_string():
    assert normalize_missing_value("") is None


def test_normalize_missing_value_returns_none_for_whitespace():
    assert normalize_missing_value("   ") is None


def test_normalize_missing_value_strips_whitespace():
    assert normalize_missing_value("  mail  ") == "mail"


def test_normalize_missing_value_passes_through_normal_value():
    assert normalize_missing_value("172.17.0.1") == "172.17.0.1"


# ---------------------------------------------------------------------------
# IncomingAlert.from_row
# ---------------------------------------------------------------------------


def test_from_row_creates_incoming_alert_from_dict():
    row = {
        "time": 1642213952,
        "signature": "Wazuh: ClamAV database update",
        "ip": "172.17.131.81",
        "host": "mail",
        "short": "W-Sys-Cav",
        "label": "false_positive",
        "event_label": "-",
    }

    alert = IncomingAlert.from_row(row)

    assert alert.time == 1642213952
    assert alert.signature == "Wazuh: ClamAV database update"
    assert alert.ip == "172.17.131.81"
    assert alert.host == "mail"
    assert alert.short == "W-Sys-Cav"
    assert alert.label == "false_positive"
    assert alert.event_label == "-"


def test_from_row_allows_missing_optional_label_fields():
    row = {
        "time": 1642213952,
        "signature": "Wazuh: ClamAV database update",
        "ip": "172.17.131.81",
        "host": "mail",
        "short": "W-Sys-Cav",
    }

    alert = IncomingAlert.from_row(row)

    assert alert.label is None
    assert alert.event_label is None


def test_from_row_raises_on_missing_required_field():
    row = {
        "signature": "Wazuh: ClamAV database update",
        "ip": "172.17.131.81",
        "host": "mail",
        "short": "W-Sys-Cav",
    }

    with pytest.raises(ValueError, match="missing fields"):
        IncomingAlert.from_row(row)

import pandas as pd

from thesis.preprocessing.tokenization import (
    build_feature_tokens,
    extract_signature_tokens,
    normalize_text,
    tokenize_alert,
    tokenize_name_to_signature_substrings,
)
from thesis.schemas.preprocessing import ParsedAlert, TokenizedAlert


def make_parsed_alert(
    *,
    signature: str | None = "Wazuh: ClamAV database update",
    ip: str | None = "172.17.131.81",
    host: str | None = "mail",
    short: str | None = "W-Sys-Cav",
) -> ParsedAlert:
    return ParsedAlert(
        alert_id="abc123",
        ts=1642213952,
        time_norm=pd.Timestamp("2022-01-15 09:12:32+00:00"),
        signature=signature,
        ip=ip,
        host=host,
        short=short,
        raw={
            "time": 1642213952,
            "signature": signature,
            "ip": ip,
            "host": host,
            "short": short,
            "label": "false_positive",
            "event_label": "-",
        },
    )


def test_normalize_text():
    text = "Wazuh: ClamAV database update"
    assert normalize_text(text) == "wazuh clamav database update"


def test_extract_signature_tokens_filters_by_whitelist_and_blacklist():
    signature = "Wazuh: ClamAV database update"
    tokens = extract_signature_tokens(signature)

    assert tokens == {"database", "update"}


def test_extract_signature_tokens_returns_empty_set_for_missing_value():
    assert extract_signature_tokens(None) == set()


def test_tokenize_name_to_signature_substrings():
    name = "Wazuh: ClamAV database update"
    tokens = tokenize_name_to_signature_substrings(name)

    assert tokens == {"sig:database", "sig:update"}


def test_build_feature_tokens():
    alert = make_parsed_alert()

    tokens = build_feature_tokens(alert)

    assert tokens == {
        "short:W-Sys-Cav",
        "host:mail",
        "sig:database",
        "sig:update",
    }


def test_build_feature_tokens_omits_missing_fields():
    alert = make_parsed_alert(signature=None, host="mail", short=None)

    tokens = build_feature_tokens(alert)

    assert tokens == {"host:mail"}


def test_tokenize_alert_returns_tokenized_alert_with_expected_values():
    alert = make_parsed_alert()

    tokenized = tokenize_alert(alert)

    assert isinstance(tokenized, TokenizedAlert)

    assert tokenized.alert_id == alert.alert_id
    assert tokenized.ts == alert.ts
    assert tokenized.time_norm == alert.time_norm
    assert tokenized.signature == alert.signature
    assert tokenized.ip == alert.ip
    assert tokenized.host == alert.host
    assert tokenized.short == alert.short

    assert tokenized.tokens == {
        "short:W-Sys-Cav",
        "host:mail",
        "sig:database",
        "sig:update",
    }

    assert tokenized.raw == alert.raw
    assert tokenized.raw is not alert.raw

from __future__ import annotations

from typing import Any
import hashlib
import pandas as pd
from thesis.schemas.preprocessing import IncomingAlert, ParsedAlert


def normalize_missing_value(value: Any) -> str | None:
    """
    Normalize missing-like values to None and cast others to stripped strings.
    """
    if value is None:
        return None
    if pd.isna(value):
        return None

    value_str = str(value).strip()
    if value_str == "":
        return None

    return value_str


def normalize_row_timestamp(value: object) -> tuple[int, pd.Timestamp]:
    ts = pd.to_numeric(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value!r}")

    ts_int = int(ts)
    time_norm = pd.to_datetime(ts_int, unit="s", utc=True, errors="coerce")

    if pd.isna(time_norm):
        raise ValueError(f"Could not normalize timestamp: {value!r}")

    return ts_int, time_norm


def assign_window_id(ts: int, window_size_seconds: int = 2) -> int:
    """
    Assign a fixed window ID based on epoch seconds.
    """
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be > 0")

    return ts // window_size_seconds


def make_alert_id(
    ts: int,
    alert: IncomingAlert,
    scenario: str,
) -> str:
    """
    Build a stable alert ID for IncomingAlert object and scenario name.
    Returns a SHA-1 hash of the concatenated scenario and alert fields.
    """
    key = "|".join(
        [
            str(scenario),
            str(ts),
            "" if alert.name is None else str(alert.name),
            "" if alert.ip is None else str(alert.ip),
            "" if alert.host is None else str(alert.host),
            "" if alert.short is None else str(alert.short),
        ]
    )

    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def parse_alert_row(
    alert: IncomingAlert,
    scenario: str,
    window_size_seconds: int = 2,
    keep_raw: bool = True,
) -> ParsedAlert:
    """
    Parse one incoming alert into a ParsedAlert object.
    """
    ts, time_norm = normalize_row_timestamp(alert.time)
    window_id = assign_window_id(ts=ts, window_size_seconds=window_size_seconds)

    alert_id = make_alert_id(
        ts=ts,
        alert=alert,
        scenario=scenario,
    )
    raw = (
        {
            "time": alert.time,
            "name": alert.name,
            "ip": alert.ip,
            "host": alert.host,
            "short": alert.short,
            "time_label": alert.time_label,
            "event_label": alert.event_label,
        }
        if keep_raw
        else {}
    )

    parsed = ParsedAlert(
        alert_id=alert_id,
        ts=ts,
        time_norm=time_norm,
        window_id=window_id,
        name=normalize_missing_value(alert.name),
        ip=normalize_missing_value(alert.ip),
        host=normalize_missing_value(alert.host),
        short=normalize_missing_value(alert.short),
        time_label=normalize_missing_value(alert.time_label),
        event_label=normalize_missing_value(alert.event_label),
        raw=raw,
    )

    return parsed

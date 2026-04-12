from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import pandas as pd


@dataclass(slots=True)
class ParsedAlert:
    """
    Canonical parsed representation of one incoming alert.

    This object should contain only normalized alert-level fields.
    The alert_id is assigned by the parser and is unique for each alert.
    """

    alert_id: str
    ts: float
    window_id: int

    name: str | None = None
    ip: str | None = None
    host: str | None = None
    short: str | None = None

    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "ts": self.ts,
            "window_id": self.window_id,
            "name": self.name,
            "ip": self.ip,
            "host": self.host,
            "short": self.short,
            "raw": self.raw,
        }


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


def normalize_row_timestamp(
    row: dict[str, Any] | pd.Series,
    time_col: str = "time",
) -> tuple[int, pd.Timestamp]:
    """
    Normalize one row timestamp to integer epoch seconds and UTC datetime.
    """
    if time_col not in row:
        raise ValueError(
            f"normalize_row_timestamp() missing required field: '{time_col}'"
        )

    ts = pd.to_numeric(row[time_col], errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp in field '{time_col}': {row[time_col]!r}")

    ts_int = int(ts)
    time_norm = pd.to_datetime(ts_int, unit="s", utc=True, errors="coerce")

    if pd.isna(time_norm):
        raise ValueError(f"Could not normalize timestamp: {row[time_col]!r}")

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
    row: dict[str, Any] | pd.Series,
    scenario: str,
) -> str:
    """
    Build a stable alert ID for one parsed incoming alert row (dict object) and scenario name.
    Returns a SHA-1 hash of the concatenated scenario and alert fields.

    Expected row keys:
    - time
    - name
    - ip
    - host
    - short
    """
    required = ["time", "name", "ip", "host", "short"]
    missing = [c for c in required if c not in row]
    if missing:
        raise ValueError(
            f"make_alert_id() missing required fields: {missing}. "
            f"Available fields: {list(row.keys())}"
        )

    key = "|".join(
        [
            str(scenario),
            str(ts),  # use normalized timestamp
            str(row.get("name", "")),
            str(row.get("ip", "")),
            str(row.get("host", "")),
            str(row.get("short", "")),
        ]
    )

    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def parse_alert_row(
    row: dict[str, Any] | pd.Series,
    scenario: str,
    time_col: str = "time",
    window_size_seconds: int = 2,
    keep_raw: bool = True,
) -> ParsedAlert:
    """
    Parse one incoming alert row into a ParsedAlert object.

    Expected fields:
    - time
    - name
    - ip
    - host
    - short
    """
    required = [time_col, "name", "ip", "host", "short"]
    missing = [c for c in required if c not in row]
    if missing:
        available = list(row.index) if hasattr(row, "index") else list(row.keys())
        raise ValueError(
            f"parse_alert_row() missing required fields: {missing}. "
            f"Available fields: {available}"
        )

    ts, time_norm = normalize_row_timestamp(row=row, time_col=time_col)
    window_id = assign_window_id(ts=ts, window_size_seconds=window_size_seconds)

    ts, time_norm = normalize_row_timestamp(row, time_col)

    alert_id = make_alert_id(
        ts=ts,
        row=row,
        scenario=scenario,
    )

    parsed = ParsedAlert(
        alert_id=alert_id,
        ts=ts,
        time_norm=time_norm,
        window_id=window_id,
        name=normalize_missing_value(row["name"]),
        ip=normalize_missing_value(row["ip"]),
        host=normalize_missing_value(row["host"]),
        short=normalize_missing_value(row["short"]),
        raw=dict(row) if keep_raw else {},
    )

    return parsed

import pandas as pd
from typing import Any
from dataclasses import dataclass, field


@dataclass(slots=True)
class IncomingAlert:
    """
    Raw incoming alert as received by the preprocessing module.

    External data format before parsing/normalization.
    """

    time: Any
    name: str | None
    ip: str | None
    host: str | None
    short: str | None
    time_label: str | None
    event_label: str | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "IncomingAlert":
        """
        Create IncomingAlert from a dict-like row.
        """
        required = [
            "time",
            "name",
            "ip",
            "host",
            "short",
            # "time_label",
            # "event_label",
        ]
        missing = [c for c in required if c not in row]
        if missing:
            raise ValueError(
                f"IncomingAlert.from_row() missing fields: {missing}. "
                f"Available: {list(row.keys())}"
            )

        return cls(
            time=row["time"],
            name=row.get("name"),
            ip=row.get("ip"),
            host=row.get("host"),
            short=row.get("short"),
            time_label=row.get("time_label"),
            event_label=row.get("event_label"),
        )


@dataclass(slots=True)
class ParsedAlert:
    """
    Canonical parsed representation of one incoming alert.

    This object should contain only normalized alert-level fields.
    The alert_id is assigned by the parser and is unique for each alert.
    """

    alert_id: str
    ts: float
    time_norm: pd.Timestamp
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
            "time_norm": self.time_norm,
            "window_id": self.window_id,
            "name": self.name,
            "ip": self.ip,
            "host": self.host,
            "short": self.short,
            "raw": self.raw,
        }


@dataclass(slots=True)
class TokenizedAlert:
    """
    Parsed alert enriched with token views for downstream use.
    """

    alert_id: str
    ts: int
    time_norm: Any
    window_id: int

    name: str | None
    ip: str | None
    host: str | None
    short: str | None

    repr_tokens: set[str] = field(default_factory=set)
    mining_tokens: set[str] = field(default_factory=set)

    raw: dict[str, Any] = field(default_factory=dict)

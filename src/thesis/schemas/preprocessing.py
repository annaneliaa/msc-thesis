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
    signature: str | None
    ip: str | None
    host: str | None
    short: str | None
    label: str | None
    event_label: str | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "IncomingAlert":
        """
        Create IncomingAlert from a dict-like row.

        Accepts both the canonical alerts.json field names and the raw
        AIT-ADS CSV column names ("name", "time_label"), so stale
        alerts.json caches written before the canonical rename existed
        still parse correctly.
        """
        if "signature" not in row and "name" in row:
            row = {**row, "signature": row["name"]}
        if "label" not in row and "time_label" in row:
            row = {**row, "label": row["time_label"]}

        required = [
            "time",
            "signature",
            "ip",
            "host",
            "short",
            # "label",
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
            signature=row.get("signature"),
            ip=row.get("ip"),
            host=row.get("host"),
            short=row.get("short"),
            label=row.get("label"),
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

    signature: str | None = None
    ip: str | None = None
    host: str | None = None
    short: str | None = None
    label: str | None = None
    event_label: str | None = None

    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "ts": self.ts,
            "time_norm": self.time_norm,
            "signature": self.signature,
            "ip": self.ip,
            "host": self.host,
            "short": self.short,
            "label": self.label,
            "event_label": self.event_label,
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

    signature: str | None
    ip: str | None
    host: str | None
    short: str | None

    tokens: set[str] = field(default_factory=set)

    label: str | None = None
    event_label: str | None = None

    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IncomingSuricataGroup:
    """
    One row from the Suricata pre-grouped dataset.

    Each row represents a cluster of alerts sharing the same signature and
    external IP, already aggregated by the source dataset. There are no
    individual alert IDs — only the aggregate count and label are available.
    """

    timestamp: str  # ISO 8601 with timezone, e.g. "2022-01-20T00:00:03+02:00"
    signature_text: str  # e.g. "ET EXPLOIT D-Link ..."
    signature_id: int
    alert_count: int
    proto: int
    ext_ip: str  # anonymised external IP, e.g. "extip1"
    ext_port: int
    int_ip: str | None  # anonymised internal IP; None when not available
    int_port: int
    label: int  # 0 = benign, 1 = attack

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "IncomingSuricataGroup":
        required = [
            "Timestamp",
            "SignatureText",
            "SignatureID",
            "AlertCount",
            "Proto",
            "ExtIP",
            "ExtPort",
            "IntIP",
            "IntPort",
            "Label",
        ]
        missing = [c for c in required if c not in row]
        if missing:
            raise ValueError(
                f"SuricataGroupRow.from_row() missing fields: {missing}. "
                f"Available: {list(row.keys())}"
            )
        int_ip_raw = str(row["IntIP"]).strip()
        return cls(
            timestamp=str(row["Timestamp"]),
            signature_text=str(row["SignatureText"]),
            signature_id=int(row["SignatureID"]),
            alert_count=int(row["AlertCount"]),
            proto=int(row["Proto"]),
            ext_ip=str(row["ExtIP"]),
            ext_port=int(row["ExtPort"]),
            int_ip=None if int_ip_raw in ("-1", "") else int_ip_raw,
            int_port=int(row["IntPort"]),
            label=int(row["Label"]),
        )


@dataclass(slots=True)
class ParsedSuricataGroup:
    """
    Parsed representation of one pre-grouped Suricata row.

    Produced by parse_suricata_group_row; consumed by the caching module to
    build a GroupCacheEntry. No cache-specific fields are present here.
    """

    group_id: str
    ts: int
    n_alerts: int
    tokens: set[str]
    proto: int
    ext_ip: str
    ext_ip_is_multiple: bool
    ext_port: int
    int_ip: str | None
    int_ip_is_multiple: bool
    int_port: int
    label: str  # "benign" | "attack"

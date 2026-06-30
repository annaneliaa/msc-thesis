import pandas as pd
from typing import Any, Optional
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

    name: str | None = None
    ip: str | None = None
    host: str | None = None
    short: str | None = None
    time_label: str | None = None
    event_label: str | None = None

    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "ts": self.ts,
            "time_norm": self.time_norm,
            "name": self.name,
            "ip": self.ip,
            "host": self.host,
            "short": self.short,
            "time_label": self.time_label,
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

    name: str | None
    ip: str | None
    host: str | None
    short: str | None

    tokens: set[str] = field(default_factory=set)

    time_label: str | None = None
    event_label: str | None = None

    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GroupingRecord:
    alert_id: str
    group_id: str
    method: str  # "fixed_window"


@dataclass(slots=True)
class GroupSnapshot:  # stable snapshot
    # identity
    group_id: str
    method: str  # "fixed_window" | "alertbert"
    version: int

    # temporal scope
    start_ts: int
    end_ts: int

    # membership
    alert_ids: list[str] = field(default_factory=list)
    n_alerts: int = 0
    items: set[str] = field(
        default_factory=set
    )  # raw group items, pre-abstraction # unordered, deduplicated, for itemset mining
    sorted_items: list[set[str]] = field(
        default_factory=list
    )  # ordered list of per-alert itemsets, for sequence mining
    alert_ips: set[str] = field(default_factory=set)
    # labels (for evaluation)
    alert_labels: Optional[set[str]] = None
    group_label: Optional[str] = None

    # lifecycle
    status: str = "closed"  # expected: "closed" when emitted


@dataclass(slots=True)
class AlertGroup:  # mining input (with weight)
    alert_group_id: str
    group_id: str
    method: str  # "fixed_window" | "alertbert"

    start_ts: int
    end_ts: int

    n_alerts: int
    alert_ids: Optional[list[str]] = None
    abs_items: set[str] = field(default_factory=set)  # mining-ready abstracted itemset
    raw_items: Optional[set[str]] = None  # pre-abstraction mining items
    sorted_items: list[set[str]] = field(
        default_factory=list
    )  # ordered list of per-alert itemsets, for sequence mining
    alert_ips: set[str] = field(default_factory=set)

    group_label: Optional[str] = None
    alert_labels: Optional[set[str]] = None

    weight: float = 1.0

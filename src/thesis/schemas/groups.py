from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class GroupingRecord:
    alert_id: str
    group_id: str
    method: str  # "fixed_window"


@dataclass(slots=True)
class GroupSnapshot:  # stable snapshot
    # identity
    group_id: str
    method: str  # "fixed_window" | "alertbert" | "cscas_pregrouped"

    # temporal scope
    start_ts: int
    end_ts: int

    # membership
    version: int = 0
    alert_ids: list[str] = field(default_factory=list)
    n_alerts: int = 0
    items: set[str] = field(default_factory=set)
    sorted_items: list[set[str]] = field(
        default_factory=list
    )  # ordered list of per-alert itemsets, for sequence mining
    alert_ips: set[str] = field(default_factory=set)

    # labels (for evaluation)
    alert_labels: Optional[set[str]] = None
    group_label: Optional[str] = None

    # lifecycle
    status: str = "closed"  # expected: "closed" when emitted

    def to_alert_group(self) -> AlertGroup:
        return AlertGroup(
            alert_group_id=self.group_id,
            group_id=self.group_id,
            method=self.method,
            start_ts=self.start_ts,
            end_ts=self.end_ts,
            n_alerts=self.n_alerts,
            abs_items=set(self.items),
            raw_items=set(self.items),
            sorted_items=self.sorted_items,
            alert_ids=list(self.alert_ids),
            alert_ips=set(self.alert_ips),
            group_label=self.group_label,
            alert_labels=set(self.alert_labels)
            if self.alert_labels is not None
            else None,
            weight=1.0,
        )


@dataclass(slots=True)
class AlertGroup:  # encoding/experiment input (with weight)
    alert_group_id: str
    group_id: str
    method: str  # "fixed_window" | "alertbert" | "cscas_pregrouped"

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

    # CSCAS-only network metadata (None for AIT-ADS scenarios)
    proto: Optional[int] = None
    int_ip: Optional[str] = None
    int_port: Optional[int] = None
    ext_port: Optional[int] = None
    int_ip_is_multiple: bool = False
    ext_ip_is_multiple: bool = False

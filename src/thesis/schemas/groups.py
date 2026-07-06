from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class GroupingRecord:
    alert_id: str
    group_id: str
    method: str  # "fixed_window" | "cscas_pregrouped | cscas_grouping"


@dataclass(slots=True)
class GroupSnapshot:  # stable snapshot
    # identity
    group_id: str
    method: str  # "fixed_window" | "cscas_pregrouped | cscas_grouping"

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
    method: str  # "fixed_window" | "cscas_pregrouped"
    start_ts: int
    end_ts: Optional[int]
    n_alerts: int

    group_label: Optional[str] = None
    weight: float = 1.0
    raw_items: Optional[set[str]] = None  # mining items (tokens / signature words)

    # AIT-ADS only (None for CSCAS)
    alert_ids: Optional[list[str]] = None  # AIT-ADS only (None for CSCAS)
    sorted_items: Optional[list[set[str]]] = (
        None  # AIT-ADS only: ordered per-alert itemsets for sequence mining
    )
    alert_ips: Optional[set[str]] = None
    alert_labels: Optional[set[str]] = None

    # CSCAS-only network metadata (None for AIT-ADS scenarios)
    proto: Optional[int] = None
    ext_ip: Optional[str] = None
    int_ip: Optional[str] = None
    int_port: Optional[int] = None
    ext_port: Optional[int] = None
    int_ip_is_multiple: Optional[bool] = None
    ext_ip_is_multiple: Optional[bool] = None

    # CSCAS-only attribute-mining fields (None for AIT-ADS scenarios)
    category: Optional[str] = None
    ruleset: Optional[str] = None
    cve_refs: Optional[set[str]] = None
    qualifiers: Optional[set[str]] = None
    signature_matches_per_day: Optional[float] = None
    similarity: Optional[float] = None
    signature_id_similarity: Optional[float] = None
    attr_similarities: Optional[dict[str, float]] = None
    scas: Optional[int] = None
    ext_port_is_multiple: Optional[bool] = None
    int_port_is_multiple: Optional[bool] = None

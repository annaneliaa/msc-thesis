from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class GroupCacheEntry:
    group_id: str
    method: str  # "fixed_window | cscas_pregrouped | cscas_grouping"
    status: str  # "open" | "stale" | "closed" | "mined"

    last_update_ts: int
    start_ts: int
    end_ts: int

    alert_ids: list[str] = field(default_factory=list)
    n_alerts: int = 0

    items: set[str] = field(
        default_factory=set
    )  # union of raw mining items from member alerts
    sorted_items: list[set[str]] = field(default_factory=list)

    alert_ips: set[str] = field(default_factory=set)
    # group_features_summary: dict[str, set[str]] = field(default_factory=dict)
    # embedding_centroid: Optional[list[float]] = None

    alert_labels: Optional[set[str]] = None
    group_label: Optional[str] = None

    version: int = 1
    mined_at: Optional[int] = None


@dataclass(slots=True)
class CacheQuery:
    # grouping / experiment control
    allowed_methods: Optional[set[str]] = None  # {"fixed_window"}

    # lifecycle filtering
    only_closed: bool = True
    allowed_statuses: Optional[set[str]] = None  # {"open", "stale", "closed", "mined"}

    # time-based filtering
    min_start_ts: Optional[int] = None
    max_end_ts: Optional[int] = None

    # optional limits
    limit: Optional[int] = None  # max number of groups to return


@dataclass(slots=True)
class CacheResponse:
    groups: list["GroupCacheEntry"] = field(default_factory=list)

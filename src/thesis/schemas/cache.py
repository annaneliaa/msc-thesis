from dataclasses import dataclass, field


@dataclass(slots=True)
class AlertCacheEntry:
    alert_id: str
    ts: int
    window_id: int
    repr_tokens: set[str] = field(default_factory=set)
    mining_tokens: set[str] = field(default_factory=set)
    ip: str | None = None
    host: str | None = None
    short: str | None = None


@dataclass(slots=True)
class WindowCacheEntry:
    window_id: int
    start_ts: int
    end_ts: int
    alert_ids: list[str] = field(default_factory=list)
    items: set[str] = field(default_factory=set)
    hosts: set[str] = field(default_factory=set)
    signatures: set[str] = field(default_factory=set)
    closed: bool = False

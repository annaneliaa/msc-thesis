"""Per-host look-back sequences for the temporal arm (AIT-ADS), built once and
shared by every temporal miner (PrefixSpan, MCTSExtent).

One sequence per alert group: the alerts on the group's host from
`window_minutes` before the group's start up to and including its end,
bucketed into fixed time buckets (1 minute by default). Each bucket is one
step, holding the set of `short:<code>` items of its alerts; empty buckets are
skipped. The sequence carries the group's label.

Why not the group's own sorted_items: cscas_grouping groups hold one signature
on one host (median duration 0 s on fox), so there is no order inside a group
to mine. The order lives in what happened on the host before it.

w is fixed once for the comparison and not tuned (plan section 4).
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from thesis.paths import ARTIFACTS_DIR
from thesis.schemas.groups import AlertGroup

DEFAULT_WINDOW_MINUTES = 10
DEFAULT_BUCKET_SECONDS = 60
ITEM_PREFIX = "short:"


def load_ait_ads_raw_alerts(scenario: str, path: Path | None = None) -> pd.DataFrame:
    """Raw per-alert time/host/short for one AIT-ADS scenario, from the
    canonical alerts.json pipeline.convert_ait_alerts_to_json writes."""
    path = path or ARTIFACTS_DIR / "processed-data" / scenario / "alerts.json"
    with Path(path).open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(
        {
            "time": [int(r["time"]) for r in rows],
            "host": [r.get("host") or "" for r in rows],
            "short": [r.get("short") or "" for r in rows],
        }
    )


class HostAlertIndex:
    """Alerts per host, sorted by time, for fast window lookups."""

    def __init__(self, alerts: pd.DataFrame) -> None:
        by_host: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for t, h, s in zip(alerts["time"], alerts["host"], alerts["short"]):
            if h and s:
                by_host[h].append((int(t), str(s)))
        self._times: dict[str, list[int]] = {}
        self._shorts: dict[str, list[str]] = {}
        for h, rows in by_host.items():
            rows.sort()
            self._times[h] = [t for t, _ in rows]
            self._shorts[h] = [s for _, s in rows]

    def window(self, host: str, start: int, end: int) -> list[tuple[int, str]]:
        """(time, short) of alerts on `host` with start <= time <= end."""
        times = self._times.get(host)
        if not times:
            return []
        lo, hi = bisect_left(times, start), bisect_right(times, end)
        shorts = self._shorts[host]
        return [(times[i], shorts[i]) for i in range(lo, hi)]


def group_hosts(tx: AlertGroup) -> set[str]:
    """The group's host(s), from its `host:<name>` raw items."""
    return {
        item[len("host:") :]
        for item in (tx.raw_items or ())
        if item.startswith("host:")
    }


def lookback_steps(
    tx: AlertGroup,
    index: HostAlertIndex,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> list[set[str]]:
    """One group's look-back sequence: non-empty time buckets, in order."""
    end = tx.end_ts if tx.end_ts is not None else tx.start_ts
    start = tx.start_ts - window_minutes * 60
    buckets: dict[int, set[str]] = defaultdict(set)
    for host in group_hosts(tx):
        for t, short in index.window(host, start, end):
            buckets[t // bucket_seconds].add(f"{ITEM_PREFIX}{short}")
    return [buckets[b] for b in sorted(buckets)]


def attach_lookback_steps(
    alert_groups: Sequence[AlertGroup],
    alerts: pd.DataFrame,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> None:
    """Set `context_steps` on every group in place. The symbolic encoder then
    evaluates sequence features on it, so mining and encoding see the same
    sequence."""
    index = HostAlertIndex(alerts)
    for tx in alert_groups:
        tx.context_steps = lookback_steps(tx, index, window_minutes, bucket_seconds)

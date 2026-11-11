"""Per-host look-back sequences for the temporal arm (AIT-ADS), built once and
shared by every temporal miner (PrefixSpan, MCTSExtent).

One sequence per alert group: the alerts on the group's host from
`window_minutes` before the group's start up to and including its end,
bucketed into fixed time buckets (1 minute by default). Each bucket is one
step, holding the set of items of its alerts; empty buckets are skipped. The
sequence carries the group's label.

Items per alert follow the token arm (`token_fields`): `short:<code>` (T0) plus,
for the coarse arms, `short_src:<src>`, `short_cat:<src-cat>`,
`short_family:<family>` and `sig_family:<family>` -- the same coarse levels the
atemporal item table gets (common/attribute_features.py), so every method mines
from the same tokens.

Why not the group's own sorted_items: cscas_grouping groups hold one signature
on one host (median duration 0 s on fox), so there is no order inside a group
to mine. The order lives in what happened on the host before it.

w is fixed once for the comparison and not tuned (plan section 4).
"""

from __future__ import annotations

import json
from functools import lru_cache
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
# Token fields an alert can contribute to a look-back step.
LOOKBACK_TOKEN_FIELDS = frozenset(
    {"short", "short_src", "short_cat", "short_family", "sig_family"}
)
DEFAULT_TOKEN_FIELDS = frozenset({"short"})


@lru_cache(maxsize=None)
def _sig_families(signature: str) -> frozenset[str]:
    """Abstraction-map families of an alert name's whitelisted sig keywords."""
    from thesis.mining.common.attribute_features import _abstraction_families
    from thesis.preprocessing.tokenization import (
        tokenize_name_to_signature_substrings,
    )

    families = _abstraction_families()
    return frozenset(
        families[t]
        for t in tokenize_name_to_signature_substrings(signature)
        if t in families
    )


def alert_tokens(
    short: str, signature: str = "", fields: frozenset[str] = DEFAULT_TOKEN_FIELDS
) -> frozenset[str]:
    """One alert's look-back items for the given token fields."""
    from thesis.mining.common.attribute_features import (
        _abstraction_families,
        _short_parts,
    )

    unknown = set(fields) - LOOKBACK_TOKEN_FIELDS
    if unknown:
        raise ValueError(f"unknown look-back token fields: {sorted(unknown)}")
    out: set[str] = set()
    if short:
        src, cat = _short_parts(short)
        if "short" in fields:
            out.add(f"{ITEM_PREFIX}{short}")
        if "short_src" in fields:
            out.add(f"short_src:{src}")
        if "short_cat" in fields:
            out.add(f"short_cat:{cat}")
        if "short_family" in fields:
            fam = _abstraction_families().get(f"short:{short}")
            if fam:
                out.add(f"short_family:{fam}")
    if "sig_family" in fields and signature:
        out |= {f"sig_family:{f}" for f in _sig_families(signature)}
    return frozenset(out)


@lru_cache(maxsize=4)
def load_ait_ads_raw_alerts(scenario: str, path: Path | None = None) -> pd.DataFrame:
    """Raw per-alert time/host/short/signature for one AIT-ADS scenario, from
    the canonical alerts.json pipeline.convert_ait_alerts_to_json writes.
    Falls back to the raw CSV (data/alerts_csv/<scenario>_alerts.txt; the alert
    name is its `name` column) when that JSON hasn't been generated. Cached:
    treat the returned frame as read-only."""
    path = path or ARTIFACTS_DIR / "processed-data" / scenario / "alerts.json"
    if not Path(path).exists():
        csv = ARTIFACTS_DIR.parent / "data" / "alerts_csv" / f"{scenario}_alerts.txt"
        df = pd.read_csv(
            csv,
            usecols=["time", "name", "host", "short"],
            dtype={"name": str, "host": str, "short": str},
        )
        return pd.DataFrame(
            {
                "time": df["time"].astype(int),
                "host": df["host"].fillna(""),
                "short": df["short"].fillna(""),
                "signature": df["name"].fillna(""),
            }
        )
    with Path(path).open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(
        {
            "time": [int(r["time"]) for r in rows],
            "host": [r.get("host") or "" for r in rows],
            "short": [r.get("short") or "" for r in rows],
            "signature": [r.get("signature") or "" for r in rows],
        }
    )


class HostAlertIndex:
    """Alerts per host, sorted by time, each with its look-back items for the
    given token fields, for fast window lookups."""

    def __init__(
        self, alerts: pd.DataFrame, token_fields: frozenset[str] = DEFAULT_TOKEN_FIELDS
    ) -> None:
        token_fields = frozenset(token_fields)
        signatures = (
            alerts["signature"] if "signature" in alerts else [""] * len(alerts)
        )
        cache: dict[tuple[str, str], frozenset[str]] = {}
        by_host: dict[str, list[tuple[int, frozenset[str]]]] = defaultdict(list)
        for t, h, s, sig in zip(
            alerts["time"], alerts["host"], alerts["short"], signatures
        ):
            if not (h and s):
                continue
            key = (str(s), str(sig or ""))
            if key not in cache:
                cache[key] = alert_tokens(key[0], key[1], token_fields)
            by_host[h].append((int(t), cache[key]))
        self._times: dict[str, list[int]] = {}
        self._items: dict[str, list[frozenset[str]]] = {}
        for h, rows in by_host.items():
            rows.sort(key=lambda r: r[0])
            self._times[h] = [t for t, _ in rows]
            self._items[h] = [i for _, i in rows]

    def window(
        self, host: str, start: int, end: int
    ) -> list[tuple[int, frozenset[str]]]:
        """(time, items) of alerts on `host` with start <= time <= end."""
        times = self._times.get(host)
        if not times:
            return []
        lo, hi = bisect_left(times, start), bisect_right(times, end)
        items = self._items[host]
        return [(times[i], items[i]) for i in range(lo, hi)]


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
        for t, items in index.window(host, start, end):
            buckets[t // bucket_seconds] |= items
    return [buckets[b] for b in sorted(buckets)]


def attach_lookback_steps(
    alert_groups: Sequence[AlertGroup],
    alerts: pd.DataFrame | None,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    token_fields: frozenset[str] = DEFAULT_TOKEN_FIELDS,
    index: HostAlertIndex | None = None,
) -> None:
    """Set `context_steps` on every group in place. The symbolic encoder then
    evaluates sequence features on it, so mining and encoding see the same
    sequence. Pass a prebuilt `index` to reuse it (its token fields win)."""
    if index is None:
        if alerts is None:
            raise ValueError("pass alerts or a prebuilt index")
        index = HostAlertIndex(alerts, token_fields)
    for tx in alert_groups:
        tx.context_steps = lookback_steps(tx, index, window_minutes, bucket_seconds)

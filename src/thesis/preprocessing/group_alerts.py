from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from thesis.schemas.preprocessing import GroupingRecord, TokenizedAlert

if TYPE_CHECKING:
    from thesis.preprocessing.grouping.alertbert_grouper import AlertBERTGrouper

FIXED_WINDOW_SECONDS = 2
FIXED_WINDOW_METHOD = "fixed_window"
FIXED_WINDOW_HOST_METHOD = "fixed_window_host"
TIME_DELTA_METHOD = "time_delta"
TIME_DELTA_HOST_METHOD = "time_delta_host"
TIME_DELTA_SECONDS = 2.0
ALERTBERT_METHOD = "alertbert"


def fixed_window_group_id(ts: int, window_size: int = FIXED_WINDOW_SECONDS) -> str:
    window_id = ts // window_size
    return f"fixed_window:{window_id}"


def group_alert_fixed_window(
    alert: TokenizedAlert,
    window_size: int = FIXED_WINDOW_SECONDS,
) -> GroupingRecord:
    return GroupingRecord(
        alert_id=alert.alert_id,
        group_id=fixed_window_group_id(alert.ts, window_size=window_size),
        method=FIXED_WINDOW_METHOD,
    )


def group_alerts_fixed_window(
    alerts: list[TokenizedAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> list[GroupingRecord]:
    return [
        group_alert_fixed_window(alert, window_size=window_size) for alert in alerts
    ]


def group_alerts_fixed_window_by_group(
    alerts: list[TokenizedAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> dict[str, list[TokenizedAlert]]:
    groups: dict[str, list[TokenizedAlert]] = defaultdict(list)

    for alert in alerts:
        group_id = fixed_window_group_id(alert.ts, window_size=window_size)
        groups[group_id].append(alert)

    return {gid: sorted(grp, key=lambda a: a.ts) for gid, grp in groups.items()}


def group_alerts_fixed_window_host(
    alerts: list[TokenizedAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> list[GroupingRecord]:
    """Per-host variant of fixed_window.

    Each alert is assigned to a group identified by (host, window_id) so that
    alerts from different machines are never merged even when they fall in the
    same calendar window. The host component is taken from alert.host; alerts
    with no host value are treated as a single anonymous host.
    """
    records: list[GroupingRecord] = []
    for alert in alerts:
        host = alert.host or "_unknown"
        window_id = alert.ts // window_size
        group_id = f"fixed_window_host:{host}:{window_id}"
        records.append(
            GroupingRecord(
                alert_id=alert.alert_id,
                group_id=group_id,
                method=FIXED_WINDOW_HOST_METHOD,
            )
        )
    return records


def group_alerts_time_delta(
    alerts: list[TokenizedAlert],
    delta: float = TIME_DELTA_SECONDS,
) -> list[GroupingRecord]:
    """Landauer et al. (2022) time-delta method.

    Sorts alerts by timestamp and starts a new group whenever the gap to the
    previous alert exceeds delta seconds. Equivalent to connected-components
    clustering with a time-only distance threshold on a 1-D sorted sequence.
    Unlike fixed_window, groups are bounded by stream gaps rather than absolute
    calendar windows, so a continuous stream with gaps < delta forms one group.
    """
    if not alerts:
        return []

    sorted_alerts = sorted(alerts, key=lambda a: a.ts)
    anchor_id = sorted_alerts[0].alert_id
    prev_ts: float = sorted_alerts[0].ts

    records: list[GroupingRecord] = []
    for alert in sorted_alerts:
        if alert.ts - prev_ts > delta:
            anchor_id = alert.alert_id
        prev_ts = alert.ts
        records.append(
            GroupingRecord(
                alert_id=alert.alert_id,
                group_id=f"time_delta:{anchor_id}",
                method=TIME_DELTA_METHOD,
            )
        )
    return records


def group_alerts_time_delta_host(
    alerts: list[TokenizedAlert],
    delta: float = TIME_DELTA_SECONDS,
) -> list[GroupingRecord]:
    """Per-host variant of the Landauer et al. time-delta method.

    Runs the same stream-gap scan as group_alerts_time_delta but independently
    per host, so a dense stream on one machine cannot absorb alerts from other
    machines. The host component is taken from alert.host; alerts with no host
    value are treated as a single anonymous host.
    """
    if not alerts:
        return []

    by_host: dict[str, list[TokenizedAlert]] = defaultdict(list)
    for alert in alerts:
        by_host[alert.host or "_unknown"].append(alert)

    records: list[GroupingRecord] = []
    for host, host_alerts in by_host.items():
        sorted_alerts = sorted(host_alerts, key=lambda a: a.ts)
        anchor_id = sorted_alerts[0].alert_id
        prev_ts: float = sorted_alerts[0].ts
        for alert in sorted_alerts:
            if alert.ts - prev_ts > delta:
                anchor_id = alert.alert_id
            prev_ts = alert.ts
            records.append(
                GroupingRecord(
                    alert_id=alert.alert_id,
                    group_id=f"time_delta_host:{host}:{anchor_id}",
                    method=TIME_DELTA_HOST_METHOD,
                )
            )
    return records


def group_alerts(
    alerts: list[TokenizedAlert],
    method: str = FIXED_WINDOW_METHOD,
    grouper: AlertBERTGrouper | None = None,
    **kwargs,
) -> list[GroupingRecord]:
    if method == FIXED_WINDOW_METHOD:
        return group_alerts_fixed_window(alerts, **kwargs)
    elif method == FIXED_WINDOW_HOST_METHOD:
        return group_alerts_fixed_window_host(alerts, **kwargs)
    elif method == TIME_DELTA_METHOD:
        return group_alerts_time_delta(alerts, **kwargs)
    elif method == TIME_DELTA_HOST_METHOD:
        return group_alerts_time_delta_host(alerts, **kwargs)
    elif method == ALERTBERT_METHOD:
        if grouper is None:
            raise ValueError(
                "grouper must be a loaded AlertBERTGrouper when method='alertbert'"
            )
        return grouper.group(alerts)
    else:
        raise ValueError(f"Unsupported grouping method: {method}")

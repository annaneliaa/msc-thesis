from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from thesis.schemas.preprocessing import GroupingRecord, TokenizedAlert

if TYPE_CHECKING:
    from thesis.preprocessing.grouping.alertbert_grouper import AlertBERTGrouper

FIXED_WINDOW_SECONDS = 2
FIXED_WINDOW_METHOD = "fixed_2s"
ALERTBERT_METHOD = "alertbert"


def fixed_2s_group_id(ts: int, window_size: int = FIXED_WINDOW_SECONDS) -> str:
    window_id = ts // window_size
    return f"fixed_2s:{window_id}"


def group_alert_fixed_2s(
    alert: TokenizedAlert,
    window_size: int = FIXED_WINDOW_SECONDS,
) -> GroupingRecord:
    return GroupingRecord(
        alert_id=alert.alert_id,
        group_id=fixed_2s_group_id(alert.ts, window_size=window_size),
        method=FIXED_WINDOW_METHOD,
    )


def group_alerts_fixed_2s(
    alerts: list[TokenizedAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> list[GroupingRecord]:
    return [group_alert_fixed_2s(alert, window_size=window_size) for alert in alerts]


def group_alerts_fixed_2s_by_group(
    alerts: list[TokenizedAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> dict[str, list[TokenizedAlert]]:
    groups: dict[str, list[TokenizedAlert]] = defaultdict(list)

    for alert in alerts:
        group_id = fixed_2s_group_id(alert.ts, window_size=window_size)
        groups[group_id].append(alert)

    return {gid: sorted(grp, key=lambda a: a.ts) for gid, grp in groups.items()}


def group_alerts(
    alerts: list[TokenizedAlert],
    method: str = FIXED_WINDOW_METHOD,
    grouper: AlertBERTGrouper | None = None,
    **kwargs,
) -> list[GroupingRecord]:
    if method == FIXED_WINDOW_METHOD:
        return group_alerts_fixed_2s(alerts, **kwargs)
    elif method == ALERTBERT_METHOD:
        if grouper is None:
            raise ValueError(
                "grouper must be a loaded AlertBERTGrouper when method='alertbert'"
            )
        return grouper.group(alerts)
    else:
        raise ValueError(f"Unsupported grouping method: {method}")

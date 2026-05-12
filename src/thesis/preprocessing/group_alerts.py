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
    """
    Map an alert timestamp to a fixed-width 2-second group id.
    """
    window_id = ts // window_size
    return f"fixed_2s:{window_id}"


def group_alert_fixed_2s(
    alert: TokenizedAlert,
    window_size: int = FIXED_WINDOW_SECONDS,
) -> GroupingRecord:
    """
    Group a single TokenizedAlert into a fixed-width time window.
    """
    return GroupingRecord(
        alert_id=alert.alert_id,
        group_id=fixed_2s_group_id(alert.ts, window_size=window_size),
        method=FIXED_WINDOW_METHOD,
    )


def group_alerts_fixed_2s(
    alerts: list[TokenizedAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> list[GroupingRecord]:
    """
    Group a batch of TokenizedAlerts into fixed-width 2-second windows.
    Returns one GroupingRecord per alert.
    """
    return [group_alert_fixed_2s(alert, window_size=window_size) for alert in alerts]


def group_alerts_fixed_2s_by_group(
    alerts: list[TokenizedAlert],
    window_size: int = FIXED_WINDOW_SECONDS,
) -> dict[str, list[TokenizedAlert]]:
    """
    Convenience helper:
    returns alerts bucketed by fixed-width 2-second group id.
    """
    groups: dict[str, list[TokenizedAlert]] = defaultdict(list)

    for alert in alerts:
        group_id = fixed_2s_group_id(alert.ts, window_size=window_size)
        groups[group_id].append(alert)

    # return sorted group based on timestamp
    return {gid: sorted(grp, key=lambda a: a.ts) for gid, grp in groups.items()}


def group_alerts_alertbert(
    alerts: list[TokenizedAlert],
    grouper: AlertBERTGrouper,
) -> list[GroupingRecord]:
    """Group alerts using a pre-loaded AlertBERTGrouper."""
    return grouper.group(alerts)


def group_alerts(
    alerts: list[TokenizedAlert],
    method: str = FIXED_WINDOW_METHOD,
    grouper: AlertBERTGrouper | None = None,
    **kwargs,
) -> list[GroupingRecord]:
    """Main entry point for alert grouping.

    Dispatches to the chosen grouping strategy.  Pass a pre-loaded
    AlertBERTGrouper as `grouper` when method='alertbert'.
    """
    if method == FIXED_WINDOW_METHOD:
        return group_alerts_fixed_2s(alerts, **kwargs)
    elif method == ALERTBERT_METHOD:
        if grouper is None:
            raise ValueError(
                "grouper must be a loaded AlertBERTGrouper when method='alertbert'"
            )
        return group_alerts_alertbert(alerts, grouper)
    else:
        raise ValueError(f"Unsupported grouping method: {method}")

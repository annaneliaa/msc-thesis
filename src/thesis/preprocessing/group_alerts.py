from collections import defaultdict

from thesis.schemas.preprocessing import TokenizedAlert, GroupingRecord


FIXED_WINDOW_SECONDS = 2
FIXED_WINDOW_METHOD = "fixed_2s"


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

    return dict(groups)


def group_alerts(
    alerts: list[TokenizedAlert],
    method: str = FIXED_WINDOW_METHOD,
    **kwargs,
) -> list[GroupingRecord]:
    """
    Main entry point for alert grouping.
    Dispatches to specific grouping methods based on the 'method' argument.
    """
    if method == FIXED_WINDOW_METHOD:
        return group_alerts_fixed_2s(alerts, **kwargs)
    else:
        raise ValueError(f"Unsupported grouping method: {method}")

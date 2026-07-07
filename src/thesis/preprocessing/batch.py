from __future__ import annotations

from thesis.grouping import group_alerts
from thesis.preprocessing.parsing import parse_incoming_alert
from thesis.preprocessing.tokenization import tokenize_alert
from thesis.schemas.groups import GroupingRecord
from thesis.schemas.preprocessing import IncomingAlert, ParsedAlert, TokenizedAlert


def process_alert_rows(
    rows: list[dict],
    scenario: str,
    grouping_mode: str | None = None,
    **grouping_kwargs,
) -> tuple[list[TokenizedAlert], list[GroupingRecord]]:
    """
    Parse incoming alert rows and tokenize them, with an optional grouping
    step in between: parse -> group -> tokenize.

    Grouping runs on the parsed alerts, before tokenization, via
    thesis.grouping. Pass grouping_mode=None (the default) to skip it
    entirely, e.g. for sources that arrive as pre-closed groups (CSCAS),
    where alert-level grouping doesn't apply.
    """
    parsed_alerts: list[ParsedAlert] = []
    for row in rows:
        try:
            alert = IncomingAlert.from_row(row)
            parsed_alerts.append(parse_incoming_alert(alert=alert, scenario=scenario))
        except Exception as e:
            print(f"Skipping row due to parsing error: {e}")
            continue

    grouping_records: list[GroupingRecord] = []
    if grouping_mode is not None:
        grouping_records = group_alerts(
            parsed_alerts, method=grouping_mode, **grouping_kwargs
        )

    tokenized_alerts = [tokenize_alert(alert) for alert in parsed_alerts]
    return tokenized_alerts, grouping_records

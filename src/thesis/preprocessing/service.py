from thesis.schemas.parsing import IncomingAlert, ParsedAlert
from thesis.preprocessing.parsing import parse_alert_row


def preprocess_alert(
    alert: IncomingAlert,
    scenario: str,
) -> ParsedAlert:
    return parse_alert_row(alert=alert, scenario=scenario)

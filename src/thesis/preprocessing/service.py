from thesis.schemas.preprocessing import IncomingAlert, ParsedAlert, TokenizedAlert
from thesis.preprocessing.parsing import parse_alert_row
from thesis.preprocessing.tokenization import tokenize_alert
from thesis.preprocessing.cache_ingestor import (
    ingest_tokenized_alert,
    ingest_tokenized_alert_batch,
)
from thesis.preprocessing.cache import TokenCache


def parse_alert(
    alert: IncomingAlert,
    scenario: str,
) -> ParsedAlert:
    return parse_alert_row(alert=alert, scenario=scenario)


def run_preprocessing_pipeline(
    alert: IncomingAlert,
    scenario: str,
    cache: TokenCache,
) -> TokenizedAlert:
    parsed = parse_alert_row(alert=alert, scenario=scenario)
    tokenized = tokenize_alert(parsed)
    ingest_tokenized_alert(cache=cache, alert=tokenized)
    return tokenized


def process_one_alert(row: dict, scenario: str, cache: TokenCache) -> None:
    alert = IncomingAlert.from_row(row)
    parsed = parse_alert_row(alert=alert, scenario=scenario)
    tokenized = tokenize_alert(parsed)
    ingest_tokenized_alert(cache=cache, alert=tokenized)
    return tokenized


def process_alert_batch(rows: list[dict], scenario: str, cache: TokenCache) -> None:
    tokenize_alerts = []
    for row in rows:
        alert = IncomingAlert.from_row(row)
        parsed = parse_alert_row(alert=alert, scenario=scenario)
        tokenized = tokenize_alert(parsed)
        tokenize_alerts.append(tokenized)

    ingest_tokenized_alert_batch(
        cache=cache, alerts=tokenize_alerts, alert_batch_name=scenario
    )

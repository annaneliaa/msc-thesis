from datetime import datetime, timezone
from typing import Any, Iterable
import pandas as pd

from thesis.schemas.groups import AlertGroup

# IANA port ranges: well-known, registered, dynamic/ephemeral.
_PORT_CLASS_UNKNOWN = 3
_PORT_CLASS_WELL_KNOWN = 0
_PORT_CLASS_REGISTERED = 1
_PORT_CLASS_DYNAMIC = 2


def _count_items_with_prefix(items: set[str], prefix: str) -> int:
    return sum(1 for item in items if item.startswith(prefix))


def _port_class(port: int | None) -> int:
    if port is None or port < 0:
        return _PORT_CLASS_UNKNOWN
    if port <= 1023:
        return _PORT_CLASS_WELL_KNOWN
    if port <= 49151:
        return _PORT_CLASS_REGISTERED
    return _PORT_CLASS_DYNAMIC


def compute_baseline_features(tx: AlertGroup) -> dict[str, Any]:
    items = set(tx.abs_items or tx.raw_items or [])
    hour_of_day = datetime.fromtimestamp(int(tx.start_ts), tz=timezone.utc).hour

    return {
        "hour_of_day": hour_of_day,
        "n_alerts": int(tx.n_alerts),
        "n_hosts": _count_items_with_prefix(items, "host:"),
        "n_shorts": _count_items_with_prefix(items, "short:"),
        "n_sigs": _count_items_with_prefix(items, "sig:"),
        "proto": tx.proto if tx.proto is not None else -1,
        "int_port_class": _port_class(tx.int_port),
        "ext_port_class": _port_class(tx.ext_port),
        "int_ip_is_multiple": int(tx.int_ip_is_multiple),
        "ext_ip_is_multiple": int(tx.ext_ip_is_multiple),
    }


class BaselineFeatureEncoder:
    """
    Stateless baseline feature encoder for training and inference.
    """

    def transform_one(self, tx: AlertGroup) -> pd.DataFrame:
        """
        Encode one alert_group into a 1-row feature DataFrame.
        """
        features = compute_baseline_features(tx)
        return pd.DataFrame([features])

    def transform(self, alert_groups: Iterable[AlertGroup]) -> pd.DataFrame:
        """
        Encode many alert_groups into a feature DataFrame.
        """
        rows = [compute_baseline_features(tx) for tx in alert_groups]
        return pd.DataFrame(rows)

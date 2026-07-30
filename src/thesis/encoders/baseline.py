from datetime import datetime, timezone
from typing import Any, Iterable
import pandas as pd

from thesis.schemas.groups import AlertGroup

# AlertGroup.method values produced by CSCAS grouping (group_alerts.py); every
# other method comes from the AIT-ADS alert-level grouping phase.
_CSCAS_METHOD_PREFIX = "cscas"


def _count_items_with_prefix(items: set[str], prefix: str) -> int:
    return sum(1 for item in items if item.startswith(prefix))


def compute_ait_ads_baseline_features(tx: AlertGroup) -> dict[str, Any]:
    """
    Baseline features for AIT-ADS: summary stats over one grouping-phase
    basket (fixed_window/time_delta/alertbert/...), since a basket there
    aggregates many individual alerts rather than mirroring one CSV row.
    """
    items = set(tx.raw_items or [])
    hour_of_day = datetime.fromtimestamp(int(tx.start_ts), tz=timezone.utc).hour

    return {
        "hour_of_day": hour_of_day,
        "n_alerts": int(tx.n_alerts),
        "n_hosts": _count_items_with_prefix(items, "host:"),
        "n_shorts": _count_items_with_prefix(items, "short:"),
        "n_sigs": _count_items_with_prefix(items, "sig:"),
    }


def compute_cscas_baseline_features(tx: AlertGroup) -> dict[str, Any]:
    """
    Baseline features for CSCAS: the paper's own per-row columns, taken as-is
    off the CSV -- no bucketing, no derived multi_* flags. Those derived
    features live in mining/attribute_features.py as mining candidates instead.

    Deliberately excludes, same reduced-feature-set reasoning as
    baselines/cscas_base.py's module docstring (none of these are things a
    real deployment could compute for a fresh alert without already knowing
    the answer or running CSCAS's offline similarity pipeline):
      - SignatureID: a nominal identifier, not a real signal, and feeding
        its raw integer value to RF/logreg risks encoding an arbitrary ID
        ordering rather than anything meaningful. Not even carried through
        the ingestion pipeline past IncomingSuricataGroup as a result (see
        schemas/preprocessing.py).
      - SCAS: the paper's own outlier/inlier flag, computed from the same
        offline similarity pipeline as the *Similarity columns below.
      - similarity, signature_id_similarity, and the 33 attr_value:*
        columns (from ATTR_SIMILARITY_COLUMNS): CSCAS's own offline,
        per-field similarity scores -- unrealistic for a real deployment to
        have on hand for a fresh alert.
    """
    return {
        "proto": tx.proto if tx.proto is not None else -1,
        "ext_port": tx.ext_port if tx.ext_port is not None else -1,
        "int_port": tx.int_port if tx.int_port is not None else -1,
        "n_alerts": int(tx.n_alerts),
        "signature_matches_per_day": (
            tx.signature_matches_per_day
            if tx.signature_matches_per_day is not None
            else 0.0
        ),
    }


def compute_baseline_features(tx: AlertGroup) -> dict[str, Any]:
    if tx.method.startswith(_CSCAS_METHOD_PREFIX):
        return compute_cscas_baseline_features(tx)
    return compute_ait_ads_baseline_features(tx)


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

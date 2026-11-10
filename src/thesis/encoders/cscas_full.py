"""
CSCAS "full" feature encoder -- the paper's own feature set, as far as the
ingestion pipeline preserves it.

The CSCAS paper (Table IV) trains on 42 raw columns: SignatureID,
SignatureMatchesPerDay, AlertCount, Proto, ExtPort, IntPort, Similarity,
SCAS, and 34 per-attribute *Similarity columns. This encoder reproduces
that set with two unavoidable differences:

  * SignatureID is dropped at ingest (schemas/preprocessing.py keeps it only
    as far as IncomingSuricataGroup -- it's a nominal id, not a signal), so
    it can't be reconstructed here. 41 columns, not 42.
  * SignatureIDSimilarity is carried as its own AlertGroup field
    (`signature_id_similarity`), separate from the 33-entry
    ATTR_SIMILARITY_COLUMNS list, so the "34 *Similarity columns" become
    33 `attr_value:*` + `signature_id_similarity`.

Unlike encoders/baseline.py's `compute_cscas_baseline_features` -- the
deployment-realistic reduced set -- this encoder deliberately includes SCAS
and the offline *Similarity scores. Those come from CSCAS's own offline
similarity pipeline and a real deployment could not compute them for a
fresh alert, so this feature set is a non-deployable *reference ceiling*
for the temporal-decay experiment ("does the frozen model decay even with
the paper's full oracle features?"), not a candidate deployment schema.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from thesis.encoders.baseline import compute_cscas_baseline_features
from thesis.schemas.groups import AlertGroup
from thesis.schemas.preprocessing import ATTR_SIMILARITY_COLUMNS

# Same sentinel the mining candidate space uses for an inapplicable
# similarity column (mining/common/attribute_features.py `_NOT_APPLICABLE`).
_NOT_APPLICABLE = -1.0

# The 5 deployment-realistic base columns (encoders/baseline.py) come first,
# then CSCAS's offline scores.
CSCAS_FULL_FEATURES: list[str] = [
    "proto",
    "ext_port",
    "int_port",
    "n_alerts",
    "signature_matches_per_day",
    "similarity",
    "signature_id_similarity",
    "scas",
    *(f"attr_value:{name}" for name in ATTR_SIMILARITY_COLUMNS),
]


def compute_cscas_full_features(tx: AlertGroup) -> dict[str, Any]:
    """The 5 base columns plus CSCAS's own offline scores: `similarity`,
    `signature_id_similarity`, `scas`, and one `attr_value:<col>` per
    ATTR_SIMILARITY_COLUMNS entry. Missing values follow the same
    conventions as the mining candidate space (similarity -> 0.0, an
    inapplicable *Similarity column -> -1.0, SCAS -> -1.0 when unset)."""
    out = dict(compute_cscas_baseline_features(tx))
    out["similarity"] = tx.similarity if tx.similarity is not None else 0.0
    out["signature_id_similarity"] = (
        tx.signature_id_similarity if tx.signature_id_similarity is not None else 0.0
    )
    out["scas"] = float(tx.scas) if tx.scas is not None else _NOT_APPLICABLE
    sims = tx.attr_similarities or {}
    for name in ATTR_SIMILARITY_COLUMNS:
        value = sims.get(name, _NOT_APPLICABLE)
        out[f"attr_value:{name}"] = float(value)
    return out


class CscasFullFeatureEncoder:
    """Stateless CSCAS-full encoder, same interface as
    encoders.baseline.BaselineFeatureEncoder so
    encode_alert_groups_for_schema can pick between them on
    `schema.base.kind`."""

    def transform_one(self, tx: AlertGroup) -> pd.DataFrame:
        return pd.DataFrame([compute_cscas_full_features(tx)])

    def transform(self, alert_groups: Iterable[AlertGroup]) -> pd.DataFrame:
        return pd.DataFrame([compute_cscas_full_features(tx) for tx in alert_groups])

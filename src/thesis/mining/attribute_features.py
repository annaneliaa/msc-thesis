from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from thesis.schemas.groups import AlertGroup
from thesis.schemas.preprocessing import ATTR_SIMILARITY_COLUMNS
from thesis.preprocessing.suricata_tokenization import QUALIFIER_WORDS

# Candidate categorical fields whose *values* need one-hot expansion into
# per-value predicate columns (e.g. category=EXPLOIT, category=WEB_SERVER, ...).
MULTI_VALUED_CATEGORICAL_FIELDS: tuple[str, ...] = (
    "category",
    "ruleset",
    "proto",
    "scas",
)

# Candidate categorical fields that are already single binary predicates.
BINARY_CATEGORICAL_FIELDS: tuple[str, ...] = (
    "cve_present",
    "multi_target",
    "multi_port",
    *(f"qualifier_{w}" for w in sorted(QUALIFIER_WORDS)),
    *(f"attr_populated:{name}" for name in ATTR_SIMILARITY_COLUMNS),
)

# Candidate numeric base features handed to Step 2 regardless of Step 1's outcome.
NUMERIC_FIELDS: tuple[str, ...] = (
    "signature_matches_per_day",
    "alert_count",
    "similarity",
    "signature_id_similarity",
    "cve_age_years",
    *(f"attr_value:{name}" for name in ATTR_SIMILARITY_COLUMNS),
)

_NOT_APPLICABLE = -1.0


def _cve_year(cve_ref: str) -> int | None:
    """Extract the year embedded in a 'CVE-YYYY-NNNN' identifier."""
    parts = cve_ref.split("-")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def cve_age_years(cve_refs: set[str] | None, as_of_ts: int) -> float | None:
    """
    Years between the oldest referenced CVE's publication year and the alert
    group's own timestamp. No external CVE database needed -- the year is
    already encoded in the CVE-YYYY-NNNN identifier itself.
    """
    if not cve_refs:
        return None
    years = [y for y in (_cve_year(c) for c in cve_refs) if y is not None]
    if not years:
        return None
    alert_year = datetime.fromtimestamp(as_of_ts, tz=timezone.utc).year
    return float(alert_year - min(years))


def compute_candidate_attribute_features(tx: AlertGroup) -> dict[str, Any]:
    """
    Single source of truth for the candidate per-alert-group attribute space.

    Called both at mining time (building the training matrix over a window)
    and at encode time (evaluating a schema's predicates against one new
    alert group), so the two never drift apart.
    """
    cve_refs = tx.cve_refs or set()
    qualifiers = tx.qualifiers or set()
    attr_similarities = tx.attr_similarities or {}

    age = cve_age_years(cve_refs, tx.start_ts)

    features: dict[str, Any] = {
        "category": tx.category or "",
        "ruleset": tx.ruleset or "",
        "proto": tx.proto if tx.proto is not None else -1,
        "scas": tx.scas if tx.scas is not None else -1,
        "cve_present": bool(cve_refs),
        "multi_target": bool(tx.int_ip_is_multiple),
        "multi_port": bool(tx.ext_port_is_multiple),
        "signature_matches_per_day": (
            tx.signature_matches_per_day
            if tx.signature_matches_per_day is not None
            else 0.0
        ),
        "alert_count": float(tx.n_alerts),
        "similarity": tx.similarity if tx.similarity is not None else 0.0,
        "signature_id_similarity": (
            tx.signature_id_similarity
            if tx.signature_id_similarity is not None
            else 0.0
        ),
        "cve_age_years": age if age is not None else _NOT_APPLICABLE,
    }

    for word in QUALIFIER_WORDS:
        features[f"qualifier_{word}"] = word in qualifiers

    for name in ATTR_SIMILARITY_COLUMNS:
        value = attr_similarities.get(name, _NOT_APPLICABLE)
        populated = value != _NOT_APPLICABLE
        features[f"attr_populated:{name}"] = populated
        features[f"attr_value:{name}"] = value

    return features

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

# Application-layer *Similarity columns grouped by protocol, for the
# applicable_layer:* binary features below -- which app-layer protocol(s)
# actually fired for this alert group. AppProto/Ext*/Int*/Proto columns are
# excluded: they're generic connection metadata, not tied to one app layer.
_LAYER_PREFIXES: tuple[str, ...] = ("Dns", "Email", "Http", "Smtp", "Ssh", "Tls")
_LAYER_COLUMNS: dict[str, tuple[str, ...]] = {
    prefix.lower(): tuple(c for c in ATTR_SIMILARITY_COLUMNS if c.startswith(prefix))
    for prefix in _LAYER_PREFIXES
}

# Empirically dominant protocol (IANA number: 1=ICMP, 6=TCP, 17=UDP) per
# Suricata rule category, derived from the observed proto distribution within
# each category in the CSCAS cache (see attribute_features.py history for the
# breakdown). This is a data-driven convention, not a protocol spec -- MALWARE
# and ADWARE_PUP in particular are genuinely mixed in the data, so
# proto_mismatch is a much weaker signal for those two than for e.g. DNS/SNMP
# or WEB_SERVER/USER_AGENTS, which are close to unanimous.
_CATEGORY_EXPECTED_PROTO: dict[str, int] = {
    "DNS": 17,
    "SNMP": 17,
    "RPC": 17,
    "DOS": 17,
    "VOIP": 17,
    "MALWARE": 17,
    "WEB_SERVER": 6,
    "USER_AGENTS": 6,
    "EXPLOIT": 6,
    "WEB_SPECIFIC_APPS": 6,
    "ADWARE_PUP": 6,
    "SQL": 6,
    "NETBIOS": 6,
    "COINMINER": 6,
    "POLICY": 6,
    "PHISHING": 6,
    "JA3": 6,
    "ATTACK_RESPONSE": 6,
    "FTP": 6,
    "WORM": 6,
    "WEB_CLIENT": 6,
    "HUNTING": 6,
    "TELNET": 6,
    "EXPLOIT_KIT": 6,
    "CHAT": 6,
    "MOBILE_MALWARE": 6,
    "INFO": 6,
}

# Candidate categorical fields that are already single binary predicates.
BINARY_CATEGORICAL_FIELDS: tuple[str, ...] = (
    "cve_present",
    "multi_target",
    "multi_ext_port",
    "multi_int_port",
    "proto_mismatch",
    *(f"qualifier_{w}" for w in sorted(QUALIFIER_WORDS)),
    *(f"attr_populated:{name}" for name in ATTR_SIMILARITY_COLUMNS),
    *(f"applicable_layer:{layer}" for layer in _LAYER_COLUMNS),
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


def _proto_mismatch(category: str | None, proto: int | None) -> bool:
    """
    True iff this category has a known dominant protocol (see
    _CATEGORY_EXPECTED_PROTO) and the alert group's actual proto differs from
    it. False (not a mismatch) whenever we can't tell either way: unknown
    category, or proto == -1 (CSCAS's "multiple protocols collapsed" sentinel,
    same convention as ext_port/int_port).
    """
    if not category or proto is None or proto == -1:
        return False
    expected = _CATEGORY_EXPECTED_PROTO.get(category)
    if expected is None:
        return False
    return proto != expected


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
        "multi_ext_port": bool(tx.ext_port_is_multiple),
        "multi_int_port": bool(tx.int_port_is_multiple),
        "proto_mismatch": _proto_mismatch(tx.category, tx.proto),
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

    for layer, cols in _LAYER_COLUMNS.items():
        features[f"applicable_layer:{layer}"] = any(
            attr_similarities.get(c, _NOT_APPLICABLE) != _NOT_APPLICABLE for c in cols
        )

    return features

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from thesis.schemas.groups import AlertGroup
from thesis.schemas.preprocessing import ATTR_SIMILARITY_COLUMNS
from thesis.preprocessing.suricata_tokenization import QUALIFIER_WORDS

# Candidate categorical fields whose *values* need one-hot expansion into
# per-value predicate columns (e.g. category=EXPLOIT, category=WEB_SERVER, ...).
# Despite the name, these are all CSCAS fields that take exactly ONE value
# per alert group in practice (a CSCAS group is exactly one signature
# matching one external host, so e.g. category can't be both EXPLOIT and
# WEB_SERVER for the same group) -- build_categorical_predicate_matrix sets
# exactly one column to 1 per row for these, and
# compute_predicate_contrast_stats's _mutually_exclusive check relies on
# that being true. For fields where multiple values genuinely CAN co-occur
# in one group (AIT-ADS's own grouping, see SET_VALUED_CATEGORICAL_FIELDS
# below), that assumption doesn't hold -- don't add a field here unless a
# group can only ever have one value for it.
MULTI_VALUED_CATEGORICAL_FIELDS: tuple[str, ...] = (
    "category",
    "ruleset",
    "proto",
    "scas",
)

# AIT-ADS counterpart to MULTI_VALUED_CATEGORICAL_FIELDS, for fields that
# genuinely CAN take several simultaneous values in one alert group -- an
# AIT-ADS grouping-phase basket (fixed_window/time_delta/...) bundles many
# individual alerts, unlike a CSCAS group's single signature. Verified
# against real cached data (fox/harrison, both grouping methods): 97-99% of
# groups have more than one distinct "sig" token, 11-82% more than one
# distinct "short" code (scenario/method-dependent) -- collapsing either to
# a single reduced value would throw away real, common heterogeneity, not
# an edge case. Cardinality is bounded either way (~66 short codes, 26 sig
# keywords = the SIGNATURE_TOKEN_WHITELIST size, 11-13 hosts per scenario),
# so one-hot expansion is cheap. Each is a SET of the distinct values
# present in the group (see compute_ait_ads_attribute_features), not a
# scalar -- build_categorical_predicate_matrix and
# compute_predicate_contrast_stats._mutually_exclusive both branch on
# membership in this tuple to handle that correctly (multiple columns set
# to 1 per row; cross-value pairs NOT assumed mutually exclusive).
SET_VALUED_CATEGORICAL_FIELDS: tuple[str, ...] = ("short", "sig", "host")

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

# CSCAS-only "oracle" fields: derived from the paper's own offline
# outlier-cluster verdict (SCAS) and its Similarity-scoring pipeline -- not
# something a real deployment could compute for a fresh alert (see
# baselines/cscas_base.py's 5-column deployment-realistic schema, which
# already excludes every one of these). Mirrors the "scas" entry in
# MULTI_VALUED_CATEGORICAL_FIELDS above and the similarity-derived entries
# in NUMERIC_FIELDS/BINARY_CATEGORICAL_FIELDS -- named here as one set so a
# caller can exclude all of them from mining in a single call instead of
# re-deriving the list.
CSCAS_ORACLE_FIELDS: frozenset[str] = frozenset(
    {
        "scas",
        "similarity",
        "signature_id_similarity",
        *(f"attr_value:{name}" for name in ATTR_SIMILARITY_COLUMNS),
        *(f"attr_populated:{name}" for name in ATTR_SIMILARITY_COLUMNS),
        *(f"applicable_layer:{layer}" for layer in _LAYER_COLUMNS),
    }
)


def default_leaky_attribute_fields(scenario: str) -> set[str]:
    """Attribute-mining candidate fields to exclude by default for
    `scenario`, because they're derived from a non-deployable offline
    oracle rather than something computable for a fresh alert.

    Found missing (2026-09-13): attribute_schema_cache.mine_or_reuse_attribute_schema
    (and therefore every caller behind it -- window_schema_cache.py, used by
    temporal_decay.py/monitor_drift.py/screening_sweep.py, plus
    experiments/symbolic.py's "attribute" mining_strategy and cli.py's
    mine-symbolic command) never passed exclude_fields to
    run_alert_group_attribute_mining_job at all, so CSCAS_ORACLE_FIELDS
    (including SCAS itself -- a near-oracle proxy for the label, ~99%
    recall / 29% precision on its own) leaked straight into the mined
    "symbolic" schema for every one of those experiments. The baseline
    scripts (baselines/cscas_mining.py / cscas_mining_anomaly.py /
    cscas_mining_anomaly_iforest.py) already excluded these correctly, but
    only because each one separately builds and passes its own identical
    field set directly to run_alert_group_attribute_mining_job, bypassing
    this cache layer entirely -- this function is what those scripts'
    LEAKY_ATTRIBUTE_FIELDS should delegate to instead of duplicating the
    list a fourth time.

    Every attribute-mining entry point that goes through
    attribute_schema_cache.mine_or_reuse_attribute_schema now calls this
    automatically when its own `exclude_fields` argument is left as None,
    so a caller can't silently forget it again -- pass an explicit
    (possibly empty) set there to override.

    Returns an empty set for every scenario other than "cscas": these
    fields don't exist for non-CSCAS AlertGroups
    (compute_candidate_attribute_features fills them with a constant
    sentinel there instead), so excluding them would be a no-op anyway --
    but scoping this to "cscas" explicitly keeps the function honest about
    what it actually knows, rather than silently assuming every future
    scenario shares the same oracle fields."""
    if scenario == "cscas":
        return set(CSCAS_ORACLE_FIELDS)
    return set()


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


def _tokens_with_prefix(raw_items: set[str] | None, prefix: str) -> set[str]:
    """The distinct values of a raw_items token family, prefix stripped --
    e.g. {"short:A-Acc-Chr1", "short:W-Err-Fbd1"} -> {"A-Acc-Chr1", "W-Err-Fbd1"}
    for prefix="short:". Empty set (not an error) when raw_items is None/empty
    or has no tokens of this family -- true for every CSCAS alert group,
    whose raw_items use a completely different, unprefixed token vocabulary
    (verified: zero "short:"/"sig:"/"host:" tokens in the CSCAS cache), so
    these fields are harmlessly empty-and-constant there rather than
    colliding with anything CSCAS-specific."""
    if not raw_items:
        return set()
    plen = len(prefix)
    return {item[plen:] for item in raw_items if item.startswith(prefix)}


def _normalize_host(host: str) -> str:
    """Collapse a per-employee mail host (e.g. "smith_mail",
    "taylorcruz_mail") to a generic "user_mail" token, so mined host=*
    predicates generalize across AIT-ADS scenarios instead of keying on one
    company's specific employee names -- which not only don't exist in any
    other scenario, but can coincidentally collide between unrelated people
    in different scenarios who happen to share a surname (verified:
    "smith_mail" is a real, distinct employee in three different companies'
    data -- harrison, shaw, and wardbeck -- not the same person). Checked
    against every cached scenario (8 scenarios x 3 grouping methods): every
    per-employee host follows the "<name>_mail" pattern, and the shared
    infra hosts (mail, mail4, vpn, webserver, monitoring, cloud_share,
    internal_share, intranet_server, inet-dns, inet-firewall) never match
    it -- "mail" itself is too short to end with the "_mail" suffix, so it
    passes through unchanged alongside every other generic host."""
    if host.endswith("_mail"):
        return "user_mail"
    return host


def compute_ait_ads_attribute_features(tx: AlertGroup) -> dict[str, set[str]]:
    """AIT-ADS's SET_VALUED_CATEGORICAL_FIELDS, derived from the raw_items
    tokens every AIT-ADS grouping method already produces per alert (see
    preprocessing/tokenization.py's build_feature_tokens: "short:<code>",
    "host:<name>", "sig:<keyword>"). Each is the SET of distinct values
    present across every alert in this group, not a single reduced value --
    see SET_VALUED_CATEGORICAL_FIELDS' own docstring for why that matters.
    "host" is additionally normalized (see _normalize_host) so per-employee
    mailboxes don't leak scenario-specific identities into the mined schema.
    Merged into compute_candidate_attribute_features's returned dict below,
    same "called at mining time and at encode time" contract."""
    return {
        "short": _tokens_with_prefix(tx.raw_items, "short:"),
        "sig": _tokens_with_prefix(tx.raw_items, "sig:"),
        "host": {
            _normalize_host(h) for h in _tokens_with_prefix(tx.raw_items, "host:")
        },
    }


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

    features.update(compute_ait_ads_attribute_features(tx))

    return features

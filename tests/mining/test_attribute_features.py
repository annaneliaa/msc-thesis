from datetime import datetime, timezone

from thesis.mining.attribute_features import (
    compute_candidate_attribute_features,
    cve_age_years,
)
from thesis.schemas.groups import AlertGroup


def _make_alert_group(**overrides) -> AlertGroup:
    defaults = dict(
        alert_group_id="g1",
        group_id="g1",
        method="cscas_pregrouped",
        start_ts=int(datetime(2022, 1, 20, tzinfo=timezone.utc).timestamp()),
        end_ts=int(datetime(2022, 1, 20, tzinfo=timezone.utc).timestamp()),
        n_alerts=5,
        category="EXPLOIT",
        ruleset="GPL",
        proto=6,
        scas=2,
        cve_refs=set(),
        qualifiers=set(),
        signature_matches_per_day=120.0,
        similarity=0.8,
        signature_id_similarity=0.9,
        attr_similarities={"AppProtoSimilarity": -1.0, "ExtIPSimilarity": 0.75},
        int_ip_is_multiple=False,
        ext_port_is_multiple=False,
    )
    defaults.update(overrides)
    return AlertGroup(**defaults)


def test_cve_age_years_uses_year_embedded_in_cve_id():
    as_of = int(datetime(2022, 1, 20, tzinfo=timezone.utc).timestamp())
    assert cve_age_years({"CVE-2015-1234"}, as_of) == 7.0


def test_cve_age_years_uses_oldest_cve_when_multiple_present():
    as_of = int(datetime(2022, 1, 20, tzinfo=timezone.utc).timestamp())
    assert cve_age_years({"CVE-2015-1234", "CVE-2020-0001"}, as_of) == 7.0


def test_cve_age_years_none_when_no_cve_refs():
    as_of = int(datetime(2022, 1, 20, tzinfo=timezone.utc).timestamp())
    assert cve_age_years(set(), as_of) is None
    assert cve_age_years(None, as_of) is None


def test_compute_candidate_attribute_features_basic_fields():
    tx = _make_alert_group()
    features = compute_candidate_attribute_features(tx)

    assert features["category"] == "EXPLOIT"
    assert features["ruleset"] == "GPL"
    assert features["proto"] == 6
    assert features["scas"] == 2
    assert features["cve_present"] is False
    assert features["signature_matches_per_day"] == 120.0
    assert features["alert_count"] == 5.0
    assert features["similarity"] == 0.8
    assert features["signature_id_similarity"] == 0.9
    assert features["cve_age_years"] == -1.0  # no CVE refs -> not-applicable sentinel


def test_compute_candidate_attribute_features_multi_target_and_multi_port():
    tx = _make_alert_group(
        int_ip_is_multiple=True,
        ext_port_is_multiple=True,
        int_port_is_multiple=True,
    )
    features = compute_candidate_attribute_features(tx)

    assert features["multi_target"] is True
    assert features["multi_ext_port"] is True
    assert features["multi_int_port"] is True


def test_compute_candidate_attribute_features_qualifier_flags():
    tx = _make_alert_group(qualifiers={"possible"})
    features = compute_candidate_attribute_features(tx)

    assert features["qualifier_possible"] is True
    assert features["qualifier_observed"] is False


def test_compute_candidate_attribute_features_cve_present_and_age():
    as_of = int(datetime(2022, 1, 20, tzinfo=timezone.utc).timestamp())
    tx = _make_alert_group(cve_refs={"CVE-2010-0001"}, start_ts=as_of, end_ts=as_of)
    features = compute_candidate_attribute_features(tx)

    assert features["cve_present"] is True
    assert features["cve_age_years"] == 12.0


def test_compute_candidate_attribute_features_populated_vs_sentinel_similarity():
    tx = _make_alert_group(
        attr_similarities={"AppProtoSimilarity": -1.0, "ExtIPSimilarity": 0.75}
    )
    features = compute_candidate_attribute_features(tx)

    assert features["attr_populated:AppProtoSimilarity"] is False
    assert features["attr_value:AppProtoSimilarity"] == -1.0
    assert features["attr_populated:ExtIPSimilarity"] is True
    assert features["attr_value:ExtIPSimilarity"] == 0.75
    # Column not present in the group's dict at all -> defaults to not-applicable.
    assert features["attr_populated:DnsRrnameSimilarity"] is False
    assert features["attr_value:DnsRrnameSimilarity"] == -1.0


def test_compute_candidate_attribute_features_defaults_for_none_fields():
    tx = _make_alert_group(
        category=None,
        ruleset=None,
        proto=None,
        scas=None,
        signature_matches_per_day=None,
        similarity=None,
        signature_id_similarity=None,
        attr_similarities=None,
    )
    features = compute_candidate_attribute_features(tx)

    assert features["category"] == ""
    assert features["ruleset"] == ""
    assert features["proto"] == -1
    assert features["scas"] == -1
    assert features["signature_matches_per_day"] == 0.0
    assert features["similarity"] == 0.0
    assert features["signature_id_similarity"] == 0.0
    assert features["attr_populated:AppProtoSimilarity"] is False

from thesis.mining.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
    compute_predicate_contrast_stats,
    filter_contrast_survivors,
    surviving_single_columns,
)
from thesis.schemas.groups import AlertGroup


def _make_alert_group(group_id: str, label: str, **overrides) -> AlertGroup:
    defaults = dict(
        alert_group_id=group_id,
        group_id=group_id,
        method="cscas_pregrouped",
        start_ts=1_642_636_800,
        end_ts=1_642_636_800,
        n_alerts=1,
        group_label=label,
        category="POLICY",
        ruleset="ET",
        proto=6,
        scas=0,
        cve_refs=set(),
        qualifiers=set(),
        signature_matches_per_day=10.0,
        similarity=0.5,
        signature_id_similarity=0.5,
        attr_similarities={},
        int_ip_is_multiple=False,
        ext_port_is_multiple=False,
    )
    defaults.update(overrides)
    return AlertGroup(**defaults)


def _build_categorical_survival_groups() -> list[AlertGroup]:
    """
    10 attack + 10 benign groups. category=EXPLOIT is attack-discriminative,
    category=SNMP is benign-discriminative, category=NEUTRAL fires about
    equally in both classes and should not survive.
    """
    groups = []
    # attack: 8 EXPLOIT, 1 SNMP, 1 NEUTRAL
    for i in range(8):
        groups.append(_make_alert_group(f"a_exploit_{i}", "attack", category="EXPLOIT"))
    groups.append(_make_alert_group("a_snmp_0", "attack", category="SNMP"))
    groups.append(_make_alert_group("a_neutral_0", "attack", category="NEUTRAL"))
    # benign: 1 EXPLOIT, 8 SNMP, 1 NEUTRAL
    groups.append(_make_alert_group("b_exploit_0", "benign", category="EXPLOIT"))
    for i in range(8):
        groups.append(_make_alert_group(f"b_snmp_{i}", "benign", category="SNMP"))
    groups.append(_make_alert_group("b_neutral_0", "benign", category="NEUTRAL"))
    return groups


def test_compute_predicate_contrast_stats_categorical_singles():
    groups = _build_categorical_survival_groups()
    X, _X_num, y, _ = build_categorical_predicate_matrix(groups)
    stats_df = compute_predicate_contrast_stats(X, y)

    exploit_row = stats_df[stats_df["itemset"] == ("category=EXPLOIT",)].iloc[0]
    assert exploit_row["confidence_attack"] == 0.8
    assert abs(exploit_row["confidence_benign"] - 0.1) < 1e-9
    assert exploit_row["growth_rate"] > 3.0
    assert exploit_row["n_attack"] == 8
    assert exploit_row["n_benign"] == 1

    snmp_row = stats_df[stats_df["itemset"] == ("category=SNMP",)].iloc[0]
    assert abs(snmp_row["confidence_attack"] - 0.1) < 1e-9
    assert snmp_row["confidence_benign"] == 0.8
    assert snmp_row["growth_rate"] < 0.333
    assert snmp_row["n_attack"] == 1
    assert snmp_row["n_benign"] == 8

    neutral_row = stats_df[stats_df["itemset"] == ("category=NEUTRAL",)].iloc[0]
    assert abs(neutral_row["growth_rate"] - 1.0) < 1e-6


def test_filter_contrast_survivors_keeps_both_directions_drops_neutral():
    groups = _build_categorical_survival_groups()
    X, _X_num, y, _ = build_categorical_predicate_matrix(groups)
    stats_df = compute_predicate_contrast_stats(X, y)
    survivors = filter_contrast_survivors(
        stats_df,
        min_attack_coverage=0.05,
        min_benign_coverage=0.05,
        min_growth_rate=3.0,
    )
    survivor_itemsets = set(survivors["itemset"])

    assert ("category=EXPLOIT",) in survivor_itemsets
    assert ("category=SNMP",) in survivor_itemsets
    assert ("category=NEUTRAL",) not in survivor_itemsets


def test_filter_contrast_survivors_drops_low_coverage_high_growth_predicate():
    # 3 attack groups have category=RARE out of 2000 attack groups; growth
    # rate can be enormous but coverage is negligible -- must not survive.
    groups = []
    for i in range(3):
        groups.append(_make_alert_group(f"a_rare_{i}", "attack", category="RARE"))
    for i in range(1997):
        groups.append(_make_alert_group(f"a_common_{i}", "attack", category="COMMON"))
    for i in range(2000):
        groups.append(_make_alert_group(f"b_common_{i}", "benign", category="COMMON"))

    X, _X_num, y, _ = build_categorical_predicate_matrix(groups)
    stats_df = compute_predicate_contrast_stats(X, y)
    survivors = filter_contrast_survivors(
        stats_df, min_attack_coverage=0.05, min_growth_rate=3.0
    )

    assert ("category=RARE",) not in set(survivors["itemset"])


def _build_pairwise_only_survival_groups() -> list[AlertGroup]:
    """
    Neither cve_present nor multi_target alone is discriminative, but their
    AND combination is -- this is the "emerging pattern" case Step 1 must
    catch by enumerating pairs, not just singles.
    """
    groups = []
    # attack (n=20): 8 both True, 6 cve_present only, 6 multi_target only
    for i in range(8):
        groups.append(
            _make_alert_group(
                f"a_both_{i}",
                "attack",
                cve_refs={"CVE-2020-0001"},
                int_ip_is_multiple=True,
            )
        )
    for i in range(6):
        groups.append(
            _make_alert_group(f"a_cve_{i}", "attack", cve_refs={"CVE-2020-0001"})
        )
    for i in range(6):
        groups.append(_make_alert_group(f"a_mt_{i}", "attack", int_ip_is_multiple=True))

    # benign (n=20): 2 both True, 12 cve_present only, 6 multi_target only
    for i in range(2):
        groups.append(
            _make_alert_group(
                f"b_both_{i}",
                "benign",
                cve_refs={"CVE-2020-0001"},
                int_ip_is_multiple=True,
            )
        )
    for i in range(12):
        groups.append(
            _make_alert_group(f"b_cve_{i}", "benign", cve_refs={"CVE-2020-0001"})
        )
    for i in range(6):
        groups.append(_make_alert_group(f"b_mt_{i}", "benign", int_ip_is_multiple=True))
    return groups


def test_pairwise_predicate_survives_when_singles_do_not():
    groups = _build_pairwise_only_survival_groups()
    X, _X_num, y, _ = build_categorical_predicate_matrix(groups)
    stats_df = compute_predicate_contrast_stats(X, y)

    cve_row = stats_df[stats_df["itemset"] == ("cve_present",)].iloc[0]
    mt_row = stats_df[stats_df["itemset"] == ("multi_target",)].iloc[0]
    assert not (cve_row["growth_rate"] >= 3.0 or cve_row["growth_rate"] <= 1 / 3)
    assert not (mt_row["growth_rate"] >= 3.0 or mt_row["growth_rate"] <= 1 / 3)

    survivors = filter_contrast_survivors(
        stats_df, min_attack_coverage=0.05, min_growth_rate=3.0
    )
    survivor_itemsets = set(survivors["itemset"])

    assert ("cve_present",) not in survivor_itemsets
    assert ("multi_target",) not in survivor_itemsets
    assert ("cve_present", "multi_target") in survivor_itemsets

    single_cols = surviving_single_columns(survivors)
    assert "cve_present" in single_cols
    assert "multi_target" in single_cols

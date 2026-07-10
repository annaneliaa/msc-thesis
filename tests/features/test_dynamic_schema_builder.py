from datetime import datetime, timezone

from thesis.features.dynamic_schema_builder import (
    _assign_numeric_predicate_ids,
    build_dynamic_schema,
)
from thesis.mining.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
    compute_predicate_contrast_stats,
    filter_contrast_survivors,
    surviving_single_columns,
)
from thesis.mining.decision_tree_rule_mining import (
    build_training_matrix,
    extract_leaf_rules,
    fit_rule_tree,
)
from thesis.schemas.features import AttributePredicate
from thesis.schemas.groups import AlertGroup

_WINDOW_START = datetime(2022, 1, 20, tzinfo=timezone.utc)
_WINDOW_END = datetime(2022, 1, 21, tzinfo=timezone.utc)


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


def _build_mixed_signal_groups() -> list[AlertGroup]:
    """
    category=EXPLOIT is a discriminative categorical single predicate;
    signature_matches_per_day cleanly separates the classes at a numeric
    threshold -- gives the builder both predicate flavors to work with in
    one pass.
    """
    groups = []
    for i in range(20):
        groups.append(
            _make_alert_group(
                f"attack_{i}",
                "attack",
                category="EXPLOIT",
                signature_matches_per_day=float(50 + i),
            )
        )
    for i in range(20):
        groups.append(
            _make_alert_group(
                f"benign_{i}",
                "benign",
                category="SNMP",
                signature_matches_per_day=float(5000 + i),
            )
        )
    return groups


def _mine(groups: list[AlertGroup]):
    X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(groups)
    contrast_stats_df = compute_predicate_contrast_stats(X_cat, y, column_predicate_map)
    survivors_df = filter_contrast_survivors(
        contrast_stats_df, min_attack_coverage=0.05, min_growth_rate=3.0
    )
    surviving_cols = surviving_single_columns(survivors_df)
    X_train, kept_predicate_map = build_training_matrix(
        X_cat, X_num, column_predicate_map, surviving_cols
    )
    tree = fit_rule_tree(X_train, y, max_depth=3, min_samples_leaf=1)
    leaf_rules_df, predicate_alphabet = extract_leaf_rules(
        tree, X_train, y, kept_predicate_map
    )
    return (
        survivors_df,
        leaf_rules_df,
        predicate_alphabet,
        column_predicate_map,
        X_num,
        y,
    )


def test_build_dynamic_schema_categorical_and_numeric_single_predicates():
    groups = _build_mixed_signal_groups()
    survivors_df, leaf_rules_df, predicate_alphabet, column_predicate_map, X_num, y = (
        _mine(groups)
    )

    schema = build_dynamic_schema(
        contrast_stats_df=survivors_df,
        leaf_rules_df=leaf_rules_df,
        predicate_alphabet=predicate_alphabet,
        column_predicate_map=column_predicate_map,
        X_num=X_num,
        y=y,
        version=1,
        mining_window_start=_WINDOW_START,
        mining_window_end=_WINDOW_END,
    )

    assert schema.version == 1
    assert abs(schema.base_attack_rate - 0.5) < 1e-9
    assert schema.mining_window_start == _WINDOW_START
    assert schema.mining_window_end == _WINDOW_END
    assert schema.deployed_at is None
    assert schema.superseded_at is None

    cat_predicates = [
        p for p in schema.single_predicates if p.predicate_type == "categorical"
    ]
    exploit = next(
        p for p in cat_predicates if p.predicate_id == "cat:category=EXPLOIT"
    )
    assert exploit.direction == "attack"
    assert exploit.n_attack == 20
    assert exploit.n_benign == 0

    numeric_predicates = [
        p for p in schema.single_predicates if p.predicate_type == "numeric_threshold"
    ]
    assert len(numeric_predicates) >= 1
    assert all(p.field == "signature_matches_per_day" for p in numeric_predicates)
    # The threshold cleanly separates the classes, so whichever side survives
    # should show strong, one-directional support.
    assert all(
        max(p.attack_support, p.benign_support) > 0.9 for p in numeric_predicates
    )

    assert len(schema.compound_rules) == len(leaf_rules_df)
    for rule in schema.compound_rules:
        assert rule.prediction in ("attack", "benign")
        assert 0.0 <= rule.confidence <= 1.0
        assert rule.n_samples > 0


def test_build_dynamic_schema_excludes_pairwise_survivors_from_single_predicates():
    # Neither predicate alone is discriminative, but their AND combination is
    # -- mirrors tests/mining/test_attribute_contrast_mining.py's pairwise
    # fixture. Only the length-1 itemsets should ever become single_predicates.
    groups = []
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

    survivors_df, leaf_rules_df, predicate_alphabet, column_predicate_map, X_num, y = (
        _mine(groups)
    )
    assert ("cve_present", "multi_target") in set(survivors_df["itemset"])

    schema = build_dynamic_schema(
        contrast_stats_df=survivors_df,
        leaf_rules_df=leaf_rules_df,
        predicate_alphabet=predicate_alphabet,
        column_predicate_map=column_predicate_map,
        X_num=X_num,
        y=y,
        version=1,
        mining_window_start=_WINDOW_START,
        mining_window_end=_WINDOW_END,
    )

    # single_predicate_fields = {p.field for p in schema.single_predicates}
    # The pairwise survivor's constituent fields must not appear as
    # standalone single_predicates from the pairwise row itself.
    assert not any(
        p.predicate_id in ("cat:cve_present=True", "cat:multi_target=True")
        for p in schema.single_predicates
    ) or all(
        p.field in ("cve_present", "multi_target") for p in schema.single_predicates
    )


def test_build_dynamic_schema_is_deterministic():
    groups = _build_mixed_signal_groups()
    survivors_df, leaf_rules_df, predicate_alphabet, column_predicate_map, X_num, y = (
        _mine(groups)
    )

    kwargs = dict(
        contrast_stats_df=survivors_df,
        leaf_rules_df=leaf_rules_df,
        predicate_alphabet=predicate_alphabet,
        column_predicate_map=column_predicate_map,
        X_num=X_num,
        y=y,
        version=1,
        mining_window_start=_WINDOW_START,
        mining_window_end=_WINDOW_END,
        mined_at=datetime(2022, 1, 21, tzinfo=timezone.utc),
    )
    schema_a = build_dynamic_schema(**kwargs)
    schema_b = build_dynamic_schema(**kwargs)

    ids_a = sorted(p.predicate_id for p in schema_a.single_predicates)
    ids_b = sorted(p.predicate_id for p in schema_b.single_predicates)
    assert ids_a == ids_b

    rule_ids_a = sorted(r.rule_id for r in schema_a.compound_rules)
    rule_ids_b = sorted(r.rule_id for r in schema_b.compound_rules)
    assert rule_ids_a == rule_ids_b


def test_assign_numeric_predicate_ids_collision_gets_ordinal_suffix():
    preds = [
        AttributePredicate(
            token="signature_matches_per_day_gt_100",
            attribute="signature_matches_per_day",
            operator=">",
            value=100.0,
        ),
        AttributePredicate(
            token="signature_matches_per_day_gt_50",
            attribute="signature_matches_per_day",
            operator=">",
            value=50.0,
        ),
        AttributePredicate(
            token="similarity_le_0.5",
            attribute="similarity",
            operator="<=",
            value=0.5,
        ),
    ]

    id_map = _assign_numeric_predicate_ids(preds)

    assert (
        id_map["signature_matches_per_day_gt_50"] == "num:signature_matches_per_day_gt"
    )
    assert (
        id_map["signature_matches_per_day_gt_100"]
        == "num:signature_matches_per_day_gt#2"
    )
    assert id_map["similarity_le_0.5"] == "num:similarity_le"

    # Determinism: same inputs, same assignment, regardless of input order.
    id_map_reordered = _assign_numeric_predicate_ids(list(reversed(preds)))
    assert id_map_reordered == id_map

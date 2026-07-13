from thesis.mining.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
    compute_predicate_contrast_stats,
    filter_contrast_survivors,
    surviving_single_columns,
)
from thesis.mining.decision_tree_rule_mining import (
    build_training_matrix,
    extract_leaf_rules,
    fit_and_extract_rules,
    fit_rule_tree,
)
from thesis.schemas.features import AttributePredicate
from thesis.schemas.groups import AlertGroup
from thesis.schemas.mining import DecisionTreeRuleConfig


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


def _build_numeric_threshold_groups() -> list[AlertGroup]:
    """
    signature_matches_per_day cleanly separates the classes at a threshold
    around 500: attack groups sit low, benign groups sit high (mirrors the
    paper's own top feature -- high daily match rate = routine/benign noise).
    """
    groups = []
    for i in range(30):
        groups.append(
            _make_alert_group(
                f"attack_{i}", "attack", signature_matches_per_day=float(50 + i)
            )
        )
    for i in range(30):
        groups.append(
            _make_alert_group(
                f"benign_{i}", "benign", signature_matches_per_day=float(5000 + i)
            )
        )
    return groups


def test_fit_rule_tree_and_extract_leaf_rules_numeric_threshold():
    groups = _build_numeric_threshold_groups()
    X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(groups)
    X, column_predicate_map = build_training_matrix(
        X_cat, X_num, column_predicate_map, surviving_categorical_columns=[]
    )

    tree = fit_rule_tree(X, y, max_depth=2, min_samples_leaf=1)
    leaf_df, predicates = extract_leaf_rules(tree, X, y, column_predicate_map)

    assert len(leaf_df) >= 2
    # Every original alert group must land in exactly one leaf -> supports sum to 1.
    assert abs(leaf_df["support"].sum() - 1.0) < 1e-9

    # At least one leaf should be near-pure attack, one near-pure benign.
    assert leaf_df["confidence_attack"].max() > 0.9
    assert leaf_df["confidence_benign"].max() > 0.9

    # The numeric threshold condition should show up as a predicate on
    # signature_matches_per_day with an operator, not a categorical token.
    numeric_predicates = [
        p for p in predicates if p.attribute == "signature_matches_per_day"
    ]
    assert len(numeric_predicates) >= 1
    assert all(p.operator in (">", "<=") for p in numeric_predicates)
    assert all(isinstance(p.value, float) for p in numeric_predicates)


def test_extract_leaf_rules_categorical_predicate_round_trips_through_map():
    # 20 attack groups with category=EXPLOIT, 20 benign with category=POLICY.
    groups = [
        _make_alert_group(f"a_{i}", "attack", category="EXPLOIT") for i in range(20)
    ] + [_make_alert_group(f"b_{i}", "benign", category="POLICY") for i in range(20)]

    X_cat, X_num, y_cat, column_predicate_map = build_categorical_predicate_matrix(
        groups
    )
    stats_df = compute_predicate_contrast_stats(X_cat, y_cat)
    survivors = filter_contrast_survivors(stats_df, min_growth_rate=3.0)
    surviving_cols = surviving_single_columns(survivors)
    assert "category=EXPLOIT" in surviving_cols

    X, column_predicate_map = build_training_matrix(
        X_cat, X_num, column_predicate_map, surviving_cols
    )
    y = y_cat
    tree = fit_rule_tree(X, y, max_depth=2, min_samples_leaf=1)
    leaf_df, predicates = extract_leaf_rules(tree, X, y, column_predicate_map)

    assert len(leaf_df) == 2
    assert leaf_df["confidence_attack"].max() == 1.0
    assert leaf_df["confidence_benign"].max() == 1.0

    category_predicates = [p for p in predicates if p.attribute == "category"]
    assert len(category_predicates) >= 1
    assert all(isinstance(p, AttributePredicate) for p in category_predicates)
    # Either category=EXPLOIT or category=POLICY perfectly separates the two
    # classes here, so either column may be the one the tree actually split
    # on -- what matters is that the predicate round-trips consistently.
    fires_pred = next(p for p in category_predicates if p.operator == "==")
    assert fires_pred.value in ("EXPLOIT", "POLICY")
    assert fires_pred.token == f"category={fires_pred.value}"


def test_extract_leaf_rules_supports_sum_to_one_and_are_mutually_exclusive():
    groups = _build_numeric_threshold_groups()
    X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(groups)
    X, column_predicate_map = build_training_matrix(
        X_cat, X_num, column_predicate_map, surviving_categorical_columns=[]
    )
    tree = fit_rule_tree(X, y, max_depth=3, min_samples_leaf=1)
    leaf_df, _ = extract_leaf_rules(tree, X, y, column_predicate_map)

    assert leaf_df["support_count"].sum() == len(groups)
    assert leaf_df["leaf_id"].is_unique
    assert (
        (leaf_df["n_attack"] + leaf_df["n_benign"]) == leaf_df["support_count"]
    ).all()
    assert leaf_df["n_attack"].sum() == 30
    assert leaf_df["n_benign"].sum() == 30


def test_fit_and_extract_rules_single_tree_mode_matches_direct_call():
    groups = _build_numeric_threshold_groups()
    X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(groups)
    X, column_predicate_map = build_training_matrix(
        X_cat, X_num, column_predicate_map, surviving_categorical_columns=[]
    )

    tree_config = DecisionTreeRuleConfig(max_depth=2, min_samples_leaf=1)
    leaf_df, predicates = fit_and_extract_rules(X, y, column_predicate_map, tree_config)

    tree = fit_rule_tree(X, y, max_depth=2, min_samples_leaf=1)
    expected_leaf_df, expected_predicates = extract_leaf_rules(
        tree, X, y, column_predicate_map
    )

    assert len(leaf_df) == len(expected_leaf_df)
    assert leaf_df["support_count"].sum() == expected_leaf_df["support_count"].sum()
    assert {p.token for p in predicates} == {p.token for p in expected_predicates}
    # fit_and_extract_rules additionally tags source_label, unlike a bare
    # extract_leaf_rules call.
    assert set(leaf_df["source_label"]) <= {"attack", "benign"}


def test_fit_and_extract_rules_two_tree_mode_keeps_only_each_trees_own_class():
    groups = _build_numeric_threshold_groups()
    X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(groups)
    X, column_predicate_map = build_training_matrix(
        X_cat, X_num, column_predicate_map, surviving_categorical_columns=[]
    )

    # max_depth (benign-facing) shallow, max_depth_attack deeper -- deliberately
    # different so the two trees are not identical fits.
    tree_config = DecisionTreeRuleConfig(
        max_depth=1, max_depth_attack=3, min_samples_leaf=1
    )
    leaf_df, predicates = fit_and_extract_rules(X, y, column_predicate_map, tree_config)

    assert not leaf_df.empty
    # Every kept leaf is either a benign leaf from the max_depth=1 tree or an
    # attack leaf from the max_depth_attack=3 tree -- never the discarded
    # opposite-class leaves of either fit.
    assert set(leaf_df["source_label"]) <= {"attack", "benign"}
    assert (leaf_df.loc[leaf_df["source_label"] == "benign", "n_attack"] == 0).all()
    assert (leaf_df.loc[leaf_df["source_label"] == "attack", "n_benign"] == 0).all()

    # The two trees are fit independently -- the attack tree (depth 3) should
    # be able to find at least as many attack-leaning leaves as the shallow
    # depth-1 tree could, since depth 1 only has two leaves total.
    depth1_tree = fit_rule_tree(X, y, max_depth=1, min_samples_leaf=1)
    depth1_leaf_df, _ = extract_leaf_rules(depth1_tree, X, y, column_predicate_map)
    depth1_attack_leaves = (
        depth1_leaf_df["confidence_attack"] > depth1_leaf_df["confidence_benign"]
    ).sum()
    n_attack_leaves = (leaf_df["source_label"] == "attack").sum()
    assert n_attack_leaves >= depth1_attack_leaves

    # Every predicate returned actually appears in a kept leaf's itemset --
    # tokens that only ever showed up on a discarded leaf's path should be
    # dropped, not carried through from either tree's full alphabet.
    kept_tokens = {tok for itemset in leaf_df["itemset"] for tok in itemset}
    assert {p.token for p in predicates} <= kept_tokens


def test_fit_and_extract_rules_two_tree_mode_is_deterministic():
    groups = _build_numeric_threshold_groups()
    X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(groups)
    X, column_predicate_map = build_training_matrix(
        X_cat, X_num, column_predicate_map, surviving_categorical_columns=[]
    )
    tree_config = DecisionTreeRuleConfig(
        max_depth=1, max_depth_attack=3, min_samples_leaf=1
    )
    leaf_df_1, _ = fit_and_extract_rules(X, y, column_predicate_map, tree_config)
    leaf_df_2, _ = fit_and_extract_rules(X, y, column_predicate_map, tree_config)
    assert leaf_df_1["itemset"].tolist() == leaf_df_2["itemset"].tolist()
    assert leaf_df_1["support_count"].tolist() == leaf_df_2["support_count"].tolist()

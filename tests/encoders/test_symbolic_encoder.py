from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.schemas.features import (
    AttributePredicate,
    SymbolicFeature,
    SymbolicFeatureSchema,
)
from thesis.schemas.groups import AlertGroup


def _make_alert_group(**overrides) -> AlertGroup:
    defaults = dict(
        alert_group_id="g1",
        group_id="g1",
        method="cscas_pregrouped",
        start_ts=1_642_636_800,
        end_ts=1_642_636_800,
        n_alerts=1,
        category="EXPLOIT",
        ruleset="GPL",
        proto=6,
        scas=0,
        cve_refs=set(),
        qualifiers=set(),
        signature_matches_per_day=1000.0,
        similarity=0.5,
        signature_id_similarity=0.5,
        attr_similarities={},
        int_ip_is_multiple=False,
        ext_port_is_multiple=False,
    )
    defaults.update(overrides)
    return AlertGroup(**defaults)


def test_encoder_still_evaluates_plain_itemset_features_without_predicates():
    schema = SymbolicFeatureSchema(
        schema_name="symbolic",
        schema_version="0.1.0",
        features=[
            SymbolicFeature(
                feature_name="sym__cat_exploit",
                itemset=("cat:EXPLOIT",),
                source_label="attack",
            )
        ],
    )
    tx = _make_alert_group()
    tx.raw_items = {"cat:EXPLOIT", "qual:possible"}

    encoder = SymbolicFeatureEncoder(schema)
    row = encoder.transform_one(tx)

    assert row.iloc[0]["sym__cat_exploit"] == 1


def test_encoder_evaluates_categorical_decision_tree_predicate():
    predicate = AttributePredicate(
        token="category=EXPLOIT", attribute="category", operator="==", value="EXPLOIT"
    )
    schema = SymbolicFeatureSchema(
        schema_name="symbolic",
        schema_version="0.1.0",
        features=[
            SymbolicFeature(
                feature_name="dtr__category_exploit",
                itemset=("category=EXPLOIT",),
                source_label="attack",
                mining_type="decision_tree_rule",
            )
        ],
        predicates=[predicate],
    )
    encoder = SymbolicFeatureEncoder(schema)

    fires = _make_alert_group(category="EXPLOIT")
    no_fire = _make_alert_group(category="POLICY")

    assert encoder.transform_one(fires).iloc[0]["dtr__category_exploit"] == 1
    assert encoder.transform_one(no_fire).iloc[0]["dtr__category_exploit"] == 0


def test_encoder_evaluates_numeric_threshold_predicate():
    predicate = AttributePredicate(
        token="signature_matches_per_day_gt_500",
        attribute="signature_matches_per_day",
        operator=">",
        value=500.0,
    )
    schema = SymbolicFeatureSchema(
        schema_name="symbolic",
        schema_version="0.1.0",
        features=[
            SymbolicFeature(
                feature_name="dtr__sigrate_gt_500",
                itemset=("signature_matches_per_day_gt_500",),
                source_label="benign",
                mining_type="decision_tree_rule",
            )
        ],
        predicates=[predicate],
    )
    encoder = SymbolicFeatureEncoder(schema)

    high = _make_alert_group(signature_matches_per_day=900.0)
    low = _make_alert_group(signature_matches_per_day=10.0)

    assert encoder.transform_one(high).iloc[0]["dtr__sigrate_gt_500"] == 1
    assert encoder.transform_one(low).iloc[0]["dtr__sigrate_gt_500"] == 0


def test_encoder_evaluates_conjunction_of_predicates():
    predicates = [
        AttributePredicate(
            token="category=EXPLOIT",
            attribute="category",
            operator="==",
            value="EXPLOIT",
        ),
        AttributePredicate(
            token="signature_matches_per_day_le_500",
            attribute="signature_matches_per_day",
            operator="<=",
            value=500.0,
        ),
    ]
    schema = SymbolicFeatureSchema(
        schema_name="symbolic",
        schema_version="0.1.0",
        features=[
            SymbolicFeature(
                feature_name="dtr__exploit_and_low_rate",
                itemset=("category=EXPLOIT", "signature_matches_per_day_le_500"),
                source_label="attack",
                mining_type="decision_tree_rule",
            )
        ],
        predicates=predicates,
    )
    encoder = SymbolicFeatureEncoder(schema)

    both = _make_alert_group(category="EXPLOIT", signature_matches_per_day=100.0)
    only_one = _make_alert_group(category="EXPLOIT", signature_matches_per_day=900.0)

    assert encoder.transform_one(both).iloc[0]["dtr__exploit_and_low_rate"] == 1
    assert encoder.transform_one(only_one).iloc[0]["dtr__exploit_and_low_rate"] == 0


def test_encoder_transform_batch_matches_transform_one():
    predicate = AttributePredicate(
        token="category=EXPLOIT", attribute="category", operator="==", value="EXPLOIT"
    )
    schema = SymbolicFeatureSchema(
        schema_name="symbolic",
        schema_version="0.1.0",
        features=[
            SymbolicFeature(
                feature_name="dtr__category_exploit",
                itemset=("category=EXPLOIT",),
                source_label="attack",
                mining_type="decision_tree_rule",
            )
        ],
        predicates=[predicate],
    )
    encoder = SymbolicFeatureEncoder(schema)

    groups = [
        _make_alert_group(category="EXPLOIT"),
        _make_alert_group(category="POLICY"),
    ]
    df = encoder.transform(groups)

    assert list(df["dtr__category_exploit"]) == [1, 0]


def test_sequence_features_fire_only_in_order():
    # An item_sequence feature a -> b must respect order across alerts, with
    # the same rule the miner counted support with (strictly later alert).
    schema = SymbolicFeatureSchema(
        schema_name="symbolic",
        schema_version="0.1.0",
        features=[
            SymbolicFeature(
                feature_name="seq__a__b",
                itemset=("a", "b"),
                source_label="attack",
                mining_type="item_sequence",
            ),
            SymbolicFeature(
                feature_name="sym__a__b",
                itemset=("a", "b"),
                source_label="attack",
                mining_type="itemset",
            ),
        ],
    )
    encoder = SymbolicFeatureEncoder(schema)

    def row(steps):
        tx = _make_alert_group(sorted_items=steps, n_alerts=len(steps))
        return encoder.transform_one(tx).iloc[0]

    in_order = row([{"a"}, {"c"}, {"b"}])
    reversed_ = row([{"b"}, {"a"}])
    same_alert = row([{"a", "b"}])
    assert list(in_order.index) == ["seq__a__b", "sym__a__b"]
    assert in_order["seq__a__b"] == 1
    assert reversed_["seq__a__b"] == 0
    assert same_alert["seq__a__b"] == 0
    # The itemset feature over the same items stays order-blind.
    assert reversed_["sym__a__b"] == 1


def test_repeated_item_sequence_matches_non_consecutive_repeats():
    # a -> a is mined as a subsequence (two alerts holding a, anything in
    # between); it must fire on [a], [b], [a].
    schema = SymbolicFeatureSchema(
        schema_name="symbolic",
        schema_version="0.1.0",
        features=[
            SymbolicFeature(
                feature_name="seq__a__a",
                itemset=("a", "a"),
                source_label="attack",
                mining_type="item_sequence",
            )
        ],
    )
    tx = _make_alert_group(sorted_items=[{"a"}, {"b"}, {"a"}], n_alerts=3)
    assert SymbolicFeatureEncoder(schema).transform_one(tx).iloc[0]["seq__a__a"] == 1


def test_itemset_sequence_features_match_steps_as_subsets_in_order():
    schema = SymbolicFeatureSchema(
        schema_name="symbolic",
        schema_version="0.1.0",
        features=[
            SymbolicFeature(
                feature_name="seq__ab_then_c",
                itemset=("a & b", "c"),
                source_label="attack",
                mining_type="itemset_sequence",
            )
        ],
    )
    encoder = SymbolicFeatureEncoder(schema)

    def fires(steps):
        tx = _make_alert_group(context_steps=steps, n_alerts=len(steps))
        return encoder.transform_one(tx).iloc[0]["seq__ab_then_c"]

    assert fires([{"a", "b", "x"}, {"c"}]) == 1
    assert fires([{"a"}, {"b"}, {"c"}]) == 0  # a and b must share a step
    assert fires([{"c"}, {"a", "b"}]) == 0  # order matters
    assert fires([{"a", "b", "c"}]) == 0  # c needs a later step

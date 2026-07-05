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

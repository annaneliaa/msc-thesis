import pandas as pd

from thesis.features.schema_builder import build_symbolic_feature_schema


def test_sequence_and_itemset_over_same_items_get_distinct_names():
    # a -> b (sequence) and {a, b} (itemset) are different features; before,
    # both were named sym__a__b and one was dropped as a "duplicate".
    df = pd.DataFrame(
        {
            "itemset": [("a", "b"), ("a", "b")],
            "mining_type": ["itemset", "item_sequence"],
            "source_label": ["attack", "attack"],
        }
    )
    schema = build_symbolic_feature_schema(df, "attack", "s", "1")
    names = {f.feature_name: f.mining_type for f in schema.features}
    assert names == {"sym__a__b": "itemset", "seq__a__b": "item_sequence"}
    seq = next(f for f in schema.features if f.mining_type == "item_sequence")
    assert seq.itemset == ("a", "b")  # order kept

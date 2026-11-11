"""Phase 4b coarse AIT-ADS token levels: computed always, mined only on opt-in."""

from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.mining.atemporal.contrast_tree.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
    compute_predicate_contrast_stats,
)
from thesis.mining.common.attribute_features import (
    COARSE_SET_VALUED_FIELDS,
    compute_candidate_attribute_features,
)
from thesis.mining.common.item_table import build_item_table
from thesis.mining.common.patterns import add_pattern_stats, patterns_to_schema
from thesis.schemas.groups import AlertGroup

import pandas as pd


def _group(i, shorts, sigs=(), label="benign"):
    return AlertGroup(
        alert_group_id=f"g{i}",
        group_id=f"g{i}",
        method="fixed_window",
        start_ts=i,
        end_ts=i,
        n_alerts=len(shorts),
        group_label=label,
        raw_items={f"short:{c}" for c in shorts}
        | {f"sig:{k}" for k in sigs}
        | {"host:web"},
    )


def test_coarse_levels_from_codes_and_abstraction_map():
    feats = compute_candidate_attribute_features(
        _group(0, ["A-Acc-Chr1", "W-Aut-Ssh2"], sigs=["login"])
    )
    assert feats["short_src"] == {"A", "W"}
    assert feats["short_cat"] == {"A-Acc", "W-Aut"}
    assert "apache_access" in feats["short_family"]
    assert feats["sig_family"]  # login maps to a family in the map


def test_coarse_fields_are_opt_in():
    groups = [
        _group(
            i,
            ["A-Acc-Chr1"] if i % 2 else ["W-Aut-Ssh2"],
            label="attack" if i % 3 == 0 else "benign",
        )
        for i in range(30)
    ]
    X_default, *_ = build_categorical_predicate_matrix(groups)
    assert not any(
        c.split("=")[0] in COARSE_SET_VALUED_FIELDS for c in X_default.columns
    )
    X_opt, *_ = build_categorical_predicate_matrix(
        groups, include_fields=set(COARSE_SET_VALUED_FIELDS)
    )
    assert "short_cat=A-Acc" in X_opt.columns and "short_src=W" in X_opt.columns


def test_coarse_pairs_are_not_assumed_mutually_exclusive():
    # Two codes from different categories in one group: short_cat=A-Acc AND
    # short_cat=W-Aut is a real candidate, like short/sig pairs.
    groups = [
        _group(
            i,
            ["A-Acc-Chr1", "W-Aut-Ssh2"]
            if i < 15
            else (["A-Acc-Chr1"] if i < 30 else ["S-Tls-Hnd"]),
            label="attack" if i < 10 else "benign",
        )
        for i in range(40)
    ]
    X, _, y, cmap = build_categorical_predicate_matrix(
        groups, include_fields={"short_cat"}
    )
    stats = compute_predicate_contrast_stats(X, y, cmap)
    pairs = {tuple(sorted(it)) for it in stats["itemset"] if len(it) == 2}
    assert ("short_cat=A-Acc", "short_cat=W-Aut") in pairs


def test_t3_item_table_has_no_full_codes_and_round_trips():
    groups = [
        _group(
            i,
            ["A-Acc-Chr1"] if i % 2 else ["A-Acc-Ent2", "W-Aut-Ssh2"],
            label="attack" if i % 2 == 0 else "benign",
        )
        for i in range(40)
    ]
    table = build_item_table(
        groups,
        include_fields=set(COARSE_SET_VALUED_FIELDS),
        exclude_fields={"short", "sig"},
    )
    cols = set(table.items().columns)
    assert not any(c.startswith(("short=", "sig=")) for c in cols)
    assert "short_cat=W-Aut" in cols
    # A coarse item written into the schema fires where the table says.
    frame = pd.DataFrame(
        {
            "items": [("short_cat=W-Aut",)],
            "mining_type": "itemset",
            "source_label": "attack",
            "n_attack": [20],
            "n_benign": [0],
        }
    )
    schema = patterns_to_schema(
        add_pattern_stats(frame, 20, 20), "s", "1", item_table=table
    )
    X = SymbolicFeatureEncoder(schema).transform(groups)
    assert X.iloc[:, 0].tolist() == table.items()["short_cat=W-Aut"].tolist()


def test_dedupe_identical_firing_drops_hierarchy_duplicates():
    from thesis.mining.common.patterns import dedupe_conditions
    from thesis.system_eval.mining_mini_experiment import dedupe_identical_firing

    # Only one A-Acc code occurs, so short=A-Acc-Chr1 and short_cat=A-Acc
    # fire on the same groups: different conditions, one feature.
    groups = [
        _group(
            i,
            ["A-Acc-Chr1"] if i % 2 else ["W-Aut-Ssh2"],
            label="attack" if i % 4 == 0 else "benign",
        )
        for i in range(40)
    ]
    table = build_item_table(groups, include_fields={"short_cat"})
    frame = pd.DataFrame(
        {
            "items": [
                ("short=A-Acc-Chr1",),
                ("short_cat=A-Acc",),
                ("short=W-Aut-Ssh2",),
            ],
            "mining_type": "itemset",
            "source_label": ["benign", "benign", "attack"],
            "n_attack": [0, 0, 10],
            "n_benign": [20, 20, 10],
        }
    )
    gated = dedupe_conditions(add_pattern_stats(frame, 10, 30), table)
    assert len(gated) == 3  # three distinct conditions...
    out = dedupe_identical_firing(gated, table, groups)
    assert len(out) == 2  # ...but only two distinct firing patterns
    assert ("short=W-Aut-Ssh2",) in set(out["items"])


def test_dedupe_firing_drops_near_duplicates_only():
    from thesis.mining.common.patterns import dedupe_conditions
    from thesis.system_eval.mining_mini_experiment import dedupe_firing

    # short_cat=A-Acc and short=A-Acc-Chr1 fire on the same 39 groups
    # (A-Acc-Chr1 is the only A-Acc code); short=W-Aut-Ssh2 fires on 20 of
    # them (Jaccard 20/39 = 0.51 with the A-Acc features).
    groups = [
        _group(
            i,
            ["A-Acc-Chr1", "W-Aut-Ssh2"]
            if i % 2
            else (["A-Acc-Chr1"] if i else ["S-Tls-Hnd"]),
            label="attack" if i % 4 == 0 else "benign",
        )
        for i in range(40)
    ]
    table = build_item_table(groups, include_fields={"short_cat"})
    frame = pd.DataFrame(
        {
            "items": [
                ("short_cat=A-Acc",),
                ("short=A-Acc-Chr1",),
                ("short=W-Aut-Ssh2",),
            ],
            "mining_type": "itemset",
            "source_label": ["benign", "benign", "benign"],
            "n_attack": [9, 9, 0],
            "n_benign": [30, 30, 20],
        }
    )
    gated = dedupe_conditions(add_pattern_stats(frame, 10, 30), table)
    exact = dedupe_firing(gated, table, groups, jaccard=1.0)
    near = dedupe_firing(gated, table, groups, jaccard=0.95)
    # short_cat=A-Acc and short=A-Acc-Chr1 fire identically -> one goes even
    # at 1.0; nothing else is within 0.95 of the kept features.
    assert len(exact) == 2 and len(near) == 2
    near90 = dedupe_firing(gated, table, groups, jaccard=0.4)
    assert len(near90) == 1  # Jaccard 0.5 >= 0.4: the Ssh2 feature goes too

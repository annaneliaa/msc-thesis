"""Contrast-set (Step 1) features must actually fire at encode time.

Regression test: the schema's predicate alphabet used to hold only the
decision tree's path predicates, so a contrast_categorical feature whose
column wasn't also a tree split encoded as a constant 0 column.
"""

import numpy as np

from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.features.schema_builder import build_symbolic_feature_schema
from thesis.mining.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
    compute_predicate_contrast_stats,
    contrast_survivor_predicates,
    filter_contrast_survivors,
    merge_predicate_alphabets,
)
from thesis.mining.attribute_mining_job import run_alert_group_attribute_mining_job
from thesis.pipeline.pipeline import save_alert_groups_json
from thesis.schemas.features import AttributePredicate
from thesis.schemas.groups import AlertGroup
from thesis.schemas.mining import AttributeMiningConfig, DecisionTreeRuleConfig


def _group(i: int, label: str, category: str, proto: int, cve: bool) -> AlertGroup:
    return AlertGroup(
        alert_group_id=f"g{i}",
        group_id=f"g{i}",
        method="cscas_pregrouped",
        start_ts=1_642_636_800 + i,
        end_ts=1_642_636_800 + i,
        n_alerts=1,
        group_label=label,
        raw_items={f"cat:{category}"},  # raw_items never carry "category=..."
        category=category,
        ruleset="ET",
        proto=proto,
        scas=0,
        cve_refs={"CVE-2020-0001"} if cve else set(),
        qualifiers=set(),
        signature_matches_per_day=10.0,
        similarity=0.5,
        signature_id_similarity=0.5,
        attr_similarities={},
        int_ip_is_multiple=False,
        ext_port_is_multiple=False,
    )


def _groups() -> list[AlertGroup]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(60):  # attacks: EXPLOIT, mostly tcp, often a CVE
        out.append(
            _group(
                i,
                "attack",
                "EXPLOIT",
                6 if rng.random() < 0.8 else 17,
                rng.random() < 0.7,
            )
        )
    for i in range(60, 400):  # benign: mostly SNMP/udp, some POLICY/tcp
        cat = "SNMP" if rng.random() < 0.7 else "POLICY"
        out.append(
            _group(i, "benign", cat, 17 if cat == "SNMP" else 6, rng.random() < 0.05)
        )
    return out


def _encode_and_check(groups, schema, X_cat):
    enc = SymbolicFeatureEncoder(schema).transform(groups)
    contrast = [f for f in schema.features if f.mining_type == "contrast_categorical"]
    assert contrast, "fixture should produce contrast survivors"
    for f in contrast:
        expected = np.ones(len(groups), dtype=bool)
        for tok in f.itemset:
            expected &= X_cat[tok].to_numpy() == 1
        assert (
            enc[f.feature_name].to_numpy() == expected.astype(int)
        ).all(), f.feature_name
    return contrast


def test_contrast_features_fire_with_survivor_predicates():
    groups = _groups()
    X_cat, _, y, cpm = build_categorical_predicate_matrix(groups)
    survivors = filter_contrast_survivors(
        compute_predicate_contrast_stats(X_cat, y, cpm)
    )
    survivors["source_label"] = np.where(
        survivors["confidence_attack"] > survivors["confidence_benign"],
        "attack",
        "benign",
    )
    survivors["mining_type"] = "contrast_categorical"
    preds = contrast_survivor_predicates(survivors, cpm)
    schema = build_symbolic_feature_schema(
        df=survivors,
        source_label="attack",
        schema_name="s",
        schema_version="0",
        predicates=preds,
    )
    contrast = _encode_and_check(groups, schema, X_cat)
    # and they're not all dead: SNMP / EXPLOIT columns really fire
    enc = SymbolicFeatureEncoder(schema).transform(groups)
    assert enc[[f.feature_name for f in contrast]].to_numpy().sum() > 0


def test_mining_job_alphabet_covers_contrast_survivors(tmp_path):
    groups = _groups()
    path = tmp_path / "groups.json"
    save_alert_groups_json(groups, path)
    result = run_alert_group_attribute_mining_job(
        alert_groups_path=path,
        scenario_name="cscas",
        config=AttributeMiningConfig(
            tree=DecisionTreeRuleConfig(max_depth=1, min_samples_leaf=5)
        ),
        run_dir=tmp_path / "run",
    )
    tokens = {p.token for p in result.predicates}
    contrast = result.mined_df[result.mined_df["mining_type"] == "contrast_categorical"]
    for itemset in contrast["itemset"]:
        assert set(itemset) <= tokens, itemset


def test_merge_predicate_alphabets_dedupes_by_token():
    a = AttributePredicate(
        token="category=SNMP", attribute="category", operator="==", value="SNMP"
    )
    b = AttributePredicate(
        token="category=SNMP", attribute="category", operator="==", value="SNMP"
    )
    c = AttributePredicate(
        token="cve_present", attribute="cve_present", operator="==", value=True
    )
    assert merge_predicate_alphabets([a], [b, c]) == [a, c]

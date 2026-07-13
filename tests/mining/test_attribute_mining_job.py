import json
from pathlib import Path

from thesis.mining.attribute_mining_job import run_alert_group_attribute_mining_job
from thesis.pipeline.pipeline import alert_group_to_dict
from thesis.schemas.groups import AlertGroup
from thesis.schemas.mining import AttributeMiningConfig, DecisionTreeRuleConfig


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


def _write_alert_groups_json(path: Path, groups: list[AlertGroup]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([alert_group_to_dict(tx) for tx in groups], indent=2))


def _build_groups() -> list[AlertGroup]:
    groups = []
    for i in range(20):
        groups.append(
            _make_alert_group(
                f"a_{i}",
                "attack",
                category="EXPLOIT",
                signature_matches_per_day=50.0 + i,
            )
        )
    for i in range(20):
        groups.append(
            _make_alert_group(
                f"b_{i}",
                "benign",
                category="POLICY",
                signature_matches_per_day=5000.0 + i,
            )
        )
    # a few unlabeled/mixed groups that must be dropped before mining
    groups.append(_make_alert_group("u_0", "mixed"))
    return groups


def test_run_alert_group_attribute_mining_job_end_to_end(tmp_path):
    alert_groups_path = tmp_path / "alert_groups_raw.json"
    _write_alert_groups_json(alert_groups_path, _build_groups())

    result = run_alert_group_attribute_mining_job(
        alert_groups_path=alert_groups_path,
        scenario_name="cscas_test",
        run_name="pytest_attribute_mining",
        config=AttributeMiningConfig(),
        run_dir=tmp_path / "run",
    )

    assert result.scenario_name == "cscas_test"
    assert result.run_dir.exists()
    assert (result.run_dir / "contrast_stats_all.csv").exists()
    assert (result.run_dir / "contrast_survivors.csv").exists()
    assert (result.run_dir / "decision_tree_rules.csv").exists()
    assert (result.run_dir / "mined_attribute_features.csv").exists()
    assert (result.run_dir / "metadata.json").exists()

    mined_df = result.mined_df
    assert "contrast_categorical" in set(mined_df["mining_type"])
    assert "decision_tree_rule" in set(mined_df["mining_type"])

    # category=EXPLOIT/category=POLICY are perfectly discriminative here, so
    # at least one predicate should reference the "category" attribute.
    assert any(p.attribute == "category" for p in result.predicates)


def test_run_alert_group_attribute_mining_job_two_tree_mode_end_to_end(tmp_path):
    alert_groups_path = tmp_path / "alert_groups_raw.json"
    _write_alert_groups_json(alert_groups_path, _build_groups())

    config = AttributeMiningConfig(
        tree=DecisionTreeRuleConfig(max_depth=1, max_depth_attack=3)
    )
    result = run_alert_group_attribute_mining_job(
        alert_groups_path=alert_groups_path,
        scenario_name="cscas_test",
        run_name="pytest_attribute_mining_two_tree",
        config=config,
        run_dir=tmp_path / "run_two_tree",
    )

    assert (result.run_dir / "mined_attribute_features.csv").exists()
    mined_df = result.mined_df
    tree_rows = mined_df[mined_df["mining_type"] == "decision_tree_rule"]
    assert not tree_rows.empty
    # Every Step-2 row's source_label is consistent with which tree it came
    # from -- an attack-leaning leaf only exists because the max_depth_attack
    # tree contributed it, a benign-leaning leaf only from the max_depth tree.
    assert (tree_rows.loc[tree_rows["source_label"] == "benign", "n_attack"] == 0).all()
    assert (tree_rows.loc[tree_rows["source_label"] == "attack", "n_benign"] == 0).all()

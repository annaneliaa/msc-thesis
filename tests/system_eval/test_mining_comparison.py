"""Mining comparison runner on a synthetic CSCAS-like step: every arm runs
end to end through the shared treatment, plus the summary statistics."""

import json
import random

import numpy as np
import pandas as pd
import pytest

from thesis.encoders.baseline import BaselineFeatureEncoder
from thesis.schemas.features import BaseFeatureSchema, FeatureSchema
from thesis.schemas.groups import AlertGroup
from thesis.system_eval.mining_comparison import (
    NO_MINING,
    ComparisonConfig,
    DatasetSpec,
    Step,
    ait_ads_base_schema,
    attack_phases,
    loso_folds,
    merge_runs,
    phase_cuts,
    phase_split,
    firing_overlap,
    friedman_nemenyi,
    helps_hurts,
    redundancy_share,
    run_step,
    stability,
    summarize_comparison,
)
from thesis.system_eval.mining_mini_experiment import MiniConfig

from test_mining_mini_experiment import _ait_groups, _cscas_group, _methods


@pytest.fixture(scope="module")
def step_rows():
    rng = random.Random(0)
    groups = [_cscas_group(i, rng) for i in range(900)]
    cols = list(BaselineFeatureEncoder().transform(groups[:5]).columns)
    base = FeatureSchema("base", "1", base=BaseFeatureSchema(features=cols))
    config = ComparisonConfig(
        seeds=(0, 1),
        mining=MiniConfig(k=10, n_bins=4),
    )
    step = Step(0, groups[:600], groups[600:], test_days=1.0)
    spec = DatasetSpec("cscas", base, lambda g: (g.category, g.ruleset))
    rows, feats = run_step(config, step, spec)
    return pd.DataFrame(rows), pd.concat(feats, ignore_index=True)


def test_every_arm_and_seed_scored(step_rows):
    rows, _ = step_rows
    assert set(rows["arm"]) == {NO_MINING, "eclat", "contrast_tree", "ripper"}
    assert (rows.groupby("arm").size() == 2).all()
    assert rows["f1"].notna().all() and rows["pr_auc"].notna().all()
    mined = rows[rows["arm"] != NO_MINING]
    assert (mined["n_symbolic"] > 0).all()
    assert (mined["n_symbolic"] <= 10).all()  # budget K
    assert (rows.loc[rows["arm"] == NO_MINING, "n_symbolic"] == 0).all()


def test_stochastic_miner_reseeded_deterministic_mined_once(step_rows):
    rows, feats = step_rows
    assert set(feats.loc[feats["arm"] == "ripper", "mining_seed"]) == {0, 1}
    assert set(feats.loc[feats["arm"] == "eclat", "mining_seed"]) == {0}
    ripper = rows[rows["arm"] == "ripper"]
    assert list(ripper["mining_seed"]) == [0, 1]


def test_firing_overlap_and_redundancy():
    F = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
    # pairs: (0,1)=1, (0,2)=0, (1,2)=0
    assert firing_overlap(F) == pytest.approx(1 / 3)
    assert np.isnan(firing_overlap(F[:, :1]))
    stages = {"distinct_condition": {1, 2, 3, 4}, "distinct_firing": {1}}
    assert redundancy_share(stages) == 0.75


def _per_step(deltas):
    rows = []
    for step, d in enumerate(deltas):
        for arm, f1 in ((NO_MINING, 0.5), ("a", 0.5 + d), ("b", 0.4)):
            rows.append(
                {
                    "arm": arm,
                    "seed": 0,
                    "step_i": step,
                    "f1": f1,
                    "pr_auc": f1,
                    "tp_seen": 1,
                    "n_attack_seen": 2,
                    "tp_unseen": 0,
                    "n_attack_unseen": 0,
                }
            )
    return pd.DataFrame(rows)


def test_helps_hurts_and_friedman():
    per_step = _per_step([0.1, 0.2, -0.1, 0.0, 0.1])
    hh = helps_hurts(per_step).set_index("arm")
    assert hh.loc["a", ["helps", "hurts", "ties"]].tolist() == [3, 1, 1]
    assert hh.loc["b", "hurts"] == 5
    res = friedman_nemenyi(per_step)
    assert res["n_steps"] == 5 and res["n_arms"] == 3
    assert min(res["avg_rank"], key=res["avg_rank"].get) == "a"
    assert res["nemenyi_cd"] > 0
    summary = summarize_comparison(per_step).set_index("arm")
    assert summary.loc["a", "n_steps"] == 5
    assert np.isnan(summary.loc["a", "recall_unseen_pooled"])


def test_stability_share_of_next_step():
    feats = pd.DataFrame(
        {
            "arm": "a",
            "mining_seed": 0,
            "step_i": [0, 0, 1, 1, 2],
            "condition": ["x", "y", "x", "z", "q"],
        }
    )
    # step0->1: 1/2 reappear; step1->2: 0/2
    assert stability(feats).loc[0, "stability"] == pytest.approx(0.25)


def test_comparison_budget_default():
    assert ComparisonConfig().mining.k == 60
    assert ComparisonConfig().mining.per_class_budget


# --- AIT-ADS attack-phase walk ---------------------------------------------


def _timed(ts_labels):
    return [
        AlertGroup(
            alert_group_id=f"g{i}",
            group_id=f"g{i}",
            method="cscas_grouping",
            start_ts=t,
            end_ts=t,
            n_alerts=1,
            group_label="attack" if atk else "benign",
        )
        for i, (t, atk) in enumerate(ts_labels)
    ]


def test_attack_phases_split_on_gaps():
    h = 3600
    groups = _timed(
        [
            (0, 0),
            (10 * h, 1),
            (10 * h + 1800, 1),
            (11 * h, 1),
            (20 * h, 0),
            (30 * h, 1),
            (31 * h + 1, 0),
            (40 * h, 1),
            (50 * h, 0),
        ]
    )
    phases = attack_phases(groups, gap_seconds=h)
    assert phases == [(10 * h, 11 * h), (30 * h, 30 * h), (40 * h, 40 * h)]
    assert phase_cuts(phases) == [20.5 * h, 35 * h]

    split = phase_split(groups, gap_seconds=h)
    assert [p for p, _, _ in split] == [1, 2]
    _, tr1, te1 = split[0]
    assert [g.start_ts for g in tr1] == [0, 10 * h, 10 * h + 1800, 11 * h, 20 * h]
    assert [g.start_ts for g in te1] == [30 * h, 31 * h + 1]
    _, tr2, te2 = split[1]  # expanding window; last test runs to the end
    assert len(tr2) == 7 and [g.start_ts for g in te2] == [40 * h, 50 * h]
    # every training window holds an earlier phase's attacks
    assert all(any(g.group_label == "attack" for g in tr) for _, tr, _ in split)


def test_single_phase_gives_no_step():
    groups = _timed([(0, 0), (100, 1), (200, 1), (10_000, 0)])
    assert phase_split(groups, gap_seconds=3600) == []


def test_ait_ads_base_schema_is_the_thesis_base():
    from thesis.configs import load_base_features

    schema = ait_ads_base_schema()
    assert list(schema.base.features) == load_base_features("ait-ads")


def test_ait_ads_step_runs_temporal_methods_with_attach():
    groups = _ait_groups()
    cols = list(BaselineFeatureEncoder().transform(groups[:5]).columns)
    base = FeatureSchema("base", "1", base=BaseFeatureSchema(features=cols))
    attached = []
    spec = DatasetSpec(
        "ait_ads",
        base,
        lambda g: g.group_id.split(":")[2],
        benign_gate_g=1.0,
        # steps are already on the synthetic groups; just record the call
        attach_steps=lambda gs: attached.append(len(gs)),
    )
    methods = tuple(m for m in _methods() if m in ("eclat", "prefixspan", "mctsextent"))
    config = ComparisonConfig(
        dataset="ait_ads", seeds=(0,), methods=methods, mining=MiniConfig(k=10)
    )
    step = Step(3, groups[:110], groups[110:], 1.0, scenario="syn", phase=1)
    rows, feats = run_step(config, step, spec)
    rows = pd.DataFrame(rows)
    assert attached == [160]  # once per step, train + test
    assert set(rows["arm"]) == {NO_MINING, *methods}
    assert (rows["scenario"] == "syn").all() and (rows["phase"] == 1).all()
    feats = pd.concat(feats, ignore_index=True)
    assert set(feats["scenario"]) == {"syn"}
    seq = feats[feats["arm"] == "prefixspan"]
    assert not seq.empty and seq["condition"].str.contains(" -> ").all()


def test_temporal_method_without_raw_alerts_is_an_error():
    groups = _ait_groups()
    cols = list(BaselineFeatureEncoder().transform(groups[:5]).columns)
    base = FeatureSchema("base", "1", base=BaseFeatureSchema(features=cols))
    spec = DatasetSpec("ait_ads", base, lambda g: g.group_id)
    config = ComparisonConfig(dataset="ait_ads", seeds=(0,), methods=("prefixspan",))
    with pytest.raises(ValueError, match="no raw alerts"):
        run_step(config, Step(0, groups[:110], groups[110:], 1.0), spec)


def test_stability_only_pairs_steps_within_a_scenario():
    feats = pd.DataFrame(
        {
            "arm": "a",
            "mining_seed": 0,
            "scenario": ["s1", "s1", "s2"],
            "step_i": [0, 1, 2],
            "condition": ["x", "x", "x"],
        }
    )
    out = stability(feats).set_index("arm")
    # (0, 1) in s1 counts; (1, 2) crosses scenarios and doesn't
    assert out.loc["a", "n_pairs"] == 1
    assert out.loc["a", "stability"] == 1.0


# --- AIT-ADS leave-one-scenario-out ----------------------------------------


def test_loso_folds_hold_out_one_scenario_each():
    pool = {
        "a": _timed([(0, 1), (1, 0)]),
        "b": _timed([(0, 0), (1, 1), (2, 0)]),
        "c": _timed([(0, 0)]),
    }
    folds = loso_folds(pool)
    assert [s for s, _, _ in folds] == ["a", "b", "c"]
    for s, train, test in folds:
        assert test is pool[s]
        assert len(train) == sum(len(g) for t, g in pool.items() if t != s)
        assert not {id(g) for g in train} & {id(g) for g in test}


def test_loso_folds_subset_of_held_out_scenarios():
    pool = {"a": _timed([(0, 1)]), "b": _timed([(0, 0)]), "c": _timed([(0, 0)])}
    folds = loso_folds(pool, ("b",))
    assert len(folds) == 1
    s, train, test = folds[0]
    assert s == "b" and len(train) == 2  # still trains on every other scenario
    with pytest.raises(ValueError, match="not in the pool"):
        loso_folds(pool, ("zzz",))


def test_merge_runs_renumbers_steps_and_recomputes_stats(tmp_path):
    runs = []
    for k, scen in enumerate(["a", "b", "c"]):
        d = tmp_path / "folds" / scen
        d.mkdir(parents=True)
        rows = [
            {
                "arm": arm,
                "seed": 0,
                "scenario": scen,
                "step_i": 0,
                "f1": f1 + 0.01 * k,
                "pr_auc": f1 + 0.01 * k,
                "tp_seen": 1,
                "n_attack_seen": 2,
                "tp_unseen": 0,
                "n_attack_unseen": 0,
            }
            for arm, f1 in ((NO_MINING, 0.2), ("m1", 0.4), ("m2", 0.3))
        ]
        pd.DataFrame(rows).to_csv(d / "per_step_results.csv", index=False)
        pd.DataFrame(
            {
                "arm": ["m1"],
                "mining_seed": [0],
                "scenario": [scen],
                "step_i": [0],
                "condition": ["x"],
            }
        ).to_csv(d / "features.csv", index=False)
        pd.DataFrame({"step_i": [0], "scenario": [scen]}).to_csv(
            d / "windows.csv", index=False
        )
        runs.append(d)
    out = merge_runs(runs, tmp_path / "merged")
    ps = pd.read_csv(out / "per_step_results.csv")
    assert dict(ps.groupby("scenario")["step_i"].first()) == {"a": 0, "b": 1, "c": 2}
    assert set(pd.read_csv(out / "features.csv")["step_i"]) == {0, 1, 2}
    hh = pd.read_csv(out / "helps_hurts.csv")
    m1 = hh[(hh["arm"] == "m1") & (hh["metric"] == "f1")].iloc[0]
    assert m1["n_steps"] == 3 and m1["helps"] == 3
    stats = json.loads((out / "stats.json").read_text())
    assert stats["f1"]["avg_rank"]["m1"] == 1.0
    # one step per scenario: no consecutive pairs for stability
    assert (pd.read_csv(out / "stability.csv")["n_pairs"] == 0).all()


def test_merge_runs_refuses_an_unfinished_fold(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="unfinished"):
        merge_runs([tmp_path / "empty"], tmp_path / "out")

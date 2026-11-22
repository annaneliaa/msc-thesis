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
    ait_ads_split_steps,
    attack_phases,
    loso_folds,
    merge_runs,
    mining_cache_key,
    phase_cuts,
    phase_split,
    firing_overlap,
    friedman_nemenyi,
    helps_hurts,
    redundancy_share,
    run_comparison,
    run_step,
    scenario_split,
    stability,
    summarize_comparison,
)
from thesis.system_eval.mining_mini_experiment import MiniConfig

from test_mining_mini_experiment import _ait_groups, _cscas_group, _methods


@pytest.fixture(scope="module")
def step_run():
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
    scores: list[pd.DataFrame] = []
    rows, feats = run_step(config, step, spec, scores)
    return (
        pd.DataFrame(rows),
        pd.concat(feats, ignore_index=True),
        pd.concat(scores, ignore_index=True),
    )


@pytest.fixture(scope="module")
def step_rows(step_run):
    rows, feats, _ = step_run
    return rows, feats


def test_run_step_saves_test_scores_matching_the_rows(step_run):
    from sklearn.metrics import average_precision_score

    rows, _, scores = step_run
    assert "_proba" not in rows.columns
    assert len(scores) == len(rows) * 300  # every (arm, seed) x 300 test groups
    for _, r in rows.iterrows():
        sc = scores[(scores["arm"] == r["arm"]) & (scores["seed"] == r["seed"])]
        assert sc["test_idx"].tolist() == list(range(300))
        assert average_precision_score(sc["y"], sc["proba"]) == pytest.approx(
            r["pr_auc"], abs=1e-6
        )
        assert int((sc["pred"] & (sc["y"] == 1)).sum()) == r["tp"]


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


def test_scenario_split_is_the_thesis_70_30_cut():
    groups = _timed([(t, t >= 6) for t in range(10)])
    train, test = scenario_split(groups)
    assert [g.start_ts for g in train] == list(range(7))  # int(10 * 0.7)
    assert [g.start_ts for g in test] == [7, 8, 9]
    train, test = scenario_split(groups[:9])
    assert len(train) == 6 and len(test) == 3  # int(9 * 0.7) = 6


def _split_pool(monkeypatch):
    from thesis.system_eval import mining_comparison as mc

    pool = {
        # attacks in both parts
        "a": _timed([(t, t in (2, 8)) for t in range(10)]),
        # attacks only in the last 30%: skipped
        "b": _timed([(t, t >= 8) for t in range(10)]),
        # attacks only in the first 70%: kept, with a warning
        "c": _timed([(t, t == 1) for t in range(10)]),
    }
    monkeypatch.setattr(mc, "_load_ait_ads_groups", lambda config, s: pool[s])
    return pool


def test_split_steps_skip_scenarios_without_training_attacks(monkeypatch):
    pool = _split_pool(monkeypatch)
    config = ComparisonConfig(
        dataset="ait_ads", scenarios=("a", "b", "c"), methods=("eclat",)
    )
    assert config.ait_ads_protocol == "split"  # the default
    steps, windows, skipped = ait_ads_split_steps(config)
    assert list(windows["scenario"]) == ["a", "c"]
    assert list(windows["step_i"]) == [0, 1]
    assert list(windows["n_train"]) == [7, 7] and list(windows["n_test"]) == [3, 3]
    assert list(windows["n_test_attack"]) == [1, 0]
    assert list(skipped["scenario"]) == ["b"]
    assert skipped["reason"].iloc[0] == "no training attacks"
    got = [(st.scenario, st.i, len(st.train), len(st.test)) for st, _ in steps()]
    assert got == [("a", 0, 7, 3), ("c", 1, 7, 3)]
    step, _ = next(iter(steps()))
    assert step.train == pool["a"][:7] and step.test == pool["a"][7:]

    # test_scenarios narrows the run to those scenarios only
    one = ComparisonConfig(
        dataset="ait_ads",
        scenarios=("a", "b", "c"),
        test_scenarios=("c",),
        methods=("eclat",),
    )
    _, windows, skipped = ait_ads_split_steps(one)
    assert list(windows["scenario"]) == ["c"] and skipped.empty


def test_split_run_of_a_skipped_scenario_and_its_merge(monkeypatch, tmp_path):
    _split_pool(monkeypatch)
    config = ComparisonConfig(
        dataset="ait_ads",
        scenarios=("a", "b", "c"),
        test_scenarios=("b",),
        methods=("eclat",),
        out_dir=tmp_path / "folds" / "b",
    )
    out = run_comparison(config)
    assert pd.read_csv(out / "skipped.csv")["scenario"].tolist() == ["b"]
    assert not (out / "per_step_results.csv").exists()

    # merge: the skipped run is listed, not an error
    done = tmp_path / "folds" / "a"
    done.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "arm": arm,
                "seed": 0,
                "scenario": "a",
                "step_i": 0,
                "f1": f,
                "pr_auc": f,
                "tp_seen": 1,
                "n_attack_seen": 2,
                "tp_unseen": 0,
                "n_attack_unseen": 0,
            }
            for arm, f in ((NO_MINING, 0.2), ("m1", 0.4))
        ]
    ).to_csv(done / "per_step_results.csv", index=False)
    merged = merge_runs([done, out], tmp_path / "merged")
    assert pd.read_csv(merged / "skipped.csv")["scenario"].tolist() == ["b"]
    assert pd.read_csv(merged / "per_step_results.csv")[
        "scenario"
    ].unique().tolist() == ["a"]
    sources = json.loads((merged / "sources.json").read_text())
    assert [x.get("skipped", False) for x in sources] == [False, True]


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


# --- Two-stage runs: mining cache -------------------------------------------


def _cscas_spec_and_step():
    rng = random.Random(0)
    groups = [_cscas_group(i, rng) for i in range(900)]
    cols = list(BaselineFeatureEncoder().transform(groups[:5]).columns)
    base = FeatureSchema("base", "1", base=BaseFeatureSchema(features=cols))
    spec = DatasetSpec("cscas", base, lambda g: (g.category, g.ruleset))
    return spec, Step(0, groups[:600], groups[600:], test_days=1.0)


def _score_cols(rows):
    cols = ["arm", "seed", "f1", "pr_auc", "n_symbolic", "n_gated", "n_raw"]
    return pd.DataFrame(rows)[cols].sort_values(["arm", "seed"]).reset_index(drop=True)


def test_two_stage_matches_one_pass(tmp_path):
    spec, step = _cscas_spec_and_step()
    base = dict(seeds=(0, 1), mining=MiniConfig(k=10, n_bins=4))
    one_pass, _ = run_step(ComparisonConfig(**base), step, spec)

    cache = tmp_path / "mcache"
    mined, _ = run_step(
        ComparisonConfig(**base, stage="mine", mining_cache_dir=cache), step, spec
    )
    assert mined == []  # mining stage trains nothing
    # deterministic methods cached once, RIPPER once per seed
    assert len(list((cache / "eclat").glob("*.pkl"))) == 1
    assert len(list((cache / "ripper").glob("*.pkl"))) == 2

    trained, _ = run_step(
        ComparisonConfig(**base, stage="train", mining_cache_dir=cache), step, spec
    )
    pd.testing.assert_frame_equal(_score_cols(one_pass), _score_cols(trained))


def test_train_stage_reuses_cache_at_another_k(tmp_path):
    spec, step = _cscas_spec_and_step()
    cache = tmp_path / "mcache"
    cfg = dict(seeds=(0,), methods=("eclat",), include_no_mining=False)
    run_step(
        ComparisonConfig(
            **cfg,
            mining=MiniConfig(k=10, n_bins=4),
            stage="mine",
            mining_cache_dir=cache,
        ),
        step,
        spec,
    )
    rows, _ = run_step(
        ComparisonConfig(
            **cfg,
            mining=MiniConfig(k=4, n_bins=4),
            stage="train",
            mining_cache_dir=cache,
        ),
        step,
        spec,
    )
    assert rows[0]["n_symbolic"] <= 4  # K applied at training time


def test_train_stage_refuses_to_mine(tmp_path):
    spec, step = _cscas_spec_and_step()
    config = ComparisonConfig(
        seeds=(0,), methods=("eclat",), stage="train", mining_cache_dir=tmp_path
    )
    with pytest.raises(FileNotFoundError, match="run --stage mine first"):
        run_step(config, step, spec)


def test_mining_cache_key_tracks_data_and_settings_not_budget():
    spec, step = _cscas_spec_and_step()
    mini = MiniConfig(k=10, n_bins=4)
    key = mining_cache_key("eclat", step, spec, mini, "T2")
    from dataclasses import replace as dc_replace

    assert key == mining_cache_key(
        "eclat", step, spec, dc_replace(mini, k=60, per_class_budget=False), "T2"
    )
    assert key != mining_cache_key("eclat", step, spec, dc_replace(mini, seed=1), "T2")
    assert key != mining_cache_key(
        "eclat", step, spec, dc_replace(mini, eclat_min_support=0.1), "T2"
    )
    assert key != mining_cache_key("ripper", step, spec, mini, "T2")
    shorter = Step(0, step.train[:-1], step.test, 1.0)
    assert key != mining_cache_key("eclat", shorter, spec, mini, "T2")


# --- Mixed groups as attack (H1 runs) ---------------------------------------


def test_relabel_mixed_as_attack_and_cache_key():
    from thesis.system_eval.mining_comparison import relabel_mixed_as_attack
    from thesis.system_eval.mining_mini_experiment import labelled

    groups = _timed([(0, 0), (1, 1), (2, 0)])
    groups[2].group_label = "mixed"
    spec, _ = _cscas_spec_and_step()
    mini = MiniConfig()
    dropped = Step(0, labelled(groups), [], 1.0)
    assert [g.group_id for g in dropped.train] == ["g0", "g1"]

    assert relabel_mixed_as_attack(groups) == 1
    assert [g.group_label for g in groups] == ["benign", "attack", "attack"]
    kept = Step(0, labelled(groups), [], 1.0)
    assert len(kept.train) == 3
    # the mixed group now enters the training rows, so the mined candidates
    # are cached under a different key: the two rules never share a cache entry
    assert mining_cache_key("eclat", kept, spec, mini, "T2") != mining_cache_key(
        "eclat", dropped, spec, mini, "T2"
    )


def test_merge_runs_combines_test_scores_and_bootstraps(tmp_path):
    from thesis.system_eval.mining_comparison import SCORES_FILE

    rng = np.random.default_rng(0)
    runs = []
    for scen in ("a", "b"):
        d = tmp_path / "folds" / scen
        d.mkdir(parents=True)
        y = np.r_[np.ones(10, int), np.zeros(40, int)]
        rows, frames = [], []
        for arm, signal in ((NO_MINING, 0.0), ("m1", 0.8)):
            proba = signal * y + rng.random(50) * (1 - signal)
            rows.append(
                {
                    "arm": arm,
                    "seed": 0,
                    "scenario": scen,
                    "step_i": 0,
                    "f1": signal,
                    "pr_auc": signal,
                    "tp_seen": 1,
                    "n_attack_seen": 2,
                    "tp_unseen": 0,
                    "n_attack_unseen": 0,
                }
            )
            frames.append(
                pd.DataFrame(
                    {
                        "step_i": 0,
                        "scenario": scen,
                        "arm": arm,
                        "seed": 0,
                        "test_idx": np.arange(50),
                        "y": y,
                        "proba": proba,
                        "pred": proba >= 0.5,
                    }
                )
            )
        pd.DataFrame(rows).to_csv(d / "per_step_results.csv", index=False)
        pd.concat(frames).to_csv(d / SCORES_FILE, index=False)
        runs.append(d)
    out = merge_runs(runs, tmp_path / "merged")
    sc = pd.read_csv(out / SCORES_FILE)
    assert dict(sc.groupby("scenario")["step_i"].first()) == {"a": 0, "b": 1}
    boot = pd.read_csv(out / "bootstrap.csv")
    assert set(boot["scenario"]) == {"a", "b", "all"}
    row = boot[(boot["scenario"] == "all") & (boot["metric"] == "pr_auc")].iloc[0]
    assert {row["a"], row["b"]} == {NO_MINING, "m1"} and row["n_steps"] == 2
    assert (out / "bootstrap.txt").exists()


# --- AIT-ADS shortcut checks (no host item, no hour_of_day) ------------------


def test_base_schema_can_drop_hour_of_day():
    full = ait_ads_base_schema()
    assert "hour_of_day" in full.base.features
    no_hour = ait_ads_base_schema(frozenset({"hour_of_day"}))
    assert no_hour.base.features == [
        f for f in full.base.features if f != "hour_of_day"
    ]
    with pytest.raises(ValueError, match="not AIT-ADS base features"):
        ait_ads_base_schema(frozenset({"hour"}))


def test_shortcut_options_reach_the_spec_and_the_cache_key():
    from dataclasses import replace as dc_replace

    from thesis.mining.common.item_table import build_item_table
    from thesis.system_eval.mining_comparison import _ait_ads_spec

    from test_mining_mini_experiment import _ait_groups

    groups = _ait_groups()
    base = ComparisonConfig(dataset="ait_ads", methods=("eclat",))
    no_host = dc_replace(base, extra_exclude_fields=frozenset({"host"}))
    no_hour = dc_replace(base, base_exclude_features=frozenset({"hour_of_day"}))
    attach = lambda gs: None  # noqa: E731
    spec, spec_nh, spec_nhr = (
        _ait_ads_spec(c, ["fox"], attach) for c in (base, no_host, no_hour)
    )
    assert "host" in spec_nh.exclude_fields and "host" not in spec.exclude_fields
    assert "hour_of_day" not in spec_nhr.base_schema.base.features
    assert "hour_of_day" in spec.base_schema.base.features

    fields = lambda s: {  # noqa: E731
        c.split("=")[0]
        for c in build_item_table(
            groups, exclude_fields=s.exclude_fields, include_fields=s.include_fields
        )
        .items()
        .columns
    }
    assert "host" in fields(spec) and "host" not in fields(spec_nh)

    step = Step(0, groups[:100], groups[100:], 1.0, scenario="fox")
    mini = MiniConfig(k=10, n_bins=4)
    key = mining_cache_key("eclat", step, spec, mini, "T2")
    assert key != mining_cache_key("eclat", step, spec_nh, mini, "T2")  # re-mine
    assert key == mining_cache_key("eclat", step, spec_nhr, mini, "T2")  # reuse

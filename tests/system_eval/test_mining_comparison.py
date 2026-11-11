"""Mining comparison runner on a synthetic CSCAS-like step: every arm runs
end to end through the shared treatment, plus the summary statistics."""

import random

import numpy as np
import pandas as pd
import pytest

from thesis.encoders.baseline import BaselineFeatureEncoder
from thesis.schemas.features import BaseFeatureSchema, FeatureSchema
from thesis.system_eval.mining_comparison import (
    NO_MINING,
    ComparisonConfig,
    Step,
    firing_overlap,
    friedman_nemenyi,
    helps_hurts,
    redundancy_share,
    run_step,
    stability,
    summarize_comparison,
)
from thesis.system_eval.mining_mini_experiment import MiniConfig

from test_mining_mini_experiment import _cscas_group


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
    rows, feats = run_step(config, step, base, lambda g: (g.category, g.ruleset), set())
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

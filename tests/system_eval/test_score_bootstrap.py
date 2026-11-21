"""Paired bootstrap over test groups (AIT-ADS has too few steps for a test
across steps)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from thesis.system_eval.score_bootstrap import (
    ALL,
    average_precision,
    paired_bootstrap,
)


@pytest.mark.parametrize("seed", range(5))
def test_average_precision_equals_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(300) < 0.1).astype(int)
    y[0] = 1
    score = np.round(rng.random(300) + 0.5 * y, 1)  # coarse: many ties
    assert average_precision(y, score) == pytest.approx(
        average_precision_score(y, score)
    )
    assert np.isnan(average_precision(np.zeros(5, int), np.ones(5)))


def _scores(n_steps=3, n=400, seeds=(0, 1)):
    """Arm "good" separates the classes, "noise" doesn't, "copy" == "noise"."""
    rng = np.random.default_rng(0)
    frames = []
    for step in range(n_steps):
        y = (rng.random(n) < 0.1).astype(int)
        for seed in seeds:
            noise = rng.random(n)
            for arm, proba in (
                ("good", 0.6 * y + 0.4 * rng.random(n)),
                ("noise", noise),
                ("copy", noise),
            ):
                frames.append(
                    pd.DataFrame(
                        {
                            "step_i": step,
                            "scenario": f"s{step}",
                            "arm": arm,
                            "seed": seed,
                            "test_idx": np.arange(n),
                            "y": y,
                            "proba": proba,
                            "pred": proba >= 0.5,
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def test_clear_winner_is_significant_identical_arms_are_not():
    out = paired_bootstrap(_scores(), n_boot=200)
    key = out.set_index(["scenario", "metric", "a", "b"])
    for scen in ("s0", "s1", "s2", ALL):
        good = key.loc[(scen, "pr_auc", "good", "noise")]
        assert good["delta"] > 0.5 and good["ci_low"] > 0 and good["significant"]
        assert good["p_boot"] < 0.05
        same = key.loc[(scen, "pr_auc", "noise", "copy")]
        assert same["delta"] == 0 and same["p_boot"] == 1.0
        assert not same["significant"]
    assert key.loc[(ALL, "f1", "good", "noise"), "n_steps"] == 3
    # point delta = seed-mean metric difference on the full test set
    sc = _scores()
    s0 = sc[sc["step_i"] == 0]
    want = np.mean(
        [
            average_precision_score(g["y"], g["proba"])
            for _, g in s0[s0["arm"] == "good"].groupby("seed")
        ]
    ) - np.mean(
        [
            average_precision_score(g["y"], g["proba"])
            for _, g in s0[s0["arm"] == "noise"].groupby("seed")
        ]
    )
    assert key.loc[("s0", "pr_auc", "good", "noise"), "delta"] == pytest.approx(want)


def test_steps_without_test_attacks_are_left_out():
    sc = _scores(n_steps=2)
    sc.loc[sc["step_i"] == 1, "y"] = 0
    out = paired_bootstrap(sc, n_boot=50)
    assert set(out["scenario"]) == {"s0", ALL}
    assert (out.loc[out["scenario"] == ALL, "n_steps"] == 1).all()

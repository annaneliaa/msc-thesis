import numpy as np
import pandas as pd

from thesis.system_eval.mining_setting_selection import (
    STAGE2_GRID,
    full_stage2_grid,
    select_by_metric,
    select_thesis_rule,
)


def _grid(prec_attack, recall_benign, f1, std=0.01) -> pd.DataFrame:
    rows = []
    for p in full_stage2_grid():
        rows.append(
            {
                **p,
                "val_precision_attack": prec_attack(p),
                "val_precision_attack_boot_std": std,
                "val_recall_benign": recall_benign(p),
                "val_recall_benign_boot_std": std,
                "val_f1_mean": f1(p),
                "val_f1_std": std,
                "n_features": 10 * p["max_diverse_rounds"],
            }
        )
    return pd.DataFrame(rows)


def test_full_grid_size():
    assert (
        len(full_stage2_grid())
        == int(np.prod([len(v) for v in STAGE2_GRID.values()]))
        == 54
    )


def test_thesis_rule_picks_shallowest_attack_depth_on_plateau():
    # precision plateau over {2,3,4} (within one std), benign recall drops with depth
    g = _grid(
        prec_attack=lambda p: {2: 0.90, 3: 0.905, 4: 0.91}[p["max_depth_attack"]],
        recall_benign=lambda p: {1: 0.99, 2: 0.6}[p["max_depth"]],
        f1=lambda p: 0.8,
    )
    pick = select_thesis_rule(g)
    assert pick["max_depth"] == 1
    assert pick["max_depth_attack"] == 2
    # flat precision across msl -> tie goes to the larger leaf
    assert pick["min_samples_leaf"] == 20
    # flat F1 across rounds -> fewest rounds
    assert pick["max_diverse_rounds"] == 1


def test_thesis_rule_keeps_deeper_attack_tree_when_clearly_better():
    g = _grid(
        prec_attack=lambda p: {2: 0.70, 3: 0.80, 4: 0.90}[p["max_depth_attack"]],
        recall_benign=lambda p: 0.99,
        f1=lambda p: 0.8,
    )
    assert select_thesis_rule(g)["max_depth_attack"] == 4


def test_f1_rule_prefers_simplest_within_one_std():
    g = _grid(
        prec_attack=lambda p: 0.9,
        recall_benign=lambda p: 0.99,
        f1=lambda p: 0.80
        + 0.005 * (p["max_diverse_rounds"] == 2)
        + 0.05 * (p["max_depth"] == 2),
    )
    pick = select_by_metric(g, "val_f1")
    # md=2 is clearly best (+0.05 >> std); within it rounds=2's +0.005 is inside one std -> rounds=1
    assert pick["max_depth"] == 2
    assert pick["max_diverse_rounds"] == 1
    assert pick["max_depth_attack"] == 2
    assert pick["min_samples_leaf"] == 20


def test_tolerance_floor_when_best_has_zero_seed_std():
    # The best setting's seeds agree exactly (std 0), but its F1 edge over the
    # simpler setting is far below the grid's typical noise -> still a tie.
    g = _grid(
        prec_attack=lambda p: 0.9,
        recall_benign=lambda p: 0.99,
        f1=lambda p: 0.90 + 0.0005 * (p["max_diverse_rounds"] == 3),
        std=0.01,
    )
    best = g["val_f1_mean"].idxmax()
    g.loc[best, "val_f1_std"] = 0.0
    pick = select_by_metric(g, "val_f1")
    assert pick["tolerance"] == 0.01
    assert pick["max_diverse_rounds"] == 1

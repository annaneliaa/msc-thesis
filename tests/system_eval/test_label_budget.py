from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis.system_eval.label_budget import (
    LabelLedger,
    budget_view,
    draw_budget,
    mask_labels,
)
from thesis.schemas.groups import AlertGroup


def _g(i: int, label: str | None) -> AlertGroup:
    return AlertGroup(
        alert_group_id=f"g{i}",
        group_id=f"g{i}",
        method="cscas_pregrouped",
        start_ts=1_000 + i,
        end_ts=1_000 + i,
        n_alerts=1,
        group_label=label,
    )


def _rows(n=1000, attack_every=10):
    return [_g(i, "attack" if i % attack_every == 0 else "benign") for i in range(n)]


def test_mask_labels_hides_unkept_rows_without_mutating():
    rows = _rows(4)
    out = mask_labels(rows, [True, False, True, False])
    assert [g.group_label for g in out] == ["attack", None, "benign", None]
    assert out[0] is rows[0]
    assert rows[1].group_label == "benign"  # original untouched


def test_full_budget_is_every_labelled_row():
    rows = _rows(10) + [_g(99, None)]
    assert draw_budget(rows, "full").tolist() == [True] * 10 + [False]


def test_stratified_budget_keeps_size_and_class_ratio():
    rows = _rows(1000, attack_every=10)  # 10% attack
    keep = draw_budget(rows, "stratified", size=200, seed=1)
    picked = [g for g, k in zip(rows, keep) if k]
    assert len(picked) == 200
    assert sum(g.group_label == "attack" for g in picked) == 20
    # Deterministic per seed, different across seeds.
    assert (draw_budget(rows, "stratified", size=200, seed=1) == keep).all()
    assert not (draw_budget(rows, "stratified", size=200, seed=2) == keep).all()


def test_stratified_budget_keeps_one_row_of_a_rare_class():
    rows = _rows(1000, attack_every=1000)  # a single attack
    picked = [g for g, k in zip(rows, draw_budget(rows, "stratified", size=50)) if k]
    assert sum(g.group_label == "attack" for g in picked) == 1


def test_deepcase_budget_modes_read_analyst_items(tmp_path):
    rows = _rows(20)
    path = tmp_path / "analyst_items.parquet"
    pd.DataFrame(
        {
            "group_id": ["g1", "g2", "g3", "g_not_in_window"],
            "kind": ["cluster_sample", "outlier", "outlier", "cluster_sample"],
        }
    ).to_parquet(path)
    items = draw_budget(rows, "deepcase_items", items_path=path)
    assert np.flatnonzero(items).tolist() == [1, 2, 3]
    samples = draw_budget(rows, "deepcase_cluster_samples", items_path=path)
    assert np.flatnonzero(samples).tolist() == [1]


def test_budget_view_keeps_escalated_and_audit_only():
    rows = _rows(2000)
    flagged = np.array([g.group_label == "attack" for g in rows])
    view, esc, aud = budget_view(rows, flagged, audit_rate=0.1)
    assert esc.sum() == flagged.sum()
    assert not (esc & aud).any()
    assert 0.05 < aud.sum() / (~flagged).sum() < 0.15
    kept = np.array([g.group_label is not None for g in view])
    assert (kept == (esc | aud)).all()


def test_ledger_pools_labels_cumulatively_with_audit_weights():
    ledger = LabelLedger(audit_rate=0.1)
    b0 = _rows(50)
    ledger.add_initial(b0 + [_g(999, None)])  # unlabelled rows never count
    assert ledger.n_b0 == 50

    w1 = [_g(100 + i, "attack" if i % 10 == 0 else "benign") for i in range(500)]
    flagged = np.array([g.group_label == "attack" for g in w1])
    _, cols1 = ledger.add_window(w1, flagged)
    assert cols1["ledger_n_escalated_labels"] == flagged.sum()
    n_aud = cols1["ledger_n_audit_labels"]
    assert cols1["ledger_pool_size"] == 50 + flagged.sum() + n_aud
    assert cols1["ledger_budget_labels"] == 50 + n_aud  # escalated = workload
    assert cols1["ledger_workload_labels"] == flagged.sum()

    # Re-adding the same window adds nothing (no double counting).
    _, cols2 = ledger.add_window(w1, flagged)
    assert cols2["ledger_n_escalated_labels"] == 0
    assert cols2["ledger_pool_size"] == cols1["ledger_pool_size"]

    groups, weights = ledger.training_set()
    assert len(groups) == len(weights) == cols1["ledger_pool_size"]
    assert sorted(set(weights.tolist())) == [1.0, pytest.approx(10.0)]
    assert [g.start_ts for g in groups] == sorted(g.start_ts for g in groups)

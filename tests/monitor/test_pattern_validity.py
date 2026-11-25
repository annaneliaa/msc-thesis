from __future__ import annotations

import numpy as np
import pytest

from thesis.monitor.noise_aware import WindowRows
from thesis.monitor.pattern_validity import (
    MIN_GROWTH_RATE,
    audit_mask,
    compare,
    pattern_stats,
)
from thesis.schemas.groups import AlertGroup

IDS = ["attack_pat", "benign_pat"]
TO_ATTACK = np.array([True, False])


def _rows(
    n_attack=200,
    n_benign=600,
    attack_fires_attack_pat=1.0,
    benign_fires_attack_pat=0.02,
    attack_fires_benign_pat=0.02,
    flagged_attack=1.0,
    seed=0,
) -> WindowRows:
    """Column 0: attack-direction predicate, column 1: benign-direction one."""
    rng = np.random.default_rng(seed)
    is_attack = np.r_[np.ones(n_attack), np.zeros(n_benign)].astype(np.float32)
    n = len(is_attack)
    a = is_attack > 0
    p0 = np.where(a, attack_fires_attack_pat, benign_fires_attack_pat)
    p1 = np.where(a, attack_fires_benign_pat, 0.8)
    fires = np.stack([rng.random(n) < p0, rng.random(n) < p1], axis=1).astype(
        np.float32
    )
    # Escalated: flagged_attack share of the attacks + the benign rows the
    # attack pattern fires on (false positives).
    flagged = np.where(a, rng.random(n) < flagged_attack, fires[:, 0] > 0)
    return WindowRows(
        ts=np.arange(n, dtype=np.int64),
        fires=fires,
        rule_match_labeled=np.zeros((n, 0), dtype=np.float32),
        rule_correct=np.zeros((n, 0), dtype=np.float32),
        score_bins=None,
        flagged=flagged.astype(np.float32),
        near_margin=None,
        novel=np.zeros(n, dtype=np.float32),
        labeled=np.ones(n, dtype=np.float32),
        is_attack=is_attack,
    )


def _compare(win, ref, mode="oracle", audit=None, rate=0.05):
    s = [pattern_stats(r, TO_ATTACK, mode, audit, 30, rate) for r in (win, ref)]
    return compare(s[0], s[1], IDS, mode)


def test_same_distribution_is_quiet():
    out = _compare(_rows(seed=1), _rows(seed=0))
    assert out["pv_oracle_n_checked"] == 2
    assert out["pv_oracle_n_invalid"] == 0
    assert out["pv_oracle_n_prec_drop"] == 0
    assert out["pv_oracle_elevated"] is False


def test_benign_pattern_firing_on_attacks_is_invalid():
    # Half the attacks now look benign: the benign pattern's growth rate
    # (0.8 / 0.5) falls below the miner's 3, and its precision drops.
    out = _compare(_rows(attack_fires_benign_pat=0.5), _rows())
    assert out["pv_oracle_invalid_patterns"] == "benign_pat"
    assert out["pv_oracle_worst_gr_ratio"] < 1
    assert "benign_pat" in out["pv_oracle_prec_drop_patterns"]
    assert out["pv_oracle_contrast_elevated"] and out["pv_oracle_elevated"]


def test_attacks_losing_pattern_coverage():
    out = _compare(_rows(attack_fires_attack_pat=0.3), _rows())
    assert out["pv_oracle_cov_attack"] < 0.5 < out["pv_oracle_cov_attack_ref"]
    assert out["pv_oracle_coverage_elevated"] is True


def test_escalated_labels_miss_suppressed_attacks_audit_sees_them():
    # The attacks that now look benign are the ones the model suppresses.
    ref = _rows(flagged_attack=1.0)
    win = _rows(attack_fires_benign_pat=0.5, flagged_attack=1.0)
    a = win.is_attack > 0
    win.flagged[a] = (win.fires[a, 1] == 0).astype(np.float32)
    esc = _compare(win, ref, "escalated")
    assert esc["pv_escalated_n_invalid"] == 0
    # Audit everything at rate 1 -> recovers the oracle answer.
    full = np.ones(len(win), dtype=np.float32)
    s_win = pattern_stats(win, TO_ATTACK, "audit", full, 30, 1.0)
    s_ref = pattern_stats(ref, TO_ATTACK, "audit", np.ones(len(ref)), 30, 1.0)
    aud = compare(s_win, s_ref, IDS, "audit")
    assert aud["pv_audit_invalid_patterns"] == "benign_pat"


def test_audit_weights_are_unbiased():
    rows = _rows(n_attack=2000, n_benign=6000, flagged_attack=0.5, seed=3)
    rng = np.random.default_rng(0)
    audit = (rng.random(len(rows)) < 0.2).astype(np.float32)
    o = pattern_stats(rows, TO_ATTACK, "oracle", None, 30)
    w = pattern_stats(rows, TO_ATTACK, "audit", audit, 30, 0.2)
    assert w.supp_dir == pytest.approx(o.supp_dir, abs=0.05)
    assert o.gr[0] >= MIN_GROWTH_RATE


def test_audit_mask_is_deterministic_and_near_rate():
    gs = [
        AlertGroup(
            alert_group_id=f"g{i}",
            group_id=f"g{i}",
            method="x",
            start_ts=0,
            end_ts=None,
            n_alerts=1,
        )
        for i in range(4000)
    ]
    m1, m2 = audit_mask(gs, 0.05), audit_mask(gs, 0.05)
    assert (m1 == m2).all()
    assert 0.03 < m1.mean() < 0.07


def test_no_scores_leaves_label_modes_empty():
    rows = _rows()
    rows.flagged = None
    assert pattern_stats(rows, TO_ATTACK, "escalated", None, 30) is None
    assert pattern_stats(rows, TO_ATTACK, "oracle", None, 30) is not None


def _stats(supp_dir, supp_oth, n):
    from thesis.monitor.pattern_validity import PatternStats

    a = np.array
    return PatternStats(
        gr=a([supp_dir / supp_oth]),
        supp_dir=a([supp_dir]),
        supp_oth=a([supp_oth]),
        n_dir=a([float(n)]),
        n_oth=a([float(n)]),
        checkable=True,
        prec=a([0.9]),
        n_fire=a([float(n)]),
        cov_attack=1.0,
        cov_benign=1.0,
        n_eff_attack=float(n),
        n_eff_benign=float(n),
        min_samples=30,
    )


def test_borderline_pattern_needs_confidence_to_fail():
    ref = _stats(0.30, 0.05, 2000)  # GR 6: valid
    # Point estimate GR 2.5 on a small sample: below 3, but not confidently.
    noisy = _stats(0.10, 0.04, 300)
    out = compare(noisy, ref, ["p"], "oracle")
    assert out["pv_oracle_n_invalid_point"] == 1
    assert out["pv_oracle_n_invalid"] == 0
    assert out["pv_oracle_contrast_elevated"] is False
    # Clearly broken (GR 1.2) on a decent sample: fails both ways.
    broken = _stats(0.30, 0.25, 2000)
    out = compare(broken, ref, ["p"], "oracle")
    assert out["pv_oracle_n_invalid"] == 1 and out["pv_oracle_contrast_elevated"]
    # Coverage clearly under 5%.
    gone = _stats(0.01, 0.001, 2000)
    assert compare(gone, ref, ["p"], "oracle")["pv_oracle_n_invalid"] == 1

import numpy as np
import pytest

from thesis.monitor.noise_aware import (
    NoiseAwareMonitor,
    WindowRows,
    build_window_rows,
    time_blocks,
)

from test_novelty import _exploit_predicate, _make_alert_group, _make_schema  # noqa: I001

HOUR = 3600


def _rows(
    n, p_fire, scores=None, rule_precision=0.9, hours=48, seed=0, t0=0, day_wobble=0.0
):
    """Synthetic rows spread evenly over `hours`: one predicate firing with
    prob p_fire (optionally wobbling +-day_wobble from day to day), one rule
    matching every labelled row with the given precision."""
    rng = np.random.default_rng(seed)
    ts = t0 + np.sort(rng.integers(0, hours * HOUR, n))
    p = p_fire + day_wobble * np.where(((ts - t0) // (24 * HOUR)) % 2 == 0, 1, -1)
    fires = (rng.random(n) < p).astype(np.float32)[:, None]
    match = np.ones((n, 1), dtype=np.float32)
    correct = (rng.random((n, 1)) < rule_precision).astype(np.float32)
    s = np.clip(rng.beta(1, 20, n) if scores is None else scores, 0, 1)
    bins = np.eye(10, dtype=np.float32)[
        np.clip(np.digitize(s, np.linspace(0, 1, 11)[1:-1]), 0, 9)
    ]
    return WindowRows(
        ts=ts,
        fires=fires,
        rule_match_labeled=match,
        rule_correct=correct,
        score_bins=bins,
        flagged=(s >= 0.5).astype(np.float32),
        near_margin=(np.abs(s - 0.5) < 0.2).astype(np.float32),
        novel=np.zeros(n, dtype=np.float32),
    )


def _monitor(ref):
    return NoiseAwareMonitor(input_reference=ref, output_reference=ref, min_samples=30)


def test_stable_window_is_not_elevated():
    mon = _monitor(_rows(20_000, 0.3, hours=72, day_wobble=0.02, seed=0))
    out = mon.evaluate(_rows(50_000, 0.3, hours=120, seed=1))
    for k in (
        "s1_max_psi",
        "s2_max_drift",
        "score_psi",
        "flag_rate_shift",
        "margin_shift",
    ):
        assert out[f"na_{k}_elevated"] is False, k


def test_threshold_reflects_day_to_day_wobble():
    calm = _monitor(_rows(20_000, 0.3, hours=72, seed=0))
    wobbly = _monitor(_rows(20_000, 0.3, hours=72, day_wobble=0.05, seed=0))
    assert wobbly.thresholds["s1_max_psi"] > calm.thresholds["s1_max_psi"]


def test_shifted_predicate_rate_is_elevated():
    mon = _monitor(_rows(20_000, 0.3, hours=72, day_wobble=0.02))
    out = mon.evaluate(_rows(50_000, 0.5, hours=120, seed=1))
    assert out["na_s1_max_psi_elevated"] is True


def test_precision_drop_is_elevated():
    mon = _monitor(_rows(20_000, 0.3, hours=72))
    out = mon.evaluate(_rows(50_000, 0.3, rule_precision=0.7, hours=120, seed=1))
    assert out["na_s2_max_drift_elevated"] is True


def test_score_shift_is_elevated():
    mon = _monitor(_rows(20_000, 0.3, hours=72))
    shifted = np.random.default_rng(1).beta(2, 6, 50_000)
    out = mon.evaluate(_rows(50_000, 0.3, scores=shifted, hours=120, seed=1))
    assert out["na_score_psi_elevated"] is True
    assert out["na_flag_rate_shift_elevated"] is True


def test_too_few_blocks_gives_no_threshold():
    mon = _monitor(_rows(5_000, 0.3, hours=10))  # one 12h/24h block only
    assert np.isnan(mon.thresholds["s1_max_psi"])
    assert mon.evaluate(_rows(5_000, 0.5, seed=1))["na_s1_max_psi_elevated"] is None


def test_time_blocks_drop_small_trailing_block():
    ts = np.concatenate(
        [np.full(100, 0), np.full(100, 13 * HOUR), np.full(5, 25 * HOUR)]
    )
    blocks = time_blocks(ts, 12 * HOUR)
    assert [int(m.sum()) for m in blocks] == [100, 100]


def test_rule_below_min_samples_gives_nan_drift():
    mon = _monitor(_rows(20_000, 0.3, hours=72))
    out = mon.evaluate(_rows(10, 0.3, seed=1))
    assert np.isnan(out["na_s2_max_drift"])
    assert out["na_s2_max_drift_elevated"] is None


def test_build_window_rows_from_groups():
    schema = _make_schema([_exploit_predicate()])
    groups = [
        _make_alert_group("a", "attack", category="EXPLOIT"),
        _make_alert_group("b", "benign", category="SNMP"),
        _make_alert_group("c", None, category="EXPLOIT"),
    ]
    from thesis.mining.common.attribute_features import (
        compute_candidate_attribute_features,
    )

    feats = [compute_candidate_attribute_features(g) for g in groups]
    ref_patterns = frozenset({frozenset({"cat:category=EXPLOIT"})})
    rows = build_window_rows(
        schema, groups, feats, ref_patterns, np.array([0.9, 0.1, 0.6]), 0.5
    )
    assert rows.fires[:, 0].tolist() == [1, 0, 1]
    assert rows.novel.tolist() == [0, 1, 0]
    assert rows.flagged.tolist() == [1, 0, 1]
    assert rows.near_margin.tolist() == [0, 0, 1]
    assert rows.ts.tolist() == [1_642_636_800] * 3


def test_concat_drops_scores_when_one_side_has_none():
    a = _rows(10, 0.3)
    b = _rows(10, 0.3, seed=1)
    b.score_bins = b.flagged = b.near_margin = None
    c = a.concat(b)
    assert len(c) == 20 and c.score_bins is None


def test_empty_reference_rejected():
    with pytest.raises(ValueError):
        _monitor(_rows(0, 0.3))

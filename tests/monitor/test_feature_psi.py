import numpy as np
import pandas as pd

from thesis.monitor.feature_psi import FIXED_PSI, FeaturePsiMonitor

DAY = 24 * 3600


def _ref(n_days=5, per_day=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    X = pd.DataFrame(
        {
            "volume": rng.lognormal(2.0, 0.5, n),
            "flag": rng.random(n) < 0.3,
            "cat": rng.choice([0.0, 1.0, 2.0], n, p=[0.6, 0.3, 0.1]),
        }
    )
    ts = np.repeat(np.arange(n_days) * DAY, per_day)
    return X, ts, rng


def test_same_distribution_stays_below_threshold():
    X, ts, rng = _ref()
    m = FeaturePsiMonitor(X, ts)
    assert m.n_blocks == 5 and (m.thresholds >= 0.01).all()
    fresh, _, _ = _ref(n_days=1, seed=1)
    out = m.evaluate(fresh)
    assert out["fpsi_elevated"] is False
    assert out["fpsi_fixed_elevated"] is False
    assert out["fpsi_n_features"] == 3


def test_scale_shift_on_one_feature_is_caught_and_named():
    X, ts, _ = _ref()
    m = FeaturePsiMonitor(X, ts)
    shifted, _, _ = _ref(n_days=1, seed=2)
    shifted["volume"] *= 10
    out = m.evaluate(shifted)
    assert out["fpsi_top_feature"] == "volume"
    assert out["fpsi_max"] > FIXED_PSI
    assert out["fpsi_elevated"] is True and out["fpsi_fixed_elevated"] is True
    assert out["fpsi_n_over_fixed"] == 1
    assert out["fpsi_top_ratio_feature"] == "volume" and out["fpsi_n_elevated"] == 1


def test_noisy_feature_gets_its_own_higher_threshold():
    X, ts, rng = _ref()
    # "noisy" flips its distribution every day; "steady" never does.
    X["noisy"] = (ts // DAY) % 2 == 0
    m = FeaturePsiMonitor(X, ts)
    assert m.thresholds["noisy"] > 1.0 > m.thresholds["volume"]
    shifted, _, _ = _ref(n_days=1, seed=4)
    shifted["noisy"] = True
    shifted["volume"] *= 10
    out = m.evaluate(shifted)
    # The noisy feature moves a lot (raw PSI > the fixed 0.2 rule) but stays
    # within its own day-to-day range; only the volume shift is flagged.
    psi = m.psi_per_feature(shifted)
    assert psi["noisy"] > 0.2
    assert psi["noisy"] <= m.thresholds["noisy"]
    assert out["fpsi_top_ratio_feature"] == "volume" and out["fpsi_n_elevated"] == 1


def test_unseen_category_value_is_caught():
    X, ts, _ = _ref()
    m = FeaturePsiMonitor(X, ts)
    new, _, _ = _ref(n_days=1, seed=3)
    new.loc[: len(new) // 2, "cat"] = 7.0
    out = m.evaluate(new)
    assert out["fpsi_top_feature"] == "cat" and out["fpsi_elevated"] is True


def test_missing_columns_are_filled_and_single_block_has_no_threshold():
    X, _, _ = _ref(n_days=1)
    m = FeaturePsiMonitor(X, np.zeros(len(X)))
    assert m.thresholds is None
    out = m.evaluate(X.drop(columns=["flag"]))
    assert out["fpsi_elevated"] is None
    assert out["fpsi_fixed_elevated"] is not None

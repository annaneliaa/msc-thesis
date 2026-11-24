import numpy as np

from thesis.monitor.signal_log import LOGGED_COLUMNS, SignalLog

from test_novelty import _exploit_predicate, _make_alert_group, _make_schema  # noqa: I001

DAY = 86_400


def _groups(prefix, n, t0, hours, explo_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    ts = t0 + np.sort(rng.integers(0, hours * 3600, n))
    return [
        _make_alert_group(
            f"{prefix}{i}",
            "attack" if i % 10 == 0 else "benign",
            start_ts=int(t),
            category="EXPLOIT" if rng.random() < explo_frac else "SNMP",
        )
        for i, t in enumerate(ts)
    ]


def _scores(groups, seed=0):
    return np.random.default_rng(seed).beta(1, 20, len(groups))


def test_logs_every_column_and_flags_initial_reference():
    schema = _make_schema([_exploit_predicate()])
    train = _groups("t", 3000, 0, 96)
    heldout = _groups("h", 1500, 4 * DAY, 40, seed=1)
    log = SignalLog(schema, train, heldout, _scores(heldout), 0.5, min_samples=30)
    first = log.log_window(schema, heldout, _scores(heldout), 0.5)
    assert set(first) == set(LOGGED_COLUMNS)
    assert first["signals_reference_window"] is True
    nxt = _groups("w", 3000, 6 * DAY, 96, seed=2)
    out = log.log_window(schema, nxt, _scores(nxt, 2), 0.5)
    assert out["signals_reference_window"] is False
    assert out["na_decision_psi"] is not None


def test_redeploy_makes_next_window_the_baseline():
    schema = _make_schema([_exploit_predicate()])
    train = _groups("t", 3000, 0, 96)
    heldout = _groups("h", 1500, 4 * DAY, 40, seed=1)
    log = SignalLog(schema, train, heldout, _scores(heldout), 0.5, min_samples=30)
    w1 = _groups("a", 3000, 6 * DAY, 96, explo_frac=0.6, seed=2)
    log.log_window(schema, w1, _scores(w1, 2), 0.5)
    log.redeploy(schema, w1)
    w2 = _groups("b", 3000, 10 * DAY, 96, explo_frac=0.6, seed=3)
    out2 = log.log_window(schema, w2, _scores(w2, 3), 0.5)
    assert out2["signals_reference_window"] is True
    assert out2["na_decision_psi"] == 0.0  # evaluated against itself
    w3 = _groups("c", 3000, 14 * DAY, 96, explo_frac=0.6, seed=4)
    out3 = log.log_window(schema, w3, _scores(w3, 4), 0.5)
    assert out3["signals_reference_window"] is False


def test_empty_window_logs_all_none():
    schema = _make_schema([_exploit_predicate()])
    log = SignalLog(schema, [], [], None, 0.5, min_samples=30)
    out = log.log_window(schema, [], None, 0.5)
    assert set(out) == set(LOGGED_COLUMNS) and all(v is None for v in out.values())

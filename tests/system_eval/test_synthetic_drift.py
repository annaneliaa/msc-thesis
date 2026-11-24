from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import thesis.system_eval.synthetic_drift as sd
from thesis.experiments._shared import ScenarioContext
from thesis.system_eval.synthetic_drift import (
    Injection,
    SyntheticDriftConfig,
    apply_injection,
    build_detection_summary,
    build_donors,
)
from test_monitor_drift import (  # tests/ is on sys.path (rootdir conftest)
    _BASE_TS,
    _STEP,
    _base_schema,
    _make_alert_group,
    _mining_settings,
)


def _timeline(n_windows: int = 6, per_window: int = 40):
    """Balanced windows; attacks are EXPLOIT/ET, benign mostly SNMP/GPL with
    every 5th benign row RPC (the minority benign category)."""
    rows = []
    for w in range(n_windows):
        for i in range(per_window):
            idx = w * per_window + i
            attack = i % 2 == 0
            cat = "EXPLOIT" if attack else ("RPC" if i % 10 == 1 else "SNMP")
            rows.append(
                _make_alert_group(
                    f"w{w}_g{i}",
                    "attack" if attack else "benign",
                    _BASE_TS + idx * _STEP,
                    category=cat,
                    ruleset="ET" if attack else "GPL",
                    proto=6 if attack else 17,
                    int_port=80,
                    ext_port=1234,
                )
            )
    return rows


def test_donors_pick_minority_benign_category():
    d = build_donors(_timeline(n_windows=1))
    assert d.minority_category == "RPC"
    assert all(g.group_label == "attack" for g in d.attack)
    assert all(g.category == "RPC" for g in d.benign_minority)


def test_injection_leaves_prefix_and_originals_untouched():
    rows = _timeline()
    donors = build_donors(rows[:40])
    start = 120
    out = apply_injection(
        rows, start, Injection("benign_mimics_attack", 1.0), donors, 0
    )
    assert out[:start] == rows[:start]
    assert len(out) == len(rows)
    for orig, new in zip(rows[start:], out[start:]):
        if orig.group_label == "benign":
            assert new is not orig
            assert new.category == "EXPLOIT" and new.ruleset == "ET"
            assert new.group_label == "benign"  # labels never change
            assert orig.category in ("SNMP", "RPC")  # original not mutated
        else:
            assert new is orig


def test_injection_fraction_and_determinism():
    rows = _timeline(n_windows=10)
    donors = build_donors(rows[:40])
    inj = Injection("new_values", 0.3)
    a = apply_injection(rows, 0, inj, donors, seed=7)
    b = apply_injection(rows, 0, inj, donors, seed=7)
    hit = sum(g.category == "SYNTHETIC_NEW" for g in a)
    assert [g.category for g in a] == [g.category for g in b]
    assert 0.2 < hit / len(rows) < 0.4


@pytest.mark.parametrize(
    "kind,check",
    [
        ("attack_mimics_benign", lambda o, n: n.category in ("SNMP", "RPC")),
        ("benign_mix_shift", lambda o, n: n.category == "RPC"),
        (
            "unseen_field_shift",
            lambda o, n: n.int_port >= 49152 and n.category == o.category,
        ),
        (
            "volume_shift",
            lambda o, n: (
                n.signature_matches_per_day == 10 * o.signature_matches_per_day
            ),
        ),
    ],
)
def test_each_kind_changes_what_it_says(kind, check):
    rows = _timeline(n_windows=2)
    donors = build_donors(rows[:40])
    out = apply_injection(rows, 40, Injection(kind, 1.0), donors, 0)
    target = sd._TARGET[kind]
    for o, n in zip(rows[40:], out[40:]):
        if target == "all" or o.group_label == target:
            assert check(o, n)


def test_end_to_end_logs_injection_and_summarises(monkeypatch, tmp_path):
    rows = _timeline(n_windows=6)
    ctx = ScenarioContext(
        alert_groups=rows,
        alert_groups_path=Path("unused"),
        n_total=len(rows),
        base_schema=_base_schema(),
        mining_settings_by_name=_mining_settings(),
        mining_settings_path=Path("unused.yaml"),
    )
    monkeypatch.setattr(sd, "load_scenario_context", lambda **_: ctx)
    config = SyntheticDriftConfig(
        scenario="test_scenario",
        mining_setting="test_setting",
        mining_settings_path=Path("unused.yaml"),
        models=("logreg",),
        granularity=1 / 6,
        injections=[Injection("none", 0.0), Injection("benign_mimics_attack", 0.5)],
        source_split_mode="window0",
        results_dir=tmp_path,
        n_jobs=1,
        monitor_min_samples_signal_2=3,
    )
    out_dir = sd.run_synthetic_drift_experiment(config)

    ph = pd.read_csv(out_dir / "per_horizon_results.csv")
    assert set(ph.injection) == {"none", "benign_mimics_attack@0.5"}
    clean = ph[ph.injection == "none"]
    assert not clean.injected.any()
    inj = ph[ph.injection != "none"]
    k0 = int(inj.inject_from_h.iloc[0])
    assert (inj.injected == (inj.horizon_window_index >= k0)).all()

    # Pre-injection windows are identical between the two runs.
    def pre(d):
        return d[d.horizon_window_index < k0].sort_values("horizon_window_index").f1

    assert list(pre(clean)) == pytest.approx(list(pre(inj)))

    summ = pd.read_csv(out_dir / "detection_summary.csv")
    row = summ[summ.injection == "benign_mimics_attack@0.5"].iloc[0]
    assert row.d_precision_vs_clean < 0  # benign groups that look like attacks
    assert "signal_1_detect" in summ.columns
    assert "feedback_precision_false_alarm" in summ.columns
    # Per-feature PSI over the model inputs is logged on every walked window.
    assert ph[ph.horizon_window_index > 0].fpsi_max.notna().all()
    assert "feature_psi_detect" in summ.columns
    # Pattern validity is logged in all three label modes.
    walked = ph[ph.horizon_window_index > 0]
    assert walked.pv_oracle_n_checked.notna().all()
    assert walked.pv_audit_elevated.notna().all()
    for m in ("oracle", "escalated", "audit"):
        assert f"pattern_validity_{m}_detect" in summ.columns


def test_detection_summary_rates():
    ph = pd.DataFrame(
        {
            "model": "m",
            "granularity": 0.1,
            "horizon_window_index": [1, 2, 3, 4] * 2,
            "injection": ["none"] * 4 + ["x@1"] * 4,
            "kind": ["none"] * 4 + ["x"] * 4,
            "fraction": [0.0] * 4 + [1.0] * 4,
            "inject_from_h": 3,
            "injected": [False] * 4 + [False, False, True, True],
            "f1": [0.9] * 4 + [0.9, 0.9, 0.5, 0.7],
            "precision": 0.9,
            "recall": 0.9,
            "signal_1_elevated": [False] * 4 + [True, False, True, True],
        }
    )
    s = build_detection_summary(ph).set_index("injection")
    assert s.loc["x@1", "signal_1_false_alarm"] == 0.5
    assert s.loc["x@1", "signal_1_detect"] == 1.0
    assert s.loc["x@1", "d_f1_vs_clean"] == pytest.approx(-0.3)

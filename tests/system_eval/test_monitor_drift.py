from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from thesis.system_eval.monitor_drift import (
    _run_one_monitor_config,
    fit_source_window_and_dynamic_schema,
)
from thesis.metrics.shortlist import ShortlistedConfig
from thesis.schemas.experiments import MonitorDriftConfig
from thesis.schemas.features import BaseFeatureSchema, FeatureSchema
from thesis.schemas.groups import AlertGroup
from thesis.schemas.mining import (
    ContrastSetFilterConfig,
    DecisionTreeRuleConfig,
    MiningSettingSpec,
)

_BASE_TS = 1_642_636_800  # 2022-01-20T00:00:00Z
_STEP = 3600


def _make_alert_group(
    group_id: str, label: str | None, start_ts: int, **overrides
) -> AlertGroup:
    defaults = dict(
        alert_group_id=group_id,
        group_id=group_id,
        method="cscas_pregrouped",
        start_ts=start_ts,
        end_ts=start_ts,
        n_alerts=1,
        group_label=label,
        category="POLICY",
        ruleset="ET",
        proto=6,
        scas=0,
        cve_refs=set(),
        qualifiers=set(),
        signature_matches_per_day=10.0,
        similarity=0.5,
        signature_id_similarity=0.5,
        attr_similarities={},
        int_ip_is_multiple=False,
        ext_port_is_multiple=False,
    )
    defaults.update(overrides)
    return AlertGroup(**defaults)


def _build_window_rows(
    win_idx: int, per_window: int, drifted: bool, unlabeled: bool = False
) -> list[AlertGroup]:
    """category is always EXPLOIT for attack rows / SNMP for benign rows --
    that relationship never breaks, so a compound rule conditioned on
    category stays well-calibrated (Signal 2) even when drifted. What
    "drifted" shifts is the *label mix* itself (from a balanced 50/50 split
    to almost-all-attack), which shifts category=EXPLOIT's activation *rate*
    away from what was mined (Signal 1), in isolation."""
    rows = []
    for i in range(per_window):
        idx = win_idx * per_window + i
        start_ts = _BASE_TS + idx * _STEP
        if unlabeled:
            label = "mixed"
        elif drifted:
            label = "benign" if i == 0 else "attack"
        else:
            label = "attack" if i % 2 == 0 else "benign"
        category = "EXPLOIT" if label == "attack" else "SNMP"
        rows.append(
            _make_alert_group(f"w{win_idx}_g{i}", label, start_ts, category=category)
        )
    return rows


def _build_timeline(
    n_windows: int = 5,
    per_window: int = 40,
    drift_from_window: int | None = None,
    unlabeled_windows: set[int] = frozenset(),
) -> list[AlertGroup]:
    rows: list[AlertGroup] = []
    for w in range(n_windows):
        drifted = drift_from_window is not None and w >= drift_from_window
        rows.extend(
            _build_window_rows(w, per_window, drifted, unlabeled=w in unlabeled_windows)
        )
    return rows


def _base_schema() -> FeatureSchema:
    return FeatureSchema(
        schema_name="base",
        schema_version="0.1.0",
        base=BaseFeatureSchema(
            features=[
                "signature_matches_per_day",
                "similarity",
                "signature_id_similarity",
                "n_alerts",
            ]
        ),
        symbolic=None,
    )


def _mining_settings() -> dict:
    spec = MiningSettingSpec(
        name="test_setting",
        contrast=ContrastSetFilterConfig(
            min_attack_coverage=0.05, min_benign_coverage=0.05, min_growth_rate=3.0
        ),
        tree=DecisionTreeRuleConfig(
            max_depth=2, min_samples_leaf=2, class_weight="balanced", random_state=0
        ),
    )
    return {"test_setting": spec}


def _config(**overrides) -> MonitorDriftConfig:
    defaults = dict(
        scenario="test_scenario",
        shortlist_path=Path("unused.csv"),
        train_frac_within_window=0.7,
        monitor_consecutive_windows=3,
        monitor_min_samples_signal_2=3,
    )
    defaults.update(overrides)
    return MonitorDriftConfig(**defaults)


def test_baseline_config_has_no_dynamic_schema_and_monitor_never_runs():
    alert_groups = _build_timeline(n_windows=3, per_window=20)
    cfg = ShortlistedConfig(
        feature_set="baseline", mining_setting=None, granularity=0.33, model="logreg"
    )

    fit = fit_source_window_and_dynamic_schema(
        cfg=cfg,
        scenario="test_scenario",
        alert_groups=alert_groups,
        n_total=len(alert_groups),
        base_schema=_base_schema(),
        mining_settings_by_name={},
        mining_settings_path=Path("unused.yaml"),
        train_frac_within_window=0.7,
        threshold_mode="fixed",
        calibrated_recall_target=0.9,
    )
    assert fit is not None
    assert fit.dynamic_schema is None

    horizon_rows, signal_rows = _run_one_monitor_config(
        cfg=cfg,
        config=_config(),
        scenario="test_scenario",
        alert_groups=alert_groups,
        n_total=len(alert_groups),
        base_schema=_base_schema(),
        mining_settings_by_name={},
        mining_settings_path=Path("unused.yaml"),
    )
    assert len(horizon_rows) > 0
    assert signal_rows == []
    for row in horizon_rows:
        assert row["action"] is None
        assert row["signal_1_elevated"] is None
        assert row["n_incoming_groups"] is None


def test_symbolic_config_mines_dynamic_schema_from_train_slice():
    alert_groups = _build_timeline(n_windows=3, per_window=40)
    n_total = len(alert_groups)
    cfg = ShortlistedConfig(
        feature_set="symbolic",
        mining_setting="test_setting",
        granularity=0.34,
        model="logreg",
    )

    fit = fit_source_window_and_dynamic_schema(
        cfg=cfg,
        scenario="test_scenario",
        alert_groups=alert_groups,
        n_total=n_total,
        base_schema=_base_schema(),
        mining_settings_by_name=_mining_settings(),
        mining_settings_path=Path("unused.yaml"),
        train_frac_within_window=0.7,
        threshold_mode="fixed",
        calibrated_recall_target=0.9,
    )
    assert fit is not None
    assert fit.dynamic_schema is not None
    assert fit.dynamic_schema.version == 1
    assert fit.schema.symbolic is not None
    assert len(fit.dynamic_schema.single_predicates) > 0
    assert len(fit.dynamic_schema.compound_rules) > 0

    exploit_predicate = next(
        p
        for p in fit.dynamic_schema.single_predicates
        if p.predicate_id == "cat:category=EXPLOIT"
    )
    assert exploit_predicate.direction == "attack"

    # win_size for gran=0.34 over n_total=120 is int(0.34*120)=40 (an exact
    # multiple of per_window=40, so window 0 = [0:40) lines up exactly with
    # the hand-built first window), train = [0:round(0.7*40)) = [0:28)).
    train_rows = alert_groups[0:28]
    assert fit.dynamic_schema.mining_window_start == datetime.fromtimestamp(
        train_rows[0].start_ts, tz=timezone.utc
    )
    assert fit.dynamic_schema.mining_window_end == datetime.fromtimestamp(
        train_rows[-1].start_ts, tz=timezone.utc
    )


def test_unlabeled_rows_in_train_slice_do_not_affect_base_attack_rate():
    alert_groups = _build_timeline(n_windows=3, per_window=40)
    # Insert extra unlabeled rows into the front of the timeline -- still
    # inside window 0's train split -- and shift every later timestamp so
    # ordering stays chronological.
    extra_unlabeled = [
        _make_alert_group(f"extra_{i}", "mixed", _BASE_TS - (5 - i) * _STEP)
        for i in range(5)
    ]
    alert_groups_with_unlabeled = extra_unlabeled + alert_groups
    n_total = len(alert_groups_with_unlabeled)
    cfg = ShortlistedConfig(
        feature_set="symbolic",
        mining_setting="test_setting",
        granularity=0.34,
        model="logreg",
    )

    fit = fit_source_window_and_dynamic_schema(
        cfg=cfg,
        scenario="test_scenario",
        alert_groups=alert_groups_with_unlabeled,
        n_total=n_total,
        base_schema=_base_schema(),
        mining_settings_by_name=_mining_settings(),
        mining_settings_path=Path("unused.yaml"),
        train_frac_within_window=0.7,
        threshold_mode="fixed",
        calibrated_recall_target=0.9,
    )
    assert fit is not None
    assert fit.dynamic_schema is not None
    # Labeled rows are perfectly balanced (alternating attack/benign) --
    # the unlabeled rows must not skew this away from 0.5.
    assert abs(fit.dynamic_schema.base_attack_rate - 0.5) < 1e-9


def test_monitor_receives_raw_unmasked_groups_not_label_masked():
    """Key regression test: a horizon containing unlabeled rows must still
    show n_incoming_groups (raw) > n_labeled_groups (masked) -- the monitor
    must never be pre-filtered to labeled rows only."""
    alert_groups = _build_timeline(n_windows=3, per_window=40, unlabeled_windows={1})
    n_total = len(alert_groups)
    cfg = ShortlistedConfig(
        feature_set="symbolic",
        mining_setting="test_setting",
        granularity=0.34,
        model="logreg",
    )

    horizon_rows, _signal_rows = _run_one_monitor_config(
        cfg=cfg,
        config=_config(),
        scenario="test_scenario",
        alert_groups=alert_groups,
        n_total=n_total,
        base_schema=_base_schema(),
        mining_settings_by_name=_mining_settings(),
        mining_settings_path=Path("unused.yaml"),
    )
    by_h = {r["horizon_window_index"]: r for r in horizon_rows}
    unlabeled_horizon = by_h[1]
    assert unlabeled_horizon["n_incoming_groups"] is not None
    assert (
        unlabeled_horizon["n_incoming_groups"] > unlabeled_horizon["n_labeled_groups"]
    )
    assert unlabeled_horizon["n_labeled_groups"] == 0
    # Model metrics fall back to nan (no labeled rows), but the monitor still
    # ran (Signal 1 needs no labels at all).
    assert unlabeled_horizon["target_single_class"] is True
    assert unlabeled_horizon["signal_1_elevated"] is not None


def test_monitor_state_accumulates_and_action_escalates_to_retrain_only():
    alert_groups = _build_timeline(n_windows=5, per_window=40, drift_from_window=1)
    n_total = len(alert_groups)
    cfg = ShortlistedConfig(
        feature_set="symbolic",
        mining_setting="test_setting",
        granularity=0.2,
        model="logreg",
    )
    config = _config(monitor_consecutive_windows=3, monitor_min_samples_signal_2=3)

    horizon_rows, signal_rows = _run_one_monitor_config(
        cfg=cfg,
        config=config,
        scenario="test_scenario",
        alert_groups=alert_groups,
        n_total=n_total,
        base_schema=_base_schema(),
        mining_settings_by_name=_mining_settings(),
        mining_settings_path=Path("unused.yaml"),
    )
    by_h = {r["horizon_window_index"]: r for r in horizon_rows}
    assert set(by_h) == {0, 1, 2, 3, 4}

    # h=0 is W_src's own held-out test split -- same (non-drifted)
    # distribution mining saw, so Signal 1 should not be elevated there.
    assert by_h[0]["signal_1_elevated"] is False
    assert by_h[0]["action"] == "NO_ACTION"

    # h=1..4 are drifted (category=EXPLOIT never fires anymore) -- Signal 1
    # elevates every drifted horizon, and consecutive counters must increase
    # monotonically since MonitorState carries over across the whole walk.
    consecutive = [by_h[h]["consecutive_signal_1_elevated"] for h in (1, 2, 3, 4)]
    assert consecutive == [1, 2, 3, 4]
    assert by_h[1]["action"] == "SOFT_ALERT"
    assert by_h[2]["action"] == "SOFT_ALERT"
    # Hits consecutive_windows=3 at h=3: single signal elevated + consecutive
    # streak -> RETRAIN_ONLY (never REMINE_AND_RETRAIN, since Signal 2 never
    # has enough matching rows in the drifted horizons to elevate).
    assert by_h[3]["action"] == "RETRAIN_ONLY"
    assert by_h[3]["trigger_remine"] is True
    assert by_h[4]["action"] == "RETRAIN_ONLY"

    assert len(signal_rows) > 0
    assert all(row["signal_type"] in ("signal_1", "signal_2") for row in signal_rows)


def test_monitor_alarms_is_exactly_the_elevated_subset_of_signals():
    alert_groups = _build_timeline(n_windows=5, per_window=40, drift_from_window=1)
    n_total = len(alert_groups)
    cfg = ShortlistedConfig(
        feature_set="symbolic",
        mining_setting="test_setting",
        granularity=0.2,
        model="logreg",
    )
    _horizon_rows, signal_rows = _run_one_monitor_config(
        cfg=cfg,
        config=_config(),
        scenario="test_scenario",
        alert_groups=alert_groups,
        n_total=n_total,
        base_schema=_base_schema(),
        mining_settings_by_name=_mining_settings(),
        mining_settings_path=Path("unused.yaml"),
    )
    signals_df = pd.DataFrame(signal_rows)
    alarms_df = signals_df[signals_df["elevated"] == True]  # noqa: E712

    assert len(alarms_df) > 0
    assert alarms_df["elevated"].all()
    assert len(alarms_df) < len(signals_df)


def test_monitor_state_resets_between_configs():
    alert_groups = _build_timeline(n_windows=5, per_window=40, drift_from_window=1)
    n_total = len(alert_groups)
    cfg = ShortlistedConfig(
        feature_set="symbolic",
        mining_setting="test_setting",
        granularity=0.2,
        model="logreg",
    )
    config = _config()

    _run_one_monitor_config(
        cfg=cfg,
        config=config,
        scenario="test_scenario",
        alert_groups=alert_groups,
        n_total=n_total,
        base_schema=_base_schema(),
        mining_settings_by_name=_mining_settings(),
        mining_settings_path=Path("unused.yaml"),
    )
    # A fresh call (as if a second shortlisted config ran) must start with
    # fresh consecutive counters -- state is a local variable per call, but
    # this is the behavior the accumulation design depends on.
    horizon_rows_2, _ = _run_one_monitor_config(
        cfg=cfg,
        config=config,
        scenario="test_scenario",
        alert_groups=alert_groups,
        n_total=n_total,
        base_schema=_base_schema(),
        mining_settings_by_name=_mining_settings(),
        mining_settings_path=Path("unused.yaml"),
    )
    by_h = {r["horizon_window_index"]: r for r in horizon_rows_2}
    assert by_h[1]["consecutive_signal_1_elevated"] == 1

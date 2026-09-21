from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis.system_eval import rolling_walk_forward as rwf
from thesis.metrics.shortlist import ShortlistedConfig
from thesis.schemas.experiments import RollingWalkForwardConfig
from thesis.schemas.features import BaseFeatureSchema, FeatureSchema
from thesis.schemas.groups import AlertGroup


# ---- _build_walk_forward_summary --------------------------------------------


def _step_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "feature_set": "symbolic",
        "mining_setting": "gr3.0_md4",
        "granularity": 0.1,
        "model": "logreg",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_build_walk_forward_summary_empty_input():
    assert rwf._build_walk_forward_summary(pd.DataFrame()).empty


def test_build_walk_forward_summary_mean_std_and_n_steps():
    df = _step_df(
        [
            {"step_i": 0, "auc": 0.9, "f1": 0.8, "fpr": 0.1},
            {"step_i": 1, "auc": 0.8, "f1": 0.7, "fpr": 0.2},
            {"step_i": 2, "auc": 1.0, "f1": 0.9, "fpr": 0.0},
        ]
    )
    summary = rwf._build_walk_forward_summary(df)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["auc_mean"] == pytest.approx(0.9)
    assert row["f1_mean"] == pytest.approx(0.8)
    assert row["fpr_mean"] == pytest.approx(0.1)
    assert row["auc_std"] == pytest.approx(np.std([0.9, 0.8, 1.0], ddof=1))
    assert row["n_steps"] == 3


def test_build_walk_forward_summary_one_row_per_config():
    df = _step_df(
        [
            {"step_i": 0, "auc": 0.9, "f1": 0.8, "fpr": 0.1, "model": "logreg"},
            {"step_i": 1, "auc": 0.8, "f1": 0.7, "fpr": 0.2, "model": "logreg"},
            {"step_i": 0, "auc": 0.7, "f1": 0.6, "fpr": 0.3, "model": "rf"},
        ]
    )
    summary = rwf._build_walk_forward_summary(df)
    assert len(summary) == 2
    assert set(summary["model"]) == {"logreg", "rf"}


def test_build_walk_forward_summary_nan_std_for_single_step():
    df = _step_df([{"step_i": 0, "auc": 0.9, "f1": 0.8, "fpr": 0.1}])
    summary = rwf._build_walk_forward_summary(df)
    assert summary.iloc[0]["n_steps"] == 1
    assert np.isnan(summary.iloc[0]["auc_std"])


# ---- _run_one_config ----------------------------------------------------------


class _FakeModel:
    def __init__(self, proba_pos: np.ndarray):
        self._proba_pos = proba_pos

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba_pos, self._proba_pos])


def _fake_fast_route_and_funnel(*args, **kwargs):
    """Stub for fast_route_and_funnel -- these tests exercise the
    walk/skip/explanation-wiring logic with fake alert_groups/schema
    ([object()] * 100), which the real function can't encode; the exact
    funnel values don't matter to what's being tested here."""
    return {
        "fast_route_wall_s": 0.0,
        "fast_route_cpu_s": 0.0,
        "n_alerts_in": 0,
        "n_groups_escalated": 0,
        "n_groups_suppressed": 0,
        "n_alerts_escalated": 0,
        "n_alerts_suppressed": 0,
    }


def test_run_one_config_walks_every_step_and_skips_gracefully(monkeypatch):
    """n_total=100, gran=0.2 -> win_size=20, n_windows=5 -> steps i=0..3.
    Step 0: fit succeeds, target window has labeled rows -> real metrics.
    Step 1: fit succeeds, but target window (win_idx=2) has no labeled rows
        -> nan metrics, no crash.
    Step 2: fit_window itself returns None (e.g. single-class Wi) -> nan
        metrics, encode_target_window never called for this step.
    Step 3: fit succeeds, target window has labeled rows -> real metrics.
    Each step's fit is independent -- a skip at one step doesn't affect the
    next, since every step re-mines/retrains from scratch."""
    cfg = ShortlistedConfig(
        feature_set="baseline", mining_setting=None, granularity=0.2, model="logreg"
    )
    # explanations off -- this test is about the walk/skip logic, not SHAP/LIME
    config = RollingWalkForwardConfig(
        scenario="test_scenario",
        shortlist_path=Path("unused.csv"),
        compute_explanations=False,
    )

    def fake_fit_window(*, win_idx, **kwargs):
        if win_idx == 2:
            return None
        return rwf.WindowFit(
            schema=object(),
            model=_FakeModel(np.array([0.2, 0.8, 0.9])),
            threshold=0.5,
            feature_names=["f1"],
            cache_hit=True,
            X_fit=pd.DataFrame({"f1": [0.1, 0.2, 0.3]}),
        )

    def fake_encode_target_window(alert_groups, n_total, gran, win_idx, schema):
        if win_idx == 2:
            return pd.DataFrame({"f1": []}), np.array([]), 0
        return pd.DataFrame({"f1": [0.1, 0.2, 0.3]}), np.array([0, 1, 1]), 3

    monkeypatch.setattr(rwf, "fit_window", fake_fit_window)
    monkeypatch.setattr(rwf, "encode_target_window", fake_encode_target_window)
    monkeypatch.setattr(rwf, "fast_route_and_funnel", _fake_fast_route_and_funnel)

    rows, explain_rows, fidelity_rows = rwf._run_one_config(
        cfg=cfg,
        config=config,
        scenario="test_scenario",
        alert_groups=[object()] * 100,
        alert_groups_path=Path("unused.json"),
        n_total=100,
        base_schema=object(),
        mining_settings_by_name={},
        mining_settings_path=Path("unused.yaml"),
        scheme=rwf.WindowScheme("window0", 100),
    )
    assert explain_rows == []
    assert fidelity_rows == []

    assert [r["step_i"] for r in rows] == [0, 1, 2, 3]
    assert not np.isnan(rows[0]["auc"])  # step 0: normal
    assert np.isnan(rows[1]["auc"])  # step 1: empty target window
    assert np.isnan(rows[2]["auc"])  # step 2: fit_window returned None
    assert not np.isnan(rows[3]["auc"])  # step 3: normal

    # Every row reflects that step's own fresh fit -- no leakage of
    # mining_cache_hit/threshold from a neighboring step.
    assert rows[0]["mining_cache_hit"] is True
    assert rows[2]["mining_cache_hit"] is None
    assert np.isnan(rows[2]["threshold"])
    assert rows[1]["threshold"] == pytest.approx(0.5)


# ---- fit_window: cscas_full / cscas_full_symbolic ---------------------------


def _rows(n_benign: int, n_attack: int) -> list[AlertGroup]:
    out = []
    for i in range(n_benign):
        out.append(
            AlertGroup(
                alert_group_id=f"b{i}",
                group_id=f"b{i}",
                method="cscas_pregrouped",
                start_ts=1_642_600_000 + i,
                end_ts=1_642_600_000 + i,
                n_alerts=1,
                group_label="benign",
                raw_items={f"tok_b{i % 4}"},
                proto=6,
                ext_port=80,
                int_port=1000 + i,
                category="WEB_SERVER",
                ruleset="ET",
                signature_matches_per_day=10.0,
            )
        )
    for i in range(n_attack):
        out.append(
            AlertGroup(
                alert_group_id=f"a{i}",
                group_id=f"a{i}",
                method="cscas_pregrouped",
                start_ts=1_642_600_500 + i,
                end_ts=1_642_600_500 + i,
                n_alerts=1,
                group_label="attack",
                raw_items={f"tok_a{i % 3}"},
                proto=17,
                ext_port=53,
                int_port=2000 + i,
                category="DNS",
                ruleset="ET",
                signature_matches_per_day=500.0,
            )
        )
    return out


class _FakeMiningResult:
    def __init__(self):
        self.schema_path = Path("unused.json")
        self.cache_hit = False


@pytest.fixture
def _symbolic_stub(monkeypatch):
    """Stub out the actual mining/loading calls with a tiny fixed schema, and
    fail the test if fit_window ever calls the mining function for a
    feature_set that shouldn't need it (cscas_full)."""
    from thesis.schemas.features import SymbolicFeature, SymbolicFeatureSchema

    sym = SymbolicFeatureSchema(
        schema_name="sym",
        schema_version="0.1.0",
        features=[
            SymbolicFeature(
                feature_name="sym__tok_a0", itemset=("tok_a0",), source_label="attack"
            )
        ],
    )
    calls = {"mine": 0}

    def fake_mine(**kwargs):
        calls["mine"] += 1
        return _FakeMiningResult()

    monkeypatch.setattr(rwf, "get_or_mine_full_window_attribute_schema", fake_mine)
    monkeypatch.setattr(rwf, "load_symbolic_feature_schema", lambda path: sym)
    return calls


def _base_schema():
    return FeatureSchema(
        schema_name="base",
        schema_version="0.1.0",
        base=BaseFeatureSchema(
            ["proto", "ext_port", "int_port", "n_alerts", "signature_matches_per_day"]
        ),
    )


def test_fit_window_cscas_full_skips_mining_and_drops_scas_nowhere(
    _symbolic_stub, monkeypatch
):
    # rf/xgboost are supervised, not one-class -- scas stays in the schema.
    rows = _rows(8, 4)
    cfg = ShortlistedConfig(
        feature_set="cscas_full", mining_setting=None, granularity=1.0, model="rf"
    )
    fit = rwf.fit_window(
        cfg=cfg,
        scenario="cscas",
        alert_groups=rows,
        alert_groups_path=Path("x.json"),
        n_total=len(rows),
        win_idx=0,
        base_schema=_base_schema(),
        mining_settings_by_name={},
        mining_settings_path=Path("x.yaml"),
        threshold_mode="fixed",
        calibrated_recall_target=0.9,
        scheme=rwf.WindowScheme("window0", len(rows)),
    )
    assert fit is not None
    assert _symbolic_stub["mine"] == 0  # cscas_full never mines
    assert fit.schema.base.kind == "cscas_full"
    assert fit.schema.symbolic is None
    assert "scas" in fit.schema.base.features


def test_fit_window_cscas_full_symbolic_mines_and_merges_bases(_symbolic_stub):
    rows = _rows(8, 4)
    cfg = ShortlistedConfig(
        feature_set="cscas_full_symbolic",
        mining_setting="gr3_md1_mda4_rounds2",
        granularity=1.0,
        model="xgboost",
    )
    fit = rwf.fit_window(
        cfg=cfg,
        scenario="cscas",
        alert_groups=rows,
        alert_groups_path=Path("x.json"),
        n_total=len(rows),
        win_idx=0,
        base_schema=_base_schema(),
        mining_settings_by_name={"gr3_md1_mda4_rounds2": _FakeSpec()},
        mining_settings_path=Path("x.yaml"),
        threshold_mode="fixed",
        calibrated_recall_target=0.9,
        scheme=rwf.WindowScheme("window0", len(rows)),
    )
    assert fit is not None
    assert _symbolic_stub["mine"] == 1  # mined once, on the full window
    assert fit.schema.base.kind == "cscas_full"
    assert fit.schema.symbolic is not None
    assert "sym__tok_a0" in fit.feature_names
    # base cols present exactly once (no baseline-vs-cscas_full double-count)
    assert fit.feature_names.count("proto") == 1


class _FakeSpec:
    def to_attribute_mining_config(self):
        from thesis.schemas.mining import AttributeMiningConfig

        return AttributeMiningConfig()


def test_fit_window_uses_fit_scored_model_for_class_weighting(monkeypatch):
    """fit_window must go through fit_scored_model (class-imbalance-aware),
    not a bare get_model_factory(...)().fit(...) -- the latter leaves
    xgboost unweighted on this scenario's imbalance."""
    rows = _rows(8, 4)
    seen = {}

    def spy_fit_scored_model(model_name, X, y):
        seen["model_name"] = model_name
        seen["n_rows"] = len(X)
        from thesis.experiments._shared import fit_scored_model as real

        return real(model_name, X, y)

    monkeypatch.setattr(rwf, "fit_scored_model", spy_fit_scored_model)
    cfg = ShortlistedConfig(
        feature_set="baseline", mining_setting=None, granularity=1.0, model="xgboost"
    )
    fit = rwf.fit_window(
        cfg=cfg,
        scenario="cscas",
        alert_groups=rows,
        alert_groups_path=Path("x.json"),
        n_total=len(rows),
        win_idx=0,
        base_schema=_base_schema(),
        mining_settings_by_name={},
        mining_settings_path=Path("x.yaml"),
        threshold_mode="fixed",
        calibrated_recall_target=0.9,
        scheme=rwf.WindowScheme("window0", len(rows)),
    )
    assert fit is not None
    assert seen["model_name"] == "xgboost"
    assert seen["n_rows"] == 12


# ---- _explanation_rows -----------------------------------------------------


def _cfg_defaults(**overrides) -> RollingWalkForwardConfig:
    defaults = dict(scenario="cscas", shortlist_path=Path("unused.csv"))
    return RollingWalkForwardConfig(**{**defaults, **overrides})


def test_explanation_rows_labels_every_feature_with_step_and_method(monkeypatch):
    monkeypatch.setattr(
        rwf,
        "compute_shap_signed_importances",
        lambda model, bg, x, names, top_n: {"f1": 0.5, "f2": -0.2},
    )

    class _Lime:
        importances = {"f1": 0.4, "f2": -0.1}
        mean_fidelity = 0.87

    monkeypatch.setattr(rwf, "compute_lime_signed_importances", lambda *a, **k: _Lime())

    X_target = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    base_row = {
        "feature_set": "symbolic",
        "model": "xgboost",
        "mining_setting": "gr3_md1_mda4_rounds2",
    }
    rows, fidelity_rows = rwf._explanation_rows(
        model=object(),
        X_background=X_target,
        X_target=X_target,
        feature_names=["f1", "f2"],
        base_row=base_row,
        step_i=3,
        config=_cfg_defaults(explain_sample_n=10),
    )

    methods = {r["method"] for r in rows}
    assert methods == {"shap", "lime"}
    assert all(r["step_i"] == 3 for r in rows)
    assert all(r["feature_set"] == "symbolic" for r in rows)  # base_row merged in
    shap_f1 = next(r for r in rows if r["method"] == "shap" and r["feature"] == "f1")
    assert shap_f1["importance"] == pytest.approx(0.5)
    assert len(fidelity_rows) == 1
    assert fidelity_rows[0]["mean_fidelity"] == pytest.approx(0.87)
    assert fidelity_rows[0]["step_i"] == 3


def test_explanation_rows_empty_target_returns_nothing(monkeypatch):
    called = {"shap": False}
    monkeypatch.setattr(
        rwf,
        "compute_shap_signed_importances",
        lambda *a, **k: called.__setitem__("shap", True) or {},
    )
    rows, fidelity_rows = rwf._explanation_rows(
        model=object(),
        X_background=pd.DataFrame({"f1": [0.1]}),
        X_target=pd.DataFrame({"f1": []}),
        feature_names=["f1"],
        base_row={},
        step_i=0,
        config=_cfg_defaults(),
    )
    assert rows == [] and fidelity_rows == []
    assert called["shap"] is False  # sample_rows on an empty frame short-circuits


def test_explanation_rows_shap_failure_does_not_drop_lime_rows(monkeypatch):
    def raising_shap(*a, **k):
        raise RuntimeError("shap blew up")

    class _Lime:
        importances = {"f1": 0.4}
        mean_fidelity = 0.9

    monkeypatch.setattr(rwf, "compute_shap_signed_importances", raising_shap)
    monkeypatch.setattr(rwf, "compute_lime_signed_importances", lambda *a, **k: _Lime())

    X_target = pd.DataFrame({"f1": [0.1, 0.2]})
    rows, fidelity_rows = rwf._explanation_rows(
        model=object(),
        X_background=X_target,
        X_target=X_target,
        feature_names=["f1"],
        base_row={},
        step_i=1,
        config=_cfg_defaults(),
    )
    assert [r["method"] for r in rows] == ["lime"]
    assert len(fidelity_rows) == 1


# ---- _run_one_config: explanations wiring -----------------------------------


def test_run_one_config_explanations_use_wi_background_and_wi1_sample(monkeypatch):
    """Background must come from the step's own fit (Wi's encoding), the
    explained sample from the step's evaluation window (W(i+1))."""
    cfg = ShortlistedConfig(
        feature_set="baseline", mining_setting=None, granularity=0.5, model="logreg"
    )
    config = _cfg_defaults(compute_explanations=True, explain_background_n=2)

    wi_fit_X = pd.DataFrame({"f1": [0.9, 0.9, 0.9]})  # distinct from the eval window
    wi1_X = pd.DataFrame({"f1": [0.1, 0.2, 0.3]})

    def fake_fit_window(*, win_idx, **kwargs):
        return rwf.WindowFit(
            schema=object(),
            model=_FakeModel(np.array([0.2, 0.8, 0.9])),
            threshold=0.5,
            feature_names=["f1"],
            cache_hit=True,
            X_fit=wi_fit_X,
        )

    def fake_encode_target_window(alert_groups, n_total, gran, win_idx, schema):
        return wi1_X, np.array([0, 1, 1]), 3

    seen = {}

    def fake_explanation_rows(
        model, X_background, X_target, feature_names, base_row, step_i, config
    ):
        seen["background"] = X_background
        seen["target"] = X_target
        return [], []

    monkeypatch.setattr(rwf, "fit_window", fake_fit_window)
    monkeypatch.setattr(rwf, "encode_target_window", fake_encode_target_window)
    monkeypatch.setattr(rwf, "_explanation_rows", fake_explanation_rows)
    monkeypatch.setattr(rwf, "fast_route_and_funnel", _fake_fast_route_and_funnel)

    rwf._run_one_config(
        cfg=cfg,
        config=config,
        scenario="test_scenario",
        alert_groups=[object()] * 100,
        alert_groups_path=Path("unused.json"),
        n_total=100,
        base_schema=object(),
        mining_settings_by_name={},
        mining_settings_path=Path("unused.yaml"),
        scheme=rwf.WindowScheme("window0", 100),
    )

    assert seen["background"]["f1"].tolist() == [0.9, 0.9]  # sampled from Wi (X_fit)
    assert (
        seen["target"] is wi1_X
    )  # the evaluation window, unsampled here (sampling is inside _explanation_rows)


def test_run_one_config_flags_skip_shap_for_oneclass_models(monkeypatch):
    cfg = ShortlistedConfig(
        feature_set="baseline", mining_setting=None, granularity=0.5, model="iforest"
    )
    config = _cfg_defaults(compute_explanations=True, oneclass_shap=False)
    fitted_model = _FakeModel(np.array([0.2, 0.8, 0.9]))

    def fake_fit_window(*, win_idx, **kwargs):
        return rwf.WindowFit(
            schema=object(),
            model=fitted_model,
            threshold=0.5,
            feature_names=["f1"],
            cache_hit=True,
            X_fit=pd.DataFrame({"f1": [0.1, 0.2, 0.3]}),
        )

    def fake_encode_target_window(alert_groups, n_total, gran, win_idx, schema):
        return pd.DataFrame({"f1": [0.1, 0.2, 0.3]}), np.array([0, 1, 1]), 3

    monkeypatch.setattr(rwf, "fit_window", fake_fit_window)
    monkeypatch.setattr(rwf, "encode_target_window", fake_encode_target_window)
    monkeypatch.setattr(rwf, "_explanation_rows", lambda *a, **k: ([], []))
    monkeypatch.setattr(rwf, "fast_route_and_funnel", _fake_fast_route_and_funnel)

    rwf._run_one_config(
        cfg=cfg,
        config=config,
        scenario="test_scenario",
        alert_groups=[object()] * 100,
        alert_groups_path=Path("unused.json"),
        n_total=100,
        base_schema=object(),
        mining_settings_by_name={},
        mining_settings_path=Path("unused.yaml"),
        scheme=rwf.WindowScheme("window0", 100),
    )
    assert fitted_model._skip_shap is True

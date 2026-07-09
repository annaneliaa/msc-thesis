"""
Experiment 2: Temporal Generalization (Rolling-Horizon Decay).

Purpose: for a shortlisted (feature_set, mining_setting, granularity, model)
config -- the output of Experiment 1 (screening_sweep.py +
thesis.metrics.config_selection) -- test whether a schema+model trained on
the *first* chronological window still discriminates well on future windows,
how AUC/F1/etc and FPR decay as temporal distance increases, and how SHAP/
LIME feature attributions drift alongside that decay.

W_src is always window 0 -- there is no other source-window role. For a
given granularity g, the timeline is carved into n(g) windows exactly as in
screening_sweep.py (pipeline.compute_window_bounds), and window 0 has its
own internal train/test split (pipeline.compute_window_train_end, same
70/30 default as the screening sweep):

  1. Mine a schema on window 0's *train* split only
     (mining.window_schema_cache.get_or_mine_window_attribute_schema --
     the same train-split-only mining screening_sweep uses, unlike the
     previous version of this experiment which mined on the full window).
  2. Fit the config's model on window 0's train split.
  3. Fix a decision threshold from that train split's own scores (flat 0.5,
     or a calibrated-recall threshold) -- computed once.
  4. Freeze schema, model, and threshold. Walk the horizon forward one
     window at a time, from h=0 (window 0's held-out *test* split) through
     h=n_windows-1 (the last window), scoring each window's alert_groups
     with the frozen schema/model/threshold. Every window is in bounds by
     construction (W_src is always the earliest window), so there is no
     boundary-skip bookkeeping to do here, unlike the multi-role design this
     replaced.
  5. At every horizon step, also compute SHAP and LIME signed feature
     importances (thesis.training.explain) on a sample of that window's rows
     -- same frozen model, same frozen SHAP/LIME background sample drawn
     once from window 0's train split -- so any change in the reported
     importances reflects the target window drifting, not the explainer's
     reference point moving.

Outputs: per_horizon_results.csv (one row per config x horizon window,
including the h=0 held-out anchor), decay_summary.csv (score/FPR at h=0 vs
the last horizon actually run, and their difference), explanations.csv
(long format: one row per config x horizon x method[shap|lime] x feature),
lime_fidelity.csv (one row per config x horizon: LIME's own local-surrogate
R^2, averaged over that horizon's explained sample -- separate from
explanations.csv since it's one number per horizon, not per feature),
summary.txt, config.json.

fit_source_window and encode_target_window (below) are also the entry
points thesis.experiments.instance_explain uses for on-demand, single-
instance SHAP/LIME case studies (e.g. "explain this specific false
positive at horizon 5") without re-running the whole sweep.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.encoders.service import encode_alert_groups_for_schema
from thesis.experiments._shared import (
    CONFIG_COLS,
    METRIC_COLS,
    decide_threshold,
    labels_and_mask,
    load_scenario_context,
    metrics_at_threshold,
    nan_metrics,
    sample_rows,
)
from thesis.features.persistence import load_symbolic_feature_schema
from thesis.metrics.shortlist import ShortlistedConfig, load_shortlist
from thesis.mining.window_schema_cache import get_or_mine_window_attribute_schema
from thesis.paths import ensure_artifact_dirs
from thesis.pipeline.pipeline import compute_window_bounds, compute_window_train_end
from thesis.schemas.experiments import TemporalDecayConfig
from thesis.schemas.features import FeatureSchema
from thesis.training.explain import (
    compute_lime_signed_importances,
    compute_shap_signed_importances,
)
from thesis.training.model_factory import get_model_factory

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS_DIR = _ROOT / "artifacts" / "experiments" / "temporal_decay"


@dataclass(slots=True)
class SourceWindowFit:
    """Everything frozen once window 0's train split is mined+fit: the
    schema, the fitted model, the decision threshold, and window 0's own
    train/test split (X_train for SHAP/LIME background sampling, X_test/
    y_test as the h=0 held-out anchor). Returned by fit_source_window so
    both the main sweep (_run_one_config) and the on-demand case-study
    tooling (experiments/instance_explain.py) mine/fit exactly once, the
    same way, instead of duplicating that logic."""

    schema: FeatureSchema
    model: object
    threshold: float
    feature_names: list[str]
    n_windows: int
    gran: float
    cache_hit: bool | None
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: np.ndarray


def fit_source_window(
    cfg: ShortlistedConfig,
    scenario: str,
    alert_groups: list,
    alert_groups_path: Path,
    n_total: int,
    base_schema: FeatureSchema,
    mining_settings_by_name: dict,
    mining_settings_path: Path,
    train_frac_within_window: float,
    threshold_mode: str,
    calibrated_recall_target: float,
    force_remine: bool = False,
) -> SourceWindowFit | None:
    """Mine (if `cfg.feature_set == "symbolic"`) on window 0's train split,
    fit `cfg.model` on that same train split, and decide a frozen threshold
    from its own scores. Returns None (with a warning printed, never raises)
    if the mining setting can't be resolved or window 0's train split turns
    out to be single-class -- both non-fatal, "this config can't run"
    conditions the caller is expected to skip past."""
    gran = cfg.granularity
    _, _, n_windows = compute_window_bounds(n_total, gran, 0)

    spec = None
    if cfg.feature_set == "symbolic":
        spec = mining_settings_by_name.get(cfg.mining_setting)
        if spec is None:
            print(
                f"  [warn] mining_setting '{cfg.mining_setting}' not found in "
                f"{mining_settings_path} -- skipping this config"
            )
            return None

    win_start, win_end, _ = compute_window_bounds(n_total, gran, 0)
    win_train_end = compute_window_train_end(
        win_start, win_end, train_frac_within_window
    )
    local_train_end = win_train_end - win_start
    window_rows = alert_groups[win_start:win_end]
    labels, mask = labels_and_mask(window_rows)
    n_attack_src = int(np.nansum(labels))

    print(
        f"  [W_src=window 0] n={len(window_rows)} attack={n_attack_src} "
        f"train_end(local)={local_train_end}"
    )

    cache_hit = None
    if cfg.feature_set == "baseline":
        schema = base_schema
    else:
        schema_result = get_or_mine_window_attribute_schema(
            scenario=scenario,
            alert_groups=alert_groups,
            alert_groups_path=alert_groups_path,
            gran=gran,
            win_idx=0,
            attribute_mining_config=spec.to_attribute_mining_config(),
            train_frac=train_frac_within_window,
            force=force_remine,
        )
        symbolic = load_symbolic_feature_schema(schema_result.schema_path)
        schema = FeatureSchema(
            schema_name="base+symbolic",
            schema_version=symbolic.schema_version,
            base=base_schema.base,
            symbolic=symbolic,
        )
        cache_hit = schema_result.cache_hit
        print(
            f"    [{cfg.mining_setting}] {'cache hit' if cache_hit else 'mined fresh'} "
            f"({len(symbolic.features)} features)"
        )

    encoded_src = encode_alert_groups_for_schema(window_rows, schema)
    y_masked = labels[mask].astype(int)
    X_masked = encoded_src.loc[mask].reset_index(drop=True)
    local_train_end_masked = int(mask[:local_train_end].sum())

    X_train = X_masked.iloc[:local_train_end_masked]
    X_test = X_masked.iloc[local_train_end_masked:].reset_index(drop=True)
    y_train = y_masked[:local_train_end_masked]
    y_test = y_masked[local_train_end_masked:]

    if len(np.unique(y_train)) < 2:
        print("    [warn] window 0 train split is single-class -- skipping")
        return None

    # "logreg"/"logreg_l1" (model_factory.py) are themselves scaled Pipelines,
    # so the fitted scaler is frozen along with the model (fit on W_src's
    # train split only, reused for every horizon's target window) like
    # everything else in this experiment's "freeze schema, model, threshold"
    # design.
    model = get_model_factory(cfg.model)()
    model.fit(X_train, y_train)
    proba_train = model.predict_proba(X_train)[:, 1]

    threshold = decide_threshold(
        y_train, proba_train, threshold_mode, calibrated_recall_target
    )

    return SourceWindowFit(
        schema=schema,
        model=model,
        threshold=threshold,
        feature_names=list(X_train.columns),
        n_windows=n_windows,
        gran=gran,
        cache_hit=cache_hit,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
    )


def encode_target_window(
    alert_groups: list,
    n_total: int,
    gran: float,
    win_idx: int,
    schema: FeatureSchema,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Encode window `win_idx` under a (frozen) schema, dropping unlabelled
    rows. Returns (X, y, n_alert_groups_in_window) -- the third value keeps
    the *unmasked* window size available for reporting even though X/y only
    cover labeled rows."""
    t_start, t_end, _ = compute_window_bounds(n_total, gran, win_idx)
    target_rows = alert_groups[t_start:t_end]
    t_labels, t_mask = labels_and_mask(target_rows)

    encoded_tgt = encode_alert_groups_for_schema(target_rows, schema)
    X_tgt = encoded_tgt.loc[t_mask].reset_index(drop=True)
    y_tgt = t_labels[t_mask].astype(int)
    return X_tgt, y_tgt, len(target_rows)


def _explanation_rows(
    model,
    X_background: pd.DataFrame,
    X_target: pd.DataFrame,
    feature_names: list[str],
    base_row: dict,
    horizon_window_index: int,
    horizon_fraction: float,
    config: TemporalDecayConfig,
) -> tuple[list[dict], list[dict]]:
    """SHAP + LIME signed importances for one horizon step, long format (one
    row per method x feature), plus a separate (0 or 1 row) list carrying
    LIME's mean local fidelity for this horizon -- fidelity is one number
    per horizon, not per feature, so it doesn't fit explanations.csv's long
    format. Each method's failure is independent -- a LIME crash shouldn't
    drop the SHAP rows already computed, and vice versa."""
    rows: list[dict] = []
    fidelity_rows: list[dict] = []
    x_explain = sample_rows(X_target, config.explain_sample_n, config.random_seed)
    if x_explain.empty:
        return rows, fidelity_rows

    horizon_meta = {
        **base_row,
        "horizon_window_index": horizon_window_index,
        "horizon_fraction": horizon_fraction,
        "n_explained": len(x_explain),
    }

    try:
        shap_importances = compute_shap_signed_importances(
            model,
            X_background,
            x_explain,
            feature_names,
            top_n=config.top_n_importances,
        )
        rows.extend(
            {
                **horizon_meta,
                "method": "shap",
                "feature": feat,
                "importance": val,
                "rank": rank,
            }
            for rank, (feat, val) in enumerate(shap_importances.items())
        )
    except Exception as exc:
        print(f"      [warn] SHAP failed at h={horizon_window_index}: {exc}")

    try:
        lime_result = compute_lime_signed_importances(
            model,
            X_background,
            x_explain,
            feature_names,
            top_n=config.top_n_importances,
            num_samples=config.lime_num_samples,
            random_state=config.random_seed,
        )
        rows.extend(
            {
                **horizon_meta,
                "method": "lime",
                "feature": feat,
                "importance": val,
                "rank": rank,
            }
            for rank, (feat, val) in enumerate(lime_result.importances.items())
        )
        fidelity_rows.append(
            {**horizon_meta, "mean_fidelity": lime_result.mean_fidelity}
        )
    except Exception as exc:
        print(f"      [warn] LIME failed at h={horizon_window_index}: {exc}")

    return rows, fidelity_rows


def _build_decay_summary(per_horizon_df: pd.DataFrame) -> pd.DataFrame:
    """One row per config: score/FPR at h=0 (W_src's own held-out test
    split) vs the last horizon actually reached, and their difference
    (decay_rate = score(h=0) - score(h=last); fpr_drift = fpr(h=last) -
    fpr(h=0))."""
    if per_horizon_df.empty:
        return pd.DataFrame()

    rows = []
    for keys, group in per_horizon_df.groupby(CONFIG_COLS, dropna=False):
        row = dict(zip(CONFIG_COLS, keys))
        pivot = group.set_index("horizon_window_index").sort_index()
        h_min, h_max = pivot.index.min(), pivot.index.max()
        row["h_max"] = int(h_max)
        for metric in METRIC_COLS:
            if metric not in pivot.columns:
                continue
            v_min = pivot[metric].get(h_min, np.nan)
            v_max = pivot[metric].get(h_max, np.nan)
            valid = pd.notna(v_min) and pd.notna(v_max)
            if metric == "fpr":
                row["fpr_at_h0"] = v_min
                row[f"fpr_at_h{h_max:g}"] = v_max
                row["fpr_drift"] = (v_max - v_min) if valid else np.nan
            else:
                row[f"{metric}_at_h0"] = v_min
                row[f"{metric}_at_h{h_max:g}"] = v_max
                row[f"decay_rate_{metric}"] = (v_min - v_max) if valid else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_temporal_decay_experiment(config: TemporalDecayConfig) -> Path:
    ensure_artifact_dirs()

    scenario = config.scenario
    ctx = load_scenario_context(
        scenario=scenario,
        cache_dir=config.cache_dir,
        grouping=config.grouping,
        alerts_json_path=config.alerts_json_path,
        mining_settings_path=config.mining_settings_path,
    )
    alert_groups = ctx.alert_groups
    alert_groups_path = ctx.alert_groups_path
    n_total = ctx.n_total
    base_schema = ctx.base_schema
    mining_settings_by_name = ctx.mining_settings_by_name
    mining_settings_path = ctx.mining_settings_path

    print("[4/4] Loading shortlist...")
    shortlist = load_shortlist(config.shortlist_path)
    print(f"  {len(shortlist)} shortlisted configs")

    horizon_rows: list[dict] = []
    explain_rows: list[dict] = []
    lime_fidelity_rows: list[dict] = []

    # Shortlisted configs are independent (each mines/fits/evaluates entirely
    # on its own windows) -- run them concurrently instead of one at a time.
    # See TemporalDecayConfig.n_jobs for why this is a thread pool, not
    # processes.
    print(
        f"  Running {len(shortlist)} shortlisted configs with n_jobs={config.n_jobs}..."
    )
    with ThreadPoolExecutor(max_workers=config.n_jobs) as pool:
        future_to_cfg = {
            pool.submit(
                _run_one_config,
                cfg=cfg,
                config=config,
                scenario=scenario,
                alert_groups=alert_groups,
                alert_groups_path=alert_groups_path,
                n_total=n_total,
                base_schema=base_schema,
                mining_settings_by_name=mining_settings_by_name,
                mining_settings_path=mining_settings_path,
            ): cfg
            for cfg in shortlist
        }
        for future in as_completed(future_to_cfg):
            cfg = future_to_cfg[future]
            try:
                cfg_horizon_rows, cfg_explain_rows, cfg_fidelity_rows = future.result()
                horizon_rows.extend(cfg_horizon_rows)
                explain_rows.extend(cfg_explain_rows)
                lime_fidelity_rows.extend(cfg_fidelity_rows)
            except Exception as exc:
                print(f"  [warn] config {cfg} failed: {exc}")
                traceback.print_exc()

    results_dir = (
        config.results_dir
        if config.results_dir is not None
        else _EXPERIMENTS_DIR / scenario
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    per_horizon_df = pd.DataFrame(horizon_rows)
    per_horizon_path = out_dir / "per_horizon_results.csv"
    per_horizon_df.to_csv(per_horizon_path, index=False)
    print(f"\n  Saved → {per_horizon_path}")

    decay_summary_df = _build_decay_summary(per_horizon_df)
    decay_summary_path = out_dir / "decay_summary.csv"
    decay_summary_df.to_csv(decay_summary_path, index=False)
    print(f"  Saved → {decay_summary_path}")

    explanations_df = pd.DataFrame(explain_rows)
    explanations_path = out_dir / "explanations.csv"
    explanations_df.to_csv(explanations_path, index=False)
    print(f"  Saved → {explanations_path}")

    lime_fidelity_df = pd.DataFrame(lime_fidelity_rows)
    lime_fidelity_path = out_dir / "lime_fidelity.csv"
    lime_fidelity_df.to_csv(lime_fidelity_path, index=False)
    print(f"  Saved → {lime_fidelity_path}")

    summary_lines = [
        f"Temporal Decay — {ts}",
        f"Scenario: {scenario}",
        f"Shortlist: {config.shortlist_path} ({len(shortlist)} configs)",
        f"Threshold mode: {config.threshold_mode}"
        + (
            f" (recall target={config.calibrated_recall_target})"
            if config.threshold_mode == "calibrated_recall"
            else ""
        ),
        f"Explanations: {'on' if config.compute_explanations else 'off'} "
        f"(background_n={config.explain_background_n}, sample_n={config.explain_sample_n}, "
        f"lime_num_samples={config.lime_num_samples})",
        f"Rows: {len(per_horizon_df)} (per_horizon), {len(explanations_df)} (explanations), "
        f"{len(lime_fidelity_df)} (lime_fidelity)",
        "",
        per_horizon_df.to_string(index=False)
        if not per_horizon_df.empty
        else "(no data)",
    ]
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"  Saved → {summary_path}")

    (out_dir / "config.json").write_text(
        pd.Series({**asdict(config), "mining_settings_path": str(mining_settings_path)})
        .apply(str)
        .to_json(indent=2)
    )

    return out_dir


def _run_one_config(
    cfg: ShortlistedConfig,
    config: TemporalDecayConfig,
    scenario: str,
    alert_groups: list,
    alert_groups_path: Path,
    n_total: int,
    base_schema: FeatureSchema,
    mining_settings_by_name: dict,
    mining_settings_path: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    print(
        f"\n[{cfg.feature_set}/{cfg.mining_setting}/gran={cfg.granularity:g}] starting"
    )

    fit = fit_source_window(
        cfg=cfg,
        scenario=scenario,
        alert_groups=alert_groups,
        alert_groups_path=alert_groups_path,
        n_total=n_total,
        base_schema=base_schema,
        mining_settings_by_name=mining_settings_by_name,
        mining_settings_path=mining_settings_path,
        train_frac_within_window=config.train_frac_within_window,
        threshold_mode=config.threshold_mode,
        calibrated_recall_target=config.calibrated_recall_target,
        force_remine=config.force_remine,
    )
    if fit is None:
        return [], [], []

    explain_background = (
        sample_rows(fit.X_train, config.explain_background_n, config.random_seed)
        if config.compute_explanations
        else None
    )

    base_row = {
        "scenario": scenario,
        "feature_set": cfg.feature_set,
        "mining_setting": cfg.mining_setting,
        "granularity": fit.gran,
        "model": cfg.model,
        "n_windows": fit.n_windows,
        "threshold_mode": config.threshold_mode,
        "threshold": fit.threshold,
        "mining_cache_hit": fit.cache_hit,
    }

    horizon_rows: list[dict] = []
    explain_rows: list[dict] = []
    fidelity_rows: list[dict] = []

    def _record_horizon(
        horizon_window_index: int,
        X_h: pd.DataFrame,
        y_h: np.ndarray,
        n_alert_groups_h: int,
    ) -> None:
        horizon_fraction = (
            horizon_window_index / (fit.n_windows - 1) if fit.n_windows > 1 else 0.0
        )
        if len(y_h) == 0:
            print(
                f"    [warn] horizon {horizon_window_index} has no labeled rows "
                "-- recording nan metrics"
            )
            horizon_rows.append(
                {
                    **base_row,
                    "horizon_window_index": horizon_window_index,
                    "horizon_fraction": horizon_fraction,
                    "is_source_window": horizon_window_index == 0,
                    "target_single_class": True,
                    "n_alert_groups": n_alert_groups_h,
                    "n_attack": 0,
                    **nan_metrics(),
                }
            )
            return

        target_single_class = len(np.unique(y_h)) < 2
        proba_h = fit.model.predict_proba(X_h)[:, 1]
        metrics = metrics_at_threshold(y_h, proba_h, fit.threshold)
        horizon_rows.append(
            {
                **base_row,
                "horizon_window_index": horizon_window_index,
                "horizon_fraction": horizon_fraction,
                "is_source_window": horizon_window_index == 0,
                "target_single_class": target_single_class,
                "n_alert_groups": n_alert_groups_h,
                "n_attack": int(np.nansum(y_h)),
                **metrics,
            }
        )
        if config.compute_explanations:
            cfg_explain_rows, cfg_fidelity_rows = _explanation_rows(
                fit.model,
                explain_background,
                X_h,
                fit.feature_names,
                base_row,
                horizon_window_index,
                horizon_fraction,
                config,
            )
            explain_rows.extend(cfg_explain_rows)
            fidelity_rows.extend(cfg_fidelity_rows)

    # h=0: W_src's own held-out test split -- never seen by mining or fitting.
    _record_horizon(0, fit.X_test, fit.y_test, len(fit.X_test))

    for k in range(1, fit.n_windows):
        X_tgt, y_tgt, n_alert_groups = encode_target_window(
            alert_groups, n_total, fit.gran, k, fit.schema
        )
        _record_horizon(k, X_tgt, y_tgt, n_alert_groups)

    return horizon_rows, explain_rows, fidelity_rows

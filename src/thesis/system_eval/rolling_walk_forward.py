"""
Experiment 3: Rolling / Walk-Forward Evaluation.

Purpose: for a shortlisted (feature_set, mining_setting, granularity, model)
config -- the output of Experiment 1 (screening_sweep.py +
thesis.metrics.config_selection), the same shortlist Experiment 2 uses --
give an aggregate, low-noise "how good is this config in general use"
estimate by sliding across the entire timeline and retraining at every
step, in contrast to Experiment 2's single fixed-source-window snapshot.
This is the "always retrain" anchor: the opposite extreme from Experiment
2's "never retrain, frozen" decay curve. The gap between the two is the
improvement a perfect always-retrain policy buys, which is what Experiment
4's drift monitor gets compared against.

feature_set is one of "baseline", "symbolic", "cscas_full", or
"cscas_full_symbolic" -- see temporal_decay.py's module docstring for what
each means; the cscas_full base-column logic (incl. dropping `scas` for a
one-class model) is imported from there rather than duplicated.

For a given granularity g, the timeline is carved into n(g) windows exactly
as in screening_sweep.py/temporal_decay.py (pipeline.compute_window_bounds).
Walking i = 0 .. n(g)-2 (n(g)-1 steps total):

  1. Mine a schema on the *full* window Wi -- no held-out split within Wi,
     unlike screening_sweep/temporal_decay's train-split-only mining
     (mining.window_schema_cache.get_or_mine_full_window_attribute_schema).
     Those experiments hold back part of a window because they evaluate on
     that same window; here the held-out evaluation set is the disjoint
     window Wi+1, so all of Wi is available to mine and train on.
  2. Fit the config's model on all of Wi's encoding.
  3. Decide a threshold from Wi's own (in-sample) scores -- same method
     (flat 0.5, or calibrated-recall) at every step, per the experiment
     spec's requirement that the threshold-decision rule be locked in once
     and applied consistently; only the *value* varies step to step, since
     it's recomputed from that step's own freshly-fit model.
  4. Encode W(i+1) under Wi's schema, evaluate the model at that threshold.
  5. Also compute SHAP and LIME signed importances (thesis.training.explain)
     for *every* schema feature, on a sample of W(i+1) -- the same window
     the metrics are scored on -- against a background sample drawn from Wi
     (the data the model was just mined+fit on). Unlike Experiment 2, there
     is no frozen W_src here: both the background and the explained sample
     are step-local, redrawn fresh every step since the model itself is
     refit every step. Together with per_step_results.csv this answers two
     questions at once: which features this step's schema mined (every
     feature gets a row in explanations.csv, whether or not it mattered),
     and which of those actually moved the model's output on the held-out
     evaluation window (the importance/rank columns). One-class models get
     LIME only unless config.oneclass_shap is set (see temporal_decay.py --
     same reasoning: no analytic SHAP explainer, PermutationExplainer
     dominates the run's cost).
  6. Discard the schema and model. The next step re-mines and retrains from
     scratch on W(i+1) -- no accumulation, no state carried forward.

Outputs: per_step_results.csv (one row per config x step), a
walk_forward_summary.csv (mean/std of auc/f1/fpr per config across all
steps -- the headline "how good is this in general" number),
explanations.csv (long format: one row per config x step x
method[shap|lime] x feature), lime_fidelity.csv (one row per config x step:
LIME's own local-surrogate R^2, averaged over that step's explained
sample), summary.txt, config.json.
"""

from __future__ import annotations

import shutil
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
    ONE_CLASS_MODELS,
    decide_threshold,
    fit_scored_model,
    labels_and_mask,
    load_scenario_context,
    metrics_at_threshold,
    nan_metrics,
    sample_rows,
)
from thesis.system_eval.temporal_decay import (
    _SYMBOLIC_FEATURE_SETS,
    _cscas_full_base,
    encode_target_window,
)
from thesis.features.persistence import load_symbolic_feature_schema
from thesis.metrics.shortlist import ShortlistedConfig, load_shortlist
from thesis.mining.window_schema_cache import get_or_mine_full_window_attribute_schema
from thesis.paths import RESULTS_DIR, ensure_artifact_dirs
from thesis.pipeline.pipeline import compute_window_bounds
from thesis.schemas.experiments import RollingWalkForwardConfig
from thesis.schemas.features import FeatureSchema
from thesis.training.explain import (
    compute_lime_signed_importances,
    compute_shap_signed_importances,
)

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS_DIR = _ROOT / "artifacts" / "experiments" / "rolling_walk_forward"
# A default run's output dir is also copied here, verbatim, as the curated,
# human-facing home for rolling-walk-forward results (see temporal_decay.py's
# _RESULTS_MIRROR_DIR -- same pattern). Skipped when the caller passes an
# explicit results_dir.
_RESULTS_MIRROR_DIR = RESULTS_DIR / "sys-eval" / "rolling-walk-forward"

SUMMARY_METRICS = ["auc", "f1", "fpr"]


@dataclass(slots=True)
class WindowFit:
    """Everything produced by mining+fitting one step's training window Wi:
    the schema, the fitted model, the decision threshold, and Wi's own
    encoding (X_fit) -- the SHAP/LIME background sample for this step is
    drawn from X_fit, since there's no frozen W_src to draw it from once
    the way Exp2 does; every step redraws its own. Unlike Exp2's
    SourceWindowFit there's no held-out X_test -- Wi has no internal split,
    the held-out evaluation set is the disjoint window Wi+1."""

    schema: FeatureSchema
    model: object
    threshold: float
    feature_names: list[str]
    cache_hit: bool | None
    X_fit: pd.DataFrame


def fit_window(
    cfg: ShortlistedConfig,
    scenario: str,
    alert_groups: list,
    alert_groups_path: Path,
    n_total: int,
    win_idx: int,
    base_schema: FeatureSchema,
    mining_settings_by_name: dict,
    mining_settings_path: Path,
    threshold_mode: str,
    calibrated_recall_target: float,
    force_remine: bool = False,
) -> WindowFit | None:
    """Mine (for `cfg.feature_set` in {"symbolic", "cscas_full_symbolic"}) on
    the *full* window `win_idx`, fit `cfg.model` on all of it, and decide a
    threshold from its own in-sample scores. Returns None (with a warning
    printed, never raises) if the mining setting can't be resolved or the
    window turns out to be single-class -- both non-fatal, "this step can't
    run" conditions the caller is expected to skip past."""
    gran = cfg.granularity
    win_start, win_end, _ = compute_window_bounds(n_total, gran, win_idx)
    window_rows = alert_groups[win_start:win_end]
    labels, mask = labels_and_mask(window_rows)
    n_attack = int(np.nansum(labels))

    print(f"  [Wi=window {win_idx}] n={len(window_rows)} attack={n_attack}")

    spec = None
    if cfg.feature_set in _SYMBOLIC_FEATURE_SETS:
        spec = mining_settings_by_name.get(cfg.mining_setting)
        if spec is None:
            print(
                f"  [warn] mining_setting '{cfg.mining_setting}' not found in "
                f"{mining_settings_path} -- skipping this config"
            )
            return None

    cache_hit = None
    if cfg.feature_set == "baseline":
        schema = base_schema
    elif cfg.feature_set == "cscas_full":
        schema = FeatureSchema(
            schema_name="cscas_full",
            schema_version="0.1.0",
            base=_cscas_full_base(cfg.model),
            symbolic=None,
        )
    else:
        # "symbolic" or "cscas_full_symbolic": mine the symbolic layer on the
        # full window, then attach it to the reduced base columns
        # ("symbolic") or the full CSCAS columns ("cscas_full_symbolic").
        # encode_alert_groups_for_schema drops any column the two halves
        # share, so the base features aren't doubled.
        schema_result = get_or_mine_full_window_attribute_schema(
            scenario=scenario,
            alert_groups=alert_groups,
            alert_groups_path=alert_groups_path,
            gran=gran,
            win_idx=win_idx,
            attribute_mining_config=spec.to_attribute_mining_config(),
            force=force_remine,
        )
        symbolic = load_symbolic_feature_schema(schema_result.schema_path)
        cache_hit = schema_result.cache_hit
        if cfg.feature_set == "cscas_full_symbolic":
            base_part = _cscas_full_base(cfg.model)
            schema_tag = "cscas_full+symbolic"
        else:
            base_part = base_schema.base
            schema_tag = "base+symbolic"
        schema = FeatureSchema(
            schema_name=schema_tag,
            schema_version=symbolic.schema_version,
            base=base_part,
            symbolic=symbolic,
        )
        print(
            f"    [{cfg.mining_setting}] {schema_tag}: "
            f"{'cache hit' if cache_hit else 'mined fresh'} "
            f"({len(base_part.features)} base + {len(symbolic.features)} symbolic)"
        )

    encoded = encode_alert_groups_for_schema(window_rows, schema)
    y = labels[mask].astype(int)
    X = encoded.loc[mask].reset_index(drop=True)

    if len(np.unique(y)) < 2:
        print(f"    [warn] window {win_idx} is single-class -- skipping")
        return None

    # fit_scored_model, not a bare get_model_factory(cfg.model)().fit(X, y):
    # it hands every supervised model class-imbalance-aware kwargs
    # (class_weight="balanced" / scale_pos_weight -- see its docstring),
    # which matters for xgboost in particular (unweighted, it sits at a
    # low-recall corner at a flat 0.5 on this scenario's imbalance). logreg
    # and rf hardcode "balanced" in their own factory entry either way, so
    # this is a no-op for them but the only correct path for xgboost.
    model = fit_scored_model(cfg.model, X, y)
    if model is None:
        print(f"    [warn] window {win_idx}: {cfg.model} could not be fit -- skipping")
        return None
    proba = model.predict_proba(X)[:, 1]

    threshold = decide_threshold(
        y, proba, threshold_mode, calibrated_recall_target, model=model
    )

    return WindowFit(
        schema=schema,
        model=model,
        threshold=threshold,
        feature_names=list(X.columns),
        cache_hit=cache_hit,
        X_fit=X,
    )


def _explanation_rows(
    model,
    X_background: pd.DataFrame,
    X_target: pd.DataFrame,
    feature_names: list[str],
    base_row: dict,
    step_i: int,
    config: RollingWalkForwardConfig,
) -> tuple[list[dict], list[dict]]:
    """SHAP + LIME signed importances for one step, long format (one row per
    method x feature) -- explains this step's freshly-fit model on a sample
    of W(i+1) (the same window the metrics are scored on), against a
    background sample drawn from Wi (what the model was just fit on). Both
    are step-local and redrawn every step -- unlike temporal_decay.py, there
    is no frozen W_src background to reuse across steps. Every schema
    feature is recorded (top_n = full width): together with
    per_step_results.csv, this answers both "what did this step's schema
    mine" (every feature gets a row here) and "which of those moved the
    model's output on the held-out window" (the importance/rank columns).
    Each method's failure is independent -- a LIME crash shouldn't drop the
    SHAP rows already computed, and vice versa."""
    rows: list[dict] = []
    fidelity_rows: list[dict] = []
    x_explain = sample_rows(X_target, config.explain_sample_n, config.random_seed)
    if x_explain.empty:
        return rows, fidelity_rows

    n_feats = len(feature_names)
    step_meta = {**base_row, "step_i": step_i, "n_explained": len(x_explain)}

    try:
        shap_importances = compute_shap_signed_importances(
            model,
            X_background,
            x_explain,
            feature_names,
            top_n=n_feats,
        )
        rows.extend(
            {
                **step_meta,
                "method": "shap",
                "feature": feat,
                "importance": val,
                "rank": rank,
            }
            for rank, (feat, val) in enumerate(shap_importances.items())
        )
    except Exception as exc:
        print(f"      [warn] SHAP failed at step {step_i}: {exc}")

    try:
        lime_result = compute_lime_signed_importances(
            model,
            X_background,
            x_explain,
            feature_names,
            top_n=n_feats,
            num_samples=config.lime_num_samples,
            random_state=config.random_seed,
        )
        rows.extend(
            {
                **step_meta,
                "method": "lime",
                "feature": feat,
                "importance": val,
                "rank": rank,
            }
            for rank, (feat, val) in enumerate(lime_result.importances.items())
        )
        fidelity_rows.append({**step_meta, "mean_fidelity": lime_result.mean_fidelity})
    except Exception as exc:
        print(f"      [warn] LIME failed at step {step_i}: {exc}")

    return rows, fidelity_rows


def _build_walk_forward_summary(per_step_df: pd.DataFrame) -> pd.DataFrame:
    """One row per config: mean/std of auc, f1, and fpr across every step
    actually run -- the headline "how good is this in general" number,
    directly comparable across configs and against Experiment 2's
    fixed-source-window results."""
    if per_step_df.empty:
        return pd.DataFrame()

    agg = per_step_df.groupby(CONFIG_COLS, dropna=False)[SUMMARY_METRICS].agg(
        ["mean", "std"]
    )
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_steps"] = per_step_df.groupby(CONFIG_COLS, dropna=False).size()
    return agg.reset_index()


def run_rolling_walk_forward_experiment(config: RollingWalkForwardConfig) -> Path:
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

    step_rows: list[dict] = []
    explain_rows: list[dict] = []
    fidelity_rows: list[dict] = []

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
                cfg_step_rows, cfg_explain_rows, cfg_fidelity_rows = future.result()
                step_rows.extend(cfg_step_rows)
                explain_rows.extend(cfg_explain_rows)
                fidelity_rows.extend(cfg_fidelity_rows)
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

    per_step_df = pd.DataFrame(step_rows)
    per_step_path = out_dir / "per_step_results.csv"
    per_step_df.to_csv(per_step_path, index=False)
    print(f"\n  Saved → {per_step_path}")

    summary_df = _build_walk_forward_summary(per_step_df)
    summary_path = out_dir / "walk_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved → {summary_path}")

    explanations_df = pd.DataFrame(explain_rows)
    explanations_path = out_dir / "explanations.csv"
    explanations_df.to_csv(explanations_path, index=False)
    print(f"  Saved → {explanations_path}")

    lime_fidelity_df = pd.DataFrame(fidelity_rows)
    lime_fidelity_path = out_dir / "lime_fidelity.csv"
    lime_fidelity_df.to_csv(lime_fidelity_path, index=False)
    print(f"  Saved → {lime_fidelity_path}")

    summary_lines = [
        f"Rolling Walk-Forward — {ts}",
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
        f"Rows: {len(per_step_df)} (per_step), {len(summary_df)} (summary), "
        f"{len(explanations_df)} (explanations), {len(lime_fidelity_df)} (lime_fidelity)",
        "",
        summary_df.to_string(index=False) if not summary_df.empty else "(no data)",
    ]
    summary_txt_path = out_dir / "summary.txt"
    summary_txt_path.write_text("\n".join(summary_lines) + "\n")
    print(f"  Saved → {summary_txt_path}")

    (out_dir / "config.json").write_text(
        pd.Series({**asdict(config), "mining_settings_path": str(mining_settings_path)})
        .apply(str)
        .to_json(indent=2)
    )

    # Mirror the finished run into the curated results/ tree (unless the
    # caller chose their own output location). Best-effort -- a copy failure
    # must not lose the primary run under out_dir.
    if config.results_dir is None:
        mirror_dir = _RESULTS_MIRROR_DIR / scenario / ts
        try:
            shutil.copytree(out_dir, mirror_dir, dirs_exist_ok=True)
            print(f"  Mirrored → {mirror_dir}")
        except OSError as exc:
            print(f"  [warn] could not mirror results to {mirror_dir}: {exc}")

    return out_dir


def _run_one_config(
    cfg: ShortlistedConfig,
    config: RollingWalkForwardConfig,
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

    _, _, n_windows = compute_window_bounds(n_total, cfg.granularity, 0)

    base_row = {
        "scenario": scenario,
        "feature_set": cfg.feature_set,
        "mining_setting": cfg.mining_setting,
        "granularity": cfg.granularity,
        "model": cfg.model,
        "n_windows": n_windows,
        "threshold_mode": config.threshold_mode,
    }

    rows: list[dict] = []
    explain_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    for i in range(n_windows - 1):
        fit = fit_window(
            cfg=cfg,
            scenario=scenario,
            alert_groups=alert_groups,
            alert_groups_path=alert_groups_path,
            n_total=n_total,
            win_idx=i,
            base_schema=base_schema,
            mining_settings_by_name=mining_settings_by_name,
            mining_settings_path=mining_settings_path,
            threshold_mode=config.threshold_mode,
            calibrated_recall_target=config.calibrated_recall_target,
            force_remine=config.force_remine,
        )
        if fit is None:
            rows.append(
                {
                    **base_row,
                    "step_i": i,
                    "threshold": np.nan,
                    "mining_cache_hit": None,
                    "n_alert_groups": 0,
                    "n_attack": 0,
                    **nan_metrics(),
                }
            )
            continue

        # One-class SHAP has no analytic explainer -> PermutationExplainer
        # over every feature at this step, the dominant cost of an
        # explanations run (see temporal_decay.py). Unless explicitly asked
        # for, flag the model so compute_shap_signed_importances raises
        # straight away (LIME still runs).
        if (
            config.compute_explanations
            and not config.oneclass_shap
            and cfg.model in ONE_CLASS_MODELS
        ):
            fit.model._skip_shap = True

        X_next, y_next, n_alert_groups_next = encode_target_window(
            alert_groups, n_total, cfg.granularity, i + 1, fit.schema
        )
        if len(y_next) == 0:
            print(
                f"    [warn] step {i}: window {i + 1} has no labeled rows -- recording nan metrics"
            )
            rows.append(
                {
                    **base_row,
                    "step_i": i,
                    "threshold": fit.threshold,
                    "mining_cache_hit": fit.cache_hit,
                    "n_alert_groups": n_alert_groups_next,
                    "n_attack": 0,
                    **nan_metrics(),
                }
            )
            continue

        proba_next = fit.model.predict_proba(X_next)[:, 1]
        metrics = metrics_at_threshold(y_next, proba_next, fit.threshold)
        rows.append(
            {
                **base_row,
                "step_i": i,
                "threshold": fit.threshold,
                "mining_cache_hit": fit.cache_hit,
                "n_alert_groups": n_alert_groups_next,
                "n_attack": int(np.nansum(y_next)),
                **metrics,
            }
        )

        if config.compute_explanations:
            explain_background = sample_rows(
                fit.X_fit, config.explain_background_n, config.random_seed
            )
            cfg_explain_rows, cfg_fidelity_rows = _explanation_rows(
                fit.model,
                explain_background,
                X_next,
                fit.feature_names,
                base_row,
                i,
                config,
            )
            explain_rows.extend(cfg_explain_rows)
            fidelity_rows.extend(cfg_fidelity_rows)

    return rows, explain_rows, fidelity_rows

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
  5. Discard the schema and model. The next step re-mines and retrains from
     scratch on W(i+1) -- no accumulation, no state carried forward.

Outputs: per_step_results.csv (one row per config x step), a
walk_forward_summary.csv (mean/std of auc/f1/fpr per config across all
steps -- the headline "how good is this in general" number), summary.txt,
config.json.
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
    decide_threshold,
    labels_and_mask,
    load_scenario_context,
    metrics_at_threshold,
    nan_metrics,
)
from thesis.system_eval.temporal_decay import encode_target_window
from thesis.features.persistence import load_symbolic_feature_schema
from thesis.metrics.shortlist import ShortlistedConfig, load_shortlist
from thesis.mining.window_schema_cache import get_or_mine_full_window_attribute_schema
from thesis.paths import ensure_artifact_dirs
from thesis.pipeline.pipeline import compute_window_bounds
from thesis.schemas.experiments import RollingWalkForwardConfig
from thesis.schemas.features import FeatureSchema
from thesis.training.model_factory import get_model_factory

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS_DIR = _ROOT / "artifacts" / "experiments" / "rolling_walk_forward"

SUMMARY_METRICS = ["auc", "f1", "fpr"]


@dataclass(slots=True)
class WindowFit:
    """Everything produced by mining+fitting one step's training window Wi:
    the schema, the fitted model, and the decision threshold. Unlike Exp2's
    SourceWindowFit, there's no frozen X_train/X_test to carry -- Wi has no
    internal split, and this experiment doesn't track SHAP/LIME."""

    schema: FeatureSchema
    model: object
    threshold: float
    feature_names: list[str]
    cache_hit: bool | None


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
    """Mine (if `cfg.feature_set == "symbolic"`) on the *full* window
    `win_idx`, fit `cfg.model` on all of it, and decide a threshold from its
    own in-sample scores. Returns None (with a warning printed, never
    raises) if the mining setting can't be resolved or the window turns out
    to be single-class -- both non-fatal, "this step can't run" conditions
    the caller is expected to skip past."""
    gran = cfg.granularity
    win_start, win_end, _ = compute_window_bounds(n_total, gran, win_idx)
    window_rows = alert_groups[win_start:win_end]
    labels, mask = labels_and_mask(window_rows)
    n_attack = int(np.nansum(labels))

    print(f"  [Wi=window {win_idx}] n={len(window_rows)} attack={n_attack}")

    spec = None
    if cfg.feature_set == "symbolic":
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
    else:
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

    encoded = encode_alert_groups_for_schema(window_rows, schema)
    y = labels[mask].astype(int)
    X = encoded.loc[mask].reset_index(drop=True)

    if len(np.unique(y)) < 2:
        print(f"    [warn] window {win_idx} is single-class -- skipping")
        return None

    model = get_model_factory(cfg.model)()
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]

    threshold = decide_threshold(y, proba, threshold_mode, calibrated_recall_target)

    return WindowFit(
        schema=schema,
        model=model,
        threshold=threshold,
        feature_names=list(X.columns),
        cache_hit=cache_hit,
    )


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
                step_rows.extend(future.result())
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
        f"Rows: {len(per_step_df)} (per_step), {len(summary_df)} (summary)",
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
) -> list[dict]:
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

    return rows

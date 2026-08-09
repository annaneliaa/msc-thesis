"""
Experiment 4: Drift-Monitor Evaluation (observe-only).

Purpose: for a shortlisted (feature_set, mining_setting, granularity, model)
config, mine a schema and fit a model once on window 0's train split (same
freeze-and-decay design as experiments/temporal_decay.py -- W_src is always
window 0, with its own internal 70/30 train/test split), and additionally --
for symbolic configs -- build a deployment-scoped DynamicSchema (Vk) from
that same mining pass. Freeze schema, model, threshold, and Vk. Walk forward
one window at a time from h=0 (W_src's own held-out test split) to
h=n_windows-1, scoring the frozen model exactly like temporal_decay.py, and
at every horizon also run thesis.monitor.monitor.run_monitor_window against
the frozen Vk over that horizon's raw incoming alert groups, logging every
signal it computes and every alarm it raises (which predicate/rule, which
field/value, which class, how far past threshold).

This is observe-only: the monitor's action (NO_ACTION/SOFT_ALERT/
RETRAIN_ONLY/REMINE_AND_RETRAIN) is recorded but never acted on -- nothing
is ever re-mined or retrained within this experiment. A future, separate
"reactive" experiment could do that; this one answers "what would the
monitor have said, and when, watching one frozen schema drift?"

The monitor must see the RAW, unmasked window/horizon AlertGroup slice, not
the label-masked one used for model scoring: run_monitor_window needs no
labels for Signal 1 (predicate activation), and does its own internal label
filtering for Signal 2, reporting n_incoming_groups vs n_labeled_groups
separately in MonitorSnapshot specifically so that distinction survives.
Pre-masking before the call would collapse n_incoming_groups ==
n_labeled_groups and misrepresent production semantics (most live traffic
won't have confirmed labels yet).

fit_source_window_and_dynamic_schema and the horizon walk below duplicate
(rather than extend) experiments/temporal_decay.py's fit_source_window /
encode_target_window: those functions are also used by
experiments/instance_explain.py for on-demand SHAP/LIME case studies, and
this experiment's mining strategy for symbolic configs differs entirely
(bypasses the cached mining wrapper to also retain the raw contrast/leaf
stats a DynamicSchema needs) -- branching that into the shared functions
would bloat them for consumers that never touch this path.

Outputs: per_horizon_results.csv (model decay metrics, same shape as
temporal_decay.py's, plus per-horizon monitor summary columns),
decay_summary.csv (reuses temporal_decay.py's _build_decay_summary
unmodified), monitor_signals.csv (dense long format: one row per config x
horizon x predicate/rule, every signal whether elevated or not),
monitor_alarms.csv (the elevated == True subset of monitor_signals.csv),
summary.txt, config.json. No SHAP/LIME in this experiment.
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
    decide_threshold,
    labels_and_mask,
    load_scenario_context,
    metrics_at_threshold,
    nan_metrics,
)
from thesis.system_eval.temporal_decay import _build_decay_summary
from thesis.features.dynamic_schema_builder import build_dynamic_schema
from thesis.features.schema_builder import build_symbolic_feature_schema
from thesis.metrics.shortlist import ShortlistedConfig, load_shortlist
from thesis.mining.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
    compute_predicate_contrast_stats,
    filter_contrast_survivors,
    surviving_single_columns,
)
from thesis.mining.decision_tree_rule_mining import (
    build_training_matrix,
    extract_leaf_rules,
    fit_rule_tree,
)
from thesis.monitor.monitor import run_monitor_window
from thesis.monitor.state import MonitorState
from thesis.paths import ensure_artifact_dirs
from thesis.pipeline.pipeline import compute_window_bounds, compute_window_train_end
from thesis.schemas.dynamic_schema import DynamicSchema
from thesis.schemas.experiments import MonitorDriftConfig
from thesis.schemas.features import FeatureSchema
from thesis.schemas.groups import AlertGroup
from thesis.training.model_factory import get_model_factory

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS_DIR = _ROOT / "artifacts" / "experiments" / "monitor_drift"


@dataclass(slots=True)
class MonitorSourceWindowFit:
    """Everything frozen once window 0's train split is mined+fit, plus the
    deployment-scoped DynamicSchema (Vk) the monitor evaluates against --
    the sibling of temporal_decay.py's SourceWindowFit for this experiment."""

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
    dynamic_schema: DynamicSchema | None
    test_groups_raw: list[AlertGroup]


def fit_source_window_and_dynamic_schema(
    cfg: ShortlistedConfig,
    scenario: str,
    alert_groups: list[AlertGroup],
    n_total: int,
    base_schema: FeatureSchema,
    mining_settings_by_name: dict,
    mining_settings_path: Path,
    train_frac_within_window: float,
    threshold_mode: str,
    calibrated_recall_target: float,
) -> MonitorSourceWindowFit | None:
    """Mine (if cfg.feature_set == "symbolic") on window 0's train split via
    the direct two-stage mining building blocks (not the cached wrapper
    temporal_decay.py uses -- that wrapper's return chain only keeps the
    post-concatenation mined_df, losing the attack/benign split
    build_dynamic_schema needs), fit cfg.model on that same train split, and
    decide a frozen threshold from its own scores. Returns None (warns,
    never raises) for the same non-fatal "this config can't run" conditions
    fit_source_window does."""
    gran = cfg.granularity
    win_start, win_end, n_windows = compute_window_bounds(n_total, gran, 0)
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
    dynamic_schema: DynamicSchema | None = None
    if cfg.feature_set == "baseline":
        schema = base_schema
    else:
        spec = mining_settings_by_name.get(cfg.mining_setting)
        if spec is None:
            print(
                f"  [warn] mining_setting '{cfg.mining_setting}' not found in "
                f"{mining_settings_path} -- skipping this config"
            )
            return None

        train_rows_labeled = [
            tx
            for tx in window_rows[:local_train_end]
            if tx.group_label in ("benign", "attack")
        ]
        if (
            not train_rows_labeled
            or len({tx.group_label for tx in train_rows_labeled}) < 2
        ):
            print(
                "    [warn] window 0 train split has no labeled rows or is "
                "single-class -- skipping"
            )
            return None

        attribute_mining_config = spec.to_attribute_mining_config()
        X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(
            train_rows_labeled
        )
        contrast_stats_df = compute_predicate_contrast_stats(
            X_cat, y, column_predicate_map
        )
        survivors_df = filter_contrast_survivors(
            contrast_stats_df,
            min_attack_coverage=attribute_mining_config.contrast.min_attack_coverage,
            min_benign_coverage=attribute_mining_config.contrast.min_benign_coverage,
            min_growth_rate=attribute_mining_config.contrast.min_growth_rate,
            max_p_value=attribute_mining_config.contrast.max_p_value,
        )
        surviving_cols = surviving_single_columns(survivors_df)
        X_train_mat, kept_predicate_map = build_training_matrix(
            X_cat, X_num, column_predicate_map, surviving_cols
        )
        tree = fit_rule_tree(
            X_train_mat,
            y,
            max_depth=attribute_mining_config.tree.max_depth,
            min_samples_leaf=attribute_mining_config.tree.min_samples_leaf,
            class_weight=attribute_mining_config.tree.class_weight,
            random_state=attribute_mining_config.tree.random_state,
            min_impurity_decrease=attribute_mining_config.tree.min_impurity_decrease,
        )
        leaf_rules_df, predicate_alphabet = extract_leaf_rules(
            tree, X_train_mat, y, kept_predicate_map
        )

        mined_at = datetime.now(timezone.utc)
        mining_window_start = datetime.fromtimestamp(
            train_rows_labeled[0].start_ts, tz=timezone.utc
        )
        mining_window_end = datetime.fromtimestamp(
            train_rows_labeled[-1].end_ts or train_rows_labeled[-1].start_ts,
            tz=timezone.utc,
        )

        # Build the DynamicSchema first, from the untagged survivors/leaf
        # frames -- build_dynamic_schema doesn't read source_label, but
        # keeping this ordering explicit avoids any future coupling surprise.
        dynamic_schema = build_dynamic_schema(
            contrast_stats_df=survivors_df,
            leaf_rules_df=leaf_rules_df,
            predicate_alphabet=predicate_alphabet,
            column_predicate_map=column_predicate_map,
            X_num=X_num,
            y=y,
            version=1,
            mining_window_start=mining_window_start,
            mining_window_end=mining_window_end,
            mined_at=mined_at,
        )

        for df in (survivors_df, leaf_rules_df):
            if not df.empty:
                df["source_label"] = np.where(
                    df["confidence_attack"] > df["confidence_benign"],
                    "attack",
                    "benign",
                )
        mined_df = pd.concat(
            [survivors_df, leaf_rules_df], ignore_index=True, sort=False
        )
        symbolic = build_symbolic_feature_schema(
            df=mined_df,
            source_label="attack",
            schema_name="symbolic",
            schema_version=f"monitor_drift-{mined_at:%Y%m%dT%H%M%S}",
            predicates=predicate_alphabet,
        )
        schema = FeatureSchema(
            schema_name="base+symbolic",
            schema_version=symbolic.schema_version,
            base=base_schema.base,
            symbolic=symbolic,
        )
        cache_hit = False
        print(
            f"    [{cfg.mining_setting}] mined fresh ({len(symbolic.features)} "
            f"features, {len(dynamic_schema.single_predicates)} single predicates, "
            f"{len(dynamic_schema.compound_rules)} compound rules)"
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

    model = get_model_factory(cfg.model)()
    model.fit(X_train, y_train)
    proba_train = model.predict_proba(X_train)[:, 1]
    threshold = decide_threshold(
        y_train, proba_train, threshold_mode, calibrated_recall_target
    )

    return MonitorSourceWindowFit(
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
        dynamic_schema=dynamic_schema,
        test_groups_raw=window_rows[local_train_end:],
    )


def _window_bounds_ts(raw_groups: list[AlertGroup]) -> tuple[datetime, datetime]:
    start = datetime.fromtimestamp(raw_groups[0].start_ts, tz=timezone.utc)
    end_ts = raw_groups[-1].end_ts or raw_groups[-1].start_ts
    end = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    return start, end


def _signal_rows_for_horizon(
    dynamic_schema: DynamicSchema,
    predicate_by_id: dict,
    rule_by_id: dict,
    base_row: dict,
    horizon_window_index: int,
    horizon_fraction: float,
    snapshot,
) -> list[dict]:
    rows: list[dict] = []
    for sig in snapshot.signal_1_results:
        pred = predicate_by_id[sig.predicate_id]
        rows.append(
            {
                **base_row,
                "horizon_window_index": horizon_window_index,
                "horizon_fraction": horizon_fraction,
                "signal_type": "signal_1",
                "identifier": sig.predicate_id,
                "field": pred.field,
                "operator": pred.operator,
                "value": pred.value,
                "predicate_type": pred.predicate_type,
                "direction": pred.direction,
                "conditions": None,
                "prediction": None,
                "mined_value": sig.p_expected,
                "observed_value": sig.p_observed,
                "metric_value": sig.psi,
                "elevated": sig.elevated,
                "significant": sig.significant,
                "n_observed": sig.n_observed,
                "n_matching": None,
            }
        )
    for sig in snapshot.signal_2_results:
        rule = rule_by_id[sig.rule_id]
        rows.append(
            {
                **base_row,
                "horizon_window_index": horizon_window_index,
                "horizon_fraction": horizon_fraction,
                "signal_type": "signal_2",
                "identifier": sig.rule_id,
                "field": None,
                "operator": None,
                "value": None,
                "predicate_type": None,
                "direction": None,
                "conditions": str(rule.conditions),
                "prediction": rule.prediction,
                "mined_value": sig.mined_confidence,
                "observed_value": sig.observed_confidence,
                "metric_value": sig.drift,
                "elevated": sig.elevated,
                "significant": None,
                "n_observed": None,
                "n_matching": sig.n_matching,
            }
        )
    return rows


def _run_one_monitor_config(
    cfg: ShortlistedConfig,
    config: MonitorDriftConfig,
    scenario: str,
    alert_groups: list[AlertGroup],
    n_total: int,
    base_schema: FeatureSchema,
    mining_settings_by_name: dict,
    mining_settings_path: Path,
) -> tuple[list[dict], list[dict]]:
    print(
        f"\n[{cfg.feature_set}/{cfg.mining_setting}/gran={cfg.granularity:g}] starting"
    )

    fit = fit_source_window_and_dynamic_schema(
        cfg=cfg,
        scenario=scenario,
        alert_groups=alert_groups,
        n_total=n_total,
        base_schema=base_schema,
        mining_settings_by_name=mining_settings_by_name,
        mining_settings_path=mining_settings_path,
        train_frac_within_window=config.train_frac_within_window,
        threshold_mode=config.threshold_mode,
        calibrated_recall_target=config.calibrated_recall_target,
    )
    if fit is None:
        return [], []

    state = (
        MonitorState(scenario_name=scenario, deployed_schema_version=1)
        if fit.dynamic_schema is not None
        else None
    )
    predicate_by_id = (
        {p.predicate_id: p for p in fit.dynamic_schema.single_predicates}
        if fit.dynamic_schema is not None
        else {}
    )
    rule_by_id = (
        {r.rule_id: r for r in fit.dynamic_schema.compound_rules}
        if fit.dynamic_schema is not None
        else {}
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
    signal_rows: list[dict] = []

    def _record_horizon(
        horizon_window_index: int,
        X_h: pd.DataFrame,
        y_h: np.ndarray,
        n_alert_groups_h: int,
        raw_groups_h: list[AlertGroup],
    ) -> None:
        horizon_fraction = (
            horizon_window_index / (fit.n_windows - 1) if fit.n_windows > 1 else 0.0
        )
        monitor_cols: dict = {
            "schema_version": None,
            "signal_1_elevated": None,
            "signal_2_elevated": None,
            "n_elevated": None,
            "trigger_remine": None,
            "action": None,
            "consecutive_signal_1_elevated": None,
            "consecutive_signal_2_elevated": None,
            "n_incoming_groups": None,
            "n_labeled_groups": None,
        }

        if fit.dynamic_schema is not None and raw_groups_h:
            window_start, window_end = _window_bounds_ts(raw_groups_h)
            snapshot = run_monitor_window(
                schema=fit.dynamic_schema,
                state=state,
                incoming_groups=raw_groups_h,
                window_start=window_start,
                window_end=window_end,
                consecutive_windows=config.monitor_consecutive_windows,
                min_samples_signal_2=config.monitor_min_samples_signal_2,
            )
            monitor_cols = {
                "schema_version": snapshot.schema_version,
                "signal_1_elevated": snapshot.signal_1_elevated,
                "signal_2_elevated": snapshot.signal_2_elevated,
                "n_elevated": snapshot.n_elevated,
                "trigger_remine": snapshot.trigger_remine,
                "action": snapshot.action,
                "consecutive_signal_1_elevated": state.consecutive_signal_1_elevated,
                "consecutive_signal_2_elevated": state.consecutive_signal_2_elevated,
                "n_incoming_groups": snapshot.n_incoming_groups,
                "n_labeled_groups": snapshot.n_labeled_groups,
            }
            signal_rows.extend(
                _signal_rows_for_horizon(
                    fit.dynamic_schema,
                    predicate_by_id,
                    rule_by_id,
                    base_row,
                    horizon_window_index,
                    horizon_fraction,
                    snapshot,
                )
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
                    **monitor_cols,
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
                **monitor_cols,
            }
        )

    # h=0: W_src's own held-out test split -- never seen by mining or fitting.
    _record_horizon(0, fit.X_test, fit.y_test, len(fit.X_test), fit.test_groups_raw)

    for k in range(1, fit.n_windows):
        t_start, t_end, _ = compute_window_bounds(n_total, fit.gran, k)
        target_rows = alert_groups[t_start:t_end]
        t_labels, t_mask = labels_and_mask(target_rows)
        encoded_tgt = encode_alert_groups_for_schema(target_rows, fit.schema)
        X_tgt = encoded_tgt.loc[t_mask].reset_index(drop=True)
        y_tgt = t_labels[t_mask].astype(int)
        _record_horizon(k, X_tgt, y_tgt, len(target_rows), target_rows)

    return horizon_rows, signal_rows


def run_monitor_drift_experiment(config: MonitorDriftConfig) -> Path:
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
    n_total = ctx.n_total
    base_schema = ctx.base_schema
    mining_settings_by_name = ctx.mining_settings_by_name
    mining_settings_path = ctx.mining_settings_path

    print("[4/4] Loading shortlist...")
    shortlist = load_shortlist(config.shortlist_path)
    print(f"  {len(shortlist)} shortlisted configs")

    horizon_rows: list[dict] = []
    signal_rows: list[dict] = []

    print(
        f"  Running {len(shortlist)} shortlisted configs with n_jobs={config.n_jobs}..."
    )
    with ThreadPoolExecutor(max_workers=config.n_jobs) as pool:
        future_to_cfg = {
            pool.submit(
                _run_one_monitor_config,
                cfg=cfg,
                config=config,
                scenario=scenario,
                alert_groups=alert_groups,
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
                cfg_horizon_rows, cfg_signal_rows = future.result()
                horizon_rows.extend(cfg_horizon_rows)
                signal_rows.extend(cfg_signal_rows)
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

    monitor_signals_df = pd.DataFrame(signal_rows)
    monitor_signals_path = out_dir / "monitor_signals.csv"
    monitor_signals_df.to_csv(monitor_signals_path, index=False)
    print(f"  Saved → {monitor_signals_path}")

    monitor_alarms_df = (
        monitor_signals_df[monitor_signals_df["elevated"] == True]  # noqa: E712
        if "elevated" in monitor_signals_df.columns
        else monitor_signals_df
    )
    monitor_alarms_path = out_dir / "monitor_alarms.csv"
    monitor_alarms_df.to_csv(monitor_alarms_path, index=False)
    print(f"  Saved → {monitor_alarms_path}")

    summary_lines = [
        f"Monitor Drift (Experiment 4) — {ts}",
        f"Scenario: {scenario}",
        f"Shortlist: {config.shortlist_path} ({len(shortlist)} configs)",
        f"Threshold mode: {config.threshold_mode}"
        + (
            f" (recall target={config.calibrated_recall_target})"
            if config.threshold_mode == "calibrated_recall"
            else ""
        ),
        f"Monitor: consecutive_windows={config.monitor_consecutive_windows}, "
        f"min_samples_signal_2={config.monitor_min_samples_signal_2}",
        f"Rows: {len(per_horizon_df)} (per_horizon), {len(monitor_signals_df)} "
        f"(monitor_signals), {len(monitor_alarms_df)} (monitor_alarms)",
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

"""
Monitor Attached: the reactive counterpart to monitor_drift.py's
observe-only walk (Experiment 4). See MonitorAttachedConfig's docstring
(schemas/experiments.py) for the thesis-facing experiment design; this
module docstring covers implementation decisions not re-derivable from
that alone.

Mirrors monitor_drift.py's h=0 setup exactly (mine+fit on W_src's train
split via fit_source_window_and_dynamic_schema, freeze, build Vk) so the
starting point is directly comparable to a monitor_drift.py /
temporal_decay.py run using the same source_split_mode/shortlist. Unlike
that experiment, every horizon's monitor decision is actually acted on:
later horizons are scored against whatever schema/model is *currently*
deployed, not the original frozen one.

Design decisions:

* Only feature_set == "symbolic" configs ever get a DynamicSchema from
  fit_source_window_and_dynamic_schema ("baseline" mines nothing;
  "cscas_full"/"cscas_full_symbolic" are explicitly rejected there). A
  monitor with no Vk to evaluate against has nothing to attach to, so any
  other config is skipped with a print, same as monitor_drift.py's
  implicit per-config skip pattern (no separate pre-filtering step).

* What an on-trigger update trains on: this horizon's own labeled rows --
  the same window whose monitor snapshot just triggered the action, never
  an accumulation of several horizons. This mirrors rolling_walk_forward.py
  (the always-retrain baseline)'s per-step "mine+fit on Wi, evaluate on
  Wi+1" design, so a monitor-triggered update is directly comparable to
  what an always-retrain step does, just gated by the monitor instead of
  unconditional. RETRAIN_ONLY calls fit_scored_model on the *current*
  schema's encoding of this horizon (schema/Vk untouched); REMINE_AND_RETRAIN
  calls monitor_drift.mine_symbolic_and_dynamic_schema fresh on this
  horizon's labeled rows (schema_version bumped) and fits on that new
  encoding instead.

* Every horizon is first SCORED under whatever schema/model was deployed
  *before* this horizon was observed (realistic decision timing: score
  incoming traffic under the current model, then decide whether to
  update; the update only affects horizon h+1 onward) -- the monitor then
  runs against that same horizon's raw groups, and only then, if
  triggered, is the update applied.

* Consecutive-elevation counters: REMINE_AND_RETRAIN always gets a brand
  new MonitorState (a new Vk deploys, and MonitorState is scoped to one
  deployed Vk's lifetime -- see monitor/state.py). RETRAIN_ONLY keeps the
  same Vk (nothing about Signal 1's mined expectations changed) but
  manually resets both consecutive counters to 0 immediately after acting
  -- otherwise, under sustained real covariate drift, the exact same
  still-elevated flags would trigger another retrain at literally every
  following horizon, drowning out "how many distinct update events did
  this cost" against the two baselines. This is a deliberate cooldown, not
  implied by monitor/triggers.py's classify_action alone.

* Trigger (config.trigger): "feedback" (default, since 2026-11-24) acts on
  analyst-feedback precision (thesis.monitor.feedback, logged by SignalLog
  as fb_precision_elevated): precision of the escalated, labelled groups
  significantly below the deployment's reference precision for
  config.feedback_consecutive_windows windows in a row ->
  config.feedback_action. After a redeploy the next window is the new
  reference (warm-up), so it can never fire there. "thesis" acts on the
  original Signal 1/Signal 2 table (monitor/triggers.py). Either way
  run_monitor_window still runs every window and its decision is logged as
  thesis_action, so both triggers stay comparable from one run.
  "combined" (2026-11-25) splits the two questions: pattern validity
  (thesis.monitor.pattern_validity, pv_<config.pattern_label_mode>_elevated:
  a mined pattern's precision or the patterns' coverage dropped) ->
  REMINE_AND_RETRAIN; otherwise feedback precision -> RETRAIN_ONLY. Each
  has its own streak of config.feedback_consecutive_windows windows;
  trigger_reason records which one acted.

* Signal thresholds: config.psi_threshold/cal_threshold are passed
  straight through to run_monitor_window at every horizon -- deploy the
  values selected by Drift Signal EDA's sweep (03_monitor_signal_drift.ipynb,
  Analysis 3), not the untuned defaults monitor_drift.py's own logged
  `elevated` column used.

System-operationality instrumentation (built in from the start, not a
separate pass -- see the "System Operationality" experiment specs this
module was extended for): every horizon's fast route (Encoder + Model,
grouping/preprocessing is already done upstream by the cached
alert_groups load, not timed here) is timed once as a single batched
encode+predict call over *all* of that horizon's incoming groups
(labeled or not -- production scores every alert, not just the ones that
later get a label), giving `fast_route_wall_s`/`fast_route_cpu_s` and the
escalated/suppressed funnel at both group and raw-alert (`n_alerts`)
granularity with zero extra model calls (the same proba array is reused
for both the funnel and the labeled-subset quality metrics). A batched
call gives a *mean* per-group cost, not a true per-instance latency
distribution, so a small fixed-size sample of individually-timed
single-alert-group calls is additionally taken every horizon
(config.latency_sample_n, split evenly across horizons) into
latency_sample.csv, for mean/p50/p95/p99 reporting. Every executed
(and skipped) retrain/remine event is timed and diffed
(thesis.monitor.schema_diff) into events.csv -- t_mining_s only applies to
REMINE_AND_RETRAIN, t_retrain_s/t_deploy_s apply to both action kinds.
Ungroupability (the CSCAS scenario is pregrouped -- see
grouping.group_alerts's CSCAS_PREGROUPED_METHOD) is out of scope: every
raw_groups_h row is a formed AlertGroup already, so G == n_alert_groups
always and there is no separate "ungroupable" bucket to track here.

Outputs: per_horizon_results.csv (one row per config x horizon -- decay
metrics, workload funnel, fast-route cost, monitor signal/action),
events.csv (one row per triggered horizon, executed or skipped, with
retrain/remine timing and schema churn), latency_sample.csv (one row per
individually-timed single-alert-group call), summary.txt, config.json. The
retrain-lag computation itself (which needs h*, the ground-truth decay
horizon from the *frozen* monitor_drift.py run of the same config) is left
to the notebook -- events.csv is designed to join against that run's
per_horizon_results.csv directly on CONFIG_COLS.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.mining.common.attribute_features import default_leaky_attribute_fields
from thesis.encoders.service import encode_alert_groups_for_schema
from thesis.experiments._shared import (
    decide_threshold,
    fit_scored_model,
    labels_and_mask,
    load_scenario_context,
    metrics_at_threshold,
    nan_metrics,
)
from thesis.pipeline.pipeline import compute_window_train_end
from thesis.system_eval.label_budget import (
    LabelLedger,
    budget_view,
    draw_budget,
    mask_labels,
)
from thesis.system_eval.monitor_drift import (
    _window_bounds_ts,
    fit_source_window_and_dynamic_schema,
    mine_symbolic_and_dynamic_schema,
)
from thesis.system_eval.temporal_decay import WindowScheme, _build_window_scheme
from thesis.metrics.shortlist import ShortlistedConfig, load_shortlist
from thesis.monitor.monitor import run_monitor_window
from thesis.monitor.signal_log import LOGGED_COLUMNS as LOGGED_SIGNAL_COLUMNS
from thesis.monitor.signal_log import SignalLog
from thesis.monitor.schema_diff import SchemaDiff, diff_dynamic_schemas, zero_diff
from thesis.monitor.state import MonitorState
from thesis.paths import ensure_artifact_dirs
from thesis.schemas.experiments import MonitorAttachedConfig
from thesis.schemas.features import FeatureSchema
from thesis.schemas.groups import AlertGroup

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS_DIR = _ROOT / "artifacts" / "experiments" / "monitor_attached"


def _diff_cols(diff: SchemaDiff) -> dict:
    return {
        "predicates_added": diff.predicates_added,
        "predicates_removed": diff.predicates_removed,
        "predicates_changed": diff.predicates_changed,
        "predicates_before": diff.predicates_before,
        "predicates_after": diff.predicates_after,
        "rules_added": diff.rules_added,
        "rules_removed": diff.rules_removed,
        "rules_changed": diff.rules_changed,
        "rules_before": diff.rules_before,
        "rules_after": diff.rules_after,
        "churn_frac": diff.churn_frac,
    }


_NO_TIMING = {
    "t_mining_s": None,
    "mining_cpu_s": None,
    "t_retrain_s": None,
    "retrain_cpu_s": None,
    "t_deploy_s": None,
}


def _sample_latency(
    raw_groups_h: list[AlertGroup],
    schema: FeatureSchema,
    model,
    n: int,
    rng: np.random.Generator,
) -> list[float]:
    """Wall-clock seconds for `n` individually-timed single-alert-group
    encode+predict calls, sampled without replacement from raw_groups_h.
    The main per-horizon fast_route_wall_s is one batched call and only
    gives a *mean* per-group cost -- this is what a real p50/p95/p99
    latency distribution needs instead."""
    if not raw_groups_h or n <= 0:
        return []
    idx = rng.choice(len(raw_groups_h), size=min(n, len(raw_groups_h)), replace=False)
    latencies = []
    for i in idx:
        t0 = time.perf_counter()
        encoded = encode_alert_groups_for_schema([raw_groups_h[int(i)]], schema)
        model.predict_proba(encoded)
        latencies.append(time.perf_counter() - t0)
    return latencies


def _run_one_config(
    cfg: ShortlistedConfig,
    config: MonitorAttachedConfig,
    scenario: str,
    alert_groups: list[AlertGroup],
    n_total: int,
    base_schema: FeatureSchema,
    mining_settings_by_name: dict,
    mining_settings_path: Path,
    scheme: WindowScheme,
    predictions_out: list | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """predictions_out, if given, receives one DataFrame of per-group
    predictions per scored horizon (config.save_predictions)."""
    print(
        f"\n[{cfg.feature_set}/{cfg.mining_setting}/gran={cfg.granularity:g}] "
        "starting (reactive)"
    )

    # Label budget (DeepCASE comparison only): the initial fit sees only the
    # B0 rows of W_src's train split; later horizons are still scored and
    # evaluated on the true alert_groups.
    budget = config.update_labels == "cumulative_budget"
    fit_groups = alert_groups
    b0_groups: list[AlertGroup] = []
    if budget or config.initial_budget_mode != "full":
        src_start, src_end = scheme.source_bounds(cfg.granularity)
        src_train_end = compute_window_train_end(
            src_start, src_end, config.train_frac_within_window
        )
        train_rows = alert_groups[src_start:src_train_end]
        keep = draw_budget(
            train_rows,
            config.initial_budget_mode,
            size=config.initial_budget_size,
            seed=config.random_seed,
            items_path=config.deepcase_items_path,
        )
        b0_groups = [g for g, k in zip(train_rows, keep) if k]
        fit_groups = list(alert_groups)
        fit_groups[src_start:src_train_end] = mask_labels(train_rows, keep)
        print(
            f"  [budget] B0 = {len(b0_groups)} labelled rows "
            f"({config.initial_budget_mode}) of {len(train_rows)} training rows; "
            f"updates: {config.update_labels}"
        )

    fit = fit_source_window_and_dynamic_schema(
        cfg=cfg,
        scenario=scenario,
        alert_groups=fit_groups,
        n_total=n_total,
        base_schema=base_schema,
        mining_settings_by_name=mining_settings_by_name,
        mining_settings_path=mining_settings_path,
        train_frac_within_window=config.train_frac_within_window,
        threshold_mode=config.threshold_mode,
        calibrated_recall_target=config.calibrated_recall_target,
        scheme=scheme,
        model_seed=config.model_seed,
    )
    if fit is None:
        return [], [], []
    if fit.dynamic_schema is None:
        print(
            "  [skip] no DynamicSchema -- monitor has nothing to attach to "
            "(feature_set must be 'symbolic')"
        )
        return [], [], []

    spec = mining_settings_by_name.get(cfg.mining_setting)
    rng = np.random.default_rng(config.random_seed)

    # Mutable "what's currently deployed" state -- updated in place by
    # _process_horizon via the closures below.
    current_schema = fit.schema
    current_model = fit.model
    current_threshold = fit.threshold
    dynamic_schema = fit.dynamic_schema
    schema_version = dynamic_schema.version
    state = MonitorState(scenario_name=scenario, deployed_schema_version=schema_version)
    feedback_streak = 0
    pattern_streak = 0

    # Every signal ever considered, logged per horizon and never acted on
    # (thesis.monitor.signal_log). Its baselines reset on every executed
    # redeploy: novelty to the rows the new deployment was fit on, the
    # noise-aware baseline to the next window (warm-up).
    heldout_scores = (
        fit.model.predict_proba(fit.X_test)[:, 1]
        if len(fit.X_test) and len(fit.X_test) == len(fit.test_groups_raw)
        else None
    )
    # The audit is monitoring work: by default ("auto") only the combined
    # trigger's pattern validity uses it, so frozen / fixed-schedule /
    # feedback runs neither pay for it nor train on it. config.audit "on" /
    # "off" force it either way (e.g. a fixed schedule with the audit labels).
    audit_on = config.audit == "on" or (
        config.audit == "auto" and config.trigger == "combined"
    )
    audit_rate = config.audit_rate if audit_on else 0.0
    ledger: LabelLedger | None = None
    heldout_view = None
    if budget:
        ledger = LabelLedger(audit_rate)
        ledger.add_initial(b0_groups)
        if heldout_scores is not None:
            heldout_view = budget_view(
                fit.test_groups_raw, heldout_scores >= fit.threshold, audit_rate
            )[0]
    signal_log = SignalLog(
        dynamic_schema,
        train_groups=fit.train_groups_raw,
        heldout_groups=heldout_view
        if heldout_view is not None
        else fit.test_groups_raw,
        heldout_scores=heldout_scores,
        threshold=fit.threshold,
        min_samples=config.monitor_min_samples_signal_2,
        train_X=fit.X_train,
        heldout_X=fit.X_test,
    )
    print(f"    [signals] {signal_log.baseline_note}")

    base_row = {
        "scenario": scenario,
        "feature_set": cfg.feature_set,
        "mining_setting": cfg.mining_setting,
        "granularity": fit.gran,
        "model": cfg.model,
        "n_windows": fit.n_windows,
        "threshold_mode": config.threshold_mode,
        "psi_threshold": config.psi_threshold,
        "cal_threshold": config.cal_threshold,
        "monitor_consecutive_windows": config.monitor_consecutive_windows,
        "trigger": config.trigger,
        "feedback_consecutive_windows": config.feedback_consecutive_windows,
        "feedback_action": config.feedback_action,
        "pattern_label_mode": config.pattern_label_mode,
        "update_labels": config.update_labels,
        "initial_budget_mode": config.initial_budget_mode,
        "initial_budget_n": len(b0_groups) if b0_groups else None,
        "audit_rate": audit_rate if budget else None,
    }
    n_per_horizon_latency = (
        max(1, config.latency_sample_n // max(fit.n_windows, 1))
        if config.latency_sample_n > 0
        else 0
    )

    horizon_rows: list[dict] = []
    event_rows: list[dict] = []
    latency_rows: list[dict] = []

    def _process_horizon(
        horizon_window_index: int, raw_groups_h: list[AlertGroup]
    ) -> None:
        nonlocal current_schema, current_model, current_threshold
        nonlocal dynamic_schema, schema_version, state, feedback_streak
        nonlocal pattern_streak

        horizon_fraction = (
            horizon_window_index / (fit.n_windows - 1) if fit.n_windows > 1 else 0.0
        )
        schema_version_active = schema_version

        if not raw_groups_h:
            horizon_rows.append(
                {
                    **base_row,
                    "horizon_window_index": horizon_window_index,
                    "horizon_fraction": horizon_fraction,
                    "is_source_window": horizon_window_index == 0,
                    "schema_version_active": schema_version_active,
                    "target_single_class": True,
                    "n_alert_groups": 0,
                    "n_alerts_in": 0,
                    "n_attack": 0,
                    "attack_rate_h": None,
                    **nan_metrics(),
                    "n_groups_escalated": None,
                    "n_groups_suppressed": None,
                    "n_alerts_escalated": None,
                    "n_alerts_suppressed": None,
                    "fast_route_wall_s": None,
                    "fast_route_cpu_s": None,
                    "signal_1_elevated": None,
                    "signal_2_elevated": None,
                    "thesis_action": None,
                    "action": None,
                    "trigger_remine": None,
                    **dict.fromkeys(LOGGED_SIGNAL_COLUMNS),
                }
            )
            return

        labels_h, mask_h = labels_and_mask(raw_groups_h)
        n_groups_h = len(raw_groups_h)
        n_alerts_in_h = int(sum(tx.n_alerts for tx in raw_groups_h))

        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()
        encoded_full = encode_alert_groups_for_schema(raw_groups_h, current_schema)
        proba_full = current_model.predict_proba(encoded_full)[:, 1]
        fast_route_wall_s = time.perf_counter() - t0_wall
        fast_route_cpu_s = time.process_time() - t0_cpu

        y_pred_full = (proba_full >= current_threshold).astype(int)
        alerts_arr = np.array([tx.n_alerts for tx in raw_groups_h])
        n_groups_escalated = int(y_pred_full.sum())
        n_groups_suppressed = n_groups_h - n_groups_escalated
        n_alerts_escalated = int(alerts_arr[y_pred_full == 1].sum())
        n_alerts_suppressed = int(alerts_arr[y_pred_full == 0].sum())

        y_h = labels_h[mask_h].astype(int)
        proba_h = proba_full[mask_h]
        if predictions_out is not None:
            predictions_out.append(
                pd.DataFrame(
                    {
                        "model": cfg.model,
                        "granularity": fit.gran,
                        "horizon_window_index": horizon_window_index,
                        "schema_version_active": schema_version_active,
                        "group_id": [str(tx.group_id) for tx in raw_groups_h],
                        "label": labels_h,
                        "score": proba_full,
                        "escalated": y_pred_full.astype(bool),
                    }
                )
            )

        # What the system knows about this window: everything (default), or
        # only the analyst's labels -- escalated + audit -- under the budget.
        if ledger is not None:
            view_h, ledger_cols = ledger.add_window(raw_groups_h, y_pred_full == 1)
            if horizon_window_index == 0 and heldout_view is not None:
                view_h = heldout_view  # same labels; keeps the reference identity
        else:
            view_h, ledger_cols = raw_groups_h, {}

        if len(y_h) == 0:
            target_single_class = True
            metrics_block = nan_metrics()
            n_attack_h = 0
            attack_rate_h = None
        else:
            target_single_class = len(np.unique(y_h)) < 2
            metrics_block = metrics_at_threshold(y_h, proba_h, current_threshold)
            n_attack_h = int(np.nansum(y_h))
            attack_rate_h = n_attack_h / len(y_h)

        horizon_rows.append(
            {
                **base_row,
                "horizon_window_index": horizon_window_index,
                "horizon_fraction": horizon_fraction,
                "is_source_window": horizon_window_index == 0,
                "schema_version_active": schema_version_active,
                "target_single_class": target_single_class,
                "n_alert_groups": n_groups_h,
                "n_alerts_in": n_alerts_in_h,
                "n_attack": n_attack_h,
                "attack_rate_h": attack_rate_h,
                **metrics_block,
                "n_groups_escalated": n_groups_escalated,
                "n_groups_suppressed": n_groups_suppressed,
                "n_alerts_escalated": n_alerts_escalated,
                "n_alerts_suppressed": n_alerts_suppressed,
                "fast_route_wall_s": fast_route_wall_s,
                "fast_route_cpu_s": fast_route_cpu_s,
                "signal_1_elevated": None,
                "signal_2_elevated": None,
                "thesis_action": None,
                "action": None,
                "trigger_remine": None,
                **ledger_cols,
                # Logged against the deployment active in this window,
                # before any redeploy this window may trigger.
                **signal_log.log_window(
                    dynamic_schema,
                    view_h,
                    proba_full,
                    current_threshold,
                    X=encoded_full,
                ),
            }
        )

        for lat_s in _sample_latency(
            raw_groups_h, current_schema, current_model, n_per_horizon_latency, rng
        ):
            latency_rows.append(
                {
                    **base_row,
                    "horizon_window_index": horizon_window_index,
                    "schema_version_active": schema_version_active,
                    "latency_s": lat_s,
                }
            )

        window_start, window_end = _window_bounds_ts(raw_groups_h)
        snapshot = run_monitor_window(
            schema=dynamic_schema,
            state=state,
            incoming_groups=view_h,
            window_start=window_start,
            window_end=window_end,
            consecutive_windows=config.monitor_consecutive_windows,
            min_samples_signal_2=config.monitor_min_samples_signal_2,
            psi_threshold=config.psi_threshold,
            cal_threshold=config.cal_threshold,
        )
        row = horizon_rows[-1]
        row["signal_1_elevated"] = snapshot.signal_1_elevated
        row["signal_2_elevated"] = snapshot.signal_2_elevated
        row["thesis_action"] = snapshot.action
        feedback_streak = feedback_streak + 1 if row.get("fb_precision_elevated") else 0
        pattern_streak = (
            pattern_streak + 1
            if row.get(f"pv_{config.pattern_label_mode}_elevated")
            else 0
        )
        n_needed = config.feedback_consecutive_windows
        reason = None
        if config.trigger == "feedback":
            trigger_remine = feedback_streak >= n_needed
            if trigger_remine:
                action, reason = config.feedback_action, "feedback"
            elif feedback_streak:
                action = "SOFT_ALERT"
            else:
                action = "NO_ACTION"
        elif config.trigger == "combined":
            trigger_remine = pattern_streak >= n_needed or feedback_streak >= n_needed
            if pattern_streak >= n_needed:
                action, reason = "REMINE_AND_RETRAIN", "pattern_validity"
            elif feedback_streak >= n_needed:
                action, reason = "RETRAIN_ONLY", "feedback"
            elif feedback_streak or pattern_streak:
                action = "SOFT_ALERT"
            else:
                action = "NO_ACTION"
        elif config.trigger == "schedule":
            trigger_remine = True
            action, reason = config.schedule_action, "schedule"
        elif config.trigger == "never":
            trigger_remine, action = False, "NO_ACTION"
        else:
            action, trigger_remine = snapshot.action, snapshot.trigger_remine
            reason = "thesis" if trigger_remine else None
        row["action"] = action
        row["trigger_reason"] = reason
        row["feedback_streak"] = feedback_streak
        row["pattern_streak"] = pattern_streak
        row["trigger_remine"] = trigger_remine

        if not trigger_remine:
            return

        if ledger is not None:
            # Cumulative pool: B0 + every window's analyst labels so far.
            train_rows_labeled, train_weights = ledger.training_set()
        else:
            train_rows_labeled = [tx for tx, keep in zip(raw_groups_h, mask_h) if keep]
            train_weights = None
        y_train = labels_and_mask(train_rows_labeled)[0].astype(int)
        n_train = len(train_rows_labeled)
        n_classes = len({tx.group_label for tx in train_rows_labeled})
        event_base = {
            **base_row,
            "horizon_window_index": horizon_window_index,
            "action": action,
            "trigger_reason": reason,
            "n_train_rows": n_train,
            "schema_version_before": schema_version,
        }

        if n_train == 0 or n_classes < 2:
            event_rows.append(
                {
                    **event_base,
                    "executed": False,
                    "reason": "insufficient labeled rows to retrain/remine",
                    "schema_version_after": schema_version,
                    **_NO_TIMING,
                    **_diff_cols(zero_diff(dynamic_schema)),
                }
            )
            return

        if action == "RETRAIN_ONLY":
            if ledger is None:
                X_train = encoded_full.loc[mask_h].reset_index(drop=True)
            else:
                X_train = encode_alert_groups_for_schema(
                    train_rows_labeled, current_schema
                ).reset_index(drop=True)
            t0 = time.perf_counter()
            tc0 = time.process_time()
            new_model = fit_scored_model(
                cfg.model,
                X_train,
                y_train,
                seed=config.model_seed,
                sample_weight=train_weights,
            )
            t_retrain_s = time.perf_counter() - t0
            retrain_cpu_s = time.process_time() - tc0

            if new_model is None:
                event_rows.append(
                    {
                        **event_base,
                        "executed": False,
                        "reason": f"{cfg.model} could not be fit",
                        "schema_version_after": schema_version,
                        "t_mining_s": None,
                        "mining_cpu_s": None,
                        "t_retrain_s": t_retrain_s,
                        "retrain_cpu_s": retrain_cpu_s,
                        "t_deploy_s": None,
                        **_diff_cols(zero_diff(dynamic_schema)),
                    }
                )
                return

            proba_train = new_model.predict_proba(X_train)[:, 1]
            new_threshold = decide_threshold(
                y_train,
                proba_train,
                config.threshold_mode,
                config.calibrated_recall_target,
                model=new_model,
            )
            diff = zero_diff(dynamic_schema)

            t0_deploy = time.perf_counter()
            current_model = new_model
            current_threshold = new_threshold
            state.consecutive_signal_1_elevated = 0
            state.consecutive_signal_2_elevated = 0
            t_deploy_s = time.perf_counter() - t0_deploy
            signal_log.redeploy(dynamic_schema, view_h)
            feedback_streak = 0
            pattern_streak = 0

            event_rows.append(
                {
                    **event_base,
                    "executed": True,
                    "reason": None,
                    "schema_version_after": schema_version,
                    "t_mining_s": None,
                    "mining_cpu_s": None,
                    "t_retrain_s": t_retrain_s,
                    "retrain_cpu_s": retrain_cpu_s,
                    "t_deploy_s": t_deploy_s,
                    **_diff_cols(diff),
                }
            )

        elif action == "REMINE_AND_RETRAIN":
            if spec is None:
                event_rows.append(
                    {
                        **event_base,
                        "executed": False,
                        "reason": "no mining_setting spec resolved for this config",
                        "schema_version_after": schema_version,
                        **_NO_TIMING,
                        **_diff_cols(zero_diff(dynamic_schema)),
                    }
                )
                return

            new_version = schema_version + 1
            t0 = time.perf_counter()
            tc0 = time.process_time()
            mined = mine_symbolic_and_dynamic_schema(
                train_rows_labeled=train_rows_labeled,
                spec=spec,
                base_schema=base_schema,
                version=new_version,
                schema_version_tag=(
                    f"monitor_attached-h{horizon_window_index}-"
                    f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
                ),
                exclude_fields=default_leaky_attribute_fields(scenario),
            )
            t_mining_s = time.perf_counter() - t0
            mining_cpu_s = time.process_time() - tc0

            encoded_new = encode_alert_groups_for_schema(
                train_rows_labeled, mined.schema
            ).reset_index(drop=True)
            t0 = time.perf_counter()
            tc0 = time.process_time()
            new_model = fit_scored_model(
                cfg.model,
                encoded_new,
                y_train,
                seed=config.model_seed,
                sample_weight=train_weights,
            )
            t_retrain_s = time.perf_counter() - t0
            retrain_cpu_s = time.process_time() - tc0

            if new_model is None:
                event_rows.append(
                    {
                        **event_base,
                        "executed": False,
                        "reason": f"{cfg.model} could not be fit on remined schema",
                        "schema_version_after": schema_version,
                        "t_mining_s": t_mining_s,
                        "mining_cpu_s": mining_cpu_s,
                        "t_retrain_s": t_retrain_s,
                        "retrain_cpu_s": retrain_cpu_s,
                        "t_deploy_s": None,
                        **_diff_cols(zero_diff(dynamic_schema)),
                    }
                )
                return

            proba_train = new_model.predict_proba(encoded_new)[:, 1]
            new_threshold = decide_threshold(
                y_train,
                proba_train,
                config.threshold_mode,
                config.calibrated_recall_target,
                model=new_model,
            )
            diff = diff_dynamic_schemas(dynamic_schema, mined.dynamic_schema)

            t0_deploy = time.perf_counter()
            current_schema = mined.schema
            dynamic_schema = mined.dynamic_schema
            current_model = new_model
            current_threshold = new_threshold
            schema_version = new_version
            state = MonitorState(
                scenario_name=scenario, deployed_schema_version=schema_version
            )
            t_deploy_s = time.perf_counter() - t0_deploy
            signal_log.redeploy(dynamic_schema, view_h)
            feedback_streak = 0
            pattern_streak = 0

            event_rows.append(
                {
                    **event_base,
                    "executed": True,
                    "reason": None,
                    "schema_version_after": schema_version,
                    "t_mining_s": t_mining_s,
                    "mining_cpu_s": mining_cpu_s,
                    "t_retrain_s": t_retrain_s,
                    "retrain_cpu_s": retrain_cpu_s,
                    "t_deploy_s": t_deploy_s,
                    **_diff_cols(diff),
                }
            )

    # h=0: W_src's own held-out test split -- never seen by mining or fitting.
    _process_horizon(0, fit.test_groups_raw)

    for k in range(1, fit.n_windows):
        t_start, t_end = scheme.target_bounds(fit.gran, k)
        _process_horizon(k, alert_groups[t_start:t_end])

    return horizon_rows, event_rows, latency_rows


def run_monitor_attached_experiment(config: MonitorAttachedConfig) -> Path:
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

    scheme = _build_window_scheme(config, alert_groups, n_total)

    horizon_rows: list[dict] = []
    event_rows: list[dict] = []
    latency_rows: list[dict] = []
    predictions: list[pd.DataFrame] | None = [] if config.save_predictions else None

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
                n_total=n_total,
                base_schema=base_schema,
                mining_settings_by_name=mining_settings_by_name,
                mining_settings_path=mining_settings_path,
                scheme=scheme,
                predictions_out=predictions,
            ): cfg
            for cfg in shortlist
        }
        for future in as_completed(future_to_cfg):
            cfg = future_to_cfg[future]
            try:
                cfg_horizon_rows, cfg_event_rows, cfg_latency_rows = future.result()
                horizon_rows.extend(cfg_horizon_rows)
                event_rows.extend(cfg_event_rows)
                latency_rows.extend(cfg_latency_rows)
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

    events_df = pd.DataFrame(event_rows)
    events_path = out_dir / "events.csv"
    events_df.to_csv(events_path, index=False)
    print(f"  Saved → {events_path}")

    latency_df = pd.DataFrame(latency_rows)
    latency_path = out_dir / "latency_sample.csv"
    latency_df.to_csv(latency_path, index=False)
    print(f"  Saved → {latency_path}")

    if predictions:
        predictions_path = out_dir / "predictions.parquet"
        pd.concat(predictions, ignore_index=True).to_parquet(
            predictions_path, index=False
        )
        print(f"  Saved → {predictions_path}")

    n_executed = (
        int(events_df["executed"].sum()) if "executed" in events_df.columns else 0
    )
    summary_lines = [
        f"Monitor Attached — {ts}",
        f"Scenario: {scenario}",
        f"Shortlist: {config.shortlist_path} ({len(shortlist)} configs)",
        f"Threshold mode: {config.threshold_mode}"
        + (
            f" (recall target={config.calibrated_recall_target})"
            if config.threshold_mode == "calibrated_recall"
            else ""
        ),
        f"Monitor: psi_threshold={config.psi_threshold}, cal_threshold={config.cal_threshold}, "
        f"consecutive_windows={config.monitor_consecutive_windows}, "
        f"min_samples_signal_2={config.monitor_min_samples_signal_2}",
        f"Trigger: {config.trigger}"
        + (
            f" (feedback precision, {config.feedback_consecutive_windows} window(s) "
            f"-> {config.feedback_action})"
            if config.trigger == "feedback"
            else f" (every window -> {config.schedule_action})"
            if config.trigger == "schedule"
            else " (frozen)"
            if config.trigger == "never"
            else (
                f" (pattern validity [{config.pattern_label_mode}] -> "
                "REMINE_AND_RETRAIN, else feedback precision -> RETRAIN_ONLY; "
                f"{config.feedback_consecutive_windows} window(s))"
                if config.trigger == "combined"
                else " (Signal 1/2 table)"
            )
        ),
        f"Labels: updates={config.update_labels}, B0={config.initial_budget_mode}"
        + (
            f" (size {config.initial_budget_size})"
            if config.initial_budget_mode == "stratified"
            else ""
        )
        + (
            f", audit={config.audit} (rate {config.audit_rate})"
            if config.update_labels == "cumulative_budget"
            else ""
        ),
        f"Rows: {len(per_horizon_df)} (per_horizon), {len(events_df)} (events, "
        f"{n_executed} executed), {len(latency_df)} (latency_sample)",
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

"""
In-Window Baseline (Screening Sweep).

Purpose: determine which (mining_setting x granularity) configurations
produce attribute-mined features that are discriminative between
benign/attack at all, in the simplest possible setting -- train and test are
always drawn from the *same* chronological window, so this experiment never
tests cross-window generalization. It is a cheap filter pass meant to cut a
full mining grid down to a shortlist before more expensive downstream
experiments, not a final result.

For each granularity g, the full (chronologically sorted) alert_group
timeline is carved into windows W1..Wn(g) (n(g) depends on g -- finer
granularity means more, smaller windows; see pipeline.compute_window_bounds).
Each window has its own internal 70/30 train/test split
(pipeline.compute_window_train_end). For every (mining_setting, granularity,
window):

  a. Mine an attribute schema on the window's train split only
     (mining.window_schema_cache.get_or_mine_window_attribute_schema --
     cached on disk, so re-running the sweep or adding more screening models
     reuses the same mined schema instead of re-mining).
  b. Encode the window's train and test splits under that schema.
  c. Train each screening model (default: LogReg) on the train split.
  d. Evaluate on the test split.

A no-symbolic baseline pass (LogReg only, per window) is recorded alongside
the symbolic passes for direct comparison.

Screening models deliberately skip permutation/SHAP importance computation
(train_eval_holdout(..., compute_importances=False)) and are not persisted
as model artifacts -- this experiment is called many times over
(mining_setting x granularity x window x model), and importances/artifacts
are not part of what this stage measures (see "Outputs to record" below).

Outputs (per config, per window, per model): mining_setting, granularity,
window_id, model, auc, f1 (both recorded -- the metric is fixed by this
column set before the sweep runs, not chosen after the fact), tp/fp/tn/fn,
plus window context (n_alert_groups, n_attack, n_train, n_test,
win_start_frac, win_end_frac) so per-window variance -- not just the
aggregate -- is visible.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.config import load_mining_settings
from thesis.configs import dataset_for_scenario, load_base_features
from thesis.encoders.service import encode_alert_groups_for_schema
from thesis.features.persistence import load_symbolic_feature_schema
from thesis.mining.window_schema_cache import get_or_mine_window_attribute_schema
from thesis.paths import ensure_artifact_dirs
from thesis.pipeline.pipeline import (
    compute_window_bounds,
    compute_window_train_end,
    ensure_feature_manifest,
    ingest_ait_scenario,
    ingest_cscas_scenario,
    load_or_build_alert_groups,
)
from thesis.schemas.experiments import ScreeningSweepConfig
from thesis.schemas.features import BaseFeatureSchema, FeatureSchema
from thesis.schemas.groups import AlertGroup
from thesis.training.model_factory import get_model_factory
from thesis.training.train import train_eval_holdout

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS_DIR = _ROOT / "artifacts" / "experiments" / "screening_sweep"

_LABEL_MAP = {"benign": 0.0, "attack": 1.0}


def _select_window_indices(n_windows: int, windows_per_gran: int | None) -> list[int]:
    """All windows by default. If windows_per_gran is set and smaller than
    n_windows, fall back to that many evenly-spaced window indices spanning
    [0, n_windows - 1] (documented sampling rule for when the full sweep is
    computationally infeasible -- see module docstring)."""
    if windows_per_gran is None or windows_per_gran >= n_windows:
        return list(range(n_windows))
    positions = np.linspace(0, n_windows - 1, windows_per_gran)
    return sorted({int(round(p)) for p in positions})


def _labels_and_mask(window_rows: list[AlertGroup]) -> tuple[np.ndarray, np.ndarray]:
    """Per-row 0/1 label and a mask of which rows carry a usable label
    (drops unlabelled/mixed alert_groups, mirroring baseline.py/symbolic.py)."""
    labels = np.array(
        [_LABEL_MAP.get(t.group_label, np.nan) for t in window_rows], dtype=float
    )
    return labels, ~np.isnan(labels)


def _mask_and_split(
    X: pd.DataFrame, labels: np.ndarray, mask: np.ndarray, local_train_end: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Drop unlabelled rows, then split at the window's train/test boundary
    -- expressed as a *local* row count within the window, adjusted for any
    rows dropped before it, so the split still lands at the same temporal
    point regardless of how many rows the mask removed."""
    y = labels[mask].astype(int)
    X_masked = X.loc[mask].reset_index(drop=True)
    train_end = int(mask[:local_train_end].sum())
    return (
        X_masked.iloc[:train_end],
        X_masked.iloc[train_end:],
        y[:train_end],
        y[train_end:],
    )


def _eval_row(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    schema: FeatureSchema,
    model_name: str,
) -> dict:
    result = train_eval_holdout(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        schema=schema,
        model_factory=get_model_factory(model_name),
        compute_importances=False,
    )
    return {
        "model": model_name,
        "n_features": len(result["feature_names"]),
        "auc": result["auc"],
        "f1": result["f1"],
        "accuracy": result["accuracy"],
        "precision": result["precision"],
        "recall": result["recall"],
        "tp": result["tp"],
        "fp": result["fp"],
        "tn": result["tn"],
        "fn": result["fn"],
        "single_class_split": result["single_class_split"],
    }


def _run_one_window(
    scenario: str,
    alert_groups: list[AlertGroup],
    alert_groups_path: Path,
    n_total: int,
    gran: float,
    win_idx: int,
    base_schema: FeatureSchema,
    mining_settings: list,
    config: ScreeningSweepConfig,
) -> list[dict]:
    """Baseline pass + every mining setting's symbolic pass for one
    (granularity, window) -- the unit of work run_screening_sweep_experiment
    parallelizes across, since different windows never share state (each
    mines/encodes/fits/evaluates entirely on its own window's rows)."""
    win_start, win_end, _ = compute_window_bounds(n_total, gran, win_idx)
    win_train_end = compute_window_train_end(
        win_start, win_end, config.train_frac_within_window
    )
    local_train_end = win_train_end - win_start
    window_rows = alert_groups[win_start:win_end]
    labels, mask = _labels_and_mask(window_rows)
    n_attack = int(np.nansum(labels))

    win_meta = {
        "scenario": scenario,
        "granularity": gran,
        "window_id": win_idx,
        "win_start_frac": win_start / n_total,
        "win_end_frac": win_end / n_total,
        "n_alert_groups": len(window_rows),
        "n_attack": n_attack,
        "n_train": local_train_end,
        "n_test": len(window_rows) - local_train_end,
    }
    print(
        f"  [win {win_idx}] [{win_meta['win_start_frac']:.0%},"
        f"{win_meta['win_end_frac']:.0%})  n={len(window_rows)} "
        f"attack={n_attack} train={win_meta['n_train']} test={win_meta['n_test']}"
    )

    rows: list[dict] = []

    # ---- Baseline pass (no symbolic features) ----
    try:
        baseline_encoded = encode_alert_groups_for_schema(window_rows, base_schema)
        X_train, X_test, y_train, y_test = _mask_and_split(
            baseline_encoded, labels, mask, local_train_end
        )
        for model_name in config.baseline_models:
            row = _eval_row(X_train, X_test, y_train, y_test, base_schema, model_name)
            rows.append(
                {
                    **win_meta,
                    "feature_set": "baseline",
                    "mining_setting": None,
                    **row,
                }
            )
    except Exception as exc:
        print(f"    [warn] baseline pass failed: {exc}")
        traceback.print_exc()

    # ---- Symbolic pass, per mining setting ----
    for spec in mining_settings:
        try:
            schema_result = get_or_mine_window_attribute_schema(
                scenario=scenario,
                alert_groups=alert_groups,
                alert_groups_path=alert_groups_path,
                gran=gran,
                win_idx=win_idx,
                attribute_mining_config=spec.to_attribute_mining_config(),
                train_frac=config.train_frac_within_window,
                force=config.force_remine,
            )
            symbolic = load_symbolic_feature_schema(schema_result.schema_path)
            schema = FeatureSchema(
                schema_name="base+symbolic",
                schema_version=symbolic.schema_version,
                base=base_schema.base,
                symbolic=symbolic,
            )
            encoded = encode_alert_groups_for_schema(window_rows, schema)
            X_train, X_test, y_train, y_test = _mask_and_split(
                encoded, labels, mask, local_train_end
            )
            print(
                f"    [{spec.name}] {'cache hit' if schema_result.cache_hit else 'mined fresh'} "
                f"({len(symbolic.features)} features) → {schema_result.schema_path.name}"
            )
            for model_name in config.models:
                row = _eval_row(X_train, X_test, y_train, y_test, schema, model_name)
                rows.append(
                    {
                        **win_meta,
                        "feature_set": "symbolic",
                        "mining_setting": spec.name,
                        "mining_cache_hit": schema_result.cache_hit,
                        **row,
                    }
                )
        except Exception as exc:
            print(f"    [warn] setting '{spec.name}' failed: {exc}")
            traceback.print_exc()

    return rows


def run_screening_sweep_experiment(config: ScreeningSweepConfig) -> Path:
    ensure_artifact_dirs()

    scenario = config.scenario
    is_cscas = dataset_for_scenario(scenario) == "cscas"

    print(f"\n[ScreeningSweep] Scenario: '{scenario}'")

    print("[1/4] Ingesting scenario...")
    if is_cscas:
        ingest_cscas_scenario(cache_dir=config.cache_dir)
    else:
        ingest_ait_scenario(
            scenario,
            alerts_json_path=config.alerts_json_path,
            cache_dir=config.cache_dir,
            grouping=config.grouping,
        )

    print("[2/4] Checking feature manifest...")
    ensure_feature_manifest(scenario)

    print("[3/4] Building alert_groups from cache...")
    alert_groups = load_or_build_alert_groups(scenario, config.cache_dir)
    alert_groups_path = config.cache_dir / "alert_groups" / "alert_groups_raw.json"
    alert_groups.sort(key=lambda t: t.start_ts or "")
    n_total = len(alert_groups)
    print(f"  {n_total} alert_groups total")

    dataset = dataset_for_scenario(scenario)
    if dataset is None:
        raise ValueError(
            f"Scenario '{scenario}' is not listed under any dataset in scenarios.json."
        )
    base_schema = FeatureSchema(
        schema_name="base",
        schema_version="0.1.0",
        base=BaseFeatureSchema(load_base_features(dataset)),
        symbolic=None,
    )

    mining_settings_path = config.mining_settings_path
    if not mining_settings_path.is_absolute():
        mining_settings_path = _ROOT / mining_settings_path
    mining_settings = load_mining_settings(mining_settings_path)
    print(f"  Mining settings: {[s.name for s in mining_settings]}")

    print("[4/4] Running sweep...")

    # (granularity, window) pairs are independent -- flatten the two nested
    # loops into one task list and run it on a thread pool (see
    # ScreeningSweepConfig.n_jobs). Threads, not processes, for the same
    # reason as temporal_decay.py: the dominant per-window costs (mining's
    # contrast-set/tree-fit numpy work, BLAS ops inside each LogReg fit,
    # vectorized pandas/numpy encoding) release the GIL, and threads avoid
    # re-pickling the multi-million-row alert_groups list per worker under
    # macOS's spawn-based multiprocessing.
    window_tasks: list[tuple[float, int]] = []
    for gran in config.granularities:
        _, _, n_windows = compute_window_bounds(n_total, gran, 0)
        win_indices = _select_window_indices(n_windows, config.windows_per_gran)
        print(
            f"[gran={gran:.2f}] {n_windows} windows total, "
            f"evaluating {len(win_indices)}: {win_indices}"
        )
        window_tasks.extend((gran, win_idx) for win_idx in win_indices)

    print(
        f"  Running {len(window_tasks)} (granularity, window) tasks with n_jobs={config.n_jobs}..."
    )
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=config.n_jobs) as pool:
        future_to_task = {
            pool.submit(
                _run_one_window,
                scenario=scenario,
                alert_groups=alert_groups,
                alert_groups_path=alert_groups_path,
                n_total=n_total,
                gran=gran,
                win_idx=win_idx,
                base_schema=base_schema,
                mining_settings=mining_settings,
                config=config,
            ): (gran, win_idx)
            for gran, win_idx in window_tasks
        }
        for future in as_completed(future_to_task):
            gran, win_idx = future_to_task[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                print(f"  [warn] gran={gran:.2f} win={win_idx} failed: {exc}")
                traceback.print_exc()

    results_dir = (
        config.results_dir
        if config.results_dir is not None
        else _EXPERIMENTS_DIR / scenario
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    per_window_df = pd.DataFrame(rows)
    per_window_path = out_dir / "per_window_results.csv"
    per_window_df.to_csv(per_window_path, index=False)
    print(f"\n  Saved → {per_window_path}")

    if not per_window_df.empty:
        aggregate_df = (
            per_window_df.groupby(
                ["feature_set", "mining_setting", "granularity", "model"], dropna=False
            )[["auc", "f1"]]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        aggregate_df.columns = [
            "_".join(c).rstrip("_") for c in aggregate_df.columns.to_flat_index()
        ]
    else:
        aggregate_df = pd.DataFrame()
    aggregate_path = out_dir / "aggregate_results.csv"
    aggregate_df.to_csv(aggregate_path, index=False)
    print(f"  Saved → {aggregate_path}")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(
        f"Screening Sweep — {ts}\n"
        f"Scenario: {scenario}\n"
        f"Granularities: {config.granularities}\n"
        f"Mining settings: {[s.name for s in mining_settings]}\n"
        f"Models: {config.models}  Baseline models: {config.baseline_models}\n"
        f"Rows: {len(per_window_df)}\n\n"
        f"{aggregate_df.to_string(index=False) if not aggregate_df.empty else '(no data)'}\n"
    )
    print(f"  Saved → {summary_path}")

    (out_dir / "config.json").write_text(
        pd.Series({**asdict(config), "mining_settings_path": str(mining_settings_path)})
        .apply(str)
        .to_json(indent=2)
    )

    return out_dir

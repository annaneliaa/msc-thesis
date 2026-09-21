"""
Experiment 2: Temporal Generalization (Rolling-Horizon Decay).

Purpose: for a (feature_set, mining_setting, granularity, model) config from
the parameter grid (configs/screening_mining_settings.yaml x granularities x
models -- no separate screening/shortlist step), test whether a schema+model
trained on the *first* chronological window still discriminates well on
future windows, how AUC/F1/etc and FPR decay as temporal distance increases,
and how per-feature importances drift alongside that decay.

feature_set is one of:
  * "baseline"  -- the deployment-realistic reduced base columns only
    (encoders.baseline), no symbolic features.
  * "symbolic"  -- base + a schema mined on W_src's train split.
  * "cscas_full" -- the CSCAS paper's own full feature set
    (encoders.cscas_full: base + SCAS + Similarity + SignatureIDSimilarity
    + 33 attr-similarity columns). SCAS and the offline *Similarity scores
    are not computable for a fresh alert, so this is a non-deployable
    *reference ceiling* -- "does the frozen model decay even with the
    paper's full oracle features?" -- not a candidate schema. SCAS is
    dropped for one-class models (it is itself an anomaly score). Added by
    the runner's --cscas-full flag; carries no mining_setting.
  * "cscas_full_symbolic" -- the union of "cscas_full" and "symbolic": the
    full CSCAS columns plus a schema mined on W_src. The shared base
    columns are encoded once, not doubled. Carries a mining_setting like
    "symbolic"; added by the runner's --cscas-full-symbolic flag.

Models: supervised classifiers (logreg, xgboost, ...) fit on the mixed W_src
train split, every one made class-imbalance-aware (class_weight="balanced"
/ scale_pos_weight); one-class anomaly detectors (iforest, ocsvm) fit
unsupervised on its benign rows and are then Platt-scaled against the
labels so they score like a classifier -- see
experiments._shared.fit_scored_model. Both kinds go through the identical
freeze/threshold/metric/SHAP/LIME path below.

threshold_mode="fixed" resolves per model to its own no-tuning operating
point: 0.5 for a (balanced) classifier, the detector's own contamination
cut for a one-class model (a flat 0.5 in Platt-probability space would
predict everything benign on this imbalance). "calibrated_recall" tunes the
threshold to a target recall on W_src instead.

The source window W_src depends on config.source_split_mode (see
WindowScheme):

  * "window0" (default): W_src is window 0. For a given granularity g the
    timeline is carved into n(g) windows exactly as in screening_sweep.py
    (pipeline.compute_window_bounds); the walk covers windows
    1..n_windows-1.
  * "baseline_split": W_src is every alert_group at or before
    config.source_split_time -- the CSCAS baseline's own chronological
    train/test boundary (baselines/cscas_base.py). The walk carves the
    post-split_time remainder (the baseline's test period) into windows at
    granularity g, so the h=1..n_windows-1 decay curve lines up one-to-one
    with the single aggregate score the baseline reports on that same test
    set. Symbolic schemas for this mode are mined via
    window_schema_cache.get_or_mine_slice_attribute_schema (explicit slice
    bounds) rather than the (gran, win_idx) windowed entry point.

W_src always has its own internal train/test split
(pipeline.compute_window_train_end, same 70/30 default as the screening
sweep):

  1. Mine a schema on window 0's *train* split only
     (mining.window_schema_cache.get_or_mine_window_attribute_schema --
     the same train-split-only mining screening_sweep uses, unlike the
     previous version of this experiment which mined on the full window).
  2. Fit the config's model on window 0's train split (fit_scored_model --
     supervised on the whole split, one-class on its benign rows + a frozen
     Platt scaler).
  3. Fix a decision threshold once -- the model's own operating point
     ("fixed": 0.5 for a classifier, the contamination cut for a one-class
     detector) or a calibrated-recall threshold from the train split's
     scores.
  4. Freeze schema, model, and threshold. Walk the horizon forward one
     window at a time, from h=0 (window 0's held-out *test* split) through
     h=n_windows-1 (the last window), scoring each window's alert_groups
     with the frozen schema/model/threshold. Every window is in bounds by
     construction (W_src is always the earliest window), so there is no
     boundary-skip bookkeeping to do here, unlike the multi-role design this
     replaced.
  5. At every horizon step, also compute SHAP and LIME signed importances
     (thesis.training.explain) for *every* schema feature on a sample of
     that window's rows -- same frozen model, same frozen background sample
     drawn once from window 0's train split -- so any change in the reported
     importances reflects the target window drifting, not the explainer's
     reference point moving. These are model-*output* attributions (which
     features move the score, and which way), not model-*performance*
     attributions. One-class models (iforest, ocsvm) get LIME only unless
     config.oneclass_shap is set -- their SHAP has no analytic explainer and
     the PermutationExplainer fallback dominates the run's cost.

Every per_horizon_results.csv row also carries alert-group *novelty*
columns (schema/model-independent, so one series covers every config at a
given granularity + source split): each horizon window's groups are keyed
at two grains -- the raw_items token-set (the unit symbolic mining
operates on) and the coarser (category, ruleset, proto) tuple -- and
scored against the set of keys seen anywhere in W_src's train span.
n_novel_items / frac_novel_items / n_new_item_types (and the _crp and
_*_attack variants) let the EDA notebook line the metric-decay rate up
against how much genuinely unseen traffic each window brings.

Outputs: per_horizon_results.csv (one row per config x horizon window,
including the h=0 held-out anchor), decay_summary.csv (score/FPR at h=0 vs
the last horizon actually run, and their difference), explanations.csv
(long format: one row per config x horizon x method[shap|lime] x feature --
every schema feature, so the EDA notebook's per-horizon importance heatmap
has no holes), lime_fidelity.csv (one row per config x horizon: LIME's own
local-surrogate R^2, averaged over that horizon's explained sample --
separate from explanations.csv since it's one number per horizon, not per
feature), summary.txt, config.json. Written under
artifacts/experiments/temporal_decay/<scenario>/<ts>/ and, for a default
run (no explicit results_dir), also copied verbatim to
results/sys-eval/temporal-decay/<scenario>/<ts>/.

fit_source_window and encode_target_window (below) are also the entry
points thesis.experiments.instance_explain uses for on-demand, single-
instance SHAP/LIME case studies (e.g. "explain this specific false
positive at horizon 5") without re-running the whole sweep.
"""

from __future__ import annotations

import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.encoders.cscas_full import CSCAS_FULL_FEATURES
from thesis.encoders.service import encode_alert_groups_for_schema
from thesis.experiments._shared import (
    CONFIG_COLS,
    METRIC_COLS,
    ONE_CLASS_MODELS,
    decide_threshold,
    fast_route_and_funnel,
    fit_scored_model,
    labels_and_mask,
    load_scenario_context,
    metrics_at_threshold,
    nan_metrics,
    sample_rows,
)
from thesis.features.persistence import load_symbolic_feature_schema
from thesis.metrics.shortlist import ShortlistedConfig, load_shortlist
from thesis.mining.window_schema_cache import (
    get_or_mine_slice_attribute_schema,
    get_or_mine_window_attribute_schema,
)
from thesis.paths import RESULTS_DIR, ensure_artifact_dirs
from thesis.pipeline.pipeline import compute_window_bounds, compute_window_train_end
from thesis.schemas.experiments import TemporalDecayConfig
from thesis.schemas.features import BaseFeatureSchema, FeatureSchema
from thesis.training.explain import (
    compute_lime_signed_importances,
    compute_shap_signed_importances,
)

_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS_DIR = _ROOT / "artifacts" / "experiments" / "temporal_decay"
# A default run's output dir is also copied here, verbatim, as the curated,
# human-facing home for temporal-decay results (kept out of the regenerable
# artifacts/ tree). Skipped when the caller passes an explicit results_dir.
_RESULTS_MIRROR_DIR = RESULTS_DIR / "sys-eval" / "temporal-decay"


@dataclass(slots=True)
class WindowScheme:
    """How the source window and the forward horizon windows are carved out
    of the full chronologically-sorted alert_groups list -- see
    TemporalDecayConfig.source_split_mode.

    "window0": W_src is window 0 of compute_window_bounds at the config's
    granularity; the walk covers windows 1..n_windows-1 of the whole
    timeline.

    "baseline_split": W_src is alert_groups[:split_idx] (every group at or
    before the baseline's split_time -- see baselines/cscas_base.py); the
    walk carves alert_groups[split_idx:] (the baseline's own test period)
    into windows at the config's granularity, indexed h=1..n_windows-1, so
    the decay curve lines up one-to-one with the single aggregate score the
    baseline reports on that same test set."""

    mode: str
    n_total: int
    split_idx: int | None = None

    def n_windows(self, gran: float) -> int:
        if self.mode == "baseline_split":
            n_fwd = compute_window_bounds(self.n_total - self.split_idx, gran, 0)[2]
            return n_fwd + 1
        return compute_window_bounds(self.n_total, gran, 0)[2]

    def source_bounds(self, gran: float) -> tuple[int, int]:
        if self.mode == "baseline_split":
            return 0, self.split_idx
        start, end, _ = compute_window_bounds(self.n_total, gran, 0)
        return start, end

    def target_bounds(self, gran: float, horizon: int) -> tuple[int, int]:
        """Absolute [start, end) row bounds of horizon window `horizon` (>= 1)."""
        if self.mode == "baseline_split":
            start, end, _ = compute_window_bounds(
                self.n_total - self.split_idx, gran, horizon - 1
            )
            return start + self.split_idx, end + self.split_idx
        start, end, _ = compute_window_bounds(self.n_total, gran, horizon)
        return start, end


def _resolve_baseline_split_idx(alert_groups: list, split_time_iso: str) -> int:
    """Count of alert_groups whose start_ts is at or before `split_time_iso`
    -- the index at which the baseline's chronological train/test boundary
    falls, matching cscas_base.py's `df["Timestamp"] <= split_time` train
    mask. alert_groups is assumed sorted ascending by start_ts (epoch
    seconds), which load_scenario_context guarantees."""
    cutoff = int(pd.Timestamp(split_time_iso).timestamp())
    lo, hi = 0, len(alert_groups)
    while lo < hi:
        mid = (lo + hi) // 2
        if (alert_groups[mid].start_ts or 0) <= cutoff:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _build_window_scheme(
    config: TemporalDecayConfig, alert_groups: list, n_total: int
) -> WindowScheme:
    if config.source_split_mode != "baseline_split":
        return WindowScheme("window0", n_total)

    split_idx = _resolve_baseline_split_idx(alert_groups, config.source_split_time)
    if not 0 < split_idx < n_total:
        raise ValueError(
            f"baseline split_time {config.source_split_time!r} puts the "
            f"train/test boundary at index {split_idx} of {n_total} -- expected "
            "it strictly inside the timeline."
        )
    print(
        f"  [source=baseline_split] split_time={config.source_split_time} → "
        f"W_src = alert_groups[:{split_idx}] ({split_idx / n_total:.1%} of "
        f"{n_total}); walk carves the remaining {n_total - split_idx}"
    )
    return WindowScheme("baseline_split", n_total, split_idx)


# feature_set values whose schema is base + a mined symbolic layer (so they
# resolve a mining_setting and go through the mining path); the base half is
# the reduced columns for "symbolic" and the full CSCAS columns for
# "cscas_full_symbolic".
_SYMBOLIC_FEATURE_SETS = frozenset({"symbolic", "cscas_full_symbolic"})


def _cscas_full_base(model_name: str) -> BaseFeatureSchema:
    """The CSCAS paper's full column list (encoders.cscas_full) as a
    BaseFeatureSchema. `scas` -- CSCAS's own precomputed outlier/inlier
    score -- is dropped for one-class models, since feeding an anomaly score
    into an anomaly detector is circular; it's kept for supervised
    classifiers, matching the paper's own protocol."""
    feats = [
        f
        for f in CSCAS_FULL_FEATURES
        if not (f == "scas" and model_name in ONE_CLASS_MODELS)
    ]
    return BaseFeatureSchema(feats, kind="cscas_full")


def _cscas_full_schema(model_name: str) -> FeatureSchema:
    """The CSCAS paper's full feature set as a base-only FeatureSchema."""
    return FeatureSchema(
        schema_name="cscas_full",
        schema_version="0.1.0",
        base=_cscas_full_base(model_name),
        symbolic=None,
    )


# --- Per-horizon alert-group novelty ------------------------------------
# "Type" of an alert group at two grains, both schema/model-independent so
# one novelty series covers every config at a given (granularity, source
# split): the raw_items token-set (the unit symbolic mining actually
# operates on -- a token-set unseen in training is a pattern the frozen
# schema never had a chance to encode) and the coarser (category, ruleset,
# proto) tuple. A group counts as "novel" at a horizon if its key was not
# present anywhere in W_src's train span.

_NOVELTY_COUNT_COLS = (
    "n_groups_win",
    "n_attack_win",
    "n_novel_items",
    "n_new_item_types",
    "n_novel_items_attack",
    "n_novel_crp",
    "n_new_crp_types",
    "n_novel_crp_attack",
)
_NOVELTY_FRAC_COLS = (
    "frac_novel_items",
    "frac_novel_items_attack",
    "frac_novel_crp",
    "frac_novel_crp_attack",
)


def _items_key(g) -> frozenset:
    return frozenset(g.raw_items or ())


def _crp_key(g) -> tuple:
    return (g.category, g.ruleset, g.proto)


def _group_type_keys(rows: list) -> tuple[set, set]:
    """(raw_items token-set keys, (category, ruleset, proto) keys) over `rows`."""
    return {_items_key(g) for g in rows}, {_crp_key(g) for g in rows}


def _novelty_metrics(rows: list, train_items: set, train_crp: set) -> dict:
    """Novelty of `rows` (one horizon window, unmasked) against W_src's
    train-span type keys -- group-level counts/fractions and the number of
    distinct new types. Attack fractions are over that window's own attack
    count (nan when it has none)."""
    n = len(rows)
    if n == 0:
        return {
            **{c: 0 for c in _NOVELTY_COUNT_COLS},
            **{c: np.nan for c in _NOVELTY_FRAC_COLS},
        }
    labels, _ = labels_and_mask(rows)
    is_atk = labels == 1
    n_atk = int(is_atk.sum())

    item_keys = [_items_key(g) for g in rows]
    crp_keys = [_crp_key(g) for g in rows]
    nov_i = np.array([k not in train_items for k in item_keys])
    nov_c = np.array([k not in train_crp for k in crp_keys])

    return {
        "n_groups_win": n,
        "n_attack_win": n_atk,
        "n_novel_items": int(nov_i.sum()),
        "frac_novel_items": float(nov_i.mean()),
        "n_new_item_types": len(set(item_keys) - train_items),
        "n_novel_items_attack": int((nov_i & is_atk).sum()),
        "frac_novel_items_attack": (
            float((nov_i & is_atk).sum() / n_atk) if n_atk else np.nan
        ),
        "n_novel_crp": int(nov_c.sum()),
        "frac_novel_crp": float(nov_c.mean()),
        "n_new_crp_types": len(set(crp_keys) - train_crp),
        "n_novel_crp_attack": int((nov_c & is_atk).sum()),
        "frac_novel_crp_attack": (
            float((nov_c & is_atk).sum() / n_atk) if n_atk else np.nan
        ),
    }


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
    scheme: WindowScheme
    # W_src train-span type keys + the raw (unmasked) h=0 held-out rows, for
    # the per-horizon novelty columns. Defaulted so callers that build a
    # SourceWindowFit directly (older tests) don't have to supply them.
    train_item_keys: set = field(default_factory=set)
    train_crp_keys: set = field(default_factory=set)
    h0_rows: list = field(default_factory=list)
    # One-time setup cost (system-operationality instrumentation): the
    # "never-retrain" anchor pays this exactly once, at h=0, then nothing
    # ever again -- the opposite extreme from rolling_walk_forward.py's
    # every-step mining_wall_s/fit_wall_s. mining_wall_s/mining_cpu_s are
    # None for feature_set in {"baseline", "cscas_full"} (no mining call).
    mining_wall_s: float | None = None
    mining_cpu_s: float | None = None
    fit_wall_s: float = 0.0
    fit_cpu_s: float = 0.0


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
    scheme: WindowScheme | None = None,
) -> SourceWindowFit | None:
    """Mine (for `cfg.feature_set` in {"symbolic", "cscas_full_symbolic"}) on
    the source window's train split, fit `cfg.model` on that same train
    split, and decide a frozen
    threshold from its own scores. `scheme` picks what the source window is
    (default: window 0 at the config's granularity -- see WindowScheme).
    Returns None (with a warning printed, never raises) if the mining
    setting can't be resolved or the train split turns out to be
    single-class -- both non-fatal, "this config can't run" conditions the
    caller is expected to skip past."""
    gran = cfg.granularity
    scheme = scheme or WindowScheme("window0", n_total)
    n_windows = scheme.n_windows(gran)

    spec = None
    if cfg.feature_set in _SYMBOLIC_FEATURE_SETS:
        spec = mining_settings_by_name.get(cfg.mining_setting)
        if spec is None:
            print(
                f"  [warn] mining_setting '{cfg.mining_setting}' not found in "
                f"{mining_settings_path} -- skipping this config"
            )
            return None

    win_start, win_end = scheme.source_bounds(gran)
    win_train_end = compute_window_train_end(
        win_start, win_end, train_frac_within_window
    )
    local_train_end = win_train_end - win_start
    window_rows = alert_groups[win_start:win_end]
    labels, mask = labels_and_mask(window_rows)
    n_attack_src = int(np.nansum(labels))

    print(
        f"  [W_src={scheme.mode}] rows[{win_start}:{win_end}] n={len(window_rows)} "
        f"attack={n_attack_src} train_end(local)={local_train_end}"
    )

    cache_hit = None
    mining_wall_s: float | None = None
    mining_cpu_s: float | None = None
    if cfg.feature_set == "baseline":
        schema = base_schema
    elif cfg.feature_set == "cscas_full":
        schema = _cscas_full_schema(cfg.model)
    else:
        # "symbolic" or "cscas_full_symbolic": mine the symbolic layer on
        # W_src's train split, then attach it to the reduced base columns
        # ("symbolic") or the full CSCAS columns ("cscas_full_symbolic").
        # encode_alert_groups_for_schema drops any column the two halves
        # share, so the base features aren't doubled.
        _t0_mine = time.perf_counter()
        _tc0_mine = time.process_time()
        if scheme.mode == "baseline_split":
            tf_tag = f"{train_frac_within_window:.6f}".rstrip("0").rstrip(".")
            schema_result = get_or_mine_slice_attribute_schema(
                scenario=scenario,
                alert_groups=alert_groups,
                alert_groups_path=alert_groups_path,
                slice_start=win_start,
                slice_end=win_train_end,
                slice_tag=f"baseline_split_train{tf_tag}",
                attribute_mining_config=spec.to_attribute_mining_config(),
                force=force_remine,
            )
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
        cache_hit = schema_result.cache_hit
        mining_wall_s = time.perf_counter() - _t0_mine
        mining_cpu_s = time.process_time() - _tc0_mine
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

    encoded_src = encode_alert_groups_for_schema(window_rows, schema)
    y_masked = labels[mask].astype(int)
    X_masked = encoded_src.loc[mask].reset_index(drop=True)
    local_train_end_masked = int(mask[:local_train_end].sum())

    X_train = X_masked.iloc[:local_train_end_masked]
    X_test = X_masked.iloc[local_train_end_masked:].reset_index(drop=True)
    y_train = y_masked[:local_train_end_masked]
    y_test = y_masked[local_train_end_masked:]

    if len(np.unique(y_train)) < 2:
        print("    [warn] W_src train split is single-class -- skipping")
        return None

    # fit_scored_model returns something exposing predict_proba(X)[:, 1] as
    # attack likelihood, frozen along with everything else in this
    # experiment's "freeze schema, model, threshold" design:
    #  - supervised models ("logreg" etc. -- themselves scaled Pipelines, so
    #    the fitted scaler is frozen too) fit on the whole train split;
    #  - one-class anomaly models ("iforest", "ocsvm") fit unsupervised on
    #    the benign rows, then get a frozen 1-D Platt scaler over the labeled
    #    split so their anomaly score reads as a probability downstream.
    _t0_fit = time.perf_counter()
    _tc0_fit = time.process_time()
    model = fit_scored_model(cfg.model, X_train, y_train)
    if model is None:
        print("    [warn] W_src train split can't fit/calibrate this model -- skipping")
        return None
    fit_wall_s = time.perf_counter() - _t0_fit
    fit_cpu_s = time.process_time() - _tc0_fit
    proba_train = model.predict_proba(X_train)[:, 1]

    threshold = decide_threshold(
        y_train, proba_train, threshold_mode, calibrated_recall_target, model=model
    )

    # Novelty reference: type keys over W_src's *train* span (unmasked --
    # every group the schema/model was derived from, labelled or not), plus
    # the raw held-out rows that h=0 is scored on.
    train_item_keys, train_crp_keys = _group_type_keys(window_rows[:local_train_end])
    h0_rows = window_rows[local_train_end:]

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
        scheme=scheme,
        train_item_keys=train_item_keys,
        train_crp_keys=train_crp_keys,
        h0_rows=h0_rows,
        mining_wall_s=mining_wall_s,
        mining_cpu_s=mining_cpu_s,
        fit_wall_s=fit_wall_s,
        fit_cpu_s=fit_cpu_s,
    )


def encode_target_window(
    alert_groups: list,
    scheme: "WindowScheme | int",
    gran: float,
    win_idx: int,
    schema: FeatureSchema,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Encode horizon window `win_idx` (>= 1) under a (frozen) schema,
    dropping unlabelled rows. Returns (X, y, n_alert_groups_in_window) --
    the third value keeps the *unmasked* window size available for
    reporting even though X/y only cover labeled rows.

    `scheme` may be a WindowScheme or, for backward compatibility with
    callers that predate it (rolling_walk_forward.py), a bare `n_total`
    int -- treated as a "window0" scheme over that many rows."""
    if not isinstance(scheme, WindowScheme):
        scheme = WindowScheme("window0", int(scheme))
    t_start, t_end = scheme.target_bounds(gran, win_idx)
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
    drop the SHAP rows already computed, and vice versa.

    Every schema feature is recorded (top_n = full width), not just a top
    slice -- the per-horizon importance heatmap in the EDA notebook needs a
    value for every (feature, horizon) cell to avoid holes; it picks its own
    top-K for display."""
    rows: list[dict] = []
    fidelity_rows: list[dict] = []
    x_explain = sample_rows(X_target, config.explain_sample_n, config.random_seed)
    if x_explain.empty:
        return rows, fidelity_rows

    n_feats = len(feature_names)

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
            top_n=n_feats,
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
            top_n=n_feats,
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

    scheme = _build_window_scheme(config, alert_groups, n_total)

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
                scheme=scheme,
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
        f"Source window: {config.source_split_mode}"
        + (
            f" (split_time={config.source_split_time})"
            if config.source_split_mode == "baseline_split"
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
    config: TemporalDecayConfig,
    scenario: str,
    alert_groups: list,
    alert_groups_path: Path,
    n_total: int,
    base_schema: FeatureSchema,
    mining_settings_by_name: dict,
    mining_settings_path: Path,
    scheme: WindowScheme | None = None,
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
        scheme=scheme,
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

    # One-class SHAP has no analytic explainer -> PermutationExplainer over
    # every feature at every horizon, the dominant cost of an explanations
    # run. Unless explicitly asked for, flag the model so
    # compute_shap_signed_importances raises straight away (LIME still runs).
    if (
        config.compute_explanations
        and not config.oneclass_shap
        and cfg.model in ONE_CLASS_MODELS
    ):
        fit.model._skip_shap = True

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
        # One-time setup cost (system-operationality instrumentation) --
        # identical on every horizon row for this config, since the
        # never-retrain design pays it exactly once, at h=0.
        "mining_wall_s": fit.mining_wall_s,
        "mining_cpu_s": fit.mining_cpu_s,
        "fit_wall_s": fit.fit_wall_s,
        "fit_cpu_s": fit.fit_cpu_s,
    }

    horizon_rows: list[dict] = []
    explain_rows: list[dict] = []
    fidelity_rows: list[dict] = []

    def _record_horizon(
        horizon_window_index: int,
        X_h: pd.DataFrame,
        y_h: np.ndarray,
        n_alert_groups_h: int,
        raw_rows_h: list,
    ) -> None:
        horizon_fraction = (
            horizon_window_index / (fit.n_windows - 1) if fit.n_windows > 1 else 0.0
        )
        novelty = _novelty_metrics(raw_rows_h, fit.train_item_keys, fit.train_crp_keys)
        funnel = fast_route_and_funnel(raw_rows_h, fit.schema, fit.model, fit.threshold)

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
                    **novelty,
                    **funnel,
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
                **novelty,
                **funnel,
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
    _record_horizon(0, fit.X_test, fit.y_test, len(fit.X_test), fit.h0_rows)

    for k in range(1, fit.n_windows):
        X_tgt, y_tgt, n_alert_groups = encode_target_window(
            alert_groups, fit.scheme, fit.gran, k, fit.schema
        )
        t_start, t_end = fit.scheme.target_bounds(fit.gran, k)
        _record_horizon(k, X_tgt, y_tgt, n_alert_groups, alert_groups[t_start:t_end])

    return horizon_rows, explain_rows, fidelity_rows

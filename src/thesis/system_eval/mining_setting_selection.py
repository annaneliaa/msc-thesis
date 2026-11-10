"""
Paper Check 1: were the mining settings picked with knowledge of the test data?

Appendix E.2.3 chose the two-tree miner's settings (Table E.13) from sweeps run
over the full timeline, which includes the windows later used as walk-forward
test windows. This module re-selects them using only the first training window
W0 (= W_src under "baseline_split", the same W0 as Check 3), split in time
order into tuning-train (first 80%) and tuning-val (last 20%). Nothing after
W0 is touched unless tuning-val has fewer than `min_val_attacks` attack
groups, in which case the tuning span grows to W0 + W1 (and the walk-forward
evaluation should then start from W2).

  Stage 1 -- growth rate. Contrast-set survivor counts on tuning-train for a
  range of min_growth_rate thresholds (Figure E.1's curve), computed straight
  from compute_predicate_contrast_stats -- one pass, no mining runs.
  Optionally the same curve on the full timeline, for the side-by-side.

  Stage 2 -- tree settings. Mine on tuning-train for every point of
  max_depth x max_depth_attack x min_samples_leaf x max_diverse_rounds
  (everything else at Table E.13), then on tuning-val measure
    * the thesis's own measures, out of sample: mean leaf precision_attack
      over the attack tree's leaves (attacks / groups the leaf fires on) and
      mean leaf recall_benign over the benign tree's leaves (benign groups the
      leaf fires on / all benign groups) -- the same definitions as
      01_mining_param_eda.ipynb's grid_point_stats, which computed them
      in-sample; with a bootstrap std over tuning-val groups;
    * the shared measure: F1 (fixed 0.5 threshold) of an RF (class-weighted)
      on Base(5) + the mined features, fit on tuning-train, mean/std over
      seeds; FP/day and PR-AUC reported alongside.

  Selection, two ways (see select_thesis_rule / select_by_metric):
    * the thesis's rule, applied coordinate-wise from the Table E.13 anchor
      exactly as the thesis swept one axis at a time;
    * best tuning-val F1, then the simplest config within one seed std
      (FP/day and PR-AUC reported alongside; PR-AUC was the first shared
      measure but saturates on tuning-val, see downstream_scores).

The tuning loop itself (run_w0_tuning) takes the miner as a callable, so the
comparison methods (Eclat, RIPPER, PrefixSpan, MCTSExtent) can go through the
same code later. A post-mining filter hook (feature cap K / precision gate,
from Check 2) is accepted but off by default.
"""

from __future__ import annotations

import itertools
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from thesis.config import GroupingConfig
from thesis.encoders.service import encode_alert_groups_for_schema
from thesis.experiments._shared import (
    decide_threshold,
    fit_scored_model,
    labels_and_mask,
    load_scenario_context,
    metrics_at_threshold,
)
from thesis.features.persistence import load_symbolic_feature_schema
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.mining.atemporal.contrast_tree.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
    compute_predicate_contrast_stats,
    filter_contrast_survivors,
)
from thesis.mining.common.attribute_features import default_leaky_attribute_fields
from thesis.mining.atemporal.contrast_tree.window_schema_cache import (
    get_or_mine_slice_attribute_schema,
)
from thesis.paths import RESULTS_DIR, ROOT
from thesis.pipeline.pipeline import compute_window_train_end
from thesis.schemas.features import FeatureSchema
from thesis.schemas.mining import (
    AttributeMiningConfig,
    ContrastSetFilterConfig,
    DecisionTreeRuleConfig,
)
from thesis.system_eval.signature_leakage import (
    CSCAS_ALERT_GROUPS_CACHE_DIR,
    CSCAS_BASELINE_SPLIT_TIME,
    PAPER_GRANULARITY,
    build_baseline_split_scheme,
    training_window_bounds,
)

# Table E.13 (= configs/monitor_eda_mining_setting.yaml gr3_md1_mda4_rounds2).
THESIS_SETTING = {
    "min_growth_rate": 3.0,
    "max_depth": 1,
    "max_depth_attack": 4,
    "min_samples_leaf": 10,
    "max_diverse_rounds": 2,
}
FIXED_CONTRAST = dict(min_attack_coverage=0.05, min_benign_coverage=0.05)
FIXED_TREE = dict(
    class_weight="balanced", round_min_coverage=0.05, round_min_growth_rate=3.0
)

STAGE1_GROWTH_RATES = (1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
STAGE2_GRID = {
    "max_depth": (1, 2),
    "max_depth_attack": (2, 3, 4),
    "min_samples_leaf": (5, 10, 20),
    "max_diverse_rounds": (1, 2, 3),
}
# Simpler-first order per axis, for tie-breaks: shallower trees, fewer
# rounds, larger leaves (coarser rules).
_SIMPLER_FIRST = {
    "max_depth": lambda v: v,
    "max_depth_attack": lambda v: v,
    "max_diverse_rounds": lambda v: v,
    "min_samples_leaf": lambda v: -v,
}

_EXPERIMENTS_DIR = ROOT / "artifacts" / "experiments" / "mining_setting_selection"
_RESULTS_MIRROR_DIR = RESULTS_DIR / "paper" / "mining-setting-selection"


def two_tree_config(
    max_depth: int,
    max_depth_attack: int,
    min_samples_leaf: int,
    max_diverse_rounds: int,
    min_growth_rate: float = THESIS_SETTING["min_growth_rate"],
) -> AttributeMiningConfig:
    return AttributeMiningConfig(
        contrast=ContrastSetFilterConfig(
            min_growth_rate=min_growth_rate, **FIXED_CONTRAST
        ),
        tree=DecisionTreeRuleConfig(
            max_depth=max_depth,
            max_depth_attack=max_depth_attack,
            min_samples_leaf=min_samples_leaf,
            max_diverse_rounds=max_diverse_rounds,
            **FIXED_TREE,
        ),
    )


# --- Tuning split -----------------------------------------------------------


@dataclass(frozen=True)
class TuningSplit:
    train_start: int
    train_end: int  # = val_start
    val_end: int
    n_windows_used: int  # 1 = W0 only, 2 = W0 + W1
    n_val_attack: int
    n_train_attack: int


def resolve_tuning_split(
    alert_groups: list,
    gran: float = PAPER_GRANULARITY,
    train_frac: float = 0.8,
    min_val_attacks: int = 50,
    split_time: str = CSCAS_BASELINE_SPLIT_TIME,
) -> TuningSplit:
    """First 80% / last 20% (time order) of W0; widened to W0 + W1 if
    tuning-val would hold fewer than `min_val_attacks` attack groups."""
    bounds = training_window_bounds(
        build_baseline_split_scheme(alert_groups, split_time), gran
    )
    for n_used in (1, 2):
        end = bounds[n_used - 1][1]
        cut = compute_window_train_end(0, end, train_frac)
        n_val_atk = int(np.nansum(labels_and_mask(alert_groups[cut:end])[0]))
        if n_val_atk >= min_val_attacks or n_used == 2:
            n_tr_atk = int(np.nansum(labels_and_mask(alert_groups[:cut])[0]))
            return TuningSplit(0, cut, end, n_used, n_val_atk, n_tr_atk)
    raise AssertionError("unreachable")


# --- Stage 1 ----------------------------------------------------------------


def stage1_survivor_curve(
    alert_groups: list,
    growth_rates: tuple[float, ...] = STAGE1_GROWTH_RATES,
    scenario: str = "cscas",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contrast-set candidate stats on `alert_groups` (one pass) and the
    number of survivors at each min_growth_rate (coverage fixed at 0.05,
    leaky oracle fields excluded -- as in mining). Returns (curve, stats)."""
    X_cat, _, y, cpm = build_categorical_predicate_matrix(
        alert_groups, exclude_fields=default_leaky_attribute_fields(scenario)
    )
    stats = compute_predicate_contrast_stats(X_cat, y, cpm)
    rows = []
    for gr in growth_rates:
        surv = filter_contrast_survivors(stats, min_growth_rate=gr, **FIXED_CONTRAST)
        gr_col = surv["growth_rate"] if len(surv) else pd.Series(dtype=float)
        rows.append(
            {
                "min_growth_rate": gr,
                "n_candidates": len(stats),
                "n_survivors": len(surv),
                "n_survivors_attack": int((gr_col >= gr).sum()),
                "n_survivors_benign": int((gr_col < gr).sum()),
            }
        )
    return pd.DataFrame(rows), stats


# --- Stage 2: one grid point ------------------------------------------------


def leaf_quality_on_val(
    schema: FeatureSchema,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Thesis measures on held-out groups: mean precision_attack over attack
    tree leaves, mean recall_benign over benign tree leaves (+ bootstrap std
    over tuning-val rows). Leaves that never fire on val have undefined
    precision and are left out of that mean (counted in n_*_silent)."""
    feats = [
        f for f in schema.symbolic.features if f.mining_type == "decision_tree_rule"
    ]
    atk_cols = [f.feature_name for f in feats if f.source_label == "attack"]
    ben_cols = [f.feature_name for f in feats if f.source_label == "benign"]
    F_atk = (
        X_val[atk_cols].to_numpy() > 0 if atk_cols else np.zeros((len(y_val), 0), bool)
    )
    F_ben = (
        X_val[ben_cols].to_numpy() > 0 if ben_cols else np.zeros((len(y_val), 0), bool)
    )
    atk, ben = y_val == 1, y_val == 0

    def _stats(idx: np.ndarray) -> tuple[float, float]:
        fa, fb, a, b = F_atk[idx], F_ben[idx], atk[idx], ben[idx]
        fires = fa.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = (fa & a[:, None]).sum(0) / fires
        rec = (
            (fb & b[:, None]).sum(0) / b.sum()
            if b.sum()
            else np.full(fb.shape[1], np.nan)
        )
        return (
            float(np.nanmean(prec)) if np.isfinite(prec).any() else np.nan,
            float(np.nanmean(rec)) if len(rec) else np.nan,
        )

    full = np.arange(len(y_val))
    p, r = _stats(full)
    rng = np.random.default_rng(seed)
    boot = np.array(
        [_stats(rng.integers(0, len(y_val), len(y_val))) for _ in range(n_boot)]
    )
    return {
        "val_precision_attack": p,
        "val_precision_attack_boot_std": float(np.nanstd(boot[:, 0])),
        "val_recall_benign": r,
        "val_recall_benign_boot_std": float(np.nanstd(boot[:, 1])),
        "n_leaves_attack": len(atk_cols),
        "n_leaves_benign": len(ben_cols),
        "n_attack_leaves_silent": int((F_atk.sum(0) == 0).sum()),
    }


def downstream_scores(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seeds: tuple[int, ...],
    val_days: float,
    model: str = "rf",
) -> dict:
    """RF (class-weighted) on Base(5) + mined features, fit on tuning-train,
    scored on tuning-val per seed: F1 at the walk-forward's own operating point
    (decide_threshold "fixed" = 0.5) -- the shared selection measure -- plus
    FP/day and PR-AUC, reported alongside. PR-AUC was the original shared
    measure; the first Check 1 run found it saturated on tuning-val (0.992-
    0.997 across all 54 settings), so it can't separate settings."""
    f1s, fpds, praucs = [], [], []
    for s in seeds:
        m = fit_scored_model(model, X_tr, y_tr, seed=s)
        thr = decide_threshold(y_tr, m.predict_proba(X_tr)[:, 1], "fixed", 0.9, model=m)
        proba = m.predict_proba(X_val)[:, 1]
        met = metrics_at_threshold(y_val, proba, thr)
        f1s.append(met["f1"])
        fpds.append(met["fp"] / val_days if val_days and val_days > 0 else np.nan)
        praucs.append(average_precision_score(y_val, proba))

    def _ms(name: str, vals: list[float]) -> dict:
        return {
            f"val_{name}_mean": float(np.mean(vals)),
            f"val_{name}_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        }

    return {
        **_ms("f1", f1s),
        **_ms("fp_per_day", fpds),
        **_ms("prauc", praucs),
        "val_f1_per_seed": json.dumps([round(v, 6) for v in f1s]),
    }


Miner = Callable[[dict], FeatureSchema]
PostFilter = Callable[[FeatureSchema, pd.DataFrame, np.ndarray], FeatureSchema]


def run_w0_tuning(
    miner: Miner,
    grid: list[dict],
    train_rows: list,
    val_rows: list,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    post_filter: PostFilter | None = None,
    leaf_quality: bool = True,
    on_row: Callable[[dict], None] | None = None,
) -> pd.DataFrame:
    """Method-agnostic W0 tuning loop: for each param dict in `grid`, mine a
    schema on tuning-train (`miner(params)`), optionally post-filter it
    (feature cap / precision gate -- Check 2), encode both halves, score
    tuning-val F1 (shared measure), FP/day and PR-AUC over `seeds` (+ thesis
    leaf measures if `leaf_quality` and the schema has decision_tree_rule
    features)."""
    val_ts = [g.start_ts for g in val_rows if g.start_ts is not None]
    val_days = (max(val_ts) - min(val_ts)) / 86400 if len(val_ts) > 1 else np.nan
    y_tr_all, m_tr = labels_and_mask(train_rows)
    y_val_all, m_val = labels_and_mask(val_rows)
    y_tr, y_val = y_tr_all[m_tr].astype(int), y_val_all[m_val].astype(int)
    rows = []
    for k, params in enumerate(grid):
        t0 = time.perf_counter()
        schema = miner(params)
        mine_s = time.perf_counter() - t0
        X_tr = (
            encode_alert_groups_for_schema(train_rows, schema)
            .loc[m_tr]
            .reset_index(drop=True)
        )
        if post_filter is not None:
            schema = post_filter(schema, X_tr, y_tr)
            X_tr = (
                encode_alert_groups_for_schema(train_rows, schema)
                .loc[m_tr]
                .reset_index(drop=True)
            )
        X_val = (
            encode_alert_groups_for_schema(val_rows, schema)
            .loc[m_val]
            .reset_index(drop=True)
        )
        row = {
            **params,
            "n_features": X_tr.shape[1],
            "n_symbolic": len(schema.symbolic.features) if schema.symbolic else 0,
            "mine_wall_s": mine_s,
        }
        if leaf_quality and schema.symbolic is not None:
            row.update(leaf_quality_on_val(schema, X_val, y_val))
        row.update(downstream_scores(X_tr, y_tr, X_val, y_val, seeds, val_days))
        row["wall_s"] = time.perf_counter() - t0
        rows.append(row)
        print(
            f"  [{k + 1}/{len(grid)}] {params} -> f1={row['val_f1_mean']:.4f} "
            f"fp/day={row['val_fp_per_day_mean']:.1f} prauc={row['val_prauc_mean']:.4f} "
            f"prec_atk={row.get('val_precision_attack', np.nan):.3f} "
            f"rec_ben={row.get('val_recall_benign', np.nan):.3f} "
            f"({row['n_symbolic']} sym, {row['wall_s']:.0f}s)"
        )
        if on_row is not None:
            on_row(row)
    return pd.DataFrame(rows)


# --- Selection rules ----------------------------------------------------------


def _key(params: dict, axes) -> tuple:
    return tuple(params[a] for a in axes)


def _simplest(df: pd.DataFrame, axes) -> pd.Series:
    order = sorted(
        df.index, key=lambda i: tuple(_SIMPLER_FIRST[a](df.at[i, a]) for a in axes)
    )
    return df.loc[order[0]]


def _tolerance(grid: pd.DataFrame, best: pd.Series, std: str) -> float:
    """Tie tolerance around the best setting: its own std, but never less than
    the typical (median) std of that measure across the whole grid. Without
    the floor, a best setting whose seeds happen to agree exactly (std 0.0000
    in the first F1 run) makes nothing a tie, and the simplest-within-noise
    rule silently degrades to a plain argmax."""
    own = best[std] if np.isfinite(best[std]) else 0.0
    typical = grid[std].median() if std in grid and grid[std].notna().any() else 0.0
    return float(max(own, typical if np.isfinite(typical) else 0.0))


def select_thesis_rule(
    grid: pd.DataFrame, anchor: dict = THESIS_SETTING, max_passes: int = 3
) -> dict:
    """The thesis's selection rule, applied one axis at a time with the other
    axes held at the current anchor (starting from Table E.13), repeated until
    nothing changes:
      max_depth          shallowest whose recall_benign is within one bootstrap
                         std of the best depth's (the thesis: "recall_benign
                         drops with depth, so the shallowest tested");
      max_depth_attack   shallowest whose precision_attack is within one
                         bootstrap std of the best (the plateau);
      min_samples_leaf   best precision_attack, ties (within one std) to the
                         larger leaf;
      max_diverse_rounds best downstream F1, ties (within one seed std) to
                         fewer rounds -- the thesis picked this on downstream
                         F1 on the test subsample; here on tuning-val F1.
    Returns the picked values plus a `trace` of each axis decision."""
    axes = list(STAGE2_GRID)
    cur = {a: anchor[a] for a in axes}
    rules = {
        "max_depth": ("val_recall_benign", "val_recall_benign_boot_std"),
        "max_depth_attack": ("val_precision_attack", "val_precision_attack_boot_std"),
        "min_samples_leaf": ("val_precision_attack", "val_precision_attack_boot_std"),
        "max_diverse_rounds": ("val_f1_mean", "val_f1_std"),
    }
    trace = []
    for p in range(max_passes):
        changed = False
        for axis in axes:
            others = {a: v for a, v in cur.items() if a != axis}
            sl = grid
            for a, v in others.items():
                sl = sl[sl[a] == v]
            if sl.empty:
                continue
            metric, std = rules[axis]
            best = sl.loc[sl[metric].idxmax()]
            tol = _tolerance(grid, best, std)
            ok = sl[sl[metric] >= best[metric] - tol]
            pick = _simplest(ok, [axis])[axis]
            trace.append(
                {
                    "pass": p,
                    "axis": axis,
                    "metric": metric,
                    "best_value": best[axis],
                    "best": best[metric],
                    "tolerance": tol,
                    "plateau": sorted(ok[axis].tolist()),
                    "picked": pick,
                }
            )
            if pick != cur[axis]:
                cur[axis], changed = pick, True
        if not changed:
            break
    return {**cur, "trace": trace}


def select_by_metric(grid: pd.DataFrame, metric: str = "val_f1") -> dict:
    """Shared measure: best mean tuning-val `metric` (default F1 at the fixed
    threshold); among configs within one seed std of it, the simplest
    (shallower trees, fewer rounds, larger leaves, then fewer features)."""
    axes = list(STAGE2_GRID)
    mean, std = f"{metric}_mean", f"{metric}_std"
    best = grid.loc[grid[mean].idxmax()]
    tol = _tolerance(grid, best, std)
    ok = grid[grid[mean] >= best[mean] - tol]
    order = sorted(
        ok.index,
        key=lambda i: (
            *(_SIMPLER_FIRST[a](ok.at[i, a]) for a in axes),
            ok.at[i, "n_features"],
        ),
    )
    pick = ok.loc[order[0]]
    return {
        **{a: pick[a] for a in axes},
        "metric": metric,
        "best": best[mean],
        "tolerance": tol,
        "n_within_tolerance": len(ok),
        "picked": pick[mean],
        "picked_fp_per_day": pick.get("val_fp_per_day_mean", np.nan),
    }


def selection_table(
    grid: pd.DataFrame, stage1_pick: float | None = None
) -> pd.DataFrame:
    thesis = select_thesis_rule(grid)
    f1 = select_by_metric(grid, "val_f1")
    rows = [
        {
            "setting": a,
            "thesis_full_timeline": THESIS_SETTING[a],
            "w0_thesis_rule": thesis[a],
            "w0_f1": f1[a],
        }
        for a in STAGE2_GRID
    ]
    if stage1_pick is not None:
        rows.insert(
            0,
            {
                "setting": "min_growth_rate",
                "thesis_full_timeline": THESIS_SETTING["min_growth_rate"],
                "w0_thesis_rule": stage1_pick,
                "w0_f1": np.nan,
            },
        )
    out = pd.DataFrame(rows)
    out["same_thesis_rule"] = out["thesis_full_timeline"] == out["w0_thesis_rule"]
    out["same_f1"] = out["thesis_full_timeline"] == out["w0_f1"]
    return out


def write_settings_yaml(table: pd.DataFrame, path: Path) -> None:
    """The thesis setting and both W0 picks as a mining-settings YAML (the
    screening_mining_settings.yaml format), so step 5's walk-forward can run
    each through run_signature_leakage.py --arms b --mining-setting <name>."""
    import yaml

    def _spec(name: str, col: str) -> dict:
        v = {r["setting"]: r[col] for _, r in table.iterrows()}
        v = {
            k: (THESIS_SETTING[k] if pd.isna(v.get(k, np.nan)) else v[k])
            for k in THESIS_SETTING
        }
        cfg = two_tree_config(
            max_depth=int(v["max_depth"]),
            max_depth_attack=int(v["max_depth_attack"]),
            min_samples_leaf=int(v["min_samples_leaf"]),
            max_diverse_rounds=int(v["max_diverse_rounds"]),
            min_growth_rate=float(v["min_growth_rate"]),
        )
        return {
            "name": name,
            "contrast": cfg.contrast.model_dump(),
            "tree": cfg.tree.model_dump(),
        }

    settings = [
        _spec("check1_thesis_full_timeline", "thesis_full_timeline"),
        _spec("check1_w0_thesis_rule", "w0_thesis_rule"),
        _spec("check1_w0_f1", "w0_f1"),
    ]
    path.write_text(yaml.safe_dump({"settings": settings}, sort_keys=False))


# --- End to end ---------------------------------------------------------------


@dataclass
class MiningSettingSelectionConfig:
    gran: float = PAPER_GRANULARITY
    train_frac: float = 0.8
    min_val_attacks: int = 50
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    stage1_full_timeline: bool = False
    # None = full STAGE2_GRID; a list of param dicts overrides (smoke tests).
    grid: list[dict] | None = None
    force_remine: bool = False
    out_dir: Path | None = None


def full_stage2_grid() -> list[dict]:
    axes = list(STAGE2_GRID)
    return [dict(zip(axes, vals)) for vals in itertools.product(*STAGE2_GRID.values())]


def run_mining_setting_selection(config: MiningSettingSelectionConfig) -> Path:
    ctx = load_scenario_context(
        scenario="cscas",
        cache_dir=CSCAS_ALERT_GROUPS_CACHE_DIR,
        grouping=GroupingConfig(mode=CSCAS_PREGROUPED_METHOD),
        alerts_json_path=None,
        mining_settings_path=ROOT
        / "src/thesis/configs/monitor_eda_mining_setting.yaml",
    )
    groups = ctx.alert_groups
    split = resolve_tuning_split(
        groups, config.gran, config.train_frac, config.min_val_attacks
    )
    print(f"Tuning split: {asdict(split)}")
    train_rows = groups[split.train_start : split.train_end]
    val_rows = groups[split.train_end : split.val_end]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(config.out_dir) if config.out_dir else _EXPERIMENTS_DIR) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str)
    )
    (out_dir / "tuning_split.json").write_text(json.dumps(asdict(split), indent=2))

    print("Stage 1: survivor curve on tuning-train...")
    curve, stats = stage1_survivor_curve(train_rows)
    curve.insert(0, "span", "w0_tuning_train")
    if config.stage1_full_timeline:
        print("Stage 1: survivor curve on the full timeline...")
        full_curve, _ = stage1_survivor_curve(groups)
        full_curve.insert(0, "span", "full_timeline")
        curve = pd.concat([curve, full_curve], ignore_index=True)
    curve.to_csv(out_dir / "stage1_survivors.csv", index=False)
    stats.to_csv(out_dir / "stage1_contrast_stats_w0.csv", index=False)
    print(curve.to_string(index=False))

    tag = f"paper_check1_w{split.n_windows_used}_tune{config.train_frac:g}"

    def miner(params: dict) -> FeatureSchema:
        res = get_or_mine_slice_attribute_schema(
            scenario="cscas",
            alert_groups=groups,
            alert_groups_path=ctx.alert_groups_path,
            slice_start=split.train_start,
            slice_end=split.train_end,
            slice_tag=tag,
            attribute_mining_config=two_tree_config(**params),
            force=config.force_remine,
        )
        symbolic = load_symbolic_feature_schema(res.schema_path)
        return FeatureSchema(
            schema_name="base+symbolic",
            schema_version=symbolic.schema_version,
            base=ctx.base_schema.base,
            symbolic=symbolic,
        )

    grid = config.grid if config.grid is not None else full_stage2_grid()
    print(f"Stage 2: {len(grid)} grid points x {len(config.seeds)} seeds")
    done: list[dict] = []

    def _checkpoint(row: dict) -> None:
        done.append(row)
        pd.DataFrame(done).to_csv(out_dir / "stage2_grid.csv", index=False)

    grid_df = run_w0_tuning(
        miner, grid, train_rows, val_rows, config.seeds, on_row=_checkpoint
    )

    thesis = select_thesis_rule(grid_df)
    pd.DataFrame(thesis["trace"]).to_csv(out_dir / "thesis_rule_trace.csv", index=False)
    table = selection_table(grid_df)
    table.to_csv(out_dir / "selection.csv", index=False)
    (out_dir / "selection_f1.json").write_text(
        json.dumps(select_by_metric(grid_df, "val_f1"), indent=2, default=float)
    )
    print(table.to_string(index=False))
    write_settings_yaml(table, out_dir / "w0_settings.yaml")

    if config.out_dir is None and config.grid is None:
        mirror = _RESULTS_MIRROR_DIR / ts
        try:
            shutil.copytree(out_dir, mirror, dirs_exist_ok=True)
            print(f"Mirrored → {mirror}")
        except OSError as exc:
            print(f"[warn] could not mirror to {mirror}: {exc}")
    return out_dir

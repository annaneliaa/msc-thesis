"""
Mining comparison (paper plan 7.6): walk-forward comparison of the mining
methods on CSCAS, every method under the same treatment.

Per walk-forward step i (the thesis's always-retrain protocol, baseline_split
scheme at the paper's granularity, identical windows to Checks 1-3): mine on
the labelled groups of training window Wi, train on Wi, test on W(i+1).

Arms:
  - "none": Base(5) only, no mining (the reference for helps/hurts).
  - one arm per method (eclat, contrast_tree, ripper): Base(5) + that
    method's features.

Shared treatment, the same for every method (plan section 3):
  - the same item table, built on Wi (numerics as quantile bins, the
    scenario's leaky fields excluded),
  - the precision gate (g = 2 on Wi), condition dedupe, near-duplicate firing
    dedupe (Jaccard >= 0.95) and budget K -- `select_features`, the same code
    path as the mini experiment,
  - class-weighted RF at a fixed 0.5 threshold, one fit per seed.

Seeds: every RF is fit once per seed. Stochastic miners (RIPPER, MCTSExtent)
are also re-mined per seed when `reseed_stochastic_miners` is on; Eclat and
the contrast tree are deterministic and mined once per step.

Per (arm, seed, step) it records: F1, precision, recall, PR-AUC, ROC-AUC,
FP/day, recall on attacks with signatures seen / unseen in Wi, the feature
count, mean conditions per feature, mean pairwise Jaccard of the features'
firing on Wi (overlap), the share of gated distinct conditions dropped as
near-duplicates (redundancy) and the mining time. Every kept feature goes
into features.csv, from which stability (share of step i's conditions that
reappear at step i+1) is computed.

`summarize_comparison` gives the per-arm table (mean/std over steps of the
seed-averaged step metrics), ΔF1 / ΔPR-AUC vs. "none" with helps/hurts
counts, and a Friedman test with average ranks and the Nemenyi critical
difference over the steps.

Method settings are the mini experiment's documented defaults, not tuned
(the tuning criterion is still open, see FINDINGS.md).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import average_precision_score

from thesis.encoders.service import encode_alert_groups_for_schema
from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.experiments._shared import (
    decide_threshold,
    fit_scored_model,
    metrics_at_threshold,
)
from thesis.mining.common.attribute_features import default_leaky_attribute_fields
from thesis.mining.common.item_table import build_item_table
from thesis.mining.common.patterns import patterns_to_schema
from thesis.paths import ROOT
from thesis.schemas.features import SEQUENCE_MINING_TYPES
from thesis.schemas.groups import AlertGroup
from thesis.system_eval.mining_mini_experiment import (
    TEMPORAL,
    MiniConfig,
    MiniWindow,
    _labels,
    _repo_commit,
    labelled,
    select_features,
)

ATEMPORAL_METHODS = ("eclat", "contrast_tree", "ripper")
STOCHASTIC_MINERS = {"ripper", "mctsextent"}
NO_MINING = "none"


@dataclass
class ComparisonConfig:
    gran: float = 0.05
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    methods: tuple[str, ...] = ATEMPORAL_METHODS
    include_no_mining: bool = True
    # Per-method settings, gate, dedupe and budget K (k, per_class_budget).
    mining: MiniConfig = field(default_factory=MiniConfig)
    reseed_stochastic_miners: bool = True
    # Extra fields left out of the item table (e.g. the signature-derived
    # items, for the "no signature items" condition of Check 3).
    extra_exclude_fields: frozenset[str] = frozenset()
    model: str = "rf"
    threshold_mode: str = "fixed"
    calibrated_recall_target: float = 0.9
    # None = every step; an int caps the walk (smoke tests).
    max_steps: int | None = None
    out_dir: Path | None = None
    experiment: str = "mining_comparison"


@dataclass
class Step:
    """One walk-forward step: labelled mining/training rows and test rows."""

    i: int
    train: list[AlertGroup]
    test: list[AlertGroup]
    test_days: float


# --- Per-step pieces ------------------------------------------------------


def _window_days(rows: list[AlertGroup]) -> float:
    ts = [g.start_ts for g in rows if g.start_ts is not None]
    return (max(ts) - min(ts)) / 86400 if len(ts) > 1 else np.nan


def firing_overlap(F: np.ndarray) -> float:
    """Mean pairwise Jaccard of the columns' firing (boolean rows x features).
    NaN with fewer than two features."""
    n = F.shape[1]
    if n < 2:
        return np.nan
    F = F.astype(np.float32)
    inter = F.T @ F
    size = np.diag(inter)
    union = size[:, None] + size[None, :] - inter
    jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    iu = np.triu_indices(n, k=1)
    return float(jac[iu].mean())


def redundancy_share(stages: dict) -> float:
    """Share of the gated distinct conditions dropped as near-duplicates."""
    n = len(stages["distinct_condition"])
    if n == 0:
        return np.nan
    return (n - len(stages["distinct_firing"])) / n


def metric_row(
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    seen: np.ndarray,
    n_days: float,
) -> dict:
    m = metrics_at_threshold(y, proba, threshold)
    atk = y == 1
    pred = proba >= threshold
    row = {
        **m,
        "pr_auc": float(average_precision_score(y, proba)) if atk.any() else np.nan,
        "n_days": n_days,
        "fp_per_day": m["fp"] / n_days if n_days and n_days > 0 else np.nan,
    }
    for part, sel in (("seen", atk & seen), ("unseen", atk & ~seen)):
        n = int(sel.sum())
        row[f"n_attack_{part}"] = n
        row[f"tp_{part}"] = int((sel & pred).sum())
        row[f"recall_{part}"] = row[f"tp_{part}"] / n if n else np.nan
    return row


def feature_frame(schema, F_tr: np.ndarray, F_te: np.ndarray) -> pd.DataFrame:
    """One row per kept feature, with its condition as the encoder reads it."""
    return pd.DataFrame(
        {
            "feature": [f.feature_name for f in schema.features],
            "condition": [
                (" -> " if f.mining_type in SEQUENCE_MINING_TYPES else " ∧ ").join(
                    f.itemset
                )
                for f in schema.features
            ],
            "source_label": [f.source_label for f in schema.features],
            "mining_type": [f.mining_type for f in schema.features],
            "n_conditions": [len(f.itemset) for f in schema.features],
            "train_fires": F_tr.sum(axis=0),
            "test_fires": F_te.sum(axis=0),
        }
    )


@dataclass
class MinedArm:
    """A method's features on one step for one mining seed."""

    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    info: dict
    features: pd.DataFrame


def mine_arm(
    method: str, step: Step, window: MiniWindow, mini: MiniConfig, table
) -> MinedArm:
    sel = select_features(method, window, mini, table)
    info = {
        **sel.counts,
        "mine_seconds": sel.mine_seconds,
        "redundancy_share": redundancy_share(sel.stages),
    }
    if sel.budget.empty:
        empty = pd.DataFrame(index=range(0))
        info.update(n_symbolic=0, mean_conditions=np.nan, overlap=np.nan)
        return MinedArm(
            empty.reindex(range(len(step.train))),
            empty.reindex(range(len(step.test))),
            info,
            pd.DataFrame(),
        )
    schema = patterns_to_schema(sel.budget, method, f"step{step.i}", item_table=table)
    enc = SymbolicFeatureEncoder(schema)
    names = [f.feature_name for f in schema.features]
    X_tr = enc.transform(step.train)[names].reset_index(drop=True)
    X_te = enc.transform(step.test)[names].reset_index(drop=True)
    F_tr = X_tr.to_numpy().astype(bool)
    feats = feature_frame(schema, F_tr, X_te.to_numpy().astype(bool))
    info.update(
        n_symbolic=len(names),
        n_symbolic_attack=int((feats["source_label"] == "attack").sum()),
        mean_conditions=float(feats["n_conditions"].mean()),
        overlap=firing_overlap(F_tr),
    )
    return MinedArm(X_tr, X_te, info, feats)


def fit_and_score(
    config: ComparisonConfig,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    seed: int,
    seen: np.ndarray,
    n_days: float,
) -> dict:
    model = fit_scored_model(config.model, X_tr, y_tr, seed=seed)
    if model is None:
        return {"status": "model could not be fit"}
    thr = decide_threshold(
        y_tr,
        model.predict_proba(X_tr)[:, 1],
        config.threshold_mode,
        config.calibrated_recall_target,
        model=model,
    )
    proba = model.predict_proba(X_te)[:, 1]
    return {"threshold": thr, **metric_row(y_te, proba, thr, seen, n_days)}


def run_step(
    config: ComparisonConfig,
    step: Step,
    base_schema,
    signature_of,
    exclude_fields: set[str],
) -> tuple[list[dict], list[pd.DataFrame]]:
    """All arms x seeds on one step."""
    y_tr = _labels(step.train).astype(int)
    y_te = _labels(step.test).astype(int)
    tr_sigs = {signature_of(g) for g in step.train}
    seen = np.array([signature_of(g) in tr_sigs for g in step.test])
    Xb_tr = encode_alert_groups_for_schema(step.train, base_schema).reset_index(
        drop=True
    )
    Xb_te = encode_alert_groups_for_schema(step.test, base_schema).reset_index(
        drop=True
    )
    common = {
        "step_i": step.i,
        "n_train": len(y_tr),
        "n_train_attack": int(y_tr.sum()),
        "n_test": len(y_te),
        "n_test_attack": int(y_te.sum()),
        "n_base": Xb_tr.shape[1],
    }
    rows: list[dict] = []
    feats: list[pd.DataFrame] = []

    if config.include_no_mining:
        for seed in config.seeds:
            rows.append(
                {
                    "arm": NO_MINING,
                    "seed": seed,
                    **common,
                    "n_symbolic": 0,
                    **fit_and_score(
                        config, Xb_tr, y_tr, Xb_te, y_te, seed, seen, step.test_days
                    ),
                }
            )

    window = MiniWindow(
        "cscas",
        f"step {step.i}",
        step.train,
        step.test,
        signature_of=signature_of,
        exclude_fields=exclude_fields,
    )
    table = None
    if any(m not in TEMPORAL for m in config.methods):
        t0 = time.perf_counter()
        table = build_item_table(
            step.train, exclude_fields=exclude_fields, n_bins=config.mining.n_bins
        )
        print(f"  item table {table.items().shape} in {time.perf_counter() - t0:.1f}s")

    for method in config.methods:
        reseed = config.reseed_stochastic_miners and method in STOCHASTIC_MINERS
        mined: MinedArm | None = None
        for seed in config.seeds:
            if mined is None or reseed:
                mined = mine_arm(
                    method, step, window, replace(config.mining, seed=seed), table
                )
                if not mined.features.empty:
                    f = mined.features.copy()
                    f.insert(0, "arm", method)
                    f.insert(1, "mining_seed", seed if reseed else config.seeds[0])
                    f.insert(2, "step_i", step.i)
                    feats.append(f)
            X_tr = pd.concat([Xb_tr, mined.X_tr], axis=1)
            X_te = pd.concat([Xb_te, mined.X_te], axis=1)
            rows.append(
                {
                    "arm": method,
                    "seed": seed,
                    **common,
                    **mined.info,
                    "mining_seed": seed if reseed else config.seeds[0],
                    **fit_and_score(
                        config, X_tr, y_tr, X_te, y_te, seed, seen, step.test_days
                    ),
                }
            )
        last = rows[-1]
        print(
            f"  step {step.i:2d} {method:<13} f1={last.get('f1', np.nan):.3f} "
            f"pr_auc={last.get('pr_auc', np.nan):.3f} "
            f"{last.get('n_symbolic', 0)} features, "
            f"redundancy {last.get('redundancy_share', np.nan):.2f}, "
            f"{last.get('mine_seconds', 0):.1f}s"
        )
    return rows, feats


# --- CSCAS walk-forward ---------------------------------------------------


def cscas_steps(config: ComparisonConfig):
    """The CSCAS walk-forward steps (same windows as Checks 1-3), the base
    schema and the SignatureID lookup."""
    from thesis.config import GroupingConfig
    from thesis.experiments._shared import load_scenario_context
    from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
    from thesis.system_eval.signature_leakage import (
        CSCAS_ALERT_GROUPS_CACHE_DIR,
        build_baseline_split_scheme,
        load_cscas_signature_lookup,
        training_window_bounds,
    )

    ctx = load_scenario_context(
        scenario="cscas",
        cache_dir=CSCAS_ALERT_GROUPS_CACHE_DIR,
        grouping=GroupingConfig(mode=CSCAS_PREGROUPED_METHOD),
        alerts_json_path=None,
        mining_settings_path=config.mining.mining_settings_path,
    )
    groups = ctx.alert_groups
    scheme = build_baseline_split_scheme(groups)
    bounds = training_window_bounds(scheme, config.gran)
    n = len(bounds) if config.max_steps is None else min(len(bounds), config.max_steps)
    lookup = load_cscas_signature_lookup()

    def steps():
        for i in range(n):
            s, e = bounds[i]
            ts, te = scheme.target_bounds(config.gran, i + 1)
            yield Step(
                i,
                labelled(groups[s:e]),
                labelled(groups[ts:te]),
                _window_days(groups[ts:te]),
            )

    windows = pd.DataFrame(
        [
            {
                "step_i": i,
                "train_start": bounds[i][0],
                "train_end": bounds[i][1],
                "test_start": scheme.target_bounds(config.gran, i + 1)[0],
                "test_end": scheme.target_bounds(config.gran, i + 1)[1],
            }
            for i in range(n)
        ]
    )
    return steps, windows, ctx.base_schema, (lambda g: lookup.get(g.group_id))


def default_out_dir(experiment: str = "mining_comparison") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "artifacts" / "experiments" / experiment / ts


def run_comparison(config: ComparisonConfig) -> Path:
    """Walk-forward over every step x arm x seed on CSCAS. Writes
    per_step_results.csv and features.csv (rewritten after every step, so a
    crashed run keeps what it finished), windows.csv, config.json, and at the
    end summary.csv, helps_hurts.csv, stability.csv and stats.json."""
    unknown = set(config.methods) - set(ATEMPORAL_METHODS)
    if unknown:
        raise ValueError(
            f"CSCAS runs the atemporal methods only; got {sorted(unknown)}"
        )
    out_dir = (
        Path(config.out_dir) if config.out_dir else default_out_dir(config.experiment)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {**asdict(config), "repo_commit": _repo_commit()}, indent=2, default=str
        )
    )
    steps, windows, base_schema, signature_of = cscas_steps(config)
    windows.to_csv(out_dir / "windows.csv", index=False)
    exclude = set(default_leaky_attribute_fields("cscas")) | set(
        config.extra_exclude_fields
    )
    print(
        f"Output → {out_dir}  ({len(windows)} steps x "
        f"{len(config.methods) + config.include_no_mining} arms x "
        f"{len(config.seeds)} seeds)"
    )

    rows: list[dict] = []
    feats: list[pd.DataFrame] = []
    for step in steps():
        print(
            f"[step {step.i}] train {len(step.train)} "
            f"({int(_labels(step.train).sum())} attack), test {len(step.test)}"
        )
        if len(set(_labels(step.train))) < 2:
            print("  [warn] single-class training window -- skipped")
            continue
        r, f = run_step(config, step, base_schema, signature_of, exclude)
        rows += r
        feats += f
        pd.DataFrame(rows).to_csv(out_dir / "per_step_results.csv", index=False)
        if feats:
            pd.concat(feats, ignore_index=True).to_csv(
                out_dir / "features.csv", index=False
            )

    write_summary(out_dir, pd.DataFrame(rows), pd.concat(feats) if feats else None)
    return out_dir


# --- Summary --------------------------------------------------------------

SUMMARY_METRICS = (
    "f1",
    "precision",
    "recall",
    "pr_auc",
    "auc",
    "fp_per_day",
    "n_symbolic",
    "mean_conditions",
    "overlap",
    "redundancy_share",
    "mine_seconds",
)


def per_step_means(per_step: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged metrics per (arm, step)."""
    cols = [c for c in SUMMARY_METRICS if c in per_step]
    return per_step.groupby(["arm", "step_i"])[cols].mean().reset_index()


def helps_hurts(per_step: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Per arm: mean Δmetric vs. "none" over steps, and the number of steps
    where mining helps / hurts / ties (seed-averaged per step)."""
    m = per_step_means(per_step).pivot(index="step_i", columns="arm", values=metric)
    if NO_MINING not in m:
        return pd.DataFrame()
    out = []
    for arm in m.columns.drop(NO_MINING):
        d = (m[arm] - m[NO_MINING]).dropna()
        out.append(
            {
                "arm": arm,
                "metric": metric,
                "n_steps": len(d),
                "mean_delta": float(d.mean()) if len(d) else np.nan,
                "median_delta": float(d.median()) if len(d) else np.nan,
                "helps": int((d > 0).sum()),
                "hurts": int((d < 0).sum()),
                "ties": int((d == 0).sum()),
            }
        )
    return pd.DataFrame(out)


def stability(features: pd.DataFrame) -> pd.DataFrame:
    """Per arm (and mining seed): the mean share of step i's conditions that
    reappear in step i+1's kept features."""
    if features is None or features.empty:
        return pd.DataFrame()
    out = []
    for (arm, seed), g in features.groupby(["arm", "mining_seed"]):
        by_step = g.groupby("step_i")["condition"].apply(set).sort_index()
        steps = list(by_step.index)
        shares = [
            len(by_step[a] & by_step[b]) / len(by_step[a])
            for a, b in zip(steps, steps[1:])
            if b == a + 1 and by_step[a]
        ]
        out.append(
            {
                "arm": arm,
                "mining_seed": seed,
                "n_pairs": len(shares),
                "stability": float(np.mean(shares)) if shares else np.nan,
            }
        )
    return (
        pd.DataFrame(out).groupby("arm")[["n_pairs", "stability"]].mean().reset_index()
    )


def friedman_nemenyi(
    per_step: pd.DataFrame, metric: str = "f1", alpha: float = 0.05
) -> dict:
    """Friedman test across arms over the steps (blocks), with average ranks
    (1 = best) and the Nemenyi critical difference (Demšar 2006)."""
    m = (
        per_step_means(per_step)
        .pivot(index="step_i", columns="arm", values=metric)
        .dropna()
    )
    k, n = m.shape[1], m.shape[0]
    if k < 2 or n < 2:
        return {"metric": metric, "n_steps": n, "n_arms": k, "status": "too few"}
    ranks = m.rank(axis=1, ascending=False).mean()
    stat, p = (
        sps.friedmanchisquare(*[m[c] for c in m.columns])
        if k >= 3
        else (np.nan, np.nan)
    )
    q = sps.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return {
        "metric": metric,
        "n_steps": int(n),
        "n_arms": int(k),
        "friedman_chi2": float(stat),
        "friedman_p": float(p),
        "avg_rank": {a: float(r) for a, r in ranks.sort_values().items()},
        "nemenyi_cd": float(cd),
        "alpha": alpha,
    }


def summarize_comparison(per_step: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: mean and std over steps of the seed-averaged metrics."""
    means = per_step_means(per_step)
    cols = [c for c in SUMMARY_METRICS if c in means]
    agg = means.groupby("arm")[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    pooled = per_step.groupby("arm")[
        ["tp_seen", "n_attack_seen", "tp_unseen", "n_attack_unseen"]
    ].sum()
    agg["recall_seen_pooled"] = pooled["tp_seen"] / pooled["n_attack_seen"]
    agg["recall_unseen_pooled"] = pooled["tp_unseen"] / pooled[
        "n_attack_unseen"
    ].replace(0, np.nan)
    agg.insert(0, "n_steps", means.groupby("arm")["step_i"].nunique())
    return agg.reset_index()


def write_summary(
    out_dir: Path, per_step: pd.DataFrame, features: pd.DataFrame | None
) -> None:
    if per_step.empty:
        return
    summary = summarize_comparison(per_step)
    hh = pd.concat(
        [helps_hurts(per_step, m) for m in ("f1", "pr_auc")], ignore_index=True
    )
    stab = stability(features)
    tests = {m: friedman_nemenyi(per_step, m) for m in ("f1", "pr_auc")}
    summary.to_csv(out_dir / "summary.csv", index=False)
    hh.to_csv(out_dir / "helps_hurts.csv", index=False)
    stab.to_csv(out_dir / "stability.csv", index=False)
    (out_dir / "stats.json").write_text(json.dumps(tests, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        text = "\n\n".join(
            [
                summary.to_string(index=False),
                hh.to_string(index=False),
                stab.to_string(index=False),
                json.dumps(tests, indent=2),
            ]
        )
    (out_dir / "summary.txt").write_text(text + "\n")
    print(text)

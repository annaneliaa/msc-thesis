"""
Mining comparison (paper plan 7.6): walk-forward comparison of the mining
methods, every method under the same treatment. Two datasets:

CSCAS (atemporal arm: eclat, contrast_tree, ripper):
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
    dedupe (Jaccard >= 0.95) and budget K -- `mine_candidates` + `apply_budget`, the same code
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

AIT-ADS (temporal arm: prefixspan, mctsextent, plus the atemporal methods on
the same steps, so Eclat vs. PrefixSpan isolates order): cscas_grouping, token
arm T2 for every method. Two protocols:

  - "loso" (default, decided 2026-11-16): leave-one-scenario-out. Fold i
    trains on the other scenarios' labelled groups and tests on scenario i's
    whole timeline. The scenarios run the same multi-step attack chain on
    different networks, so every stage of the held-out chain has been seen as
    an attack elsewhere. Look-back steps come from each group's own scenario.
  - "phase": within one scenario, one step per attack phase after the first
    (train on everything before the phase, test up to the next phase; cut
    halfway between phases). Kept for reference: later phases are different
    attack stages, so this tests transfer between stages, not between methods
    (FINDINGS.md). A rolling 5% walk-forward is not an option on AIT-ADS: the
    1-3 attack phases per scenario are < ~4 h each, so its windows almost
    never hold attacks on both sides (the collapsed cells of thesis §5.6).

Two stages (optional, with a mining cache dir): mining is split from
training. Stage 1 ("mine") runs mine -> gate -> condition dedupe -> firing
dedupe per (step, method, mining seed) and caches the ranked candidates,
keyed by the exact training rows and every mining setting except the budget.
Stage 2 ("train") loads them, applies the budget K, encodes and fits the RF.
So heavy miners can run as separate jobs, a crashed run resumes from what
was mined, and K / model variants are training-only reruns. "all" (the
default) does both, mining only what isn't cached. On AIT-ADS the per-scenario
look-back steps are cached on disk as well (CACHE_DIR/<scenario>/lookback/).

Method settings are the mini experiment's documented defaults, not tuned
(the tuning criterion is still open, see FINDINGS.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import average_precision_score

from typing import Callable, Iterator

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
from thesis.schemas.features import (
    SEQUENCE_MINING_TYPES,
    BaseFeatureSchema,
    FeatureSchema,
)
from thesis.schemas.groups import AlertGroup
from thesis.system_eval.mining_mini_experiment import (
    COMPARISON_TOKEN_ARM,
    METHODS,
    TEMPORAL,
    TOKEN_ARMS,
    MinedCandidates,
    MiniConfig,
    MiniWindow,
    _ait_signature,
    _labels,
    _repo_commit,
    apply_budget,
    labelled,
    lookback_attacher,
    lookback_token_fields,
    mine_candidates,
)

ATEMPORAL_METHODS = ("eclat", "contrast_tree", "ripper")
# AIT-ADS runs every method: the temporal pair plus the atemporal ones on the
# same steps (Eclat vs. PrefixSpan = atemporal vs. temporal, plan section 1).
AIT_ADS_METHODS = METHODS
AIT_ADS_SCENARIOS = (
    "fox",
    "harrison",
    "russellmitchell",
    "santos",
    "shaw",
    "wardbeck",
    "wheeler",
    "wilson",
)
# Attack groups more than this far apart belong to different attack phases.
# Every AIT-ADS phase lasts < ~4 h and phases are >= 10 h apart, so the phase
# count is the same for any gap between 1 h and 4 h.
PHASE_GAP_SECONDS = 3600
STOCHASTIC_MINERS = {"ripper", "mctsextent"}
NO_MINING = "none"
# Budget K for the comparison (decided 2026-11-11): about what the contrast
# tree keeps after gate + dedupe on CSCAS (55-58 on the first two steps), so
# the cap doesn't cut the thesis method down. Robustness: K in {10, 25, 50}.
COMPARISON_K = 60


@dataclass
class ComparisonConfig:
    dataset: str = "cscas"  # "cscas" | "ait_ads"
    gran: float = 0.05  # CSCAS walk-forward granularity
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    methods: tuple[str, ...] = ATEMPORAL_METHODS
    include_no_mining: bool = True
    # Per-method settings, gate, dedupe and budget K (k, per_class_budget).
    mining: MiniConfig = field(default_factory=lambda: MiniConfig(k=COMPARISON_K))
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
    # AIT-ADS only.
    scenarios: tuple[str, ...] = AIT_ADS_SCENARIOS
    grouping: str = "cscas_grouping"
    token_arm: str = COMPARISON_TOKEN_ARM
    phase_gap_seconds: int = PHASE_GAP_SECONDS
    ait_ads_protocol: str = "loso"  # "loso" | "phase"
    # LOSO: the held-out scenarios to run folds for (None = every scenario in
    # `scenarios`); training always pools all the other `scenarios`.
    test_scenarios: tuple[str, ...] | None = None
    # Two-stage runs (see run_comparison): "all" mines what isn't cached and
    # trains; "mine" only mines and caches; "train" only loads cached mining
    # output (error if missing) and trains; "prepare" only builds the
    # look-back cache (AIT-ADS).
    stage: str = "all"
    # Cache of each (step, method, mining seed)'s gated + deduped candidates,
    # before the budget K. None = no cache (mine in memory, as before).
    mining_cache_dir: Path | None = None
    # AIT-ADS: load look-back steps from CACHE_DIR/<scenario>/lookback/.
    use_lookback_cache: bool = True


@dataclass
class Step:
    """One walk-forward step: labelled mining/training rows and test rows."""

    i: int
    train: list[AlertGroup]
    test: list[AlertGroup]
    test_days: float
    scenario: str = "cscas"
    phase: int | None = None  # AIT-ADS: index of the attack phase tested


@dataclass
class DatasetSpec:
    """What a step's mining window needs beyond its rows: base features,
    signature identity, item-table fields, gate side and look-back steps."""

    dataset: str  # "cscas" | "ait_ads"
    base_schema: FeatureSchema
    # group -> the signature identity used for seen/unseen recall
    signature_of: Callable[[AlertGroup], object]
    exclude_fields: set[str] = field(default_factory=set)
    include_fields: set[str] = field(default_factory=set)
    benign_gate_g: float | None = None
    # Sets context_steps on groups (temporal methods); None = not available.
    attach_steps: Callable[[list[AlertGroup]], None] | None = None


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


# Bump when anything that changes a method's mined candidates changes in code
# (miners, gate, dedupes, item table, look-back steps) -- the key below only
# covers data and settings.
MINING_CACHE_VERSION = 2  # 2: mine_seconds covers mine + gate + dedupes
# Budget-only settings: the cached candidates come before the budget, so these
# don't enter the key (one cache serves every K).
_BUDGET_FIELDS = {"k", "per_class_budget", "methods", "mining_settings_path"}


def _sha1(obj) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def mining_cache_key(
    method: str, step: Step, spec: DatasetSpec, mini: MiniConfig, token_arm: str
) -> str:
    """What a method's mined candidates depend on: the exact training rows
    (in order), the method and its settings (incl. the mining seed), the
    item-table fields, the gate and the token arm. Not the budget K."""
    settings = {k: v for k, v in asdict(mini).items() if k not in _BUDGET_FIELDS}
    settings_file = Path(mini.mining_settings_path)
    return _sha1(
        {
            "version": MINING_CACHE_VERSION,
            "dataset": spec.dataset,
            "train": _sha1([g.group_id for g in step.train]),
            "n_train": len(step.train),
            "method": method,
            "settings": settings,
            "settings_file": (
                hashlib.sha1(settings_file.read_bytes()).hexdigest()
                if settings_file.exists()
                else None
            ),
            "exclude": sorted(spec.exclude_fields),
            "include": sorted(spec.include_fields),
            "benign_gate_g": spec.benign_gate_g,
            "token_arm": token_arm if spec.dataset == "ait_ads" else None,
        }
    )


def get_candidates(
    config: ComparisonConfig,
    spec: DatasetSpec,
    step: Step,
    window: MiniWindow,
    method: str,
    mini: MiniConfig,
    table,
) -> tuple[MinedCandidates, bool]:
    """The method's mined candidates for this step and mining seed: from the
    mining cache when present, else mined (and cached when a cache dir is
    set). stage "train" refuses to mine. Returns (candidates, cache hit)."""
    path = None
    if config.mining_cache_dir is not None:
        key = mining_cache_key(method, step, spec, mini, config.token_arm)
        path = Path(config.mining_cache_dir) / method / f"{key}.pkl"
        if path.exists():
            with path.open("rb") as f:
                payload = pickle.load(f)
            return (
                MinedCandidates(
                    None,
                    payload["stages"],
                    payload["candidates"],
                    payload["mine_seconds"],
                    payload["counts"],
                ),
                True,
            )
    if config.stage == "train":
        raise FileNotFoundError(
            f"stage 'train': no cached mining output for {method} on "
            f"{step.scenario} step {step.i} (seed {mini.seed}) at {path}; "
            "run --stage mine first"
        )
    mined = mine_candidates(method, window, mini, table)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        with tmp.open("wb") as f:
            pickle.dump(
                {
                    "candidates": mined.candidates,
                    "stages": mined.stages,
                    "counts": mined.counts,
                    "mine_seconds": mined.mine_seconds,
                    "meta": {
                        "method": method,
                        "scenario": step.scenario,
                        "step_i": step.i,
                        "mining_seed": mini.seed,
                        "repo_commit": _repo_commit(),
                        "written_at": datetime.now(timezone.utc).isoformat(),
                    },
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp.replace(path)
    return mined, False


def mine_arm(
    method: str, step: Step, mined: MinedCandidates, mini: MiniConfig, table
) -> MinedArm:
    """Budget K over the method's candidates, then encode its features."""
    sel = apply_budget(mined, mini.k, per_class=mini.per_class_budget)
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
    config: ComparisonConfig, step: Step, spec: DatasetSpec
) -> tuple[list[dict], list[pd.DataFrame]]:
    """All arms x seeds on one step."""
    base_schema, signature_of = spec.base_schema, spec.signature_of
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
        "scenario": step.scenario,
        "phase": step.phase,
        "step_i": step.i,
        "n_train": len(y_tr),
        "n_train_attack": int(y_tr.sum()),
        "n_test": len(y_te),
        "n_test_attack": int(y_te.sum()),
        "n_base": Xb_tr.shape[1],
    }
    rows: list[dict] = []
    feats: list[pd.DataFrame] = []

    if config.include_no_mining and config.stage != "mine":
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
        spec.dataset,
        f"{step.scenario} step {step.i}",
        step.train,
        step.test,
        signature_of=signature_of,
        exclude_fields=spec.exclude_fields,
        include_fields=spec.include_fields,
        token_arm=config.token_arm if spec.dataset == "ait_ads" else "T0",
        has_raw_alerts=spec.attach_steps is not None,
        benign_gate_g=spec.benign_gate_g,
    )
    table = None
    if any(m not in TEMPORAL for m in config.methods):
        t0 = time.perf_counter()
        table = build_item_table(
            step.train,
            exclude_fields=spec.exclude_fields,
            n_bins=config.mining.n_bins,
            include_fields=spec.include_fields,
        )
        print(f"  item table {table.items().shape} in {time.perf_counter() - t0:.1f}s")
    if any(m in TEMPORAL for m in config.methods):
        if spec.attach_steps is None:
            raise ValueError(f"{spec.dataset} has no raw alerts for sequences")
        t0 = time.perf_counter()
        spec.attach_steps(step.train + step.test)
        print(f"  look-back steps in {time.perf_counter() - t0:.1f}s")

    if config.stage == "mine":
        for method in config.methods:
            reseed = config.reseed_stochastic_miners and method in STOCHASTIC_MINERS
            for seed in config.seeds if reseed else config.seeds[:1]:
                cand, hit = get_candidates(
                    config,
                    spec,
                    step,
                    window,
                    method,
                    replace(config.mining, seed=seed),
                    table,
                )
                print(
                    f"  step {step.i:2d} {method:<13} seed {seed}: "
                    f"{'cached' if hit else 'mined'}, {len(cand.candidates)} "
                    f"candidates, {cand.mine_seconds:.1f}s",
                    flush=True,
                )
        return [], []

    for method in config.methods:
        reseed = config.reseed_stochastic_miners and method in STOCHASTIC_MINERS
        mined: MinedArm | None = None
        for seed in config.seeds:
            if mined is None or reseed:
                mini = replace(config.mining, seed=seed)
                cand, _ = get_candidates(
                    config, spec, step, window, method, mini, table
                )
                mined = mine_arm(method, step, cand, mini, table)
                if not mined.features.empty:
                    f = mined.features.copy()
                    f.insert(0, "arm", method)
                    f.insert(1, "mining_seed", seed if reseed else config.seeds[0])
                    f.insert(2, "scenario", step.scenario)
                    f.insert(3, "step_i", step.i)
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


def cscas_steps(
    config: ComparisonConfig,
) -> tuple[Callable[[], Iterator[tuple[Step, DatasetSpec]]], pd.DataFrame]:
    """The CSCAS walk-forward steps (same windows as Checks 1-3), each with
    the dataset spec (base schema, SignatureID lookup, leaky fields)."""
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
    spec = DatasetSpec(
        dataset="cscas",
        base_schema=ctx.base_schema,
        signature_of=lambda g: lookup.get(g.group_id),
        exclude_fields=set(default_leaky_attribute_fields("cscas"))
        | set(config.extra_exclude_fields),
    )

    def steps():
        for i in range(n):
            s, e = bounds[i]
            ts, te = scheme.target_bounds(config.gran, i + 1)
            yield (
                Step(
                    i,
                    labelled(groups[s:e]),
                    labelled(groups[ts:te]),
                    _window_days(groups[ts:te]),
                ),
                spec,
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
    return steps, windows


# --- AIT-ADS attack-phase walk --------------------------------------------


def attack_phases(groups: list[AlertGroup], gap_seconds: int) -> list[tuple[int, int]]:
    """(first, last) start_ts of each attack phase: runs of attack groups with
    no gap longer than gap_seconds. `groups` sorted by start_ts."""
    ts = [g.start_ts for g in groups if g.group_label == "attack"]
    phases: list[tuple[int, int]] = []
    for t in ts:
        if phases and t - phases[-1][1] <= gap_seconds:
            phases[-1] = (phases[-1][0], t)
        else:
            phases.append((t, t))
    return phases


def phase_cuts(phases: list[tuple[int, int]]) -> list[float]:
    """The split time before each phase after the first: halfway between the
    end of the previous phase and the start of this one."""
    return [(phases[p - 1][1] + phases[p][0]) / 2 for p in range(1, len(phases))]


def phase_split(
    groups: list[AlertGroup], gap_seconds: int
) -> list[tuple[int, list[AlertGroup], list[AlertGroup]]]:
    """(phase, train, test) per attack phase after the first: train = every
    group before the cut, test = the cut up to the next phase's cut (or the
    end). `groups` labelled and sorted by start_ts."""
    cuts = phase_cuts(attack_phases(groups, gap_seconds))
    out = []
    for p, cut in enumerate(cuts, start=1):
        end = cuts[p] if p < len(cuts) else np.inf
        train = [g for g in groups if g.start_ts < cut]
        test = [g for g in groups if cut <= g.start_ts < end]
        out.append((p, train, test))
    return out


def ait_ads_base_schema() -> FeatureSchema:
    from thesis.configs import load_base_features

    return FeatureSchema(
        schema_name="base",
        schema_version="0.1.0",
        base=BaseFeatureSchema(load_base_features("ait-ads")),
        symbolic=None,
    )


def _ait_ads_spec(config: ComparisonConfig, scenarios, attach_steps) -> DatasetSpec:
    include, exclude = TOKEN_ARMS[config.token_arm]
    leaky = set().union(*(default_leaky_attribute_fields(s) for s in scenarios))
    return DatasetSpec(
        dataset="ait_ads",
        base_schema=ait_ads_base_schema(),
        # cscas_grouping groups hold one alert name, like a CSCAS
        # SignatureID; other groupings fall back to the group id.
        signature_of=_ait_signature,
        exclude_fields=leaky | set(exclude) | set(config.extra_exclude_fields),
        include_fields=set(include),
        # Background alerts inside attack phases carry the attack label, so
        # benign rules only need attack rate <= pi (same as the mini
        # experiment and the grouping x token grid).
        benign_gate_g=1.0,
        attach_steps=attach_steps,
    )


def _load_ait_ads_groups(config: ComparisonConfig, scenario: str) -> list[AlertGroup]:
    from thesis.paths import CACHE_DIR
    from thesis.pipeline.pipeline import load_or_build_alert_groups

    groups = load_or_build_alert_groups(
        scenario, CACHE_DIR / scenario / "groups" / config.grouping
    )
    return labelled(sorted(groups, key=lambda g: g.start_ts))


def loso_folds(
    groups_by_scenario: dict[str, list[AlertGroup]],
    test_scenarios: tuple[str, ...] | None = None,
) -> list[tuple[str, list[AlertGroup], list[AlertGroup]]]:
    """(held-out scenario, train, test) per fold: train pools every other
    scenario, test is the held-out scenario's whole timeline."""
    held_out = test_scenarios or tuple(groups_by_scenario)
    unknown = set(held_out) - set(groups_by_scenario)
    if unknown:
        raise ValueError(f"test scenarios not in the pool: {sorted(unknown)}")
    return [
        (
            s,
            [g for t, gs in groups_by_scenario.items() if t != s for g in gs],
            groups_by_scenario[s],
        )
        for s in held_out
    ]


def attach_lookback(
    config: ComparisonConfig, scenario: str, groups: list[AlertGroup]
) -> None:
    """Look-back steps on one scenario's groups: from the on-disk cache when
    use_lookback_cache (building and writing it on a miss), else built from
    the raw alerts. Frees the raw alerts and host index afterwards."""
    from thesis.mining.temporal.lookback import (
        attach_cached_lookback_steps,
        load_ait_ads_raw_alerts,
        lookback_cache_path,
    )
    from thesis.paths import CACHE_DIR
    from thesis.system_eval import mining_mini_experiment as mme

    t0 = time.perf_counter()
    build = lookback_attacher(
        scenario, config.token_arm, config.mining.lookback_minutes
    )
    hit = False
    if config.use_lookback_cache:
        path = lookback_cache_path(
            CACHE_DIR,
            scenario,
            config.grouping,
            lookback_token_fields(config.token_arm),
            config.mining.lookback_minutes,
        )
        hit = attach_cached_lookback_steps(groups, scenario, path, build)
    else:
        build(groups)
    mme._LOOKBACK_INDEX.clear()
    load_ait_ads_raw_alerts.cache_clear()
    print(
        f"  look-back steps {scenario} {'from cache' if hit else 'built'} "
        f"in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )


def prepare_lookback(config: ComparisonConfig) -> None:
    """Stage "prepare": build the look-back cache for every scenario, so
    parallel jobs only ever read it."""
    for s in config.scenarios:
        attach_lookback(config, s, _load_ait_ads_groups(config, s))


def ait_ads_loso_steps(
    config: ComparisonConfig,
) -> tuple[Callable[[], Iterator[tuple[Step, DatasetSpec]]], pd.DataFrame]:
    """Leave-one-scenario-out folds over config.scenarios. Every scenario's
    groups stay in memory; their look-back steps are attached once up front,
    one scenario's raw alerts at a time, so folds share them."""
    groups_by_scenario = {s: _load_ait_ads_groups(config, s) for s in config.scenarios}
    folds = loso_folds(groups_by_scenario, config.test_scenarios)
    if config.max_steps is not None:
        folds = folds[: config.max_steps]
    windows = pd.DataFrame(
        [
            {
                "step_i": i,
                "scenario": s,
                "train_scenarios": " ".join(t for t in groups_by_scenario if t != s),
                "n_train": len(tr),
                "n_train_attack": int(_labels(tr).sum()),
                "n_test": len(te),
                "n_test_attack": int(_labels(te).sum()),
                "test_days": _window_days(te),
            }
            for i, (s, tr, te) in enumerate(folds)
        ]
    )

    def steps():
        if any(m in TEMPORAL for m in config.methods):
            for s, groups in groups_by_scenario.items():
                attach_lookback(config, s, groups)
        # Steps are already on every group; run_step's attach call is a no-op.
        spec = _ait_ads_spec(config, groups_by_scenario, lambda groups: None)
        for i, (s, tr, te) in enumerate(folds):
            yield Step(i, tr, te, _window_days(te), scenario=s), spec

    return steps, windows


def ait_ads_steps(
    config: ComparisonConfig,
) -> tuple[Callable[[], Iterator[tuple[Step, DatasetSpec]]], pd.DataFrame]:
    """The AIT-ADS attack-phase steps over config.scenarios, numbered across
    scenarios. Loads each scenario's groups twice (once here for the window
    table, once lazily in the generator) to keep only one scenario in memory."""
    from thesis.mining.temporal.lookback import load_ait_ads_raw_alerts
    from thesis.system_eval import mining_mini_experiment as mme

    def load(scenario: str) -> list[AlertGroup]:
        return _load_ait_ads_groups(config, scenario)

    plan: list[dict] = []
    for scenario in config.scenarios:
        groups = load(scenario)
        n_phases = len(attack_phases(groups, config.phase_gap_seconds))
        split = phase_split(groups, config.phase_gap_seconds)
        for p, train, test in split:
            plan.append(
                {
                    "step_i": len(plan),
                    "scenario": scenario,
                    "phase": p,
                    "n_phases": n_phases,
                    "train_end_ts": test[0].start_ts if test else np.nan,
                    "test_end_ts": test[-1].start_ts if test else np.nan,
                    "n_train": len(train),
                    "n_train_attack": int(_labels(train).sum()),
                    "n_test": len(test),
                    "n_test_attack": int(_labels(test).sum()),
                }
            )
        if not split:
            print(f"  [skip] {scenario}: {n_phases} attack phase(s), no step")
    if config.max_steps is not None:
        plan = plan[: config.max_steps]
    windows = pd.DataFrame(plan)

    def steps():
        for scenario in dict.fromkeys(r["scenario"] for r in plan):
            groups = load(scenario)
            spec = _ait_ads_spec(
                config,
                [scenario],
                lookback_attacher(
                    scenario, config.token_arm, config.mining.lookback_minutes
                ),
            )
            split = {
                p: (tr, te)
                for p, tr, te in phase_split(groups, config.phase_gap_seconds)
            }
            for r in (r for r in plan if r["scenario"] == scenario):
                tr, te = split[r["phase"]]
                yield (
                    Step(
                        r["step_i"],
                        tr,
                        te,
                        _window_days(te),
                        scenario=scenario,
                        phase=r["phase"],
                    ),
                    spec,
                )
            # One scenario's raw alerts and per-host index at a time.
            mme._LOOKBACK_INDEX.clear()
            load_ait_ads_raw_alerts.cache_clear()

    return steps, windows


def default_out_dir(experiment: str = "mining_comparison") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "artifacts" / "experiments" / experiment / ts


def run_comparison(config: ComparisonConfig) -> Path:
    """Walk-forward over every step x arm x seed (CSCAS or AIT-ADS). Writes
    per_step_results.csv and features.csv (rewritten after every step, so a
    crashed run keeps what it finished), windows.csv, config.json, and at the
    end summary.csv, helps_hurts.csv, stability.csv and stats.json."""
    allowed = ATEMPORAL_METHODS if config.dataset == "cscas" else AIT_ADS_METHODS
    if config.dataset not in ("cscas", "ait_ads"):
        raise ValueError(f"unknown dataset {config.dataset!r}")
    unknown = set(config.methods) - set(allowed)
    if unknown:
        raise ValueError(
            f"{config.dataset} runs {list(allowed)}; got {sorted(unknown)}"
        )
    if config.stage not in ("all", "mine", "train", "prepare"):
        raise ValueError(f"unknown stage {config.stage!r}")
    if config.stage in ("mine", "train") and config.mining_cache_dir is None:
        raise ValueError(f"stage {config.stage!r} needs a mining_cache_dir")
    if config.stage == "prepare":
        if config.dataset != "ait_ads":
            raise ValueError("stage 'prepare' builds the AIT-ADS look-back cache")
        from thesis.paths import CACHE_DIR

        prepare_lookback(config)
        return Path(CACHE_DIR)
    out_dir = (
        Path(config.out_dir) if config.out_dir else default_out_dir(config.experiment)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {**asdict(config), "repo_commit": _repo_commit()}, indent=2, default=str
        )
    )
    steps, windows = (
        cscas_steps(config)
        if config.dataset == "cscas"
        else ait_ads_loso_steps(config)
        if config.ait_ads_protocol == "loso"
        else ait_ads_steps(config)
    )
    windows.to_csv(out_dir / "windows.csv", index=False)
    print(
        f"Output → {out_dir}  ({len(windows)} steps x "
        f"{len(config.methods) + config.include_no_mining} arms x "
        f"{len(config.seeds)} seeds)"
    )

    rows: list[dict] = []
    feats: list[pd.DataFrame] = []
    for step, spec in steps():
        print(
            f"[step {step.i} {step.scenario}] train {len(step.train)} "
            f"({int(_labels(step.train).sum())} attack), test {len(step.test)}"
        )
        if len(set(_labels(step.train))) < 2:
            print("  [warn] single-class training window -- skipped")
            continue
        r, f = run_step(config, step, spec)
        rows += r
        feats += f
        if config.stage == "mine":
            continue
        pd.DataFrame(rows).to_csv(out_dir / "per_step_results.csv", index=False)
        if feats:
            pd.concat(feats, ignore_index=True).to_csv(
                out_dir / "features.csv", index=False
            )

    if config.stage == "mine":
        print(f"Mining cached → {config.mining_cache_dir}")
        return out_dir
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
    "miner_seconds",
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
    reappear in step i+1's kept features, within one scenario (AIT-ADS steps
    of different scenarios are not consecutive)."""
    if features is None or features.empty:
        return pd.DataFrame()
    if "scenario" not in features:
        features = features.assign(scenario="cscas")
    out = []
    for (arm, seed), g in features.groupby(["arm", "mining_seed"]):
        shares = []
        for _, gs in g.groupby("scenario"):
            by_step = gs.groupby("step_i")["condition"].apply(set).sort_index()
            steps = list(by_step.index)
            shares += [
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


# --- Merging per-fold runs --------------------------------------------------


def merge_runs(run_dirs: list[Path], out_dir: Path) -> Path:
    """Merge separate runs (e.g. one LOSO fold per process) into one run
    folder and recompute the summary, helps/hurts and statistics on the
    whole. Each run numbers its own steps from 0, so steps are renumbered
    by (run order, original step). Writes per_step_results.csv,
    features.csv, windows.csv, sources.json and the summary files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_step, feats, windows, sources = [], [], [], []
    offset = 0
    for d in map(Path, run_dirs):
        ps_path = d / "per_step_results.csv"
        if not ps_path.exists():
            raise FileNotFoundError(f"{d}: no per_step_results.csv (run unfinished?)")
        ps = pd.read_csv(ps_path)
        steps = sorted(ps["step_i"].unique())
        remap = {s: offset + k for k, s in enumerate(steps)}
        per_step.append(ps.assign(step_i=ps["step_i"].map(remap), source_run=d.name))
        if (d / "features.csv").exists():
            f = pd.read_csv(d / "features.csv")
            feats.append(f.assign(step_i=f["step_i"].map(remap)))
        if (d / "windows.csv").exists():
            w = pd.read_csv(d / "windows.csv")
            windows.append(w.assign(step_i=w["step_i"].map(remap), source_run=d.name))
        cfg = d / "config.json"
        sources.append(
            {
                "run": str(d),
                "steps": {int(k): int(v) for k, v in remap.items()},
                "config": json.loads(cfg.read_text()) if cfg.exists() else None,
            }
        )
        offset += len(steps)
    per_step_df = pd.concat(per_step, ignore_index=True)
    feats_df = pd.concat(feats, ignore_index=True) if feats else None
    per_step_df.to_csv(out_dir / "per_step_results.csv", index=False)
    if feats_df is not None:
        feats_df.to_csv(out_dir / "features.csv", index=False)
    if windows:
        pd.concat(windows, ignore_index=True).to_csv(
            out_dir / "windows.csv", index=False
        )
    (out_dir / "sources.json").write_text(json.dumps(sources, indent=2, default=str))
    write_summary(out_dir, per_step_df, feats_df)
    return out_dir

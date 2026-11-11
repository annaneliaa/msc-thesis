"""
Mining comparison, Phase 4: one mini experiment per method on both datasets.

Purpose: confirm that every mining implementation produces usable features
on the real data *before* the walk-forward comparison, i.e. that nothing like
the earlier "Eclat doesn't work on CSCAS" (single-signature restatements) or
the dead contrast columns slips through. Not a performance comparison: no
model is trained.

Windows (one mining window, one test window, per dataset):
  - CSCAS: W0 -> W1 of the paper's walk-forward (baseline_split scheme,
    gran 0.05; same windows as Checks 1-3).
  - AIT-ADS: one scenario under cscas_grouping, chronological split
    (first train_frac of groups -> rest), the AIT-ADS baselines' convention.

Methods: eclat, contrast_tree, ripper (atemporal, on the shared item table)
and prefixspan, mctsextent (temporal, on per-host look-back sequences).
Temporal methods on CSCAS are recorded as not applicable: CSCAS ships
pre-grouped rows with no raw per-alert stream, so there is no sequence to
build (the paper states this as the reason the temporal arm is AIT-ADS only).

Per (method, dataset) it records: pattern counts raw / after the precision
gate (g = 2 on the mining window) / after the budget K; pattern length; how
many budgeted features ever fire on the test window ("live"); test attack and
benign coverage; test attack rate of attack-leaning features; the share of
features that are single-signature restatements (fire on groups of exactly
one signature in the mining window); runtime. Pass criterion (PLAN.md
Phase 4): for each class, at least one budgeted feature that fires on the
test window and is not a single-signature restatement.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from thesis.config import load_mining_settings
from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.mining.atemporal.contrast_tree.miner import mine_contrast_tree
from thesis.mining.atemporal.eclat.miner import mine_eclat
from thesis.mining.atemporal.ripper.miner import mine_ripper
from thesis.mining.common.attribute_features import default_leaky_attribute_fields
from thesis.mining.common.item_table import ItemTable, build_item_table
from thesis.mining.common.patterns import (
    add_pattern_stats,
    dedupe_conditions,
    patterns_to_schema,
    select_budget,
)
from thesis.mining.temporal.lookback import (
    DEFAULT_WINDOW_MINUTES,
    HostAlertIndex,
    attach_lookback_steps,
    load_ait_ads_raw_alerts,
)
from thesis.mining.temporal.mctsextent.miner import mine_mctsextent
from thesis.mining.temporal.prefixspan.miner import mine_prefixspan
from thesis.paths import ROOT
from thesis.schemas.groups import AlertGroup
from thesis.schemas.mining import (
    AttributeMiningConfig,
    ItemsetFilterConfig,
    SequenceFilterConfig,
)
from thesis.system_eval.rule_precision_gate import MAIN_G, passes_gate

METHODS = ("eclat", "contrast_tree", "ripper", "prefixspan", "mctsextent")
TEMPORAL = {"prefixspan", "mctsextent"}
TOP_N = 10


@dataclass
class MiniWindow:
    """One dataset's mining window + test window, labelled groups only."""

    dataset: str  # "cscas" | "ait_ads"
    name: str  # e.g. "cscas W0->W1", "fox 70/30"
    train: list[AlertGroup]
    test: list[AlertGroup]
    # group -> its signature(s), for the restatement check: one id (CSCAS
    # SignatureID) or a set of ids (AIT-ADS: the group's short codes).
    signature_of: Callable[[AlertGroup], object]
    exclude_fields: set[str] = field(default_factory=set)
    # Opt-in candidate fields (coarse AIT-ADS token levels, Phase 4b).
    include_fields: set[str] = field(default_factory=set)
    # Optional second identity reported alongside (cscas_grouping: alert name).
    name_signature_of: Callable[[AlertGroup], object] | None = None
    grouping: str = ""
    token_arm: str = "T0"
    has_raw_alerts: bool = False
    # Benign side of the precision gate: benign rules need attack rate
    # <= pi / benign_gate_g. None = same g as the attack side.
    benign_gate_g: float | None = None
    # Sets context_steps on the given groups for this window's token arm.
    # Called by run_mini before the temporal methods, because windows of
    # different arms share the same group objects. None = steps already set.
    attach_steps: Callable[[list[AlertGroup]], None] | None = None


@dataclass
class MiniConfig:
    k: int = 50
    per_class_budget: bool = True
    gate_g: float = MAIN_G
    n_bins: int = 5
    seed: int = 0
    lookback_minutes: int = DEFAULT_WINDOW_MINUTES
    # Per-method settings (plan: sensible, documented, not tuned here).
    eclat_min_support: float = 0.05
    eclat_max_len: int = 3
    prefixspan_min_support: float = 0.05
    prefixspan_max_len: int = 3
    prefixspan_min_k: int = 2  # >= 2 so patterns actually use order
    mctsextent_top_k: int = 25
    mctsextent_iterations: int = 1000
    # Eclat's absolute floor on a pattern's count in its own class
    # (ItemsetFilterConfig.min_support_count). An int is used as is (10 = the
    # existing filter configs); "relative" tunes it to the mining window:
    # max(3, ceil(eclat_min_support * n_attack_train)).
    eclat_min_support_count: int | str = 10
    # Count features that fire on exactly the same mining-window groups once
    # (e.g. {short=A-Acc-Chr1} and {short=A-Acc-Chr1, short_cat=A-Acc}).
    dedupe_identical_firing: bool = True  # firing dedupe on/off
    # Firing dedupe threshold: drop a feature whose mining-window firing has
    # Jaccard >= this with a higher-scoring kept one (1.0 = exact copies only).
    dedupe_jaccard: float = 0.95
    # Equivalence restatement: Jaccard(feature's groups, groups with s) >= this.
    equivalence_jaccard: float = 0.9
    contrast_tree_setting: str = "gr3_md1_mda4_rounds2"
    mining_settings_path: Path = (
        ROOT / "src/thesis/configs/monitor_eda_mining_setting.yaml"
    )
    methods: tuple[str, ...] = METHODS


def labelled(groups: list[AlertGroup]) -> list[AlertGroup]:
    return [g for g in groups if g.group_label in ("attack", "benign")]


def _labels(groups: list[AlertGroup]) -> np.ndarray:
    return np.array([g.group_label == "attack" for g in groups])


def contrast_tree_config(config: MiniConfig) -> AttributeMiningConfig:
    specs = {s.name: s for s in load_mining_settings(config.mining_settings_path)}
    return specs[config.contrast_tree_setting].to_attribute_mining_config()


def mine(
    method: str, window: MiniWindow, table: ItemTable | None, config: MiniConfig
) -> pd.DataFrame:
    if method == "eclat":
        return mine_eclat(
            table,
            min_support=config.eclat_min_support,
            max_len=config.eclat_max_len,
            filters=ItemsetFilterConfig(
                min_support_count=eclat_count_floor(config, table)
            ),
        )
    if method == "contrast_tree":
        return mine_contrast_tree(table, contrast_tree_config(config))
    if method == "ripper":
        return mine_ripper(table, random_state=config.seed)
    if method == "prefixspan":
        return mine_prefixspan(
            window.train,
            min_support=config.prefixspan_min_support,
            max_len=config.prefixspan_max_len,
            filters=SequenceFilterConfig(min_k=config.prefixspan_min_k),
        )
    if method == "mctsextent":
        return mine_mctsextent(
            window.train,
            top_k=config.mctsextent_top_k,
            iterations=config.mctsextent_iterations,
            random_state=config.seed,
        )
    raise ValueError(f"unknown method {method!r}")


def eclat_count_floor(config: MiniConfig, table: ItemTable) -> int:
    if config.eclat_min_support_count == "relative":
        return max(3, math.ceil(config.eclat_min_support * table.n_attack))
    return int(config.eclat_min_support_count)


def _as_set(sig) -> frozenset:
    if isinstance(sig, (set, frozenset, list, tuple)) and not isinstance(sig, str):
        # tuple signatures (e.g. a composite key) are one id, not a set
        return frozenset(sig) if not isinstance(sig, tuple) else frozenset([sig])
    return frozenset([sig])


def restatement_flags(
    X_train: pd.DataFrame, train: list[AlertGroup], signature_of
) -> pd.Series:
    """Per feature: True if it restates one signature, i.e. there is a
    signature s that every mining-window group it fires on contains, so the
    feature can never fire outside s. For single-signature groups (CSCAS,
    AIT-ADS cscas_grouping) that is "every group it fires on has the same
    signature"; for multi-signature groups it generalises to "all of them
    contain s". A feature that never fires is not a restatement."""
    sigs = [_as_set(signature_of(g)) for g in train]
    out = {}
    for col in X_train.columns:
        common = None
        for i in np.flatnonzero(X_train[col].to_numpy()):
            common = sigs[i] if common is None else common & sigs[i]
            if not common:
                break
        out[col] = bool(common)
    return pd.Series(out, dtype=bool)


def unseen_attack_masks(
    window: "MiniWindow", y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Test attack groups that carry signatures never seen in the mining
    window (the generalisation test), in two strengths:

    - any: at least one of the group's signatures is unseen. The main
      metric: under multi-code groupings (fixed_window, time_delta,
      AlertBERT) almost every group also contains a common, seen code, so
      "all unseen" is empty there.
    - all: none of the group's signatures were seen (the Phase 4b fox run's
      original definition). Equal to "any" for single-signature groups.
    """
    seen = frozenset().union(*(_as_set(window.signature_of(g)) for g in window.train))
    any_unseen, all_unseen = [], []
    for g, y in zip(window.test, y_test):
        sigs = _as_set(window.signature_of(g))
        new = sigs - seen
        any_unseen.append(bool(y) and bool(new))
        all_unseen.append(bool(y) and bool(sigs) and new == sigs)
    return np.array(any_unseen, dtype=bool), np.array(all_unseen, dtype=bool)


# Backwards-compatible name (Phase 4 notebooks/tests).
single_signature_share = restatement_flags


def equivalence_flags(
    X_train: pd.DataFrame,
    train: list[AlertGroup],
    signature_of,
    jaccard: float = 0.9,
) -> pd.DataFrame:
    """Per feature: does it fire on (almost) exactly the mining-window groups
    that contain one signature s, i.e. Jaccard(F, S_s) >= `jaccard`? That is
    "this feature IS signature s", and unlike the confined check
    (restatement_flags) it isn't triggered when a frequent background code
    merely co-occurs in every group the feature fires on, so it means the
    same thing for single- and multi-signature groupings.

    Returns columns `equivalent` (bool), `equiv_signature` (best s) and
    `equiv_jaccard` (its Jaccard)."""
    sigs = [_as_set(signature_of(g)) for g in train]
    size: dict = {}
    for ss in sigs:
        for sig in ss:
            size[sig] = size.get(sig, 0) + 1
    rows = {}
    for col in X_train.columns:
        idx = np.flatnonzero(X_train[col].to_numpy())
        if len(idx) == 0:
            rows[col] = (False, None, 0.0)
            continue
        inter: dict = {}
        for i in idx:
            for sig in sigs[i]:
                inter[sig] = inter.get(sig, 0) + 1
        best, best_j = None, 0.0
        for sig, n in inter.items():
            j = n / (len(idx) + size[sig] - n)
            if j > best_j:
                best, best_j = sig, j
        rows[col] = (best_j >= jaccard, best, round(best_j, 3))
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["equivalent", "equiv_signature", "equiv_jaccard"]
    )


def dedupe_firing(
    gated: pd.DataFrame,
    table: ItemTable | None,
    train: list[AlertGroup],
    jaccard: float = 0.95,
) -> pd.DataFrame:
    """Drop near-duplicate features before the budget: walking `gated` in
    score order, a feature is dropped if its mining-window firing has
    Jaccard >= `jaccard` with a feature already kept (identical firing is
    Jaccard 1). Applied to every method alike, so no method fills its budget
    with copies of one condition -- e.g. Eclat's many coarse-token variants of
    the same big benign code under multi-code groupings. jaccard=1.0 keeps
    only exact duplicates out. `gated` must already be deduplicated by
    condition and ranked (dedupe_conditions), so its rows map 1:1, in order,
    onto patterns_to_schema's features."""
    if gated.empty:
        return gated
    schema = patterns_to_schema(gated, "dedupe", "0", item_table=table)
    if len(schema.features) != len(gated):
        raise AssertionError("gated rows and schema features are not 1:1")
    X = SymbolicFeatureEncoder(schema).transform(train)
    F = X[[f.feature_name for f in schema.features]].to_numpy(dtype=np.float32)
    inter = F.T @ F  # pairwise co-firing counts
    size = np.diag(inter)
    union = size[:, None] + size[None, :] - inter
    jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    keep: list[int] = []
    for i in range(len(schema.features)):
        if keep and jac[i, keep].max() >= jaccard:
            continue
        keep.append(i)
    return gated.iloc[keep].reset_index(drop=True)


def dedupe_identical_firing(
    gated: pd.DataFrame, table: ItemTable | None, train: list[AlertGroup]
) -> pd.DataFrame:
    """Exact-duplicate firing only (dedupe_firing with jaccard=1.0)."""
    return dedupe_firing(gated, table, train, jaccard=1.0)


def gate(
    stats: pd.DataFrame, base_rate: float, g: float, benign_g: float | None = None
) -> pd.DataFrame:
    """The paper's precision gate on the mining window (Check 2), applied to
    the miner's own counts (identical to the encoded counts: round-trip
    tested per method)."""
    n_fire = stats["n_attack"] + stats["n_benign"]
    rate = (stats["n_attack"] / n_fire.where(n_fire > 0)).astype(float)
    keep = passes_gate(stats["source_label"], rate, base_rate, g, benign_g)
    return stats[keep.to_numpy()].reset_index(drop=True)


@lru_cache(maxsize=1)
def _repo_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # not a git checkout
        return "unknown"


def method_settings(method: str, config: MiniConfig, table: ItemTable | None) -> dict:
    """The exact settings a method ran with, for mining.json."""
    if method == "eclat":
        return {
            "min_support": config.eclat_min_support,
            "max_len": config.eclat_max_len,
            "min_support_diff": 0.05,
            "min_support_count": eclat_count_floor(config, table),
            "min_support_count_rule": config.eclat_min_support_count,
            "filters": ItemsetFilterConfig(
                min_support_count=eclat_count_floor(config, table)
            ).model_dump(),
        }
    if method == "contrast_tree":
        return {
            "setting": config.contrast_tree_setting,
            "settings_file": str(config.mining_settings_path),
            "config": contrast_tree_config(config).model_dump(),
        }
    if method == "ripper":
        import wittgenstein

        return {
            "library": "wittgenstein",
            "version": getattr(wittgenstein, "__version__", "unknown"),
            "random_state": config.seed,
            "n_discretize_bins": 0,
            "other_parameters": "library defaults",
        }
    if method == "prefixspan":
        return {
            "min_support": config.prefixspan_min_support,
            "max_len": config.prefixspan_max_len,
            "min_support_diff": 0.05,
            "filters": SequenceFilterConfig(min_k=config.prefixspan_min_k).model_dump(),
            "lookback_minutes": config.lookback_minutes,
        }
    if method == "mctsextent":
        from thesis.mining.temporal.mctsextent.miner import UPSTREAM_DEFAULT_THETA
        from thesis.mining.temporal.mctsextent.vendor import COMMIT, REPO_URL

        return {
            "upstream": REPO_URL,
            "upstream_commit": COMMIT,
            "top_k": config.mctsextent_top_k,
            "theta": UPSTREAM_DEFAULT_THETA,
            "iterations": config.mctsextent_iterations,
            "quality_measure": "WRAcc",
            "random_state": config.seed,
            "lookback_minutes": config.lookback_minutes,
        }
    return {}


def cell_dir_for(out_dir: Path, window: MiniWindow, method: str) -> Path:
    """cells/<scenario>/<grouping>/<token arm>/<method>/"""
    scenario = window.name.split()[0]
    grouping = window.grouping or (
        "cscas_pregrouped" if window.dataset == "cscas" else "default"
    )
    return out_dir / "cells" / scenario / grouping / window.token_arm / method


def _pattern_stage(stats: pd.DataFrame, stages: dict[str, set[int]]) -> pd.Series:
    def stage(pid: int) -> str:
        if pid not in stages["gated"]:
            return "dropped_gate"
        if pid not in stages["distinct_condition"]:
            return "dropped_duplicate_condition"
        if pid not in stages["distinct_firing"]:
            return "dropped_duplicate_firing"
        if pid not in stages["budget"]:
            return "dropped_budget"
        return "kept"

    return stats["pattern_id"].map(stage)


def write_cell_artifacts(
    cell_dir: Path,
    method: str,
    window: MiniWindow,
    config: MiniConfig,
    table: ItemTable | None,
    stats: pd.DataFrame,
    stages: dict[str, set[int]],
    schema,
    row: dict,
) -> None:
    """Per-cell inspection files:
    patterns.csv  every mined pattern with its stats and `stage`
                  (dropped_gate / dropped_duplicate_condition /
                  dropped_duplicate_firing (Jaccard >= dedupe_jaccard with a
                  kept feature) / dropped_budget / kept) and, for
                  kept ones, the schema feature name;
    schema.json   the budgeted SymbolicFeatureSchema (reloadable with
                  features.persistence.load_symbolic_feature_schema);
    mining.json   method + its exact settings, the shared settings, the
                  window, versions and this cell's summary row."""
    from thesis.features.persistence import save_symbolic_feature_schema

    cell_dir.mkdir(parents=True, exist_ok=True)
    out = stats.copy()
    out["stage"] = _pattern_stage(out, stages)
    out["items"] = out["items"].map(lambda it: " | ".join(map(str, it)))
    if "predicates" in out:
        out["predicates"] = out["predicates"].map(
            lambda ps: " ∧ ".join(p.token for p in ps) if isinstance(ps, tuple) else ""
        )
    out["feature"] = ""
    if schema is not None:
        kept = stages["budget_order"]
        if len(kept) == len(schema.features):
            names = {pid: f.feature_name for pid, f in zip(kept, schema.features)}
            out["feature"] = out["pattern_id"].map(names).fillna("")
        save_symbolic_feature_schema(schema, cell_dir / "schema.json")
    out.to_csv(cell_dir / "patterns.csv", index=False)

    include = sorted(window.include_fields)
    meta = {
        "method": method,
        "method_settings": method_settings(method, config, table),
        "shared_settings": {
            "k": config.k,
            "per_class_budget": config.per_class_budget,
            "gate_g": config.gate_g,
            "benign_gate_g": window.benign_gate_g or config.gate_g,
            "n_bins": config.n_bins,
            "dedupe_firing": config.dedupe_identical_firing,
            "dedupe_jaccard": config.dedupe_jaccard,
            "equivalence_jaccard": config.equivalence_jaccard,
            "seed": config.seed,
        },
        "window": {
            "name": window.name,
            "dataset": window.dataset,
            "grouping": window.grouping,
            "token_arm": window.token_arm,
            "include_fields": include,
            "exclude_fields": sorted(window.exclude_fields),
            "n_items": int(table.items().shape[1]) if table is not None else None,
            "train_start_ts": min((g.start_ts for g in window.train), default=None),
            "test_start_ts": min((g.start_ts for g in window.test), default=None),
        },
        "versions": {
            "repo_commit": _repo_commit(),
            "python": sys.version.split()[0],
        },
        "summary": row,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    (cell_dir / "mining.json").write_text(json.dumps(meta, indent=2, default=str))


def evaluate(
    method: str,
    window: MiniWindow,
    config: MiniConfig,
    table: ItemTable | None,
    cell_dir: Path | None = None,
) -> tuple[dict, pd.DataFrame]:
    """One (method, window) cell: summary row + per-feature table. With
    cell_dir, also writes patterns.csv / schema.json / mining.json there."""
    row: dict = {
        "method": method,
        "dataset": window.dataset,
        "window": window.name,
        "grouping": window.grouping,
        "token_arm": window.token_arm,
    }
    if method in TEMPORAL and not window.has_raw_alerts:
        row["status"] = "n/a: no raw per-alert stream to build sequences from"
        return row, pd.DataFrame()

    y_tr, y_te = _labels(window.train), _labels(window.test)
    n_attack, n_benign = int(y_tr.sum()), int((~y_tr).sum())
    base_rate = n_attack / len(y_tr)
    row.update(
        n_train=len(y_tr),
        n_train_attack=n_attack,
        n_test=len(y_te),
        n_test_attack=int(y_te.sum()),
        benign_gate_g=window.benign_gate_g or config.gate_g,
    )

    t0 = time.perf_counter()
    frame = mine(method, window, table, config)
    row["mine_seconds"] = round(time.perf_counter() - t0, 2)

    stats = add_pattern_stats(frame, n_attack, n_benign)
    stats.insert(0, "pattern_id", range(len(stats)))
    passed = gate(stats, base_rate, config.gate_g, window.benign_gate_g)
    gated = dedupe_conditions(passed, table)
    n_gated_conditions = len(gated)
    distinct_condition = set(gated["pattern_id"])
    if config.dedupe_identical_firing:
        gated = dedupe_firing(gated, table, window.train, config.dedupe_jaccard)
    budget = select_budget(gated, config.k, per_class=config.per_class_budget)
    stages = {
        "gated": set(passed["pattern_id"]),
        "distinct_condition": distinct_condition,
        "distinct_firing": set(gated["pattern_id"]),
        "budget": set(budget["pattern_id"]),
        "budget_order": list(budget["pattern_id"]),
    }
    row.update(
        n_raw=len(stats),
        n_raw_attack=int((stats["source_label"] == "attack").sum()),
        n_gated_conditions=n_gated_conditions,  # distinct conditions passing
        n_gated=len(gated),  # ...and distinct firing patterns among those
        n_budget=len(budget),
    )
    if method == "eclat":
        row["eclat_min_support_count"] = eclat_count_floor(config, table)
    if budget.empty:
        row["status"] = "FAIL: no features after gate + budget"
        if cell_dir is not None:
            write_cell_artifacts(
                cell_dir, method, window, config, table, stats, stages, None, row
            )
        return row, pd.DataFrame()

    schema = patterns_to_schema(budget, method, "mini", item_table=table)
    encoder = SymbolicFeatureEncoder(schema)
    X_tr = encoder.transform(window.train)
    X_te = encoder.transform(window.test)

    feats = pd.DataFrame(
        {
            "feature": [f.feature_name for f in schema.features],
            "source_label": [f.source_label for f in schema.features],
            "mining_type": [f.mining_type for f in schema.features],
            "n_conditions": [len(f.itemset) for f in schema.features],
        }
    )
    # The condition as the encoder evaluates it (predicate tokens for
    # atemporal features, ordered steps for sequences).
    feats["condition"] = [
        (
            " -> " if f.mining_type in ("item_sequence", "itemset_sequence") else " ∧ "
        ).join(f.itemset)
        for f in schema.features
    ]
    tr = X_tr[feats["feature"]].to_numpy().astype(bool)
    te = X_te[feats["feature"]].to_numpy().astype(bool)
    feats["train_fires"] = tr.sum(axis=0)
    feats["train_attack_rate"] = (tr & y_tr[:, None]).sum(0) / np.maximum(tr.sum(0), 1)
    feats["test_fires"] = te.sum(axis=0)
    feats["test_attack_rate"] = np.where(
        te.sum(0) > 0, (te & y_te[:, None]).sum(0) / np.maximum(te.sum(0), 1), np.nan
    )
    feats["support"] = tr.mean(axis=0)
    feats["single_signature"] = restatement_flags(
        X_tr[feats["feature"]], window.train, window.signature_of
    ).to_numpy()
    eq = equivalence_flags(
        X_tr[feats["feature"]],
        window.train,
        window.signature_of,
        config.equivalence_jaccard,
    )
    feats["equivalent_signature"] = eq["equivalent"].to_numpy(dtype=bool)
    feats["equiv_signature"] = eq["equiv_signature"].astype(str).to_numpy()
    feats["equiv_jaccard"] = eq["equiv_jaccard"].to_numpy()
    if window.name_signature_of is not None:
        feats["single_signature_name"] = restatement_flags(
            X_tr[feats["feature"]], window.train, window.name_signature_of
        ).to_numpy()
    feats["live_test"] = feats["test_fires"] > 0

    attack = (feats["source_label"] == "attack").to_numpy()
    live = feats["live_test"].to_numpy()
    good = live & ~feats["single_signature"].to_numpy()
    unseen, unseen_all = unseen_attack_masks(window, y_te)
    row.update(
        n_budget_attack=int(attack.sum()),
        n_budget_benign=int((~attack).sum()),
        mean_conditions=round(float(feats["n_conditions"].mean()), 2),
        max_conditions=int(feats["n_conditions"].max()),
        dead_on_train=int((feats["train_fires"] == 0).sum()),
        live_share_test=round(float(feats["live_test"].mean()), 3),
        single_signature_share=round(float(feats["single_signature"].mean()), 3),
        equivalent_signature_share=round(
            float(feats["equivalent_signature"].mean()), 3
        ),
        test_attack_coverage=(
            round(float(te[:, attack].any(axis=1)[y_te].mean()), 3)
            if attack.any() and y_te.any()
            else np.nan
        ),
        test_benign_coverage=(
            round(float(te[:, ~attack].any(axis=1)[~y_te].mean()), 3)
            if (~attack).any() and (~y_te).any()
            else np.nan
        ),
        median_test_attack_rate_attack_feats=(
            round(float(np.nanmedian(feats.loc[attack, "test_attack_rate"])), 3)
            if attack.any() and feats.loc[attack, "live_test"].any()
            else np.nan
        ),
        test_base_rate=round(float(y_te.mean()), 4) if len(y_te) else np.nan,
        n_test_attack_unseen=int(unseen.sum()),
        unseen_attack_coverage=(
            round(float(te[:, attack].any(axis=1)[unseen].mean()), 3)
            if attack.any() and unseen.any()
            else np.nan
        ),
        n_test_attack_unseen_all=int(unseen_all.sum()),
        unseen_all_attack_coverage=(
            round(float(te[:, attack].any(axis=1)[unseen_all].mean()), 3)
            if attack.any() and unseen_all.any()
            else np.nan
        ),
    )
    if "single_signature_name" in feats:
        row["single_signature_share_name"] = round(
            float(feats["single_signature_name"].mean()), 3
        )
    # Verdict (decided 2026-11-10): restatement is a reported metric, not a
    # fail criterion. PASS = a live feature for each class; pass_strict
    # keeps the Phase 4 rule (live AND not a restatement) for comparison.
    ok_attack, ok_benign = bool((live & attack).any()), bool((live & ~attack).any())
    row["pass_strict"] = bool((good & attack).any() and (good & ~attack).any())
    row["status"] = (
        "PASS"
        if ok_attack and ok_benign
        else "FAIL: no live "
        + " or ".join(
            c for c, ok in (("attack", ok_attack), ("benign", ok_benign)) if not ok
        )
        + " feature"
    )
    if cell_dir is not None:
        write_cell_artifacts(
            cell_dir, method, window, config, table, stats, stages, schema, row
        )
    feats.insert(0, "method", method)
    feats.insert(1, "dataset", window.dataset)
    feats.insert(2, "grouping", window.grouping)
    feats.insert(3, "token_arm", window.token_arm)
    return row, feats


def run_mini(
    windows: list[MiniWindow], config: MiniConfig, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str)
    )
    rows, feats = [], []
    for window in windows:
        table = None
        if any(m not in TEMPORAL for m in config.methods):
            t0 = time.perf_counter()
            table = build_item_table(
                window.train,
                exclude_fields=window.exclude_fields,
                n_bins=config.n_bins,
                include_fields=window.include_fields,
            )
            print(
                f"[{window.name}] item table {table.items().shape} "
                f"in {time.perf_counter() - t0:.1f}s"
            )
        steps_attached = False
        for method in config.methods:
            if method in TEMPORAL and window.attach_steps and not steps_attached:
                window.attach_steps(window.train + window.test)
                steps_attached = True
            print(f"[{window.name}] {method} ...", flush=True)
            try:
                row, f = evaluate(
                    method,
                    window,
                    config,
                    table,
                    cell_dir=cell_dir_for(out_dir, window, method),
                )
            except Exception as exc:  # record and keep going
                traceback.print_exc()
                row = {
                    "method": method,
                    "dataset": window.dataset,
                    "window": window.name,
                    "status": f"ERROR: {type(exc).__name__}: {exc}",
                }
                f = pd.DataFrame()
            print(f"    -> {row['status']}")
            rows.append(row)
            feats.append(f)
            summary = pd.DataFrame(rows)
            summary.to_csv(out_dir / "summary.csv", index=False)
            pd.concat(feats, ignore_index=True).to_csv(
                out_dir / "features.csv", index=False
            )
    return pd.DataFrame(rows), pd.concat(feats, ignore_index=True)


# --- Window builders ---------------------------------------------------------


def cscas_window(gran: float = 0.05) -> MiniWindow:
    """CSCAS W0 -> W1 of the paper's walk-forward (same as Checks 1-3)."""
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
        mining_settings_path=MiniConfig.mining_settings_path,
    )
    groups = ctx.alert_groups
    bounds = training_window_bounds(build_baseline_split_scheme(groups), gran)
    (s0, e0), (s1, e1) = bounds[0], bounds[1]
    lookup = load_cscas_signature_lookup()
    return MiniWindow(
        dataset="cscas",
        name=f"cscas W0->W1 (gran {gran})",
        train=labelled(groups[s0:e0]),
        test=labelled(groups[s1:e1]),
        signature_of=lambda g: lookup.get(g.group_id),
        exclude_fields=default_leaky_attribute_fields("cscas"),
        has_raw_alerts=False,
    )


def _ait_signature(group: AlertGroup) -> str:
    # cscas_grouping group_id = "cscas_grouping:<host>:<signature name>:<sha1>"
    parts = group.group_id.split(":")
    return ":".join(parts[2:-1]) if len(parts) > 3 else group.group_id


def _ait_short_codes(group: AlertGroup) -> frozenset[str]:
    """Signature identity for AIT-ADS under every grouping: the group's short
    codes. Every alert name maps to exactly one code (verified 2026-11-10;
    10 of 75 codes cover 2-6 names), so this is slightly coarser than the
    name and the restatement share is slightly overstated, never understated."""
    return frozenset(
        i[len("short:") :] for i in (group.raw_items or ()) if i.startswith("short:")
    )


# Phase 4b token arms: (include_fields, extra exclude_fields).
TOKEN_ARMS: dict[str, tuple[set[str], set[str]]] = {
    "T0": (set(), set()),
    "T1": ({"short_src", "short_cat"}, set()),
    "T2": ({"short_src", "short_cat", "short_family", "sig_family"}, set()),
    "T3": ({"short_src", "short_cat", "short_family", "sig_family"}, {"short", "sig"}),
}
# The arm the comparison uses on AIT-ADS, for every method (decided
# 2026-11-11): full codes plus all coarse levels.
COMPARISON_TOKEN_ARM = "T2"


def lookback_token_fields(token_arm: str) -> frozenset[str]:
    """The look-back items an arm gives the temporal methods: the same coarse
    fields as the item table, plus the full code unless the arm excludes it."""
    include, exclude = TOKEN_ARMS[token_arm]
    return frozenset(({"short"} - exclude) | include)


_LOOKBACK_INDEX: dict[tuple[str, frozenset[str]], HostAlertIndex] = {}


def lookback_attacher(
    scenario: str, token_arm: str, window_minutes: int = DEFAULT_WINDOW_MINUTES
) -> Callable[[list[AlertGroup]], None]:
    """attach_steps for an AIT-ADS window: look-back steps with the arm's
    tokens, the per-host index built once per (scenario, token fields)."""
    fields = lookback_token_fields(token_arm)

    def attach(groups: list[AlertGroup]) -> None:
        key = (scenario, fields)
        if key not in _LOOKBACK_INDEX:
            _LOOKBACK_INDEX[key] = HostAlertIndex(
                load_ait_ads_raw_alerts(scenario), fields
            )
        attach_lookback_steps(
            groups,
            alerts=None,
            window_minutes=window_minutes,
            index=_LOOKBACK_INDEX[key],
        )

    return attach


def ait_ads_window(
    scenario: str = "fox",
    method: str = "cscas_grouping",
    train_frac: float = 0.7,
    lookback_minutes: int = DEFAULT_WINDOW_MINUTES,
    token_arm: str = "T0",
    split_ts: int | None = None,
) -> MiniWindow:
    """Mining window = groups starting before the split, test = the rest.

    split_ts None: the first train_frac of groups (by count). Pass a
    timestamp to cut every grouping at the *same moment*: a count-based cut
    lands at a different time per grouping (on fox 12:17 vs. 12:43), and a
    burst of new codes right after the cut then counts as seen for one
    grouping and unseen for another."""
    from thesis.paths import CACHE_DIR
    from thesis.pipeline.pipeline import load_or_build_alert_groups

    groups = load_or_build_alert_groups(
        scenario, CACHE_DIR / scenario / "groups" / method
    )
    groups = labelled(sorted(groups, key=lambda g: g.start_ts))
    if split_ts is None:
        cut = int(len(groups) * train_frac)
        split_desc = f"first {train_frac:.0%} -> rest"
    else:
        cut = sum(g.start_ts < split_ts for g in groups)
        split_desc = f"split at ts {split_ts}"
    include, exclude = TOKEN_ARMS[token_arm]
    return MiniWindow(
        dataset="ait_ads",
        name=f"{scenario} {method} {token_arm} {split_desc}",
        train=groups[:cut],
        test=groups[cut:],
        signature_of=_ait_short_codes,
        exclude_fields=default_leaky_attribute_fields(scenario) | exclude,
        include_fields=set(include),
        name_signature_of=_ait_signature if method == "cscas_grouping" else None,
        grouping=method,
        token_arm=token_arm,
        has_raw_alerts=True,
        attach_steps=lookback_attacher(scenario, token_arm, lookback_minutes),
        # AIT-ADS time-window labels mark background alerts during attack
        # phases as attacks (and event labels can't separate them: they're
        # absent for dnsteal / scans / reverse shell), so benign rules only
        # need to be no more attack-enriched than the base rate.
        benign_gate_g=1.0,
    )


def reference_split_ts(
    scenario: str, reference: str = "cscas_grouping", train_frac: float = 0.7
) -> int:
    """The start time of the first test group when the reference grouping is
    cut at train_frac by count; used as the common split for all groupings."""
    from thesis.paths import CACHE_DIR
    from thesis.pipeline.pipeline import load_or_build_alert_groups

    groups = labelled(
        sorted(
            load_or_build_alert_groups(
                scenario, CACHE_DIR / scenario / "groups" / reference
            ),
            key=lambda g: g.start_ts,
        )
    )
    return groups[int(len(groups) * train_frac)].start_ts


def default_out_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "artifacts" / "experiments" / "mining_mini" / ts

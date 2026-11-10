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
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
    attach_lookback_steps,
    load_ait_ads_raw_alerts,
)
from thesis.mining.temporal.mctsextent.miner import mine_mctsextent
from thesis.mining.temporal.prefixspan.miner import mine_prefixspan
from thesis.paths import ROOT
from thesis.schemas.groups import AlertGroup
from thesis.schemas.mining import AttributeMiningConfig, SequenceFilterConfig
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
    # group -> signature identity, for the single-signature restatement check
    signature_of: Callable[[AlertGroup], object]
    exclude_fields: set[str] = field(default_factory=set)
    has_raw_alerts: bool = False


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


def gate(stats: pd.DataFrame, base_rate: float, g: float) -> pd.DataFrame:
    """The paper's precision gate on the mining window (Check 2), applied to
    the miner's own counts (identical to the encoded counts: round-trip
    tested per method)."""
    n_fire = stats["n_attack"] + stats["n_benign"]
    rate = (stats["n_attack"] / n_fire.where(n_fire > 0)).astype(float)
    keep = passes_gate(stats["source_label"], rate, base_rate, g)
    return stats[keep.to_numpy()].reset_index(drop=True)


def single_signature_share(
    X_train: pd.DataFrame, train: list[AlertGroup], signature_of
) -> pd.Series:
    """Per feature: True if every mining-window group it fires on has the
    same signature (a restatement of one signature)."""
    # Fill a 1-D object array explicitly: np.array() on tuple signatures
    # would build a 2-D array.
    sigs = np.empty(len(train), dtype=object)
    sigs[:] = [signature_of(g) for g in train]
    out = {}
    for col in X_train.columns:
        fired = sigs[X_train[col].to_numpy().astype(bool)]
        out[col] = len(fired) > 0 and len(set(fired.tolist())) == 1
    return pd.Series(out, dtype=bool)


def evaluate(
    method: str,
    window: MiniWindow,
    config: MiniConfig,
    table: ItemTable | None,
) -> tuple[dict, pd.DataFrame]:
    """One (method, window) cell: summary row + per-feature table."""
    row: dict = {"method": method, "dataset": window.dataset, "window": window.name}
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
    )

    t0 = time.perf_counter()
    frame = mine(method, window, table, config)
    row["mine_seconds"] = round(time.perf_counter() - t0, 2)

    stats = add_pattern_stats(frame, n_attack, n_benign)
    gated = dedupe_conditions(gate(stats, base_rate, config.gate_g), table)
    budget = select_budget(gated, config.k, per_class=config.per_class_budget)
    row.update(
        n_raw=len(stats),
        n_raw_attack=int((stats["source_label"] == "attack").sum()),
        n_gated=len(gated),  # distinct conditions passing the gate
        n_budget=len(budget),
    )
    if budget.empty:
        row["status"] = "FAIL: no features after gate + budget"
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
    feats["single_signature"] = single_signature_share(
        X_tr[feats["feature"]], window.train, window.signature_of
    ).to_numpy()
    feats["live_test"] = feats["test_fires"] > 0

    attack = (feats["source_label"] == "attack").to_numpy()
    good = feats["live_test"].to_numpy() & ~feats["single_signature"].to_numpy()
    row.update(
        n_budget_attack=int(attack.sum()),
        n_budget_benign=int((~attack).sum()),
        mean_conditions=round(float(feats["n_conditions"].mean()), 2),
        max_conditions=int(feats["n_conditions"].max()),
        dead_on_train=int((feats["train_fires"] == 0).sum()),
        live_share_test=round(float(feats["live_test"].mean()), 3),
        single_signature_share=round(float(feats["single_signature"].mean()), 3),
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
    )
    ok_attack = bool((good & attack).any())
    ok_benign = bool((good & ~attack).any())
    row["status"] = (
        "PASS"
        if ok_attack and ok_benign
        else "FAIL: no live, non-restatement "
        + " or ".join(
            c for c, ok in (("attack", ok_attack), ("benign", ok_benign)) if not ok
        )
        + " feature"
    )
    feats.insert(0, "method", method)
    feats.insert(1, "dataset", window.dataset)
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
                window.train, exclude_fields=window.exclude_fields, n_bins=config.n_bins
            )
            print(
                f"[{window.name}] item table {table.items().shape} "
                f"in {time.perf_counter() - t0:.1f}s"
            )
        for method in config.methods:
            print(f"[{window.name}] {method} ...", flush=True)
            try:
                row, f = evaluate(method, window, config, table)
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


def ait_ads_window(
    scenario: str = "fox",
    method: str = "cscas_grouping",
    train_frac: float = 0.7,
    lookback_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> MiniWindow:
    from thesis.paths import CACHE_DIR
    from thesis.pipeline.pipeline import load_or_build_alert_groups

    groups = load_or_build_alert_groups(
        scenario, CACHE_DIR / scenario / "groups" / method
    )
    groups = labelled(sorted(groups, key=lambda g: g.start_ts))
    attach_lookback_steps(
        groups, load_ait_ads_raw_alerts(scenario), window_minutes=lookback_minutes
    )
    cut = int(len(groups) * train_frac)
    return MiniWindow(
        dataset="ait_ads",
        name=f"{scenario} {method} first {train_frac:.0%} -> rest",
        train=groups[:cut],
        test=groups[cut:],
        signature_of=_ait_signature,
        exclude_fields=default_leaky_attribute_fields(scenario),
        has_raw_alerts=True,
    )


def default_out_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "artifacts" / "experiments" / "mining_mini" / ts

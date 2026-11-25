"""
Label budget for the DeepCASE comparison (paper Sec. DeepCASE, implementation
plan B1): which labels the system actually has, and how many it cost.

Used only by monitor_attached's opt-in `update_labels="cumulative_budget"`
mode -- the default experiments keep training on full window labels.

The rule, identical for our system and DeepCASE:
* initial training gets B0 labels (`draw_budget`): a stratified random
  sample of the training rows, or exactly the items DeepCASE's simulated
  analyst inspected (`analyst_items.parquet` from deepcase_comparison);
* every window, the analyst reviews what the system escalates (free: normal
  SOC work, reported as workload) and -- only when the monitor needs it
  (audit_rate > 0) -- a random `audit_rate` sample of what it suppresses
  (extra, monitoring-only labels, counted in the budget);
* those labels accumulate in a pool (`LabelLedger`); every update trains on
  the whole pool, audit rows weighted 1/audit_rate (inverse-probability
  weights) so the rows the model misses aren't drowned out by the ones it
  already catches.

Budget = B0 + all audit labels (+ extra labels an update itself asks for --
none for our system, whose updates reuse the pool; DeepCASE's new-cluster
samples). Escalated reviews are workload, reported separately.

The audit draw reuses thesis.monitor.pattern_validity.audit_mask (hash of
alert_group_id), so the rows audited here are exactly the rows pattern
validity's "audit" label mode weights at the same rate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from thesis.monitor.pattern_validity import AUDIT_RATE, audit_mask
from thesis.schemas.groups import AlertGroup

BUDGET_MODES = ("full", "stratified", "deepcase_items", "deepcase_cluster_samples")
LEDGER_COLUMNS = (
    "ledger_n_escalated_labels",
    "ledger_n_audit_labels",
    "ledger_pool_size",
    "ledger_budget_labels",
    "ledger_workload_labels",
)
_LABELS = ("attack", "benign")


def is_labelled(g: AlertGroup) -> bool:
    return g.group_label in _LABELS


def mask_labels(groups: Sequence[AlertGroup], keep: Sequence[bool]) -> list[AlertGroup]:
    """Copy of `groups` where every row not in `keep` has no label -- what the
    system knows. Rows are never mutated; kept rows are returned as-is."""
    return [
        g if k or not is_labelled(g) else replace(g, group_label=None)
        for g, k in zip(groups, keep)
    ]


def draw_budget(
    rows: Sequence[AlertGroup],
    mode: str,
    size: int | None = None,
    seed: int = 0,
    items_path: Path | None = None,
) -> np.ndarray:
    """Boolean mask over `rows`: which rows get an initial (B0) label.

    * full: every labelled row (today's behaviour);
    * stratified: `size` labelled rows at random, keeping the attack/benign
      ratio of `rows` (each class rounded, at least one row per class);
    * deepcase_items / deepcase_cluster_samples: the rows whose group_id is
      in DeepCASE's analyst_items.parquet (all inspected items, or only the
      10-per-cluster samples). Items outside `rows` are ignored.
    """
    labelled = np.array([is_labelled(g) for g in rows])
    if mode == "full":
        return labelled
    if mode == "stratified":
        if size is None:
            raise ValueError("stratified budget needs a size")
        rng = np.random.default_rng(seed)
        is_attack = np.array([g.group_label == "attack" for g in rows])
        n_lab = int(labelled.sum())
        out = np.zeros(len(rows), dtype=bool)
        if n_lab == 0:
            return out
        for cls in (True, False):
            idx = np.flatnonzero(labelled & (is_attack == cls))
            if len(idx) == 0:
                continue
            k = int(round(size * len(idx) / n_lab))
            k = min(len(idx), max(1, k))
            out[rng.choice(idx, size=k, replace=False)] = True
        return out
    if mode in ("deepcase_items", "deepcase_cluster_samples"):
        if items_path is None:
            raise ValueError(f"{mode} budget needs items_path")
        items = pd.read_parquet(items_path, columns=["group_id", "kind"])
        if mode == "deepcase_cluster_samples":
            items = items[items["kind"] == "cluster_sample"]
        wanted = set(items["group_id"].astype(str))
        return labelled & np.array([str(g.group_id) in wanted for g in rows])
    raise ValueError(f"unknown budget mode {mode!r}; expected one of {BUDGET_MODES}")


def budget_view(
    groups: Sequence[AlertGroup],
    flagged: np.ndarray,
    audit_rate: float = AUDIT_RATE,
) -> tuple[list[AlertGroup], np.ndarray, np.ndarray]:
    """What the analyst labels in one operating window: every escalated row
    plus an `audit_rate` sample of the suppressed ones. Returns the masked
    groups and the escalated / audited masks (labelled rows only)."""
    flagged = np.asarray(flagged, dtype=bool)
    labelled = np.array([is_labelled(g) for g in groups])
    if audit_rate > 0:
        audited = (audit_mask(groups, audit_rate) > 0) & ~flagged & labelled
    else:
        audited = np.zeros(len(groups), dtype=bool)
    escalated = flagged & labelled
    return mask_labels(groups, escalated | audited), escalated, audited


@dataclass(slots=True)
class _Entry:
    group: AlertGroup
    weight: float
    source: str  # "b0" | "escalated" | "audit"


class LabelLedger:
    """Cumulative labelled pool + label accounting for one deployment run."""

    def __init__(self, audit_rate: float = AUDIT_RATE) -> None:
        self.audit_rate = audit_rate
        self._pool: dict[str, _Entry] = {}
        self.n_b0 = 0
        self.n_audit = 0
        self.n_escalated = 0

    def _add(self, g: AlertGroup, weight: float, source: str) -> bool:
        key = str(g.alert_group_id)
        if key in self._pool:
            return False
        self._pool[key] = _Entry(g, weight, source)
        return True

    def add_initial(self, groups: Sequence[AlertGroup]) -> None:
        self.n_b0 += sum(self._add(g, 1.0, "b0") for g in groups if is_labelled(g))

    def add_window(
        self, groups: Sequence[AlertGroup], flagged: np.ndarray
    ) -> tuple[list[AlertGroup], dict]:
        """Record one operating window's analyst labels; return the masked
        groups (what the monitor may see) and this window's ledger columns."""
        view, escalated, audited = budget_view(groups, flagged, self.audit_rate)
        n_esc = sum(
            self._add(g, 1.0, "escalated") for g, e in zip(groups, escalated) if e
        )
        n_aud = sum(
            self._add(g, 1.0 / self.audit_rate, "audit")
            for g, a in zip(groups, audited)
            if a
        )
        self.n_escalated += n_esc
        self.n_audit += n_aud
        return view, {
            "ledger_n_escalated_labels": n_esc,
            "ledger_n_audit_labels": n_aud,
            **self.totals(),
        }

    def totals(self) -> dict:
        return {
            "ledger_pool_size": len(self._pool),
            "ledger_budget_labels": self.n_b0 + self.n_audit,
            "ledger_workload_labels": self.n_escalated,
        }

    def training_set(self) -> tuple[list[AlertGroup], np.ndarray]:
        """Every pooled row (time order) and its sample weight."""
        entries = sorted(self._pool.values(), key=lambda e: e.group.start_ts or 0)
        return [e.group for e in entries], np.array([e.weight for e in entries])

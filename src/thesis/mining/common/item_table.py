"""Shared item table: the one atemporal input every miner in the comparison reads.

Fairness rule 1 of the mining comparison: every method sees the same binary
items plus the same numeric fields. This builds them once per mining window
from the exact candidate space the contrast-set + decision-tree miner already
uses (build_categorical_predicate_matrix over compute_candidate_attribute_features),
so the thesis method and the new methods can't drift apart on inputs.

Two views of the same window:
  - X_cat + X_num: categorical items (one per field value, "category=EXPLOIT",
    "cve_present", ...) plus raw numerics, for methods that handle numerics
    themselves (decision tree, RIPPER).
  - items(): X_cat plus the numerics as quantile-bin items, for methods that
    can't (Eclat). One item per field value, never tokenised signature words,
    which is what made Eclat restate single signatures on CSCAS.

Every item maps to the AttributePredicates whose AND is that item, so any
pattern over items can be written into the symbolic schema and evaluated at
encode time by the existing predicate path in encoders/symbolic.py. A bin item
"lo < x <= hi" is two predicates (x > lo, x <= hi). Bin edges are fitted on
this window only (the training window), never on later data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from thesis.mining.atemporal.contrast_tree.attribute_contrast_mining import (
    build_categorical_predicate_matrix,
)
from thesis.schemas.features import AttributePredicate
from thesis.schemas.groups import AlertGroup

DEFAULT_N_BINS = 5


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _le(attr: str, x: float) -> AttributePredicate:
    return AttributePredicate(
        token=f"{attr}<={_fmt(x)}", attribute=attr, operator="<=", value=float(x)
    )


def _gt(attr: str, x: float) -> AttributePredicate:
    return AttributePredicate(
        token=f"{attr}>{_fmt(x)}", attribute=attr, operator=">", value=float(x)
    )


def quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior cut points for n_bins quantile bins. Ties collapse cut points,
    so a field with heavy mass at one value gets fewer bins; a constant field
    gets none (no item, it would fire on every group)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or n_bins < 2:
        return np.array([])
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.unique(qs)
    # A cut at the maximum leaves an empty top bin; drop it.
    return edges[edges < values.max()]


def _bin_items(
    attr: str, values: np.ndarray, edges: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, tuple[AttributePredicate, ...]]]:
    cols: dict[str, np.ndarray] = {}
    preds: dict[str, tuple[AttributePredicate, ...]] = {}
    bounds = [None, *edges.tolist(), None]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = np.ones(len(values), dtype=bool)
        p: list[AttributePredicate] = []
        if lo is not None:
            mask &= values > lo
            p.append(_gt(attr, lo))
        if hi is not None:
            mask &= values <= hi
            p.append(_le(attr, hi))
        if not mask.any():
            continue
        if lo is None:
            name = f"{attr}<={_fmt(hi)}"
        elif hi is None:
            name = f"{attr}>{_fmt(lo)}"
        else:
            name = f"{_fmt(lo)}<{attr}<={_fmt(hi)}"
        cols[name] = mask.astype(np.int8)
        preds[name] = tuple(p)
    return cols, preds


@dataclass
class ItemTable:
    """One mining window's shared input. Rows align with `alert_groups`."""

    X_cat: pd.DataFrame
    X_num: pd.DataFrame
    y: pd.Series  # 1 = attack, 0 = benign
    column_predicate_map: dict[str, tuple[str, Any]]
    bin_edges: dict[str, np.ndarray]
    X_bin: pd.DataFrame
    item_predicates: dict[str, tuple[AttributePredicate, ...]] = field(
        default_factory=dict
    )

    @property
    def n_attack(self) -> int:
        return int(self.y.sum())

    @property
    def n_benign(self) -> int:
        return int(len(self.y) - self.y.sum())

    def items(self) -> pd.DataFrame:
        """Binary item matrix: categorical items + numeric bin items."""
        return pd.concat([self.X_cat, self.X_bin], axis=1)

    def baskets(self) -> list[frozenset[str]]:
        """One frozenset of present items per group (Eclat's input)."""
        X = self.items()
        cols = np.asarray(X.columns)
        values = X.to_numpy(dtype=bool)
        return [frozenset(cols[row]) for row in values]

    def predicates_for(self, items: Sequence[str]) -> tuple[AttributePredicate, ...]:
        """The AND of predicates a pattern over these items means, deduplicated
        by token, in a stable order."""
        seen: dict[str, AttributePredicate] = {}
        for item in items:
            for p in self.item_predicates[item]:
                seen.setdefault(p.token, p)
        return tuple(seen.values())


def build_item_table(
    alert_groups: Sequence[AlertGroup],
    exclude_fields: set[str] | None = None,
    n_bins: int = DEFAULT_N_BINS,
) -> ItemTable:
    """Build the shared item table for one mining window.

    Mixed-label and unlabelled groups should be dropped by the caller (the
    label is binary here: anything other than "attack" counts as benign,
    exactly as in build_categorical_predicate_matrix).
    """
    X_cat, X_num, y, column_predicate_map = build_categorical_predicate_matrix(
        alert_groups, exclude_fields=exclude_fields
    )

    item_predicates: dict[str, tuple[AttributePredicate, ...]] = {}
    for col in X_cat.columns:
        attribute, value = column_predicate_map[col]
        # Token = column name, operator "==": identical to what the contrast
        # survivors and tree splits use, so a shared token never disagrees.
        item_predicates[col] = (
            AttributePredicate(
                token=col, attribute=attribute, operator="==", value=value
            ),
        )

    bin_cols: dict[str, np.ndarray] = {}
    bin_edges: dict[str, np.ndarray] = {}
    for attr in X_num.columns:
        values = X_num[attr].to_numpy(dtype=float)
        edges = quantile_bin_edges(values, n_bins)
        bin_edges[attr] = edges
        if edges.size == 0:
            continue
        cols, preds = _bin_items(attr, values, edges)
        bin_cols.update(cols)
        item_predicates.update(preds)

    X_bin = pd.DataFrame(bin_cols, index=X_cat.index)
    return ItemTable(
        X_cat=X_cat,
        X_num=X_num,
        y=y,
        column_predicate_map=column_predicate_map,
        bin_edges=bin_edges,
        X_bin=X_bin,
        item_predicates=item_predicates,
    )

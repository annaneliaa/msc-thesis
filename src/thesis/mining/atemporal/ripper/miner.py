"""RIPPER (rule-set learner, label used *during* the search) as a comparison
miner, via the `wittgenstein` library (MIT).

Input is the shared item table:
  - categorical items stay 0/1 columns, so a rule condition is either the
    item (col=1) or its negation (col=0, written as a "!=" predicate with the
    NOT_<col> token the decision-tree rules already use);
  - each numeric field becomes ONE categorical column whose value is the
    name of its quantile-bin item, so a condition picks one bin.
wittgenstein's own numeric handling is the same mechanism (equal-frequency
bins treated as categorical values, see wittgenstein.discretize), so this
gives RIPPER the shared bin edges rather than its own, and every rule stays
an AND that the encoder can evaluate exactly. Its own discretisation is
switched off (n_discretize_bins=0).

RIPPER learns rules for one positive class, so it runs once per class to get
attack- and benign-leaning rules (plan §4). Each rule becomes one pattern.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import wittgenstein as lw

from thesis.mining.common.item_table import ItemTable
from thesis.mining.common.patterns import PATTERN_COLUMNS
from thesis.schemas.features import AttributePredicate

METHOD = "ripper"
_LABEL = "__label__"


def ripper_input(table: ItemTable) -> pd.DataFrame:
    """The table RIPPER sees: 0/1 categorical items + one bin-valued column
    per binned numeric field."""
    frame = table.X_cat.astype(np.int8).copy()
    by_field: dict[str, list[str]] = {}
    for col in table.X_bin.columns:
        field = table.item_predicates[col][0].attribute
        by_field.setdefault(field, []).append(col)
    for field, cols in by_field.items():
        bins = table.X_bin[cols].to_numpy()
        frame[field] = np.asarray(cols, dtype=object)[bins.argmax(axis=1)]
    return frame


def _condition(
    table: ItemTable, feature: str, value: Any
) -> tuple[str, tuple[AttributePredicate, ...]]:
    """(item label, predicates) for one RIPPER condition."""
    if feature in table.X_cat.columns:
        (pred,) = table.item_predicates[feature]
        if int(value) == 1:
            return feature, (pred,)
        token = f"NOT_{feature}"
        return token, (
            AttributePredicate(
                token=token, attribute=pred.attribute, operator="!=", value=pred.value
            ),
        )
    # Binned numeric field: value is the bin item's name.
    return str(value), table.item_predicates[str(value)]


def mine_ripper(
    table: ItemTable,
    random_state: int = 0,
    classes: tuple[str, ...] = ("attack", "benign"),
    **ripper_params: Any,
) -> pd.DataFrame:
    """One RIPPER fit per class in `classes`; extra keyword arguments go to
    wittgenstein.RIPPER (defaults otherwise, plan §4)."""
    frame = ripper_input(table)
    y = table.y.to_numpy()
    rows = []
    for label in classes:
        pos = 1 if label == "attack" else 0
        if not (y == pos).any() or (y == pos).all():
            continue
        model = lw.RIPPER(
            random_state=random_state, n_discretize_bins=0, **ripper_params
        )
        model.fit(frame.assign(**{_LABEL: y}), class_feat=_LABEL, pos_class=pos)
        for rule in model.ruleset_.rules:
            if not rule.conds:
                continue
            mask = np.ones(len(frame), dtype=bool)
            items, preds = [], {}
            for cond in rule.conds:
                mask &= (frame[cond.feature] == cond.val).to_numpy()
                item, cond_preds = _condition(table, cond.feature, cond.val)
                items.append(item)
                for p in cond_preds:
                    preds.setdefault(p.token, p)
            rows.append(
                {
                    "items": tuple(items),
                    "predicates": tuple(preds.values()),
                    "mining_type": "ripper_rule",
                    "source_label": label,
                    "n_attack": int((mask & (y == 1)).sum()),
                    "n_benign": int((mask & (y == 0)).sum()),
                }
            )
    return pd.DataFrame(rows, columns=[*PATTERN_COLUMNS, "predicates"])

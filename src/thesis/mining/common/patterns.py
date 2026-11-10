"""Shared output format, feature budget K, and schema writer for the mining
comparison (fairness rules 2 and 3).

Every miner emits a *pattern frame*, one row per pattern:

    items         tuple[str, ...]  item-table item names (atemporal), or the
                                   ordered `short:` items of a sequence
    mining_type   str              "itemset", "rule", "item_sequence", ...
    source_label  "attack"|"benign"  the class the miner says it leans to
    n_attack      int              mining-window attack groups it fires on
    n_benign      int              mining-window benign groups it fires on

add_pattern_stats() derives support and precision from the counts the same
way for every method; select_budget() keeps the top K; patterns_to_schema()
writes the result into the existing SymbolicFeatureSchema, so the encoder,
the precision gate (system_eval/rule_precision_gate.py) and the model treat
every method identically.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from thesis.features.schema_builder import _make_feature_name
from thesis.mining.common.item_table import ItemTable
from thesis.schemas.features import (
    SEQUENCE_MINING_TYPES,
    AttributePredicate,
    SymbolicFeature,
    SymbolicFeatureSchema,
)

PATTERN_COLUMNS = ["items", "mining_type", "source_label", "n_attack", "n_benign"]


def add_pattern_stats(
    df: pd.DataFrame, n_attack_total: int, n_benign_total: int
) -> pd.DataFrame:
    """Add support, per-class support, precision and the budget score.

    support          share of all mining-window groups the pattern fires on
    support_<class>  share of that class's groups it fires on (per-class
                     coverage; stored as confidence_<class> in the schema,
                     the same meaning the contrast-tree rows have there)
    precision_attack attack rate among groups where it fires
    lean_precision   precision in its own leaning class
    score            lean_precision * support (the budget's ranking)
    """
    missing = [c for c in PATTERN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"pattern frame is missing columns: {missing}")
    out = df.copy()
    n_total = n_attack_total + n_benign_total
    n_fire = out["n_attack"] + out["n_benign"]
    out["support"] = n_fire / n_total if n_total else 0.0
    out["support_attack"] = out["n_attack"] / n_attack_total if n_attack_total else 0.0
    out["support_benign"] = out["n_benign"] / n_benign_total if n_benign_total else 0.0
    out["precision_attack"] = (out["n_attack"] / n_fire.where(n_fire > 0)).fillna(0.0)
    out["lean_precision"] = out["precision_attack"].where(
        out["source_label"] == "attack", 1.0 - out["precision_attack"]
    )
    out.loc[n_fire == 0, "lean_precision"] = 0.0
    out["score"] = out["lean_precision"] * out["support"]
    return out


def _ranked(df: pd.DataFrame) -> pd.DataFrame:
    # Ties: shorter (more readable) patterns first, then a stable string order.
    return (
        df.assign(_len=df["items"].map(len), _key=df["items"].map(repr))
        .sort_values(["score", "_len", "_key"], ascending=[False, True, True])
        .drop(columns=["_len", "_key"])
    )


def select_budget(df: pd.DataFrame, k: int, per_class: bool = False) -> pd.DataFrame:
    """Keep the top-k patterns by score (lean_precision * support).

    per_class=False: one pool. per_class=True: ceil(k/2) attack-leaning and
    the rest benign-leaning; if one class has fewer patterns than its share,
    the other fills the remainder, so every method still gets k features when
    it has k. Which variant the paper uses is an open decision (plan §9).
    Requires add_pattern_stats() first.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if "score" not in df.columns:
        raise ValueError("call add_pattern_stats() before select_budget()")
    if df.empty or k == 0:
        return df.iloc[0:0].copy()
    if not per_class:
        return _ranked(df).head(k).reset_index(drop=True)

    attack = _ranked(df[df["source_label"] == "attack"])
    benign = _ranked(df[df["source_label"] != "attack"])
    k_attack = -(-k // 2)
    take_attack = min(len(attack), k_attack)
    take_benign = min(len(benign), k - take_attack)
    take_attack = min(len(attack), k - take_benign)
    out = pd.concat([attack.head(take_attack), benign.head(take_benign)])
    return _ranked(out).reset_index(drop=True)


_OPERATOR_WORDS = (
    ("<=", "_le_"),
    (">", "_gt_"),
    ("!=", "_ne_"),
    ("=", "_eq_"),
    ("-", "neg"),
)


def _feature_name(tokens: tuple[str, ...], prefix: str, taken: set[str]) -> str:
    """Readable, unique feature name. Operators are spelled out before
    sanitising: plain sanitising maps both "x>3" and "x<=3" (and "scas=1" and
    "scas=-1") to the same name, and the duplicate was silently dropped. Any
    remaining collision gets a short hash of the tokens."""
    readable = []
    for t in tokens:
        for op, word in _OPERATOR_WORDS:
            t = t.replace(op, word)
        readable.append(t)
    name = _make_feature_name(tuple(readable), prefix=prefix)
    if name in taken:
        digest = hashlib.sha1("\x1f".join(tokens).encode()).hexdigest()[:8]
        name = f"{name}__{digest}"
    return name


def patterns_to_schema(
    df: pd.DataFrame,
    schema_name: str,
    schema_version: str,
    item_table: ItemTable | None = None,
) -> SymbolicFeatureSchema:
    """Write a pattern frame into a SymbolicFeatureSchema.

    Atemporal patterns (items from the item table) become an AND of the
    items' AttributePredicate tokens, evaluated by the encoder's predicate
    path; their predicates go into the schema's alphabet. Sequence patterns
    (mining_type in SEQUENCE_MINING_TYPES) keep their ordered items and are
    evaluated in order on the group's look-back sequence.
    """
    features: list[SymbolicFeature] = []
    alphabet: dict[str, AttributePredicate] = {}
    seen_keys: set[tuple[bool, tuple[str, ...]]] = set()
    names: set[str] = set()
    for row in df.itertuples(index=False):
        items = tuple(row.items)
        is_seq = row.mining_type in SEQUENCE_MINING_TYPES
        if is_seq:
            tokens = items
        else:
            if item_table is None:
                raise ValueError("atemporal patterns need the item_table")
            preds = item_table.predicates_for(items)
            for p in preds:
                alphabet.setdefault(p.token, p)
            tokens = tuple(sorted(p.token for p in preds))
        key = (is_seq, tokens)
        if key in seen_keys:
            continue  # same condition reached from different items
        seen_keys.add(key)
        name = _feature_name(tokens, "seq" if is_seq else "sym", names)
        names.add(name)
        features.append(
            SymbolicFeature(
                feature_name=name,
                itemset=tokens,
                source_label=row.source_label,
                support=float(row.support),
                confidence_attack=float(row.support_attack),
                confidence_benign=float(row.support_benign),
                mining_type=row.mining_type,
            )
        )
    return SymbolicFeatureSchema(
        schema_name=schema_name,
        schema_version=schema_version,
        features=features,
        predicates=list(alphabet.values()) or None,
    )

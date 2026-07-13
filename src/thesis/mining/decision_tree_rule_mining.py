from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from thesis.schemas.features import AttributePredicate

if TYPE_CHECKING:
    from thesis.schemas.mining import DecisionTreeRuleConfig


def build_training_matrix(
    X_cat: pd.DataFrame,
    X_num: pd.DataFrame,
    column_predicate_map: dict[str, tuple[str, Any]],
    surviving_categorical_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, tuple[str, Any]]]:
    """
    Select only the categorical columns that survived Step 1 and concatenate
    with all numeric base columns (Similarity, SignatureMatchesPerDay, etc.,
    kept regardless of Step 1's outcome -- the tree does its own numeric
    feature selection via its split criterion). X_cat/X_num/column_predicate_map
    come from Step 1's single pass over alert_groups
    (attribute_contrast_mining.build_categorical_predicate_matrix) -- this is
    pure column selection, no per-group recomputation.
    """
    keep_cols = [c for c in surviving_categorical_columns if c in X_cat.columns]
    X_cat_kept = X_cat[keep_cols]
    kept_predicate_map = {c: column_predicate_map[c] for c in keep_cols}

    X = pd.concat(
        [X_cat_kept.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1
    )
    return X, kept_predicate_map


def fit_rule_tree(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 4,
    min_samples_leaf: int = 20,
    class_weight: str | dict | None = "balanced",
    random_state: int = 0,
    min_impurity_decrease: float = 1e-9,
) -> DecisionTreeClassifier:
    """
    min_impurity_decrease guards against a specific class_weight="balanced"
    artifact: a node that is truly 100% pure (e.g. 0 attack samples) can still
    report impurity ~1e-13 instead of exactly 0, because weighted Gini
    accumulates floating-point rounding error across many reweighted samples.
    That's above sklearn's internal near-zero cutoff (~2.22e-16) that would
    otherwise auto-stop a pure node, so without this the tree "splits" that
    residual noise down to 0 and reports it as a rule with two children that
    are both actually just the same pure class -- zero real discriminative
    content, despite looking like a legitimate leaf pair. 1e-9 sits far above
    that float-noise floor but far below any real split's impurity decrease
    (a genuine split here typically drops impurity by multiple orders of
    magnitude more, e.g. 0.5 -> 0.02 at the root), so real splits are
    unaffected.
    """
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        min_impurity_decrease=min_impurity_decrease,
    )
    tree.fit(X, y)
    return tree


def _split_predicate(
    feature_idx: int,
    threshold: float,
    go_left: bool,
    feature_names: list[str],
    column_predicate_map: dict[str, tuple[str, Any]],
) -> AttributePredicate:
    name = feature_names[feature_idx]
    if name in column_predicate_map:
        attribute, expected_value = column_predicate_map[name]
        fires = not go_left  # right branch (value > 0.5) means the predicate held
        operator = "==" if fires else "!="
        token = name if fires else f"NOT_{name}"
        return AttributePredicate(
            token=token, attribute=attribute, operator=operator, value=expected_value
        )
    operator = "<=" if go_left else ">"
    token = f"{name}_{'le' if go_left else 'gt'}_{threshold:.4g}"
    return AttributePredicate(
        token=token, attribute=name, operator=operator, value=float(threshold)
    )


def _merge_predicate_into_path(
    path: tuple[AttributePredicate, ...], new_pred: AttributePredicate
) -> tuple[AttributePredicate, ...]:
    """
    Add new_pred to path, collapsing it with an earlier bound on the same
    attribute + direction (">" or "<=") instead of appending a duplicate.

    A root-to-leaf path can split on the same continuous attribute more than
    once (CART re-selects whichever feature locally maximises impurity
    reduction at each node independently), e.g. "x <= 173.7" near the root and
    "x <= 143.7" further down. Keeping both is redundant -- the tighter bound
    already implies the looser one -- and reads like the rule fit an exact
    value rather than a threshold. Keep only the tightest bound per
    (attribute, operator).
    """
    if new_pred.operator not in ("<=", ">"):
        return path + (new_pred,)

    merged = []
    already_merged = False
    for pred in path:
        if pred.attribute == new_pred.attribute and pred.operator == new_pred.operator:
            tighter = (
                min(pred, new_pred, key=lambda p: p.value)
                if new_pred.operator == "<="
                else max(pred, new_pred, key=lambda p: p.value)
            )
            merged.append(tighter)
            already_merged = True
        else:
            merged.append(pred)
    if not already_merged:
        merged.append(new_pred)
    return tuple(merged)


def extract_leaf_rules(
    tree: DecisionTreeClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    column_predicate_map: dict[str, tuple[str, Any]],
) -> tuple[pd.DataFrame, list[AttributePredicate]]:
    """
    Walk the fitted tree's structure, decode each leaf's root-to-leaf path
    into AttributePredicate conditions, and compute each leaf's real
    support/confidence from actual sample assignment (tree.apply), not the
    tree's internal node values -- those are class-weighted sums when
    class_weight is set, not raw counts.
    """
    tree_ = tree.tree_
    feature_names = list(X.columns)
    leaf_ids = tree.apply(X)
    y_arr = y.to_numpy()
    n_attack = int((y_arr == 1).sum())
    n_benign = int((y_arr == 0).sum())
    n_total = len(X)

    leaf_paths: dict[int, tuple[AttributePredicate, ...]] = {}

    def _walk(node_id: int, path: tuple[AttributePredicate, ...]) -> None:
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left == right:
            leaf_paths[node_id] = path
            return
        feature_idx = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        left_pred = _split_predicate(
            feature_idx, threshold, True, feature_names, column_predicate_map
        )
        right_pred = _split_predicate(
            feature_idx, threshold, False, feature_names, column_predicate_map
        )
        _walk(left, _merge_predicate_into_path(path, left_pred))
        _walk(right, _merge_predicate_into_path(path, right_pred))

    _walk(0, ())

    rows = []
    predicate_alphabet: dict[str, AttributePredicate] = {}
    for leaf_id, path in leaf_paths.items():
        mask = leaf_ids == leaf_id
        n_samples = int(mask.sum())
        if n_samples == 0:
            continue
        n_attack_leaf = int((y_arr[mask] == 1).sum())
        n_benign_leaf = n_samples - n_attack_leaf

        for pred in path:
            predicate_alphabet.setdefault(pred.token, pred)

        rows.append(
            {
                "itemset": tuple(p.token for p in path),
                "support": n_samples / n_total if n_total else 0.0,
                "support_count": n_samples,
                "confidence_attack": n_attack_leaf / n_attack if n_attack else 0.0,
                "confidence_benign": n_benign_leaf / n_benign if n_benign else 0.0,
                "mining_type": "decision_tree_rule",
                "leaf_id": int(leaf_id),
                "depth": len(path),
                "n_attack": n_attack_leaf,
                "n_benign": n_benign_leaf,
            }
        )

    return pd.DataFrame(rows), list(predicate_alphabet.values())


def _fit_and_tag_leaves(
    X: pd.DataFrame,
    y: pd.Series,
    column_predicate_map: dict[str, tuple[str, Any]],
    max_depth: int,
    tree_config: "DecisionTreeRuleConfig",
) -> tuple[pd.DataFrame, list[AttributePredicate]]:
    tree = fit_rule_tree(
        X,
        y,
        max_depth=max_depth,
        min_samples_leaf=tree_config.min_samples_leaf,
        class_weight=tree_config.class_weight,
        random_state=tree_config.random_state,
        min_impurity_decrease=tree_config.min_impurity_decrease,
    )
    leaves_df, predicates = extract_leaf_rules(tree, X, y, column_predicate_map)
    if not leaves_df.empty:
        leaves_df["source_label"] = np.where(
            leaves_df["confidence_attack"] > leaves_df["confidence_benign"],
            "attack",
            "benign",
        )
    return leaves_df, predicates


def fit_and_extract_rules(
    X: pd.DataFrame,
    y: pd.Series,
    column_predicate_map: dict[str, tuple[str, Any]],
    tree_config: "DecisionTreeRuleConfig",
) -> tuple[pd.DataFrame, list[AttributePredicate]]:
    """
    Fit Step 2's decision tree(s) and return leaf rules, already tagged with
    source_label -- the caller doesn't need to re-derive it the way it still
    must for Step 1's survivors_df.

    Single-tree mode (tree_config.max_depth_attack is None): one tree fit at
    max_depth, every leaf kept regardless of class. Identical to calling
    fit_rule_tree + extract_leaf_rules directly, the behavior every existing
    caller (attribute_mining_job.py, monitor_drift.py,
    dynamic_schema_service.py) already relies on -- those callers are
    unaffected by this function existing since they never set
    max_depth_attack.

    Two-tree mode (tree_config.max_depth_attack is not None): two trees are
    fit on the same (X, y) -- one at max_depth (kept for its benign-leaning
    leaves), one at max_depth_attack (kept for its attack-leaning leaves) --
    so the two classes' leaf quality can be tuned independently instead of
    one shared depth trying to serve both at once. See
    attribute_mining_sweep_eda.ipynb section 5.3 for why a shared max_depth
    can't do this: mean_tree_precision_attack and mean_tree_recall_benign
    move in opposite directions as depth increases, so no single depth
    clears both a precision floor and a recall floor simultaneously. The
    returned predicate alphabet is the union of both trees' alphabets
    (deduplicated by token, same rule extract_leaf_rules itself uses),
    restricted to the predicates that survived onto a *kept* leaf's path --
    a predicate only used by a discarded leaf (e.g. an attack-leaning split
    in the benign tree) isn't included.
    """
    if tree_config.max_depth_attack is None:
        return _fit_and_tag_leaves(
            X, y, column_predicate_map, tree_config.max_depth, tree_config
        )

    benign_leaves, benign_predicates = _fit_and_tag_leaves(
        X, y, column_predicate_map, tree_config.max_depth, tree_config
    )
    benign_leaves = benign_leaves[benign_leaves["source_label"] == "benign"]

    attack_leaves, attack_predicates = _fit_and_tag_leaves(
        X, y, column_predicate_map, tree_config.max_depth_attack, tree_config
    )
    attack_leaves = attack_leaves[attack_leaves["source_label"] == "attack"]

    leaves_df = pd.concat([benign_leaves, attack_leaves], ignore_index=True)

    kept_tokens = {tok for itemset in leaves_df["itemset"] for tok in itemset}
    predicate_alphabet: dict[str, AttributePredicate] = {}
    for pred in benign_predicates + attack_predicates:
        if pred.token in kept_tokens:
            predicate_alphabet.setdefault(pred.token, pred)

    return leaves_df, list(predicate_alphabet.values())

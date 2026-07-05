from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from thesis.schemas.features import AttributePredicate


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
) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
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
        _walk(left, path + (left_pred,))
        _walk(right, path + (right_pred,))

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
            }
        )

    return pd.DataFrame(rows), list(predicate_alphabet.values())

"""Cross-model feature importance comparison utilities."""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr


def top_k_names(importances: dict[str, float], k: int) -> set[str]:
    """Top-k feature names by absolute importance, excluding zeros."""
    ranked = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {name for name, imp in ranked[:k] if imp != 0}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def spearman_rank(imp_a: dict[str, float], imp_b: dict[str, float]) -> float:
    """Spearman rank correlation of absolute importances over the union of features."""
    common = sorted(set(imp_a) | set(imp_b))
    if len(common) < 2:
        return float("nan")
    a_vals = [abs(imp_a.get(f, 0.0)) for f in common]
    b_vals = [abs(imp_b.get(f, 0.0)) for f in common]
    corr, _ = spearmanr(a_vals, b_vals)
    return float(corr)


def compare_model_importances(
    importances: dict[str, dict[str, float]],
    top_k: int = 25,
) -> pd.DataFrame:
    """
    Pairwise Jaccard overlap and Spearman rank correlation across models.

    Args:
        importances: {model_name: {feature_name: importance_value}}
        top_k: number of top features to use for Jaccard

    Returns:
        DataFrame with columns: model_a, model_b, jaccard_top_k, spearman_rho,
        shared_features, top_k
    """
    model_names = list(importances.keys())
    rows = []
    for i, ma in enumerate(model_names):
        for j, mb in enumerate(model_names):
            if i >= j:
                continue
            set_a = top_k_names(importances[ma], top_k)
            set_b = top_k_names(importances[mb], top_k)
            rows.append(
                {
                    "model_a": ma,
                    "model_b": mb,
                    "jaccard_top_k": jaccard(set_a, set_b),
                    "spearman_rho": spearman_rank(importances[ma], importances[mb]),
                    "shared_features": len(set_a & set_b),
                    "top_k": top_k,
                }
            )
    return pd.DataFrame(rows)


def feature_rank_table(
    importances: dict[str, dict[str, float]],
    top_k: int = 25,
) -> pd.DataFrame:
    """
    Side-by-side ranking table: each model gets a feature and importance column.

    Args:
        importances: {model_name: {feature_name: importance_value}}
        top_k: number of top features per model to include

    Returns:
        DataFrame with columns <model>_feature and <model>_importance for each model
    """
    cols: dict[str, list] = {}
    max_len = 0
    for model_name, imp in importances.items():
        ranked = sorted(imp.items(), key=lambda kv: abs(kv[1]), reverse=True)
        nonzero = [(name, val) for name, val in ranked if val != 0][:top_k]
        cols[f"{model_name}_feature"] = [name for name, _ in nonzero]
        cols[f"{model_name}_importance"] = [round(val, 6) for _, val in nonzero]
        max_len = max(max_len, len(nonzero))
    for key in cols:
        while len(cols[key]) < max_len:
            cols[key].append(None)
    return pd.DataFrame(cols)


def shared_core_features(
    importances: dict[str, dict[str, float]],
    top_k: int = 25,
) -> set[str]:
    """Features that appear in the top-k of every model."""
    if not importances:
        return set()
    sets = [top_k_names(imp, top_k) for imp in importances.values()]
    return set.intersection(*sets)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, roc_auc_score
import re


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _safe_name(s: str) -> str:
    # keep letters, numbers, underscore, dash
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s)).strip("_")


# ROC curve
def plot_roc(y_true, proba, d, title_suffix="", out_dir="../plots"):
    out_dir = _ensure_dir(out_dir)
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve. Lookback window {d} days")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"roc_{d}{_safe_name(title_suffix)}.png"))
    plt.show()


# Alert reduction vs missed attacks
def plot_alert_reduction(y_true, proba, d, title_suffix="", out_dir="../plots"):
    out_dir = _ensure_dir(out_dir)

    thresholds = np.linspace(0, 1, 50)
    reduction, fnr = [], []

    for th in thresholds:
        y_pred = (proba >= th).astype(int)
        reduction.append((y_pred == 0).mean())
        fnr.append(((y_pred == 0) & (y_true == 1)).sum() / max(1, (y_true == 1).sum()))

    plt.figure()
    plt.plot(reduction, fnr)
    plt.xlabel("Alert Reduction")
    plt.ylabel("False Negative Rate")
    plt.title(f"Alert Reduction vs Missed Attacks. Lookback window {d} days")

    plt.savefig(os.path.join(out_dir, f"reduction_{d}{_safe_name(title_suffix)}.png"))

    plt.show()


# Feature importance (logistic regression)
def plot_feature_importance(model, X, d, title_suffix="", out_dir="../plots"):
    out_dir = _ensure_dir(out_dir)

    coef = pd.Series(model.coef_[0], index=X.columns)
    coef.sort_values().plot(kind="barh", figsize=(8, 6))
    plt.title(f"Feature Importance. Lookback window {d} days")

    plt.savefig(os.path.join(out_dir, f"features_{d}{_safe_name(title_suffix)}.png"))
    plt.show()


# Prediction confidence distribution
def plot_confidence_distribution(proba, d, title_suffix="", out_dir="../plots"):
    out_dir = _ensure_dir(out_dir)

    plt.figure()
    plt.hist(proba, bins=50)
    plt.xlabel("Predicted attack probability")
    plt.ylabel("Count")
    plt.title(f"Prediction Confidence Distribution. Lookback window {d} days")

    plt.savefig(os.path.join(out_dir, f"confidence_{d}{_safe_name(title_suffix)}.png"))

    plt.show()


# Error analysis by alert category
def plot_top_error_categories(
    df_used, y_true, proba, d, top_k=10, out_dir="../plots", prefix=""
):
    out_dir = _ensure_dir(out_dir)

    df_err = df_used.copy()
    df_err["pred"] = (proba >= 0.5).astype(int)
    df_err["y"] = y_true

    errors = df_err[df_err["pred"] != df_err["y"]]
    errors["category"].value_counts().head(top_k).plot(kind="bar")
    plt.title("Most Misclassified Alert Categories")
    plt.savefig(os.path.join(out_dir, f"error_analysis_{d}.png"))


def plot_symbolic_performance_delta(
    results_dict, metric="auc", out_dir="../plots", prefix=""
):
    """
    results_dict: dict like
      {
        "base": {"auc": ..., "auc_auth": ...},
        "is_suspicious_auth_burst": {...},
        ...
      }
    metric: "auc" or "auc_auth"
    """
    df = pd.DataFrame(results_dict).T

    base_val = df.loc["base", metric]
    df = df.drop(index="base")

    delta = df[metric] - base_val

    plt.figure(figsize=(8, 4))
    delta.sort_values().plot(kind="bar")
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel(f"Δ {metric.upper()} vs baseline")
    plt.title(f"Impact of symbolic features on {metric.upper()}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    fname = f"{_safe_name(prefix)}sym_delta.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.show()


# def plot_symbolic_coefficients(model, feature_names, symbolic_features, out_dir="../plots", prefix=""):
#     coef = pd.Series(model.coef_[0], index=feature_names)
#     coef_sym = coef[symbolic_features].sort_values()

#     plt.figure(figsize=(6, 3))
#     coef_sym.plot(kind="barh")
#     plt.axvline(0, color="black", linewidth=1)
#     plt.xlabel("Logistic regression coefficient")
#     plt.title("Model reliance on symbolic features")
#     plt.tight_layout()

#     plt.savefig(os.path.join(out_dir, f"sym_coefficients.png"))
#     plt.show()


def plot_symbolic_coefficients(
    model, feature_names, symbolic_features, out_dir="../plots", prefix=""
):
    coef = pd.Series(model.coef_[0], index=feature_names)
    coef_sym = coef[symbolic_features].sort_values()

    print("coef_sym: ", coef_sym)

    # plt.figure(figsize=(6, 2))
    # coef_sym.plot(kind="barh")
    # plt.axvline(0, color="black", linewidth=1)
    # plt.xlabel("Logistic regression coefficient")
    # plt.title(f"Symbolic coefficient: {prefix}")
    # plt.tight_layout()

    # fname = f"sym_coeff_{prefix}.png" if prefix else "sym_coefficients.png"
    # plt.savefig(os.path.join(out_dir, fname))
    # plt.show()


def plot_symbolic_score_shift(
    X_full, res_base, res_sym, feature, out_dir="../plots", prefix=""
):
    out_dir = _ensure_dir(out_dir)

    split = res_base["test_idx_start"]
    X_test = X_full.iloc[split:].reset_index(drop=True)
    mask = X_test[feature].fillna(0).astype(int) == 1

    if mask.sum() == 0:
        print(f"No test samples for feature {feature}")
        return

    s_base = res_base["proba_test"][mask]
    s_sym = res_sym["proba_test"][mask]

    plt.figure(figsize=(6, 4))
    plt.hist(s_base, bins=30, alpha=0.5, label="base")
    plt.hist(s_sym, bins=30, alpha=0.5, label="base + symbolic")
    plt.legend()
    plt.xlabel("Predicted attack probability")
    plt.ylabel("Number of alerts")
    plt.title(f"Score shift for {feature}")
    plt.tight_layout()

    fname = f"{_safe_name(prefix)}score_shift_{_safe_name(feature)}.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.show()


def plot_all(X, results, d, run_name="default"):
    out_dir = _ensure_dir(os.path.join("../plots", _safe_name(run_name)))

    plot_roc(results["y_true"], results["proba"], d, out_dir=out_dir)
    plot_alert_reduction(results["y_true"], results["proba"], d, out_dir=out_dir)
    plot_feature_importance(results["model"], X, d, out_dir=out_dir)
    plot_confidence_distribution(results["proba"], d, out_dir=out_dir)

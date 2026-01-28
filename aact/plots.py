import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, roc_auc_score

# ROC curve
def plot_roc(y_true, proba, d, title_suffix=""):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve. Lookback window {d} days")
    plt.legend()
    plt.savefig(f"../plots/roc_{d}.png")
    plt.show()

# Alert reduction vs missed attacks
def plot_alert_reduction(y_true, proba, d):
    thresholds = np.linspace(0, 1, 50)
    reduction, fnr = [], []

    for th in thresholds:
        y_pred = (proba >= th).astype(int)
        reduction.append((y_pred == 0).mean())
        fnr.append(
            ((y_pred == 0) & (y_true == 1)).sum()
            / max(1, (y_true == 1).sum())
        )

    plt.figure()
    plt.plot(reduction, fnr)
    plt.xlabel("Alert Reduction")
    plt.ylabel("False Negative Rate")
    plt.title(f"Alert Reduction vs Missed Attacks. Lookback window {d} days")
    
    plt.savefig(f"../plots/reduction_{d}.png")

    plt.show()


# Feature importance (logistic regression)
def plot_feature_importance(model, X, d):
    coef = pd.Series(model.coef_[0], index=X.columns)
    coef.sort_values().plot(kind="barh", figsize=(8, 6))
    plt.title(f"Feature Importance. Lookback window {d} days")

    plt.savefig(f"../plots/features_{d}.png", )
    plt.show()


# Prediction confidence distribution
def plot_confidence_distribution(proba, d):
    plt.figure()
    plt.hist(proba, bins=50)
    plt.xlabel("Predicted attack probability")
    plt.ylabel("Count")
    plt.title(f"Prediction Confidence Distribution. Lookback window {d} days")

    plt.savefig(f"../plots/confidence_{d}.png")

    plt.show()


# Error analysis by alert category
def plot_top_error_categories(df_used, y_true, proba, top_k=10):
    df_err = df_used.copy()
    df_err["pred"] = (proba >= 0.5).astype(int)
    df_err["y"] = y_true

    errors = df_err[df_err["pred"] != df_err["y"]]
    errors["category"].value_counts().head(top_k).plot(kind="bar")
    plt.title("Most Misclassified Alert Categories")
    plt.show()

def plot_all(X, results, d):
    output_dir = "../plots/"
    os.makedirs(output_dir, exist_ok=True)
    plot_roc(results["y_true"], results["proba"], d)
    plot_alert_reduction(results["y_true"], results["proba"], d)
    plot_feature_importance(results["model"], X, d)
    plot_confidence_distribution(results["proba"], d)

def plot_symbolic_performance_delta(results_dict, metric="auc"):
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
    plt.show()

def plot_symbolic_coefficients(model, feature_names, symbolic_features):
    coef = pd.Series(model.coef_[0], index=feature_names)
    coef_sym = coef[symbolic_features].sort_values()

    plt.figure(figsize=(6, 3))
    coef_sym.plot(kind="barh")
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Logistic regression coefficient")
    plt.title("Model reliance on symbolic features")
    plt.tight_layout()
    plt.show()

def plot_symbolic_score_shift(X_full, res_base, res_sym, feature):
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
    plt.title(f"Score shift for {feature}")
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, roc_auc_score
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

# -------------------------
# Loading + preprocessing + formatting helpers
# -------------------------
def _load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def _safe_name(s: str) -> str:
    # keep letters, numbers, underscore, dash
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s)).strip("_")

def _as_dict(x):
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def load_fp_only_history_jsonl(jsonl_path: str) -> pd.DataFrame:
    """Load all_history.jsonl and keep only fp_only rows (if present)."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "mode" in df.columns:
        df = df[df["mode"] == "fp_only"].copy()
    return df

def prepare_fp_only_window_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses train_window -> train_start/train_end and flattens suppression_all_or into numeric columns.
    Returns a sorted dataframe ready for plotting.
    """
    d = df.copy()

    arrow = "→"
    d["train_start"] = pd.to_datetime(d["train_window"].str.split(arrow).str[0], errors="coerce")
    d["train_end"]   = pd.to_datetime(d["train_window"].str.split(arrow).str[1], errors="coerce")
    d = d.sort_values(["scenario", "train_start", "k"]).reset_index(drop=True)

    d["suppression_all_or"] = d["suppression_all_or"].apply(_as_dict)

    cols = [
        "supp_rate_total","supp_rate_benign","supp_rate_attack",
        "suppressed_next_total","suppressed_next_benign","suppressed_next_attack",
        "total_next","total_benign_next","total_attack_next"
    ]
    for col in cols:
        d[col] = d["suppression_all_or"].apply(lambda dd: dd.get(col, np.nan))

    for col in cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    return d

def flatten_fp_only_history(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide/nested FP-only history into long tidy format:

    Output columns:
      scenario, k, train_window, feature,
      suppressed_benign, suppressed_attack, suppressed_total,
      total_benign, total_attack, total,
      supp_rate_benign, supp_rate_attack, supp_rate_total
    """
    d = df_hist.copy()

    # Keep fp-only rows that actually have per-feature dicts
    d = d[d["mode"] == "fp_only"].copy()
    d["suppression_per_feat"] = d["suppression_per_feat"].apply(_as_dict)

    rows = []
    for _, r in d.iterrows():
        scen = r.get("scenario")
        k = r.get("k")
        tw = r.get("train_window")

        per_feat = r["suppression_per_feat"] or {}
        for feat, stats in per_feat.items():
            stats = _as_dict(stats)

            rows.append({
                "scenario": scen,
                "k": k,
                "train_window": tw,
                "feature": feat,

                "suppressed_benign": stats.get("suppressed_next_benign", np.nan),
                "suppressed_attack": stats.get("suppressed_next_attack", np.nan),
                "suppressed_total":  stats.get("suppressed_next_total", np.nan),

                "total_benign": stats.get("total_benign_next", np.nan),
                "total_attack": stats.get("total_attack_next", np.nan),
                "total":        stats.get("total_next", np.nan),

                "supp_rate_benign": stats.get("supp_rate_benign", np.nan),
                "supp_rate_attack": stats.get("supp_rate_attack", np.nan),
                "supp_rate_total":  stats.get("supp_rate_total", np.nan),
            })

    out = pd.DataFrame(rows)

    # Ensure numeric
    num_cols = [c for c in out.columns if c not in ("scenario", "train_window", "feature")]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

# -------------------------
# Plot functions
# -------------------------

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
    results_dict, scenario_name, metric="auc", out_dir="../plots", prefix=""
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

    # sort by signed delta for readability
    delta_sorted = delta.sort_values()

    # colors: green = improvement, red = worse
    colors = ["green" if v > 0 else "red" for v in delta_sorted]

    # plot absolute magnitude (log-safe)
    # plt.yscale("log")
    # TODO: change yticks back to positive
    plt.bar(delta_sorted.index, delta_sorted.abs(), color=colors)

    plt.ylabel(f"|Δ {metric.upper()}| vs baseline")
    plt.title(f"Delta AUC (abl.) symbolic features on {metric.upper()}, scenario={scenario_name}")

    plt.xticks(fontsize=6)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    fname = f"{scenario_name}_sym_delta.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.show()


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
    plt.title(f"Score shift ({feature}) on all test-set alerts with feature active.")
    plt.tight_layout()

    fname = f"{_safe_name(prefix)}score_shift_{_safe_name(feature)}.png"
    plt.savefig(os.path.join(out_dir, fname))


# function that plots score-shifts separately for true attacks (y=1) and benign (y=0)
# within the same symbolic subset (feature=1) on the test split
def plot_symbolic_score_shift_by_label(
    X_full, y, res_base, res_sym, feature, out_dir="../plots", prefix="", bins=30
):
    out_dir = _ensure_dir(out_dir)

    split = res_base["test_idx_start"]

    X_test = X_full.iloc[split:].reset_index(drop=True)
    y_test = np.asarray(y)[split:]
    y_test = y_test.reshape(-1)

    # symbolic subset
    feat_mask = X_test[feature].fillna(0).astype(int).values == 1

    if feat_mask.sum() == 0:
        print(f"No test samples where {feature}=1")
        return

    # scores
    s_base = np.asarray(res_base["proba_test"])
    s_sym = np.asarray(res_sym["proba_test"])

    # labels within subset
    attack_mask = feat_mask & (y_test == 1)
    benign_mask = feat_mask & (y_test == 0)

    # helper to plot one panel
    def _panel(mask, title_suffix, fname_suffix):
        n = int(mask.sum())
        if n == 0:
            print(f"No samples for {feature}=1 with y={title_suffix}")
            return

        plt.figure(figsize=(6, 4))
        plt.hist(s_base[mask], bins=bins, alpha=0.5, label="base")
        plt.hist(s_sym[mask], bins=bins, alpha=0.5, label="base + symbolic")
        plt.legend()
        plt.xlabel("Predicted attack probability")
        plt.ylabel("Number of alerts")
        plt.title(f"{feature} = 1, y = {title_suffix} (n={n})")
        plt.tight_layout()

        fname = (
            f"{_safe_name(prefix)}score_shift_{_safe_name(feature)}_{fname_suffix}.png"
        )
        plt.savefig(os.path.join(out_dir, fname))

    # y=1 (true attacks) and y=0 (benign)
    _panel(attack_mask, "1 (attack)", "y1_attack")
    _panel(benign_mask, "0 (benign)", "y0_benign")


def plot_fp_to_tn(
    X_full,
    y,
    res_base,
    res_sym,
    feature,
    threshold=0.5,
):
    """
    Compare base vs base+symbolic on test alerts where feature == 1.

    Returns:
        dict with:
        #   - fn_to_tp
          - fp_to_tn
        #   - delta_fn
          - delta_fp
          - net_alert_change
          - n_subset
    """

    split = res_base["test_idx_start"]

    X_test = X_full.iloc[split:].reset_index(drop=True)
    y_test = np.asarray(y)[split:]

    mask = X_test[feature].fillna(0).astype(int).values == 1
    if mask.sum() == 0:
        return None

    p_base = np.asarray(res_base["proba_test"])
    p_sym = np.asarray(res_sym["proba_test"])

    yb = y_test[mask]
    pb = p_base[mask]
    ps = p_sym[mask]

    pred_base = (pb >= threshold).astype(int)
    pred_sym = (ps >= threshold).astype(int)

    # error flips
    # fn_to_tp = int(((yb == 1) & (pred_base == 0) & (pred_sym == 1)).sum())
    fp_to_tn = int(((yb == 0) & (pred_base == 1) & (pred_sym == 0)).sum())

    # net changes
    # delta_fn = int(((yb == 1) & (pred_sym == 0)).sum() -
    #                ((yb == 1) & (pred_base == 0)).sum())

    delta_fp = int(
        ((yb == 0) & (pred_sym == 1)).sum() - ((yb == 0) & (pred_base == 1)).sum()
    )

    net_alert_change = int(pred_sym.sum() - pred_base.sum())

    stats = {
        "feature": feature,
        "n_subset": int(mask.sum()),
        # "fn_to_tp": fn_to_tp,
        "fp_to_tn": fp_to_tn,
        # "delta_fn": delta_fn,
        "delta_fp": delta_fp,
        "net_alert_change": net_alert_change,
    }

    return stats

def plot_per_feature_fp_suppression_timeseries(
    jsonl_path,
    scenario_name,
    out_dir,
    use_memory,
    tau,
    topk=12,
    value_key="suppressed_next_benign",   # or "suppressed_next_total"
    min_windows_on=1,                     # filter features that appear in >= this many windows
):
    
    out_path = os.path.join(out_dir, scenario_name)
    os.makedirs(out_path, exist_ok=True)
    print(f"Saving plot to {out_path}...")
    
    # -------- load jsonl --------
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    # keep fp_only + scenario
    df = df[(df["mode"] == "fp_only") & (df["scenario"] == scenario_name)].copy()
    if df.empty:
        print(f"No fp_only rows found for scenario='{scenario_name}'.")
        return

    # parse train_start for x-axis
    arrow = "→"
    df["train_start"] = pd.to_datetime(df["train_window"].str.split(arrow).str[0], errors="coerce")
    df = df.sort_values(["train_start", "k"]).reset_index(drop=True)

    # -------- flatten suppression_per_feat into long format --------
    df["suppression_per_feat"] = df["suppression_per_feat"].apply(_as_dict)

    long_rows = []
    for _, r in df.iterrows():
        per_feat = r["suppression_per_feat"] or {}
        for feat, stats in per_feat.items():
            stats = _as_dict(stats)
            long_rows.append({
                "train_start": r["train_start"],
                "k": r["k"],
                "feature": feat,
                "value": stats.get(value_key, np.nan),
            })

    long = pd.DataFrame(long_rows)
    if long.empty:
        print(f"No per-feature stats found in suppression_per_feat for scenario='{scenario_name}'.")
        return

    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # pick top-k features by total suppression over time (so plot isn't unreadable)
    feat_totals = (
        long.groupby("feature")["value"]
        .sum(min_count=1)
        .sort_values(ascending=False)
    )

    # optional: require feature to show up in enough windows
    feat_counts = long.groupby("feature")["value"].apply(lambda s: s.notna().sum())
    keep = feat_counts[feat_counts >= min_windows_on].index
    feat_totals = feat_totals.loc[feat_totals.index.intersection(keep)]

    feats = feat_totals.head(topk).index.tolist()
    plot_df = long[long["feature"].isin(feats)].copy()

    # -------- plot --------
    plt.figure(figsize=(12, 6))
    for feat, g in plot_df.groupby("feature"):
        g = g.sort_values("train_start")
        plt.plot(g["train_start"], g["value"], marker="o", linewidth=1.5, label=feat)

    plt.xlabel("Train window start date")
    plt.ylabel(f"{value_key} (single feature on NEXT window)")
    plt.title(f"{scenario_name}: per-feature FP suppression timeseries (useMem={use_memory},tau={tau})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=5, ncol=1)
    plt.tight_layout()

    plt.savefig(out_path)

def plot_suppression_rates(df_prepped: pd.DataFrame, figsize=(12, 6), legend_ncol=2):
    """Plot benign vs attack suppression rate over time per scenario (from suppression_all_or)."""
    plt.figure(figsize=figsize)
    for scenario, g in df_prepped.groupby("scenario"):
        plt.plot(g["train_start"], g["supp_rate_benign"], marker="o", linestyle="-", label=f"{scenario} benign")
        plt.plot(g["train_start"], g["supp_rate_attack"], marker="x", linestyle="--", label=f"{scenario} attack")

    plt.ylim(-0.02, 1.02)
    plt.xlabel("Train window start date")
    plt.ylabel("Suppression rate on NEXT window (OR of all candidate rules)")
    plt.title("FP-only symbolic filter performance over time (next window)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=legend_ncol, fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_suppressed_counts(df_prepped: pd.DataFrame, figsize=(12, 6), legend_ncol=2):
    """Plot benign vs attack suppressed counts over time per scenario (from suppression_all_or)."""
    plt.figure(figsize=figsize)
    for scenario, g in df_prepped.groupby("scenario"):
        plt.plot(g["train_start"], g["suppressed_next_benign"], marker="o", linestyle="-", label=f"{scenario} suppressed benign")
        plt.plot(g["train_start"], g["suppressed_next_attack"], marker="x", linestyle="--", label=f"{scenario} suppressed attack")

    plt.xlabel("Train window start date")
    plt.ylabel("# suppressed in NEXT window (OR of all candidate rules)")
    plt.title("How many alerts are suppressed by the FP-only symbolic filter (next window)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=legend_ncol, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_tradeoff_scatter(df_prepped: pd.DataFrame, figsize=(7.5, 6)):
    """Scatter of attack suppression rate vs benign suppression rate (each point = one window)."""
    plt.figure(figsize=figsize)
    for scenario, g in df_prepped.groupby("scenario"):
        plt.scatter(g["supp_rate_attack"], g["supp_rate_benign"], label=scenario, alpha=0.8)

    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Attack suppression rate (lower is better)")
    plt.ylabel("Benign suppression rate (higher is better)")
    plt.title("Filter trade-off per window (next window, OR of all candidate rules)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_fp_only_feature_suppression(
    df_hist_or_path,
    out_dir,
    use_memory,
    tau,
    topk: int = 15,
    metric: str = "suppressed_benign",   # or "supp_rate_benign"
    agg: str = "sum",                    # "sum" or "mean"
    show_attack_suppressed: bool = True  # annotate attack suppressed on bars
):
    """
    Plot per scenario: symbolic features ranked by FP suppression.

    metric:
      - "suppressed_benign" (recommended): absolute # benign suppressed across windows
      - "supp_rate_benign": average benign suppression rate (less stable across varying totals)

    agg:
      - "sum": sums metric across windows (good for absolute impact)
      - "mean": mean across windows (good if windows comparable)
    """
    if isinstance(df_hist_or_path, str):
        df_hist = _load_jsonl(df_hist_or_path)
    else:
        df_hist = df_hist_or_path

    out_dir = os.path.join(out_dir, "rankings")
    os.makedirs(out_dir, exist_ok=True)

    long = flatten_fp_only_history(df_hist)
    if long.empty:
        print("No fp_only per-feature rows found. Check that 'suppression_per_feat' is present.")
        return long

    if metric not in long.columns:
        raise KeyError(f"Unknown metric '{metric}'. Available: {list(long.columns)}")

    agg_fn = np.sum if agg == "sum" else np.mean
    group_cols = ["scenario", "feature"]

    summary = (
        long.groupby(group_cols, as_index=False)
            .agg({
                metric: agg_fn,
                "suppressed_attack": np.sum,   # useful safety signal
                "suppressed_total": np.sum,
                "total_benign": np.sum,
                "total_attack": np.sum,
                "total": np.sum,
            })
    )

    for scen in sorted(summary["scenario"].dropna().unique()):
        fname = scen
        sub = summary[summary["scenario"] == scen].copy()
        sub = sub.sort_values(metric, ascending=False).head(topk)
        sub = sub.iloc[::-1]  # nicer barh order

        plt.figure(figsize=(10, 6))
        plt.barh(sub["feature"], sub[metric])
        plt.title(f"{scen} — top {topk} sym feats by {agg}({metric}), (useMem={use_memory},tau={tau})")
        plt.xlabel(f"{agg}({metric})")
        plt.xscale("log")
        plt.grid(axis="x", alpha=0.3)

        if show_attack_suppressed and "suppressed_attack" in sub.columns:
            # annotate attacks suppressed (safety) on the bars
            for i, (_, row) in enumerate(sub.iterrows()):
                val = row[metric]
                atk = int(row["suppressed_attack"]) if np.isfinite(row["suppressed_attack"]) else 0
                plt.text(val if np.isfinite(val) else 0, i, f"  attack_supp={atk}", va="center")

        plt.tight_layout()


        plt.savefig(os.path.join(out_dir, scen))

    return long, summary

def plot_suppression_from_res(
    res,
    scenario_name: str,
    fp_metric: str = "suppressed_benign",
    tp_metric: str = "suppressed_attack",
    out_dir: str = "../plots",
    topk: int | None = None,
):
    """
    Plot per-feature FP and TP suppression from simple_ablation_experiment output.

    Expects:
        res["diag_ablation"] = {feature_name: diag_dict}
        res["diag_sym_all"] = diag_dict
    """
    rows = []

    for feat, diag in res["diag_ablation"].items():
        rows.append(
            {
                "feature": feat,
                "fp_suppressed": diag.get(fp_metric, 0),
                "tp_suppressed": diag.get(tp_metric, 0),
            }
        )

    # optional: include all-symbolic bundle
    diag_all = res.get("diag_sym_all", {})
    rows.append(
        {
            "feature": "all_symbolic",
            "fp_suppressed": diag_all.get(fp_metric, 0),
            "tp_suppressed": diag_all.get(tp_metric, 0),
        }
    )

    df_plot = pd.DataFrame(rows)

    if topk is not None:
        df_plot = df_plot.sort_values("fp_suppressed", ascending=False).head(topk)

    df_plot = df_plot.sort_values("fp_suppressed")

    os.makedirs(out_dir, exist_ok=True)

    y = np.arange(len(df_plot))
    h = 0.4

    plt.figure(figsize=(10, max(4, 0.35 * len(df_plot))))
    plt.barh(y - h / 2, df_plot["fp_suppressed"], height=h, label="FP suppressed")
    plt.barh(y + h / 2, df_plot["tp_suppressed"], height=h, label="TP suppressed")

    plt.yticks(y, df_plot["feature"])
    plt.xlabel("Count")
    plt.title(f"{scenario_name} — FP/TP suppression per symbolic feature")
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{scenario_name}_fp_tp_suppression.png"))
    plt.show()

    return df_plot

def plot_class_histogram(df, label_col="y"):
    counts = df[label_col].value_counts().sort_index()

    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.show()

def plot_token_semantic_scatter(
    ranking: pd.DataFrame,
    min_support,
    token_col: str = "token",
    p_col: str = "p_benign_given_token",
    benign_thresh: float = 0.60,
    attack_thresh: float = 0.40,
    max_points: int = 5000,
    random_state: int = 0,
):
    """
    Semantic layout: TF-IDF (char ngrams) -> cosine distances -> t-SNE 2D
    Coloring: 3 categories based on p_benign_given_token
      - benign: p >= benign_thresh
      - attack: p <= attack_thresh
      - neutral: otherwise
    """

    df = ranking[[token_col, p_col]].dropna().copy()

    # Optional: cap points for speed (keeps most supported ones if present)
    if "support_total" in ranking.columns:
        df = ranking[[token_col, p_col, "support_total"]].dropna().sort_values(
            "support_total", ascending=False
        )
        df = df.head(max_points).copy()
    else:
        df = df.head(max_points).copy()

    tokens = df[token_col].astype(str).tolist()
    p = df[p_col].astype(float).to_numpy()

    # 1) "Semantic" representation (works well for short strings like tokens)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(tokens)

    # 2) Pairwise cosine distance (t-SNE can use precomputed distances)
    D = cosine_distances(X)

    # 3) 2D embedding
    perplexity = min(30, max(5, (len(tokens) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=random_state,
    )
    Z = tsne.fit_transform(D)

    # 4) 3 categories from p_benign_given_token
    labels = np.full(len(p), "neutral", dtype=object)
    labels[p >= benign_thresh] = "benign"
    labels[p <= attack_thresh] = "attack"

    # 5) Plot (no manual colors; matplotlib picks defaults)
    plt.figure()
    for lab in ["benign", "neutral", "attack"]:
        m = labels == lab
        plt.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, label=lab)

    plt.title(f"Semantic scatter of tokens (min_support={min_support})")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df.assign(tsne_x=Z[:, 0], tsne_y=Z[:, 1], category=labels)

def plot_scenario_heatmap(
    rankings,
    attack_flags,
    output_dir,
    scenario,
    miner_name,
    scorer_name,
    min_total_support,
    top_n=25,
    order="desc",
    score_col="combined_score",
    support_col="support_total",
    vmin=None,
    vmax=None):
    """
    Plot a candidate x window heatmap, where each cell denotes the FP contrast score for that candiate (=token or token itemset) in that window, for a given scenario.

    Each row represents a mined token, each column a time window.
    Cell values correspond to the selected score (e.g., FP contrast score)
    computed for that token in that window.

    Tokens are ranked by mean absolute score across windows, and the top_n
    most important tokens are displayed.

    Windows containing attack alerts are highlighted via bold x-axis labels.

    rankings: list of ranking_k DataFrames (one per window)
    attack_flags : list of bool
        Boolean list indicating whether each window contains attack alerts.
    output_dir : str
        Directory where the figure will be saved.
    scenario : str
        Scenario name (used for title and filename).
    min_total_support : int
        Minimum support threshold used during mining (for annotation only).
    top_n : int, default=25
        Number of top tokens (by mean absolute score) to display.
    score_col : str, default="score_fp_contrast"
        Column name of the score used for visualization.
    vmin, vmax : float, optional
        Fixed color scale limits. If None, symmetric limits are computed
        from the current heatmap data.
    """

    label_map = {}
    for ranking_k in rankings:
        if "candidate_str" in ranking_k.columns:
            label_map.update(dict(zip(ranking_k["candidate"], ranking_k["candidate_str"])))
            
    # Collect all tokens across windows

    # Collect all candidates across windows
    all_tokens = sorted(set().union(*[set(r["candidate"]) for r in rankings]))

    # Build token x window score matrix (use NaN for absent)
    cols = []
    for w, ranking_k in enumerate(rankings):
        s = pd.Series(ranking_k[score_col].values, index=ranking_k["candidate"])
        s = s[~s.index.duplicated(keep="first")]   # safety
        cols.append(s.reindex(all_tokens))         # NaN if absent
    score_df = pd.concat(cols, axis=1)
    score_df.columns = range(len(cols))  # window indices 0..W-1

    # Pick tokens to show
    # For split metrics, "risk" is undefined in benign-only windows -> keep NaNs, and compute importance with skipna
    importance = score_df.abs().mean(axis=1, skipna=True)

    score_df = score_df.loc[importance.sort_values(ascending=False).index]
    top_tokens = score_df.head(top_n).index if order == "desc" else score_df.tail(top_n).index
    heatmap_df = score_df.loc[top_tokens]

    # Set fixed color scale so heatmaps are comparable across scenarios
    if vmin is None or vmax is None:
        max_abs = np.abs(heatmap_df.values).max()
        vmin = -max_abs
        vmax = max_abs
        
    # Plot heatmap
    plt.figure(figsize=(12, 6))
    im = plt.imshow(heatmap_df.values, aspect="auto", cmap="berlin", vmin=vmin, vmax=vmax)

    plt.colorbar(im, label=score_col)
    yticklabels = [label_map.get(c, str(c)) for c in heatmap_df.index]
    plt.yticks(range(len(heatmap_df.index)), yticklabels, fontsize=8)
    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns)


    ax = plt.gca()
    for i, label in enumerate(ax.get_xticklabels()):
        if i < len(attack_flags) and attack_flags[i]:
            label.set_fontweight("bold")

    plt.xlabel("Window index")
    plt.ylabel("Candidate")
    plt.title(f"Benign token importance across windows (scenario={scenario}, miner={miner_name}, scorer={scorer_name}, min_support={min_total_support}, order={order})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"token_heatmap_{scenario}"))

def plot_utility_heatmap(
    scenario_memory_entry: dict,
    scenario: str,
    output_dir: str,
    top_n: int = 50,
    fill_missing: float | None = None,  # None => show NaNs as white
    vmin=None,
    vmax=None,
    cmap="berlin",
):
    """
    Heatmap of utility over windows.
    Expects scenario_memory_entry["score_trace"] from window_based_mining.
    """
    score_trace = scenario_memory_entry["score_trace"]
    if not score_trace:
        raise ValueError("score_trace is empty")

    # windows + build long df
    win_labels = [f"{u['start']:%Y-%m-%d %H:%M}" for u in score_trace]
    frames = []
    for w_idx, u in enumerate(score_trace):
        tmp = u["values"].copy()
        tmp["window"] = w_idx
        frames.append(tmp)
    long = pd.concat(frames, ignore_index=True)  # columns: candidate, utility, window

    # candidate importance: mean abs utility across windows (skip NaNs)
    imp = long.groupby("candidate")["score"].apply(lambda s: s.abs().mean())
    top_cands = imp.sort_values(ascending=False).head(top_n).index

    # pivot to matrix
    mat = (
        long[long["candidate"].isin(top_cands)]
        .pivot(index="candidate", columns="window", values="score")
        .reindex(top_cands)
    )

    if fill_missing is not None:
        mat = mat.fillna(fill_missing)

    vals = mat.values
    if vmin is None or vmax is None:
        max_abs = np.nanmax(np.abs(vals))
        vmin = -max_abs
        vmax = max_abs

    plt.figure(figsize=(12, 0.35 * len(mat) + 3))
    im = plt.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="score")
    plt.yticks(range(len(mat.index)), [str(c) for c in mat.index], fontsize=8)
    plt.xticks(range(len(win_labels)), win_labels, rotation=90, fontsize=8)
    plt.title(f"Score heatmap (mem+raw) — {scenario} (top {top_n})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"token_score_heatmap_{scenario}"))


def plot_active_set(
    scenario_memory_entry: dict,
    scenario: str,
    output_dir: str,
    top_n_rows: int = 50,
    show_counts: bool = True,
    cmap="berlin",
):
    """
    Plot active-set evolution.
    - If show_counts: line plot of |active_set| per window.
    - Always: binary heatmap (top_n_rows most frequently active tokens).
    Expects scenario_memory_entry["active_trace"] from your window_based_mining.
    """
    active_trace = scenario_memory_entry["active_trace"]
    if not active_trace:
        raise ValueError("active_trace is empty")

    win_labels = [f"{a['start']:%Y-%m-%d %H:%M}" for a in active_trace]

    # counts plot
    if show_counts:
        counts = [len(a["active_candidates"]) for a in active_trace]
        plt.figure(figsize=(12, 3))
        plt.plot(range(len(counts)), counts, marker="o")
        plt.xticks(range(len(win_labels)), win_labels, rotation=90, fontsize=8)
        plt.ylabel("|active set|")
        plt.title(f"Active set size over time — {scenario}")
        plt.tight_layout()
        plt.show()

    # binary heatmap
    active_sets = [set(a["active_candidates"]) for a in active_trace]
    all_active = sorted(set().union(*active_sets)) if active_sets else []

    if not all_active:
        print("No active candidates in any window.")
        return

    # pick top rows by frequency of activation
    freq = {c: sum(c in s for s in active_sets) for c in all_active}
    top_cands = [c for c, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n_rows]]

    bin_mat = np.array([[1 if c in active_sets[w] else 0 for w in range(len(active_sets))] for c in top_cands])

    plt.figure(figsize=(12, 0.35 * len(top_cands) + 3))
    im = plt.imshow(bin_mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, label="active (1/0)")
    plt.yticks(range(len(top_cands)), [str(c) for c in top_cands], fontsize=8)
    plt.xticks(range(len(win_labels)), win_labels, rotation=90, fontsize=8)
    plt.title(f"Active set membership — {scenario} (top {top_n_rows} by activation freq)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"active_set_{scenario}"))

def plot_all(X, results, d, run_name="default"):
    out_dir = _ensure_dir(os.path.join("../plots", _safe_name(run_name)))

    plot_roc(results["y_true"], results["proba"], d, out_dir=out_dir)
    plot_alert_reduction(results["y_true"], results["proba"], d, out_dir=out_dir)
    plot_feature_importance(results["model"], X, d, out_dir=out_dir)
    plot_confidence_distribution(results["proba"], d, out_dir=out_dir)

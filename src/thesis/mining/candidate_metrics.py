import pandas as pd
from typing import Callable, Union
import numpy as np

# -----------------------------------
# Interfaces
# -----------------------------------
# scorer(c0, c1, n0, n1) -> pd.Series aligned to c0.index
ScoreFunction = Callable[
    [pd.Series, pd.Series, int, int], Union[pd.Series, pd.DataFrame]
]


def compute_candidate_metrics(
    counts_df: pd.DataFrame,
    candidate_col: str = "candidate",
) -> pd.DataFrame:
    """
    Compute per-window and candidate-level metrics on top of a counts dataframe.

    Expected input columns:
        - candidate
        - c0
        - c1
        - n0
        - n1
        - count_total
        - window_id

    Definitions:
        - benign-only window: n0 > 0 and n1 == 0
        - validation window: n0 > 0 and n1 > 0
        - candidate appears in a window: count_total > 0
        - risky validation window: attack_prevalence > benign_prevalence

    TF/IDF:
        - tf = count_total / (n0 + n1)
        - idf = log((N + 1) / (1 + window_frequency))
        - tfidf = tf * idf

    Returns:
        A copy of counts_df with all computed metrics added as columns.
    """
    required = {candidate_col, "c0", "c1", "n0", "n1", "count_total", "window_id"}
    missing = required - set(counts_df.columns)
    if missing:
        raise ValueError(f"counts_df is missing required columns: {sorted(missing)}")

    df = counts_df.copy()

    # --------------------------------------------------------------
    # basic flags
    # --------------------------------------------------------------
    df["is_benign_only_window"] = (df["n0"] > 0) & (df["n1"] == 0)
    df["is_validation_window"] = (df["n0"] > 0) & (df["n1"] > 0)
    df["appears"] = df["count_total"] > 0

    # --------------------------------------------------------------
    # candidate-level support stats
    # --------------------------------------------------------------
    min_support = (
        df[df["count_total"] > 0]
        .groupby(candidate_col)["count_total"]
        .min()
        .rename("min_total_support_count")
        .reset_index()
    )
    df = df.merge(min_support, on=candidate_col, how="left")
    df["min_total_support_count"] = df["min_total_support_count"].fillna(0).astype(int)

    window_freq = (
        df[df["count_total"] > 0]
        .groupby(candidate_col)["window_id"]
        .nunique()
        .rename("window_frequency")
        .reset_index()
    )
    df = df.merge(window_freq, on=candidate_col, how="left")
    df["window_frequency"] = df["window_frequency"].fillna(0).astype(int)

    # --------------------------------------------------------------
    # per-window prevalences
    # --------------------------------------------------------------
    df["benign_prevalence"] = (df["c0"] / df["n0"].replace(0, np.nan)).fillna(0.0)
    df["attack_prevalence"] = (df["c1"] / df["n1"].replace(0, np.nan)).fillna(0.0)

    df["is_risky_window"] = (
        df["is_validation_window"] & (df["attack_prevalence"] > df["benign_prevalence"])
    ).astype(int)

    # --------------------------------------------------------------
    # benign-only aggregates
    # --------------------------------------------------------------
    benign_only = df[df["is_benign_only_window"]].copy()

    if len(benign_only) > 0:
        benign_agg = (
            benign_only.groupby(candidate_col)
            .agg(
                avg_benign_prevalence=("benign_prevalence", "mean"),
                n_benign_only_windows=("appears", "sum"),
            )
            .reset_index()
        )
    else:
        benign_agg = pd.DataFrame(
            columns=[candidate_col, "avg_benign_prevalence", "n_benign_only_windows"]
        )

    # --------------------------------------------------------------
    # validation aggregates
    # --------------------------------------------------------------
    validation = df[df["is_validation_window"]].copy()

    if len(validation) > 0:
        validation_agg = (
            validation.groupby(candidate_col)
            .agg(
                avg_attack_prevalence=("attack_prevalence", "mean"),
                n_candidate_validation_windows=("appears", "sum"),
                n_validation_windows=("window_id", "nunique"),
                n_risky_windows=("is_risky_window", "sum"),
                c0_total_validation=("c0", "sum"),
                c1_total_validation=("c1", "sum"),
            )
            .reset_index()
        )

        validation_agg["risky_window_ratio"] = (
            validation_agg["n_risky_windows"]
            / validation_agg["n_validation_windows"].replace(0, np.nan)
        ).fillna(0.0)
    else:
        validation_agg = pd.DataFrame(
            columns=[
                candidate_col,
                "avg_attack_prevalence",
                "n_candidate_validation_windows",
                "n_validation_windows",
                "n_risky_windows",
                "c0_total_validation",
                "c1_total_validation",
                "risky_window_ratio",
            ]
        )

    # --------------------------------------------------------------
    # merge candidate-level aggregates back
    # --------------------------------------------------------------
    df = df.merge(benign_agg, on=candidate_col, how="left")
    df = df.merge(validation_agg, on=candidate_col, how="left")

    fill_zero_cols = [
        "avg_benign_prevalence",
        "n_benign_only_windows",
        "avg_attack_prevalence",
        "n_candidate_validation_windows",
        "n_validation_windows",
        "n_risky_windows",
        "risky_window_ratio",
        "c0_total_validation",
        "c1_total_validation",
    ]
    for col in fill_zero_cols:
        df[col] = df[col].fillna(0.0)

    int_like_cols = [
        "min_total_support_count",
        "window_frequency",
        "n_benign_only_windows",
        "n_candidate_validation_windows",
        "n_validation_windows",
        "n_risky_windows",
        "c0_total_validation",
        "c1_total_validation",
        "is_risky_window",
    ]
    for col in int_like_cols:
        df[col] = df[col].astype(int)

    # --------------------------------------------------------------
    # contrast metrics
    # --------------------------------------------------------------
    df["contrast_score"] = np.where(
        df["is_validation_window"],
        df["avg_benign_prevalence"] - df["attack_prevalence"],
        np.nan,
    )

    contrast_agg = (
        df[df["is_validation_window"]]
        .groupby(candidate_col)["contrast_score"]
        .agg(
            mean_contrast_score="mean",
            min_contrast_score="min",
        )
        .reset_index()
    )

    df = df.merge(contrast_agg, on=candidate_col, how="left")
    df["mean_contrast_score"] = df["mean_contrast_score"].fillna(0.0)
    df["min_contrast_score"] = df["min_contrast_score"].fillna(0.0)

    # --------------------------------------------------------------
    # TF / IDF / TF-IDF
    # exactly like your compute_tfidf_score
    # --------------------------------------------------------------
    N = df["window_id"].nunique()
    df["N"] = N

    df["tf"] = (df["count_total"] / (df["n0"] + df["n1"]).replace(0, np.nan)).fillna(
        0.0
    )

    df["idf"] = np.log((N + 1) / (1 + df["window_frequency"]))
    df["idf"] = df["idf"].clip(lower=0)

    df["tfidf"] = df["tf"] * df["idf"]

    tfidf_summary = (
        df.groupby(candidate_col, dropna=False)
        .agg(
            mean_tfidf=("tfidf", "mean"),
            max_tfidf=("tfidf", "max"),
        )
        .reset_index()
    )

    df = df.merge(tfidf_summary, on=candidate_col, how="left")

    return df


def compute_stability_metrics_over_windows(
    mined_df: pd.DataFrame,
    eps: float = 1e-12,
    score_col: str = "contrast_score",  # default is now contrast = coverage - risk
) -> pd.DataFrame:
    """
    Add stability metrics per candidate across windows to the input dataframe.

    Expects columns:
        - candidate
        - c0
        - c1
        - count_total
        - n0
        - n1

    Computes columns:
        - score -> custom, can be user input
        - benign_coverage
        - attack_coverage (risk)
        - window_frequency
        - mean_score
        - positive_ratio
        - score_cv
        - mean_benign_coverage
        - min_support_count
    """
    df = mined_df.copy()

    df["benign_coverage"] = (df["c0"] / df["n0"].replace(0, pd.NA)).fillna(0.0)
    df["attack_coverage"] = (df["c1"] / df["n1"].replace(0, pd.NA)).fillna(0.0)

    if score_col == "contrast_score":
        df["score"] = df["benign_coverage"] - df["attack_coverage"]
    else:
        raise ValueError(f"Unsupported score_col: {score_col}")

    df["appeared"] = df["count_total"] > 0
    df["score_positive"] = df["score"] > 0

    stats = (
        df.groupby("candidate", dropna=False)
        .agg(
            mean_score=("score", "mean"),
            positive_ratio=("score_positive", "mean"),
            score_std=("score", "std"),
            mean_benign_coverage=("benign_coverage", "mean"),
            min_support_count=("count_total", lambda x: x[x > 0].min()),
        )
        .reset_index()
    )

    window_freq = (
        df.loc[df["count_total"] > 0]
        .groupby("candidate", dropna=False)["window_id"]
        .nunique()
        .rename("window_frequency")
        .reset_index()
    )

    stats = stats.merge(window_freq, on="candidate", how="left")
    stats["window_frequency"] = stats["window_frequency"].fillna(0).astype(int)

    stats["score_cv"] = stats["score_std"] / (stats["mean_score"].abs() + eps)

    return df.merge(stats, on="candidate", how="left")


def compute_tfidf_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    N = df["window_id"].nunique()

    df["N"] = N

    # term frequency
    df["tf"] = df["count_total"] / (df["n0"] + df["n1"])

    # document frequency (windows containing candidate)

    df["idf"] = np.log((N + 1) / (1 + df["window_frequency"]))

    # prevent negative idf (overly generic tokens)
    df["idf"] = df["idf"].clip(lower=0)

    df["tfidf"] = df["tf"] * df["idf"]

    # use mean and max tfidf across windows as candidate-level metrics for filtering
    tfidf_summary = (
        df.groupby("candidate", dropna=False)
        .agg(
            mean_tfidf=("tfidf", "mean"),
            max_tfidf=("tfidf", "max"),
        )
        .reset_index()
    )

    return df.merge(tfidf_summary, on="candidate", how="left")


def compute_discriminative_power_metrics(
    df: pd.DataFrame,
    scorer: ScoreFunction,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Add discriminative power metrics to a candidate-per-window dataframe.

    Benign prevalence is aggregated over all windows.
    Attack prevalence and Bayes log-odds are aggregated only over dual-class windows.
    This is for accurate estimation of how well a candidate can discriminate between benign and attack contexts, without being skewed by single-class windows where risk is not measurable.
    """
    out = df.copy()

    # mark dual-class windows
    out["is_dual_class_window"] = out["n1"] > 0

    # Bayesian log-odds metrics, computed per window
    scorer = scorer(alpha=alpha)

    bayes_parts = []
    for window_id, df_w in out.groupby("window_id", sort=False):
        c0 = df_w.set_index("candidate")["c0"]
        c1 = df_w.set_index("candidate")["c1"]
        n0 = int(df_w["n0"].iloc[0])
        n1 = int(df_w["n1"].iloc[0])

        score_w = scorer(c0, c1, n0, n1).reset_index()
        score_w = score_w.rename(columns={"index": "candidate"})
        score_w["window_id"] = window_id
        bayes_parts.append(score_w)

    bayes_df = pd.concat(bayes_parts, ignore_index=True)

    out = out.merge(
        bayes_df,
        on=["candidate", "window_id"],
        how="left",
    )

    # Estimated FP reduction as filter, per window
    # "Suppress all alerts containing this candidate"
    out["fp_removed_est"] = out["c0"]
    out["tp_removed_est"] = out["c1"]

    out["fp_reduction_rate_est"] = out["benign_coverage"]
    out["tp_loss_rate_est"] = out["risk"]

    # Aggregate per candidate across windows
    agg_all = (
        out.groupby("candidate", dropna=False)
        .agg(
            mean_benign_prevalence=("benign_coverage", "mean"),
            # mean_attack_prevalence=("risk", "mean"),
            # mean_bayes_log_odds_ratio=("bayes_log_odds_ratio", "mean"),
            total_fp_removed_est=("fp_removed_est", "sum"),
            total_tp_removed_est=("tp_removed_est", "sum"),
            total_benign_alerts=("n0", "sum"),
            total_attack_alerts=("n1", "sum"),
        )
        .reset_index()
    )

    # aggregate only over dual-class windows: attack risk + discrimination (log odds ratio)
    dual = out[out["is_dual_class_window"]].copy()

    if len(dual) > 0:
        agg_dual = (
            dual.groupby("candidate", dropna=False)
            .agg(
                mean_attack_prevalence=("risk", "mean"),
                mean_bayes_log_odds_ratio=("bayes_log_odds_ratio", "mean"),
            )
            .reset_index()
        )
    else:
        agg_dual = pd.DataFrame(
            {
                "candidate": out["candidate"].drop_duplicates(),
                "mean_attack_prevalence": pd.NA,
                "mean_bayes_log_odds_ratio": pd.NA,
            }
        )

    agg = agg_all.merge(agg_dual, on="candidate", how="left")

    agg["total_fp_reduction_rate_est"] = (
        agg["total_fp_removed_est"] / agg["total_benign_alerts"].replace(0, pd.NA)
    ).fillna(0.0)

    agg["total_tp_loss_rate_est"] = (
        agg["total_tp_removed_est"] / agg["total_attack_alerts"].replace(0, pd.NA)
    ).fillna(0.0)

    agg = agg.drop(columns=["total_benign_alerts", "total_attack_alerts"])

    out = out.merge(agg, on="candidate", how="left")

    return out

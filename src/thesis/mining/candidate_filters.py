import pandas as pd
# -----------------------------------
# Filters
# -----------------------------------
def filter_stable_candidates(
    df: pd.DataFrame,
    output_dir: str,
    scenario_name: str,
    min_window_frequency: int = 2,
    min_mean_score: float = 0.0,
    min_positive_ratio: float = 0.7,
    max_score_cv: float = 1.0,
    min_mean_benign_coverage: float = 0.01,
    min_support_count: int = 1,
) -> pd.DataFrame:
    """
    Filter a candidate-per-window dataframe to only stable candidates.

    Expected columns already present in df:
        - candidate
        - window_frequency
        - mean_score
        - positive_ratio
        - score_cv
        - mean_benign_coverage
        - min_support_count

    Returns:
        The input dataframe, filtered to only rows whose candidate is stable.
    """
    stable_mask = (
        (df["window_frequency"] >= min_window_frequency)
        & (df["mean_score"] > min_mean_score)
        & (df["positive_ratio"] >= min_positive_ratio)
        & (df["score_cv"] <= max_score_cv)
        & (df["mean_benign_coverage"] >= min_mean_benign_coverage)
        & (df["min_support_count"] >= min_support_count)  # remove very rare tokens
    )

    filtered_df = df.loc[stable_mask].copy()

    print(
        "Removed ",
        (~stable_mask).sum(),
        " unstable candidates, kept ",
        stable_mask.sum(),
    )

    os.makedirs(output_dir, exist_ok=True)
    file = os.path.join(output_dir, f"{scenario_name}_stable_features.csv")

    filtered_df.to_csv(file)

    print(f"Saved stable features for scenario '{scenario_name}' to {file}")

    return filtered_df


def filter_discriminative_candidates(
    df: pd.DataFrame,
    min_benign_prevalence: float = 0.01,
    max_attack_prevalence: float = 0.001,
    min_fp_reduction_rate: float = 0.0,
    max_tp_loss_rate: float = 0.05,
    # min_log_odds_ratio: float = 0.0,
):
    mask = (
        (df["mean_benign_prevalence"] >= min_benign_prevalence)
        # inf means that if risk is not measurable (single-class windows)
        # we dont filter out the candidate based on attack prevalence
        # only based on benign prevalence
        & (df["mean_attack_prevalence"].fillna(float("inf")) <= max_attack_prevalence)
        & (df["total_fp_reduction_rate_est"] >= min_fp_reduction_rate)
        & (df["total_tp_loss_rate_est"] <= max_tp_loss_rate)
        # & (df["mean_bayes_log_odds_ratio"].fillna(float("-inf")) > min_log_odds_ratio)
    )

    filtered_df = df.loc[mask].copy()
    print(
        "Removed ",
        (~mask).sum(),
        " candidates, kept ",
        mask.sum(),
    )

    return filtered_df

def filter_tfidf_candidates(
    df: pd.DataFrame,
    tfidf_col: str = "mean_tfidf",
    min_tfidf: float = 0.001,
    max_window_frequency_ratio: float = 0.9,
) -> pd.DataFrame:
    """
    Filter candidates using TF-IDF and simple frequency constraints.

    Expected columns:
        - tfidf
        - window_frequency
        - window_id
        - min_support_count
    """

    df = df.copy()

    n_windows = df["window_id"].nunique()
    df["window_frequency_ratio"] = df["window_frequency"] / n_windows

    tfidf_mask = (df[tfidf_col] >= min_tfidf) & (  # remove very weak tokens
        df["window_frequency_ratio"] <= max_window_frequency_ratio
    )  # remove overly generic tokens

    filtered_df = df.loc[tfidf_mask].copy()

    print(
        "Removed",
        df.loc[~tfidf_mask, "candidate"].nunique(),
        "candidates, kept",
        df.loc[tfidf_mask, "candidate"].nunique(),
    )

    return filtered_df


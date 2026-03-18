import pandas as pd
import numpy as np
import os
from typing import Callable, Optional, Any, Dict, Union
from pathlib import Path
import inspect
from scipy import sparse
import json

from util import make_time_windows
from mining.alert_tokenization import tokenize_window
from classes import *
from mining.alert_tokenization import iter_precached_windows

# -----------------------------------
# Interfaces
# -----------------------------------
# counter(tokens, y, **kwargs) -> (c0, c1, n0, n1)
# c0, c1 : pd.Series indexed by candidate (token string or itemset tuple)
# n0, n1 : total benign / attack TRANSACTIONS (alerts)
CountFunction = Callable[..., tuple[pd.Series, pd.Series, int, int]]

# scorer(c0, c1, n0, n1) -> pd.Series aligned to c0.index
ScoreFunction = Callable[
    [pd.Series, pd.Series, int, int], Union[pd.Series, pd.DataFrame]
]


# -----------------------------------
# Behavioral features
# -----------------------------------
def add_behavioral_features(df, time_col="timestamp", src_col="srcip", dst_col="dstip"):
    """
    Adds window-relative behavioral features (counts + bins) to df:
    - src_freq_bin: activity level of a source within the current window
    - dst_fanin_bin: number of unique sources per destination within the window
    - src_fanout_bin: number of unique destinations per source within the window
    """
    df = df.copy()

    src_counts = df[src_col].value_counts()
    df["count_src_window"] = df[src_col].map(src_counts)
    df["src_freq_bin"] = pd.cut(
        df["count_src_window"],
        bins=[-1, 5, 20, 100, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )

    fan_in = df.groupby(dst_col)[src_col].nunique()
    df["unique_src_per_dst"] = df[dst_col].map(fan_in)
    df["dst_fanin_bin"] = pd.cut(
        df["unique_src_per_dst"],
        bins=[-1, 3, 10, 50, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )

    fan_out = df.groupby(src_col)[dst_col].nunique(dropna=False)
    df["unique_dst_per_src"] = df[src_col].map(fan_out)
    df["src_fanout_bin"] = pd.cut(
        df["unique_dst_per_src"],
        bins=[-1, 3, 10, 50, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )

    return df


# -----------------------------------
# Helpers
# -----------------------------------
def _prepare_counter_kwargs(counter, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs accepted by the selected counter.

    If the counter has **kwargs, pass everything through.
    """
    sig = inspect.signature(counter)
    params = sig.parameters

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_var_kwargs:
        return base_kwargs

    accepted_names = set(params.keys())
    return {k: v for k, v in base_kwargs.items() if k in accepted_names}


def format_candidate(c) -> str:
    """
    Helper for printing a candidate for plots/logs.
    Nice when candidate is a set of tokens.

    - token (str) -> "token"
    - itemset (tuple/list) -> "a & b & c"
    - fallback -> str(c)
    """
    if isinstance(c, str):
        return c
    if isinstance(c, (tuple, list)):
        return " & ".join(map(str, c))
    return str(c)


def candidate_to_tokens(candidate_str: str) -> list[str]:
    """
    Turn 'tokA&tokB' or 'tokA & tokB' into ['tokA','tokB'].
    """
    if candidate_str is None or (
        isinstance(candidate_str, float) and pd.isna(candidate_str)
    ):
        return []

    s = str(candidate_str).strip()
    if not s:
        return []

    s = s.replace(" & ", "&")
    toks = [normalize_token(t) for t in s.split("&") if t.strip()]
    return toks


def normalize_token(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip()
    return t


def get_window_df(df_s, t_s, start_k, end_k):
    """
    Extract a single time window from a scenario-specific dataframe and
    compute basic class statistics. Returns the alert window in the window, along with the statistics.

    Args:
        df_s (pd.Dataframe): The full dataframe of alerts from scenario S.
        t_s (pd.Series): Time axis for the scenario dataframe used for window slicing.
        start_k (pd.Timestamp):
            Start time of the window (inclusive).
        end_k (pd.Timestamp):
            End time of the window (exclusive).

    Returns:
        Tuple containing:
            df_k (pd.DataFrame or None):
                Windowed dataframe with behavioral features added.
                Returns None if the window contains no alerts.
            n_benign (int):
                Number of benign alerts (y == 0) in the window.
            n_attack (int):
                Number of attack alerts (y == 1) in the window.
            has_attack (bool):
                True if the window contains at least one attack alert.
    """
    df_k = df_s[(t_s >= start_k) & (t_s < end_k)]
    if df_k.empty:
        return None, 0, 0, False
    df_k = add_behavioral_features(df_k)
    n_benign = int((df_k["y"] == 0).sum())
    n_attack = int((df_k["y"] == 1).sum())
    return df_k, n_benign, n_attack, (n_attack > 0)


import json
import numpy as np
import pandas as pd
from scipy import sparse


def attach_tidsets_to_survivors(
    survivors: pd.DataFrame,
    x_tokens_path: str,
    vocab_path: str,
) -> pd.DataFrame:
    """
    Add global tidset (tx_ids) to each surviving candidate.
    Supports vocab stored as either:
      - dict: token -> col_id
      - list: index = col_id, value = token
    """
    X_tokens = sparse.load_npz(x_tokens_path).tocsr()

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)

    if isinstance(vocab_raw, dict):
        token_to_id = vocab_raw
    elif isinstance(vocab_raw, list):
        token_to_id = {tok: i for i, tok in enumerate(vocab_raw)}
    else:
        raise TypeError(f"Unsupported vocab format: {type(vocab_raw)}")

    tidsets = []

    for cand_str in survivors["candidate_str"].astype(str):
        cand_tokens = [normalize_token(t) for t in candidate_to_tokens(cand_str)]
        token_ids = [token_to_id[t] for t in cand_tokens if t in token_to_id]

        if len(token_ids) != len(cand_tokens) or len(token_ids) == 0:
            tidsets.append(np.array([], dtype=np.int64))
            continue

        nnz = X_tokens[:, token_ids].getnnz(axis=1)
        fires = np.flatnonzero(nnz == len(token_ids)).astype(np.int64)
        tidsets.append(fires)

    survivors = survivors.copy()
    survivors["tidset"] = tidsets
    survivors["tidset_size"] = survivors["tidset"].apply(len)

    return survivors


# -----------------------------------
# Filters
# -----------------------------------
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

    # per-window metrics
    df["benign_coverage"] = (df["c0"] / df["n0"].replace(0, pd.NA)).fillna(0.0)
    df["attack_coverage"] = (df["c1"] / df["n1"].replace(0, pd.NA)).fillna(0.0)
    # simple score definition
    if score_col == "contrast_score":
        df["score"] = df["benign_coverage"] - df["attack_coverage"]
    else:
        raise ValueError(f"Unsupported score_col: {score_col}")

    # helper columns
    df["appeared"] = df["count_total"] > 0
    df["score_positive"] = df["score"] > 0

    # aggregate per candidate
    stats = (
        df.groupby("candidate", dropna=False)
        .agg(
            window_frequency=("appeared", "sum"),
            mean_score=("score", "mean"),
            positive_ratio=("score_positive", "mean"),
            score_std=("score", "std"),
            mean_benign_coverage=("benign_coverage", "mean"),
            min_support_count=(
                "count_total",
                lambda x: x[x > 0].min(),
            ),  # dont consider windows where the count was zero for min support, otherwise it would be always 0
        )
        .reset_index()
    )

    # coefficient of variation
    stats["score_std"] = stats["score_std"].fillna(0.0)
    stats["score_cv"] = stats["score_std"] / stats["mean_score"].abs().clip(lower=eps)

    # keep only requested columns for merge
    stats = stats[
        [
            "candidate",
            "window_frequency",
            "mean_score",
            "positive_ratio",
            "score_cv",
            "mean_benign_coverage",
            "min_support_count",
        ]
    ]

    # add stats back to every row
    df = df.merge(stats, on="candidate", how="left")

    return df


def filter_stable_candidates(
    df: pd.DataFrame,
    output_dir: str,
    scenario_name: str,
    min_window_frequency: int = 3,
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


def compute_tfidf_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    N = df["window_id"].nunique()

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


# -----------------------------------
# Miners
# -----------------------------------
def mine_candidates(
    tokens: pd.Series,
    y: pd.Series,
    counter: CountFunction,
    counter_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, int, int]:
    """
    Count candidate occurrences in a collection of tokenized alerts.

    This function computes raw counts of each candidate across the provided
    tokens and labels. It does not assume any temporal structure: the input
    may represent a full timeline, a single window, or any arbitrary subset
    of alerts.

    Returns:
        counts_df: pd.DataFrame
            One row per candidate with raw counts:
            - c0: number of benign alerts containing the candidate
            - c1: number of attack alerts containing the candidate
            - count_total: total occurrences (c0 + c1)

        n0: int
            Total number of benign alerts in the input.

        n1: int
            Total number of attack alerts in the input.
    """
    if counter_kwargs is None:
        counter_kwargs = {}

    y = y.reindex(tokens.index)

    # raw counts for this window
    c0, c1, n0, n1 = counter(tokens, y, **counter_kwargs)

    total = c0 + c1

    # keep raw per-candidate counts only
    out = pd.DataFrame(
        {
            "candidate": c0.index,
            "candidate_str": [format_candidate(c) for c in c0.index],
            "c0": c0.values,
            "c1": c1.values,
            "count_total": total.values,
            "n0": n0,
            "n1": n1,
        }
    )

    return out.reset_index(drop=True), n0, n1


# -----------------------------------
# Window-based mining
# -----------------------------------
def window_based_mining(
    scenario_name: str,
    run_name: str,
    counter: CountFunction,
    counter_kwargs: Optional[Dict[str, Any]] = None,
    out_base: Optional[str] = None,
    time_col: str = "timestamp",
    label_col: str = "y",
    window_size: str = "12H",
    step_size: str = "12H",
    align_to: str = "h",
):
    """
    Run mining over time windows for one precached scenario.

    The selected counter determines which extra kwargs are used,
    so the  the same mining can work for different counters such as:
        - count_itemsets_eclat(tokens, y, ...)
        - count_itemsets_matmul(tokens, y, X_tokens=..., vocab=..., ...)

    Returns:
        scenario_counts:
            {scenario_name: [counts_df_per_window, ...]}

        scenario_attack_flags:
            {scenario_name: [window_has_attack, ...]}
    """

    if out_base is None:
        # Compute relative to the project root (msc-thesis)
        out_base = str(Path(__file__).parents[2] / "out")

    if counter_kwargs is None:
        counter_kwargs = {}

    print(f"Running mining for precached scenario '{scenario_name}'...")

    counts = []
    attack_flags = []

    for (
        start_k,
        end_k,
        meta_k,
        X_k,
        y_k,
        window_has_attack,
        vocab,
    ) in iter_precached_windows(
        scenario_name=scenario_name,
        run_name=run_name,
        out_base=out_base,
        time_col=time_col,
        label_col=label_col,
        window_size=window_size,
        step_size=step_size,
        align_to=align_to,
    ):

        print(f"Processing window {start_k} to {end_k}...")

        attack_flags.append(window_has_attack)

        # Choose row-level transactions
        if "tokens" in meta_k.columns:
            tokens_k = meta_k["tokens"].reset_index(drop=True)
        elif "alert_id" in meta_k.columns:
            tokens_k = meta_k["alert_id"].astype(str).reset_index(drop=True)
        else:
            tokens_k = pd.Series(range(len(meta_k))).reset_index(drop=True)

        y_k = y_k.reset_index(drop=True)

        # Build a superset of possible kwargs
        counter_kwargs_k = dict(counter_kwargs)
        counter_kwargs_k.update(
            {
                "X_tokens": X_k,
                "vocab": vocab,
                "meta_k": meta_k,
                "start_k": start_k,
                "end_k": end_k,
            }
        )

        # Keep only what this counter accepts
        counter_kwargs_k = _prepare_counter_kwargs(counter, counter_kwargs_k)

        counts_k = mine_candidates(
            tokens=tokens_k,
            y=y_k,
            counter=counter,
            counter_kwargs=counter_kwargs_k,
        )

        counts.append(counts_k)

        n_attack = int((y_k == 1).sum())
        n_benign = int((y_k == 0).sum())

        print(
            f"[{scenario_name}] {start_k} -> {end_k} | "
            f"n={len(meta_k)} | benign={n_benign} | attack={n_attack}"
        )

    scenario_counts = {scenario_name: counts}
    scenario_attack_flags = {scenario_name: attack_flags}

    print(f"Completed mining for scenario '{scenario_name}'.")
    return scenario_counts, scenario_attack_flags

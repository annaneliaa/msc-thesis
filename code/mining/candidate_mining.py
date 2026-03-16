import pandas as pd
import numpy as np
import os
from typing import Callable, Optional, Any, Dict, Union
from pathlib import Path
import inspect


from util import make_time_windows
from mining.alert_tokenization import tokenize_window
from classes import *
import shutil
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


def mem_score(cov_mem, risk_mem, cand, l=1.0):
    """
    Compute the symbolic memory score for a candidate feature.

    The score combines two memory components:
    - Coverage memory: how consistently the candidate explains benign alerts.
    - Risk memory: how strongly the candidate is associated with attack windows.

    The final score is computed as:
        coverage_score − λ * risk_score

    where λ controls how strongly past risk associations suppress activation
    of otherwise benign-looking candidates, i.e. how conservative the system is.

    Args:
        cov_mem:
            Coverage memory object storing benign-related feature scores.
        risk_mem:
            Risk memory object storing attack-related feature scores.
        cand (str):
            Candidate token or itemset identifier.
        l (float, optional):
            Risk penalty weight (λ). Higher values increase suppression
            from risk memory. Defaults to 1.0.

    Returns:
        float:
            Memory-based adjustment score for the candidate.
            Positive values favor activation; negative values suppress it.
    """
    c = cov_mem.scores.get(f"cov::{cand}", 0.0)
    r = risk_mem.scores.get(f"risk::{cand}", 0.0)
    return (
        c - l * r
    )  # minus because risk should limit activation of seemingly benign candidates


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


def compute_memory_scores(ranking_k, cov_mem, risk_mem, mem_lambda=1.0):
    """
    For each candidate, compute a memory-based score using coverage and risk memories.
    Returns dataframe with mem_score added.

    Args:
        ranking_k (pd.DataFrame):
            Current window ranking with at least columns:
            ["candidate", "contrast_score"].
        cov_mem:
            Coverage memory object.
        risk_mem:
            Risk memory object.
        mem_lambda (float):
            Weighting factor passed to memory scoring function.

    Returns:
        pd.DataFrame:
            Same dataframe with computed memory scores added as a column.
    """
    ranking_k = ranking_k.copy()
    ranking_k["mem_score"] = ranking_k["candidate"].map(
        lambda cand: mem_score(cov_mem, risk_mem, cand, l=mem_lambda)
    )
    return ranking_k


def apply_utility_rerank(ranking_k, mem_beta=0.1):
    """
    Computes utility score for each proposed candidate in window k as
    score = contrast_score + mem_beta * mem_score

    Re-rank mined candidates using this computed utility score.

    For each candidate:
    - Combines the raw mining score with the memory score.
    - Returns a re-ranked dataframe.

    Args:
        ranking_k (pd.DataFrame):
            Current window ranking with at least columns:
            ["candidate", "contrast_score"].
        cov_mem:
            Coverage memory object.
        risk_mem:
            Risk memory object.
        mem_lambda (float):
            Weighting factor passed to memory scoring function.
        mem_beta (float):
            Scaling factor controlling influence of memory score
            on final ranking.

    Returns:
        pd.DataFrame:
            Re-ranked dataframe sorted by ("combined_score"), which is the contrast score in the current window combined
            with the memory score. Contains all original columns plus "mem_score" and "combined_score".
    """
    ranking_k = ranking_k.copy()
    ranking_k["combined_score"] = (
        ranking_k["contrast_score"] + mem_beta * ranking_k["mem_score"]
    )
    return ranking_k.sort_values("combined_score", ascending=False).reset_index(
        drop=True
    )


# TODO: add here removal of candidates whose score drops below threshold?
def update_memories_and_snapshot(
    ranking_k,
    cov_mem,
    risk_mem,
    n_benign,
    window_has_attack,
    start_k,
    end_k,
    top_cov=50,
    top_risk=50,
):
    """
    Update coverage and risk memories of candidates based on current window
    ranking, and return a snapshot of memory state.

    Steps:
    - Apply decay to both memories.
    - Reward top coverage candidates if benign alerts exist.
    - Reward top risk candidates if attacks occurred.
    - Return metadata and current memory state.

    Args:
        ranking_k (pd.DataFrame):
            Current window ranking containing candidate scores.
        cov_mem:
            Coverage memory object (to track benign-associated features).
        risk_mem:
            Risk memory object (to track attack-associated features).
        n_benign (int):
            Number of benign alerts in the window.
        window_has_attack (bool):
            Whether the window contains attack alerts.
        start_k (pd.Timestamp):
            Window start time.
        end_k (pd.Timestamp):
            Window end time.
        top_cov (int):
            Number of top coverage candidates to reward.
        top_risk (int):
            Number of top risk candidates to reward.

    Returns:
        dict:
            Snapshot containing window bounds, attack flag,
            active memory entries, and current memory score maps.
    """
    cov_mem.step_decay()
    risk_mem.step_decay()

    if n_benign > 0 and "coverage" in ranking_k.columns:
        cov_top = ranking_k.nlargest(top_cov, "coverage")["candidate"]
        cov_mem.reward_feats([f"cov::{it}" for it in cov_top])

    # window is dual-class
    if window_has_attack and "risk" in ranking_k.columns:
        tmp = ranking_k.dropna(subset=["risk"])
        if not tmp.empty:
            risk_top = tmp.nlargest(top_risk, "risk")["candidate"]
            risk_mem.reward_feats([f"risk::{it}" for it in risk_top])

    return {
        "start": start_k,
        "end": end_k,
        "has_attack": window_has_attack,
        "coverage_active": cov_mem.active(),
        "risk_active": risk_mem.active(),
        "coverage_scores": dict(cov_mem.scores),
        "risk_scores": dict(risk_mem.scores),
    }


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
    df["risk"] = (df["c1"] / df["n1"].replace(0, pd.NA)).fillna(0.0)
    # simple score definition
    if score_col == "contrast_score":
        df["score"] = df["benign_coverage"] - df["risk"]
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


def window_based_mining_old(
    df,
    scenario_name: str,
    run_name: str,
    counter: CountFunction,
    counter_kwargs: Optional[Dict[str, Any]] = None,
    min_support: int = 1,
):
    """
    Single-scenario window-based mining.

    Returns:
        scenario_rankings: dict
            {scenario_name: [ranking_df_per_window, ...]}

        scenario_attack_flags: dict
            {scenario_name: [window_has_attack, ...]}

        scenario_memory: dict
            Kept for interface consistency with the memory version.
            Contains score_trace and active_trace, but mem_trace stays empty.
    """
    dataframe = df.copy()

    if counter_kwargs is None:
        counter_kwargs = {}

    # clear old outputs for this run
    shutil.rmtree(f"../out/{run_name}/tokens", ignore_errors=True)
    shutil.rmtree(f"../out/{run_name}/rankings", ignore_errors=True)

    scenario_counts = {}
    scenario_attack_flags = {}

    # keep only the requested scenario
    df_s = dataframe[dataframe["scenario"] == scenario_name].copy()
    print(f"Running mining for scenario {scenario_name}....")

    if df_s.empty:
        raise ValueError(f"No rows found for scenario '{scenario_name}'")

    counts, attack_flags = [], []

    # sort and create windows
    df_s = df_s.sort_values("timestamp")
    t_s = df_s["timestamp"]
    windows = make_time_windows(t_s, window_size="12H", step_size="12H", align_to="h")

    # prepare token output files
    out_dir = f"../out/{run_name}/tokens/{scenario_name}"
    os.makedirs(out_dir, exist_ok=True)

    tok_path = os.path.join(out_dir, "tokens.csv")
    xtok_path = os.path.join(out_dir, "X_tokens.csv")

    if os.path.exists(tok_path):
        os.remove(tok_path)
    if os.path.exists(xtok_path):
        os.remove(xtok_path)

    # collect tokens per alert over the full scenario
    tok_acc = {}

    # loop over windows
    for start_k, end_k in windows:
        df_k, n_benign, n_attack, window_has_attack = get_window_df(
            df_s, t_s, start_k, end_k
        )
        if df_k is None:
            continue

        attack_flags.append(window_has_attack)

        # tokenize alerts in this window
        tokens_k = tokenize_window(df_k)

        # accumulate tokens per alert_id
        for aid, toks in zip(df_k["alert_id"].astype(str).values, tokens_k.values):
            if aid not in tok_acc:
                tok_acc[aid] = set()

            if isinstance(toks, list):
                tok_acc[aid].update(toks)
            else:
                if pd.notna(toks):
                    tok_acc[aid].add(str(toks))

        # create cache key for this window
        counter_kwargs_k = dict(counter_kwargs)
        # counter_kwargs_k["cache_key"] = (
        #     f"{scenario_name}_{start_k.strftime('%Y%m%d_%H%M%S')}_{end_k.strftime('%Y%m%d_%H%M%S')}"
        # )

        # mine candidates in this window
        counts_k = mine_candidates(
            tokens=tokens_k,
            y=df_k["y"],
            counter=counter,
            counter_kwargs=counter_kwargs_k,
        )

        counts.append(counts_k)

    # save scenario-wide deduplicated tokens
    tok_df_all = pd.DataFrame(
        {
            "alert_id": list(tok_acc.keys()),
            "tokens": [sorted(list(s)) for s in tok_acc.values()],
        }
    )

    tok_df_all.to_csv(tok_path, index=False)

    X_tokens_all = (
        tok_df_all.set_index("alert_id")["tokens"]
        .apply(lambda L: "|".join(L))
        .str.get_dummies(sep="|")
    )
    X_tokens_all.to_csv(xtok_path)

    print(
        scenario_name,
        "unique ids df:",
        df_s["alert_id"].astype(str).nunique(),
        "unique ids tok:",
        tok_df_all["alert_id"].nunique(),
    )

    scenario_counts[scenario_name] = counts
    scenario_attack_flags[scenario_name] = attack_flags

    return scenario_counts, scenario_attack_flags


def window_based_mining_mem(
    df,
    run_name: str,
    scorer: ScoreFunction,
    counter: CountFunction,
    counter_kwargs: Optional[Dict[str, Any]] = None,
    min_support: int = 50,
    use_memory: bool = True,
    mem_lambda: float = 1.0,
    mem_beta: float = 0.1,
    top_cov: int = 50,
    top_risk: int = 50,
    utility_threshold: float = 0.0,
    active_top_k: Optional[int] = None,
):
    """
    Run windowed token/itemset mining per scenario with optional symbolic memory.

    For each scenario:
    - Split alerts into sliding time windows.
    - Mine candidates using the provided counter and scorer.
    - Optionally re-rank candidates using symbolic memory from previous windows.
    - Update memory based on top coverage/risk candidates in the current window.
    - Track per-window rankings, attack presence, utility trajectories, and active sets.

    Returns:
        scenario_rankings: dict
            scenario -> list of per-window ranking DataFrames. # current view

        scenario_attack_flags: dict
            scenario -> list of booleans indicating whether each window contains attacks. # single class window or not

        scenario_memory: dict
            scenario -> {
                "mem_trace": memory state snapshots per window,
                "utility_trace": per-window candidate utility values,
                "active_trace": per-window active candidate sets # what the miner proposes for symbolic features
            }
    """

    dataframe = df.copy()

    if counter_kwargs is None:
        counter_kwargs = {}

    # clear old outputs for this run to avoid mixing/appending stale ids
    shutil.rmtree(f"../out/{run_name}/tokens", ignore_errors=True)
    shutil.rmtree(f"../out/{run_name}/rankings", ignore_errors=True)

    scenario_rankings = {}
    scenario_attack_flags = {}

    # This dict will store information about the full mining run for scenario S
    scenario_memory = {}

    for scenario, df_s in dataframe.groupby("scenario", sort=False):
        print(f"Running mining for scenario {scenario}....")

        # Initialize memories for coverage and risk scores of mined candidates
        cov_mem = SymbolicMemory() if use_memory else None
        risk_mem = SymbolicMemory() if use_memory else None

        rankings, attack_flags = [], []
        mem_trace, score_trace, active_trace = [], [], []

        # Split up the dataset for scenario S according to time windows
        df_s = df_s.sort_values("timestamp")
        t_s = df_s["timestamp"]
        windows = make_time_windows(
            t_s, window_size="12H", step_size="12H", align_to="h"
        )

        # reset per-scenario token outputs so we don't append across reruns
        out_dir = f"../out/{run_name}/tokens/{scenario}"
        os.makedirs(out_dir, exist_ok=True)
        tok_path = os.path.join(out_dir, "tokens.csv")
        xtok_path = os.path.join(out_dir, "X_tokens.csv")
        if os.path.exists(tok_path):
            os.remove(tok_path)
        if os.path.exists(xtok_path):
            os.remove(xtok_path)

        tok_acc = (
            {}
        )  # accumulator for tokens in the scenario, to compute global frequencies if needed
        out_dir = f"../out/{run_name}/tokens/{scenario}"
        os.makedirs(out_dir, exist_ok=True)

        # Loop over all windows to do token mining
        for start_k, end_k, i in enumerate(windows):
            print(
                f"Processing window {i} out of {len(windows)}: {start_k} to {end_k}..."
            )
            # Get all alerts for the current window
            df_k, n_benign, n_attack, window_has_attack = get_window_df(
                df_s, t_s, start_k, end_k
            )
            if df_k is None:
                continue

            # Check if we are in a single class window
            attack_flags.append(window_has_attack)

            # Convert all alerts in window to list-of-tokens representation
            tokens_k = tokenize_window(df_k)  # Series indexed like df_k

            # accumulate (union) tokens per alert_id
            for aid, toks in zip(df_k["alert_id"].astype(str).values, tokens_k.values):
                if aid not in tok_acc:
                    # For each new alert ID create a new empty set to store unique tokens
                    tok_acc[aid] = set()
                if isinstance(toks, list):
                    # If toks is a list, add all token in the list to the set for that alert ID
                    tok_acc[aid].update(toks)
                else:
                    # If toks is a single value, convert token to string and add to the set for that alert ID
                    tok_acc[aid].add(str(toks)) if pd.notna(toks) else None

            # Mining step on all alerts in window returns a ranking of candidates according to scoring mechanism used
            ranking_k = mine_candidates(
                tokens=tokens_k,
                y=df_k["y"],
                scorer=scorer,
                counter=counter,
                counter_kwargs=counter_kwargs,
                top_k=None,
                min_support=min_support,
            )

            # Compute contrast score post hoc (contrast = coverage - risk)
            if {"coverage", "risk"}.issubset(ranking_k.columns):
                ranking_k["contrast_score"] = ranking_k["coverage"] - ranking_k[
                    "risk"
                ].fillna(0.0)

            # Apply a reranking of the proposed candidates using coverage and risk scores in memory rerank
            # Evaluate candidates in window k using windows [0...k-1)]
            if use_memory:
                # Compute memory score for each
                ranking_k = compute_memory_scores(
                    ranking_k, cov_mem, risk_mem, mem_lambda=mem_lambda
                )

                ranking_k = apply_utility_rerank(
                    ranking_k,
                    mem_beta=mem_beta,
                )

            # Choose metric that we want to base activation of a candidate on
            # If useMem = False, system will use only the raw scores of candidates in window k
            util_col = "combined_score" if use_memory else "contrast_score"
            if util_col not in ranking_k.columns:
                raise KeyError(
                    f"Expected '{util_col}' in ranking_k columns, got: {list(ranking_k.columns)}"
                )

            # Option here to store different types of scores (now utility score and contrast score)
            cols_to_store = ["candidate", util_col]

            score_trace.append(
                {
                    "start": start_k,
                    "end": end_k,
                    "score_col": util_col,
                    "values": ranking_k[cols_to_store].copy(),
                }
            )

            # TODO: check for adding removal from active set here
            # compute active set
            if active_top_k is not None:
                active_set = ranking_k.nlargest(active_top_k, util_col)[
                    "candidate"
                ].tolist()
            else:
                active_set = ranking_k.loc[
                    ranking_k[util_col] > utility_threshold, "candidate"
                ].tolist()

            active_trace.append(
                {
                    "start": start_k,
                    "end": end_k,
                    "utility_col": util_col,
                    "active_candidates": active_set,
                }
            )

            rankings.append(ranking_k)

            # Update memory with new scores for each candidate
            if use_memory:
                snap = update_memories_and_snapshot(
                    ranking_k=ranking_k,
                    cov_mem=cov_mem,
                    risk_mem=risk_mem,
                    n_benign=n_benign,
                    window_has_attack=window_has_attack,
                    start_k=start_k,
                    end_k=end_k,
                    top_cov=top_cov,
                    top_risk=top_risk,
                )
                mem_trace.append(snap)

        tok_df_all = pd.DataFrame(
            {
                "alert_id": list(tok_acc.keys()),
                "tokens": [sorted(list(s)) for s in tok_acc.values()],
            }
        )

        # overwrite tokens.csv with the scenario-wide, de-duplicated version (recommended)
        tok_df_all.to_csv(tok_path, index=False)

        X_tokens_all = (
            tok_df_all.set_index("alert_id")["tokens"]
            .apply(lambda L: "|".join(L))
            .str.get_dummies(sep="|")
        )
        X_tokens_all.to_csv(xtok_path)

        print(
            scenario,
            "unique ids df:",
            df_s["alert_id"].astype(str).nunique(),
            "unique ids tok:",
            tok_df_all["alert_id"].nunique(),
        )

        scenario_rankings[scenario] = rankings
        scenario_attack_flags[scenario] = attack_flags
        scenario_memory[scenario] = {
            "mem_trace": mem_trace,
            "score_trace": score_trace,
            "active_trace": active_trace,
        }

    return scenario_rankings, scenario_attack_flags, scenario_memory

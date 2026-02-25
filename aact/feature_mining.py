import pandas as pd
import numpy as np
from typing import List, Callable, Optional, Any, Dict, Union
from itertools import combinations

from util import make_time_windows
from classes import *

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
# Base token fields
# -----------------------------------

base_fields = [
    "category_norm",
    "decoder",
    "decoder_parent",
    "proto",
    # wazuh
    "wazuh_level",
    "groups_str",
    "is_ids_alert",
    "ids_category",
    "ids_severity",
    "mitre_tactic",
    "mitre_technique",
    # aminer
    "aminer_component_type",
    "aminer_training_mode",
    "aminer_new_event",
]


# -----------------------------------
# Helpers
# -----------------------------------
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


def tokenize_window(df_k):
    """
    Tokenize a single window of alerts into transactional token lists.

    Extends the base semantic fields with window-relative behavioral
    features (e.g., source frequency bin, fan-in, fan-out).

    Args:
        df_k (pd.DataFrame):
            Alert dataframe for one time window.

    Returns:
        pd.Series:
            Series of list-of-token representations per alert.
    """
    return tokenize_alerts(
        df_k,
        base_fields + ["src_freq_bin", "dst_fanin_bin", "src_fanout_bin"],
    )


# TODO: rewrite this function. extract mem score computation. Rename to utility score calculator or something.
def apply_memory_rerank(ranking_k, cov_mem, risk_mem, mem_lambda=1.0, mem_beta=0.1):
    """
    Re-rank mined candidates using symbolic memory signals.

    For each candidate:
    - Computes a memory-based score using coverage and risk memories.
    - Combines the raw mining score with the memory score.
    - Returns a re-ranked dataframe.

    Args:
        ranking_k (pd.DataFrame):
            Current window ranking with at least columns:
            ["candidate", "score"].
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
            Re-ranked dataframe sorted by updated score.
    """
    ranking_k = ranking_k.copy()
    ranking_k["mem_score"] = ranking_k["candidate"].map(
        lambda cand: mem_score(cov_mem, risk_mem, cand, l=mem_lambda)
    )
    ranking_k["score_raw"] = ranking_k["score"]
    ranking_k["score"] = ranking_k["score_raw"] + mem_beta * ranking_k["mem_score"]
    return ranking_k.sort_values("score", ascending=False).reset_index(drop=True)


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
# Tokenization
# -----------------------------------


def tokenize_alerts(
    df: pd.DataFrame,
    fields: List[str],
    source_col: str = "source",
    add_source_prefix: bool = True,
    max_unique_per_field: int = 5000,
) -> pd.Series:
    """
    Convert structured alert rows into a list-of-tokens per row.

    Token format: "<source>:<field>=<value>" (if add_source_prefix=True)

    - Only uses fields present in df.
    - Skips high-cardinality fields (nunique > max_unique_per_field).
    - Skips missing/empty values (does NOT emit <NA> tokens).
    """
    fields = [f for f in fields if f in df.columns]

    usable_fields = []
    for f in fields:
        nunique = df[f].nunique(dropna=True)
        if nunique <= max_unique_per_field:
            usable_fields.append(f)

    tokens_per_row = []
    for _, row in df.iterrows():
        src_prefix = ""
        if (
            add_source_prefix
            and source_col in df.columns
            and not pd.isna(row[source_col])
        ):
            src_prefix = f"{row[source_col]}:"

        row_tokens: list[str] = []
        for f in usable_fields:
            val = row[f]
            if pd.isna(val):
                continue
            v = str(val).strip()
            if not v:
                continue
            row_tokens.append(f"{src_prefix}{f}={v}")

        tokens_per_row.append(row_tokens)

    return pd.Series(tokens_per_row, index=df.index)


# -----------------------------------
# Scorers
# -----------------------------------


def benign_prev_scorer():
    def _benign_prevalence_score(c0, c1, n0, n1):
        """
        Compute the percentage of benign alerts covered by a candidate as a measure of token importance.
        Benign prevalence is measurable anytime, also in single-class windows.
        This benign prevalance alone is not a good measure of safety.

        - c0: the number of benign alerts in window k containing an itemset X.
        - n0: the total number of benign alerts in window k.

        c1 and n1 not used but needed to match the scorer signature
        """
        idx = c0.index.union(c1.index)
        c0 = c0.reindex(idx, fill_value=0)

        if n0 == 0:
            return pd.Series(0.0, index=idx, dtype=float)

        return c0 / n0

    return _benign_prevalence_score


def fp_contrast_scorer(alpha: float = 0.5):
    def _log_odds_contrast_score(c0, c1, n0, n1):
        """
        Smoothed log-odds contrast:

            log((c0+α)/(n0-c0+α)) - log((c1+α)/(n1-c1+α))

        Positive values → more benign-associated.
        """
        idx = c0.index.union(c1.index)
        c0 = c0.reindex(idx, fill_value=0)
        c1 = c1.reindex(idx, fill_value=0)

        if n0 == 0:
            left = pd.Series(0.0, index=idx, dtype=float)
        else:
            left = np.log((c0 + alpha) / ((n0 - c0) + alpha))

        if n1 == 0:
            right = pd.Series(0.0, index=idx, dtype=float)
        else:
            right = np.log((c1 + alpha) / ((n1 - c1) + alpha))

        return left - right

    return _log_odds_contrast_score


def split_metric_scorer(alpha: float = 0.5):
    def _coverage_risk_score(c0, c1, n0, n1):
        """
        Compute per-candidate coverage and risk log-odds scores.

        - Coverage: smoothed log-odds of appearing in benign alerts.
        - Risk: smoothed log-odds of appearing in attack alerts.

        Coverage is defined when n0 > 0; risk when n1 > 0.
        Returns a DataFrame with columns ["coverage", "risk"] indexed by candidate.
        """
        idx = c0.index.union(c1.index)
        c0 = c0.reindex(idx, fill_value=0)
        c1 = c1.reindex(idx, fill_value=0)

        if n0 > 0:
            coverage = np.log((c0 + alpha) / ((n0 - c0) + alpha))
        else:
            coverage = pd.Series(0.0, index=idx, dtype=float)

        if n1 > 0:
            risk = np.log((c1 + alpha) / ((n1 - c1) + alpha))
        else:
            risk = pd.Series(np.nan, index=idx, dtype=float)

        return pd.DataFrame({"coverage": coverage, "risk": risk})

    return _coverage_risk_score


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
# Counters
# -----------------------------------
def mine_token_counts(
    tokens: pd.Series,
    y: pd.Series,
) -> tuple[pd.Series, pd.Series, int, int]:
    """
    Transactional single-token counting.

    - Treats each alert as one transaction containing a set of tokens.
    - c0[token] = number of benign alerts that contain token (presence, not occurrences)
    - c1[token] = number of attack alerts that contain token
    - n0/n1 = number of benign/attack alerts (transactions)
    """
    y = y.loc[tokens.index]

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    counts0: dict[str, int] = {}
    counts1: dict[str, int] = {}

    for idx, row_tokens in tokens.items():
        if not row_tokens:
            continue
        present = set(row_tokens)  # presence per alert
        target = counts0 if y.loc[idx] == 0 else counts1
        for tok in present:
            target[tok] = target.get(tok, 0) + 1

    c0 = pd.Series(counts0, dtype=int)
    c1 = pd.Series(counts1, dtype=int)

    all_tokens = c0.index.union(c1.index)
    c0 = c0.reindex(all_tokens, fill_value=0)
    c1 = c1.reindex(all_tokens, fill_value=0)

    return c0, c1, n0, n1


def mine_itemset_counts(
    tokens: pd.Series,
    y: pd.Series,
    k: int = 2,
):
    """
    Fixed-size k-itemset mining with transactional support.
    Returns (c0, c1, n0, n1) where c0/c1 count alerts containing the itemset.
    """
    y = y.loc[tokens.index]

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    itemset_counts_0 = {}
    itemset_counts_1 = {}

    for idx, row_tokens in tokens.items():
        if not row_tokens or len(row_tokens) < k:
            continue

        unique_tokens = sorted(set(row_tokens))
        for itemset in combinations(unique_tokens, k):
            if y.loc[idx] == 0:
                itemset_counts_0[itemset] = itemset_counts_0.get(itemset, 0) + 1
            else:
                itemset_counts_1[itemset] = itemset_counts_1.get(itemset, 0) + 1

    c0 = pd.Series(itemset_counts_0, dtype=int)
    c1 = pd.Series(itemset_counts_1, dtype=int)

    all_itemsets = c0.index.union(c1.index)
    c0 = c0.reindex(all_itemsets, fill_value=0)
    c1 = c1.reindex(all_itemsets, fill_value=0)

    return c0, c1, n0, n1


def mine_itemset_counts_apriori(
    tokens: pd.Series,
    y: pd.Series,
    k: int = 2,
    min_support: int = 50,
):
    """
    Apriori-style frequent itemset mining (up to fixed size k), with transaction-level support.

    - Each row is a transaction (set of tokens).
    - Finds frequent 1-itemsets, then iteratively builds candidates of size 2..k.
    - Uses Apriori pruning: a candidate is kept only if all its (m-1)-subsets are frequent.
    - Counts support as: number of transactions containing the itemset (not total occurrences).
    - Also returns per-class transaction counts (benign vs attack) for size-k itemsets only.

    Returns:
        c0, c1: pd.Series indexed by itemset tuples (length k) with class-specific supports
        n0, n1: number of benign / attack transactions
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    # Align labels to the same rows as tokens
    y = y.loc[tokens.index]

    # Count benign and true alerts
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    # Preprocess transactions
    # remove duplicate tokens, sort for stable ordering, and store as tuple
    transactions = {
        idx: tuple(sorted(set(row_tokens))) if row_tokens else tuple()
        for idx, row_tokens in tokens.items()
    }

    # Build frequent 1-item sets
    # Counts in how many alerts each single token appears
    counts_1 = {}
    for t in transactions.values():
        for item in t:
            counts_1[(item,)] = counts_1.get((item,), 0) + 1

    # Create the set of frequent item sets of the previous size
    previous_set = {itemset for itemset, cnt in counts_1.items() if cnt >= min_support}

    if k == 1:
        # Easiest case, stop early and compute class counts
        # also compute per-class supports for size-1
        c0 = {}
        c1 = {}
        for idx, t in transactions.items():
            present = set((i,) for i in t)
            present &= previous_set
            target = c0 if y.loc[idx] == 0 else c1
            for it in present:
                target[it] = target.get(it, 0) + 1
        c0 = pd.Series(c0, dtype=int).reindex(sorted(previous_set), fill_value=0)
        c1 = pd.Series(c1, dtype=int).reindex(sorted(previous_set), fill_value=0)
        return c0, c1, n0, n1

    # Build frequent item set up untill size k
    for m in range(2, k + 1):
        # Generate candidates
        previous_set_sorted = sorted(previous_set)
        C_m = set()

        # Combine two (m-1)-itemsets if they share the same prefix of sixe m-2
        # Ex: for m=3, join (a,b) and (a,c) to (a,b,c)
        for i in range(len(previous_set_sorted)):
            for j in range(i + 1, len(previous_set_sorted)):
                a = previous_set_sorted[i]
                b = previous_set_sorted[j]
                if a[:-1] != b[:-1]:
                    break  # because sorted => prefixes stop matching
                cand = tuple(sorted(set(a) | set(b)))

                # ensure candidate size is equal to m
                if len(cand) != m:
                    continue

                # Apriori pruning: if the m-subset is frequent, then all (m-1)-subsets must be frequent
                ok = True
                for sub in combinations(cand, m - 1):
                    if sub not in previous_set:
                        ok = False
                        break
                if ok:
                    C_m.add(cand)

        if not C_m:
            # no candidates survive -> stop early
            return (
                pd.Series(dtype=int),
                pd.Series(dtype=int),
                n0,
                n1,
            )

        # Count candidate suports
        # For each alert check which candidates it contains, then increment supprt once per alert
        counts_m = {c: 0 for c in C_m}
        for t in transactions.values():
            if len(t) < m:
                continue
            tset = set(t)
            for c in C_m:
                # subset test
                if set(c).issubset(tset):
                    counts_m[c] += 1

        previous_set = {it for it, cnt in counts_m.items() if cnt >= min_support}
        if not previous_set:
            return (
                pd.Series(dtype=int),
                pd.Series(dtype=int),
                n0,
                n1,
            )

        # If we've reached size k, compute per-class supports for previous set (size-k)
        if m == k:
            # Count for each frequent k-itemset how many benign and true alerts contain it
            c0 = {it: 0 for it in previous_set}
            c1 = {it: 0 for it in previous_set}

            for idx, t in transactions.items():
                if len(t) < k:
                    continue
                tset = set(t)
                present = [it for it in previous_set if set(it).issubset(tset)]
                if not present:
                    continue
                target = c0 if y.loc[idx] == 0 else c1
                for it in present:
                    target[it] += 1

            c0 = pd.Series(c0, dtype=int).reindex(sorted(previous_set), fill_value=0)
            c1 = pd.Series(c1, dtype=int).reindex(sorted(previous_set), fill_value=0)
            return c0, c1, n0, n1

    # Code should never reach this point
    return pd.Series(dtype=int), pd.Series(dtype=int), n0, n1


# -----------------------------------
# Generic miner
# -----------------------------------
def mine_candidates(
    tokens: pd.Series,
    y: pd.Series,
    scorer: ScoreFunction,
    counter: CountFunction,
    score_name: str = "score",
    top_k: Optional[int] = None,
    min_support: Optional[int] = None,
    counter_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Generic miner supporting different candidate generators (single tokens, itemsets, etc.).
    Handles split metric scorers that return a DataFrame with columns ['coverage','risk'].

    Returns a ranked DataFrame with per-candidate counts, support, score, and optional coverage/risk.
    """
    if counter_kwargs is None:
        counter_kwargs = {}

    c0, c1, n0, n1 = counter(tokens, y, **counter_kwargs)

    total = c0 + c1
    if min_support is not None:
        keep = total[total >= min_support].index
        c0, c1, total = c0.loc[keep], c1.loc[keep], total.loc[keep]

    # ---- empty guard (prevents shape mismatches downstream)
    if len(c0) == 0:
        return pd.DataFrame(
            columns=[
                "candidate",
                "candidate_str",
                "count_benign",
                "count_attack",
                "support_total",
                score_name,
                "p_benign_given_candidate",
                "coverage",
                "risk",
            ]
        )

    score_out = scorer(c0, c1, n0, n1)

    coverage = risk = None
    if isinstance(score_out, pd.DataFrame):
        # expected columns
        if not {"coverage", "risk"}.issubset(score_out.columns):
            raise ValueError(
                "Split scorer must return a DataFrame with columns {'coverage','risk'}"
            )
        coverage = score_out["coverage"].reindex(c0.index)
        risk = score_out["risk"].reindex(c0.index)
        score = coverage - risk.fillna(0.0)  # single ranking scalar
    else:
        score = (
            score_out.reindex(c0.index)
            if isinstance(score_out, pd.Series)
            else pd.Series(score_out, index=c0.index)
        )

    den = (c0 + c1).replace(0, np.nan)
    p_benign = (c0 / den).fillna(0.0)

    out = pd.DataFrame(
        {
            "candidate": c0.index,
            "candidate_str": [format_candidate(c) for c in c0.index],
            "count_benign": c0.values,
            "count_attack": c1.values,
            "support_total": total.values,
            score_name: score.values,
            "p_benign_given_candidate": p_benign.values,
        }
    )

    if coverage is not None:
        out["coverage"] = coverage.values
    if risk is not None:
        out["risk"] = risk.values

    out = out.sort_values(score_name, ascending=False)
    if top_k is not None:
        out = out.head(top_k)

    return out.reset_index(drop=True)


# -----------------------------------
# Window-based mining
# -----------------------------------
def window_based_mining(
    df,
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

    if counter_kwargs is None:
        counter_kwargs = {}

    scenario_rankings = {}
    scenario_attack_flags = {}
    scenario_memory = {}

    for scenario, df_s in df.groupby("scenario", sort=False):
        print(f"Running mining for scenario {scenario}....")

        cov_mem = SymbolicMemory() if use_memory else None
        risk_mem = SymbolicMemory() if use_memory else None

        rankings, attack_flags = [], []
        mem_trace, utility_trace, active_trace = [], [], []

        df_s = df_s.sort_values("timestamp")
        t_s = df_s["timestamp"]
        windows = make_time_windows(
            t_s, window_size="12H", step_size="12H", align_to="h"
        )

        for start_k, end_k in windows:
            df_k, n_benign, n_attack, window_has_attack = get_window_df(
                df_s, t_s, start_k, end_k
            )
            if df_k is None:
                continue

            attack_flags.append(window_has_attack)

            tokens_k = tokenize_window(df_k)

            ranking_k = mine_candidates(
                tokens=tokens_k,
                y=df_k["y"],
                scorer=scorer,
                counter=counter,
                counter_kwargs=counter_kwargs,
                score_name="score",
                top_k=None,
                min_support=min_support,
            )

            # ---- apply memory rerank (uses previous windows only)
            if use_memory:
                ranking_k = apply_memory_rerank(
                    ranking_k,
                    cov_mem,
                    risk_mem,
                    mem_lambda=mem_lambda,
                    mem_beta=mem_beta,
                )
                ranking_k["mem_utility"] = ranking_k["mem_score"]

            # ---- split-metric utility (only if present)
            if {"coverage", "risk"}.issubset(ranking_k.columns):
                ranking_k["split_utility"] = ranking_k["coverage"] - ranking_k[
                    "risk"
                ].fillna(0.0)

            # ---- choose activation utility
            util_col = (
                "mem_utility"
                if (use_memory and "mem_utility" in ranking_k.columns)
                else "score"
            )

            # save full utility snapshot for plotting
            utility_trace.append(
                {
                    "start": start_k,
                    "end": end_k,
                    "utility_col": util_col,
                    "values": ranking_k[["candidate", util_col]].rename(
                        columns={util_col: "utility"}
                    ),
                }
            )

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

            # ---- update memory after using it
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

        scenario_rankings[scenario] = rankings
        scenario_attack_flags[scenario] = attack_flags
        scenario_memory[scenario] = {
            "mem_trace": mem_trace,
            "utility_trace": utility_trace,
            "active_trace": active_trace,
        }

    return scenario_rankings, scenario_attack_flags, scenario_memory

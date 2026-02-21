import pandas as pd
import numpy as np
from typing import List, Callable, Optional, Any, Dict
from itertools import combinations

from util import make_time_windows

# -----------------------------------
# Interfaces
# -----------------------------------

# counter(tokens, y, **kwargs) -> (c0, c1, n0, n1)
# c0, c1 : pd.Series indexed by candidate (token string or itemset tuple)
# n0, n1 : total benign / attack TRANSACTIONS (alerts)
CountFunction = Callable[..., tuple[pd.Series, pd.Series, int, int]]

# scorer(c0, c1, n0, n1) -> pd.Series aligned to c0.index
ScoreFunction = Callable[[pd.Series, pd.Series, int, int], pd.Series]

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

# Helpers
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


def log_odds_contrast_score(
    c0: pd.Series, c1: pd.Series, n0: int, n1: int, alpha: float = 0.5
) -> pd.Series:
    """
    Smoothed log-odds contrast (transactional counts):
      log((c0+α)/(n0-c0+α)) - log((c1+α)/(n1-c1+α))
    Higher => candidate (token or token set) is more benign-associated.
    """
    return np.log((c0 + alpha) / ((n0 - c0) + alpha)) - np.log(
        (c1 + alpha) / ((n1 - c1) + alpha)
    )


def fp_contrast_scorer(alpha: float = 0.5):
    return lambda c0, c1, n0, n1: log_odds_contrast_score(c0, c1, n0, n1, alpha=alpha)


def coverage_risk_score(
    c0: pd.Series, c1: pd.Series, n0: int, n1: int, alpha: float = 0.5
) -> pd.Series:
    # TODO: implement this scorer
    return pd.Series(0.0, index=c0.index)


def split_metric_scorer(alpha: float = 0.5):
    return lambda c0, c1, n0, n1: coverage_risk_score(c0, c1, n0, n1, alpha=alpha)


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
    df = df.sort_values(time_col).copy()

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
    tokens: pd.Series, y: pd.Series, k: int = 2, min_support: int = 50
):
    """
    Frequent fixed-size k-itemset mining with transactional support.
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
        itemsets = combinations(unique_tokens, k)

        target_dict = itemset_counts_0 if y.loc[idx] == 0 else itemset_counts_1
        for itemset in itemsets:
            target_dict[itemset] = target_dict.get(itemset, 0) + 1

    c0 = pd.Series(itemset_counts_0, dtype=int)
    c1 = pd.Series(itemset_counts_1, dtype=int)

    all_itemsets = c0.index.union(c1.index)
    c0 = c0.reindex(all_itemsets, fill_value=0)
    c1 = c1.reindex(all_itemsets, fill_value=0)

    total_support = c0 + c1
    keep = total_support[total_support >= min_support].index

    return c0.loc[keep], c1.loc[keep], n0, n1


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
):
    """
    Generic miner supporting different candidate generators (single tokens, itemsets, etc.).

    1) counter(...) -> (c0, c1, n0, n1) using TRANSACTIONAL semantics
    2) optional min_support filtering (if not already handled in counter)
    3) scorer(c0, c1, n0, n1) -> score per candidate (token/token set)
    4) returns ranked dataframe
    """
    if counter_kwargs is None:
        counter_kwargs = {}

    c0, c1, n0, n1 = counter(tokens, y, **counter_kwargs)

    total = c0 + c1
    if min_support is not None:
        keep = total[total >= min_support].index
        c0, c1, total = c0.loc[keep], c1.loc[keep], total.loc[keep]

    score = scorer(c0, c1, n0, n1)

    out = pd.DataFrame(
        {
            "candidate": c0.index,
            "candidate_str": [format_candidate(c) for c in c0.index],
            "count_benign": c0.values,
            "count_attack": c1.values,
            "support_total": total.values,
            score_name: score.values,
            "p_benign_given_candidate": (c0 / (c0 + c1)).values,
        }
    ).sort_values(score_name, ascending=False)

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
):  
    if counter_kwargs is None:
        counter_kwargs = {}

    scenario_rankings = dict()
    scenario_attack_flags = dict()

    df_copy = add_behavioral_features(df)

    for scenario, df_s in df_copy.groupby("scenario", sort=False):
        print(f"running mining for scenario {scenario}....")
        rankings = []
        attack_flags = []

        df_s = df_s.sort_values("timestamp")
        t_s = df_s["timestamp"]

        windows = make_time_windows(
            t_s, window_size="12H", step_size="12H", align_to="H"
        )

        for start_k, end_k in windows:
            df_k = df_s[(t_s >= start_k) & (t_s < end_k)]
            if df_k.empty:
                continue

            n_benign = (df_k["y"] == 0).sum()
            n_attack = (df_k["y"] == 1).sum()
            attack_flags.append(n_attack > 0)

            print(f"Window [{start_k}, {end_k})")
            print(f"  Benign alerts : {n_benign}")
            print(f"  Attack alerts : {n_attack}")
            print(f"  Total alerts  : {len(df_k)}")

            tokens_k = tokenize_alerts(
                df_k,
                base_fields
                + [
                    "src_freq_bin",
                    "dst_fanin_bin",
                    "src_fanout_bin",
                ],
            )

            ranking_k = mine_candidates(
                tokens=tokens_k,
                y=df_k["y"],
                scorer=scorer,
                counter=counter,
                counter_kwargs=counter_kwargs,
                score_name="score",
                min_support=100,
                top_k=None,
            )

            rankings.append(ranking_k)

        scenario_rankings[scenario] = rankings
        scenario_attack_flags[scenario] = attack_flags

    return scenario_rankings, scenario_attack_flags

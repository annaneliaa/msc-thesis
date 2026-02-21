import pandas as pd
import numpy as np
from typing import List, Callable, Optional
from itertools import combinations

from util import make_time_windows

base_fields = [
    "category_norm",  # high level semantic signal
    "decoder",
    "decoder_parent",  # which subsystem produced the alert, encoded detection pipeline
    "proto",  # network protocol
    # wazuh
    "wazuh_level",  # alert severity
    "groups_str",  # alert category groups, noisy? clean?
    "is_ids_alert",  # high level concept
    "ids_category",  # attack type classification, may correlate too strongly with label
    "ids_severity",  # redundant bc of wazuh level?
    "mitre_tactic",  # high level attack abstraction
    "mitre_technique",
    # aminer
    "aminer_component_type",  # what system component changed
    "aminer_training_mode",  # check usefullness of this one
    "aminer_new_event",  # novelty flag
]


def tokenize_alerts(
    df: pd.DataFrame,
    fields: List[str],
    source_col: str = "source",  # e.g. "aminer" / "wazuh"
    add_source_prefix: bool = True,
    null_token: str = "<NA>",
    max_unique_per_field: int = 5000,  # guardrail against explosions (IPs, paths, etc.)
) -> pd.Series:
    """
    Convert structured alert records into tokenized representations per row.

    For each alert (row), this function extracts selected fields and converts
    their (field, value) pairs into string tokens of the form: "<source>:<field>=<value>"

    Example:
        wazuh:rule_level=7
        aminer:component=ssh

    Key behavior:
    - Only fields present in `df` are considered.
    - Fields with very high cardinality (nunique > max_unique_per_field)
    are skipped to prevent token explosion (e.g., IPs, file paths).
    - Missing or empty values are ignored.
    - Optionally prefixes each token with the alert source (e.g., "wazuh:", "aminer:").

    Returns:
        pd.Series where each row contains a bag-of-tokens representation of alert.

    """
    # Guardrail: drop fields that don't exist
    fields = [f for f in fields if f in df.columns]

    # Optional: cap high-cardinality fields to avoid blowing up tokens
    usable_fields = []
    for f in fields:
        nunique = df[f].nunique(dropna=True)
        if nunique <= max_unique_per_field:
            usable_fields.append(f)

    src = (
        df[source_col].fillna("unknown")
        if source_col in df.columns
        else pd.Series(["unknown"] * len(df), index=df.index)
    )

    tokens_per_row = []
    for idx, row in df.iterrows():
        row_tokens = []
        src_prefix = (
            f"{row[source_col]}:"
            if (add_source_prefix and source_col in df.columns)
            else ""
        )
        for f in usable_fields:
            val = row[f]

            # skip NA and emptry string for some base fields
            if pd.isna(val) or str(val).strip() == "":
                continue
            if pd.isna(val):
                v = null_token
            else:
                v = str(val).strip()
                if not v:
                    v = null_token
            row_tokens.append(f"{src_prefix}{f}={v}")
        tokens_per_row.append(row_tokens)

    return pd.Series(tokens_per_row, index=df.index)


def log_odds_contrast_score(
    c0: pd.Series, c1: pd.Series, n0: int, n1: int, alpha: float = 0.5
) -> pd.Series:
    """
    Smoothed log-odds contrast:
    log( (c0+α)/(n0-c0+α) ) - log( (c1+α)/(n1-c1+α) )
    Higher => token is more benign-associated
    """
    return np.log((c0 + alpha) / ((n0 - c0) + alpha)) - np.log(
        (c1 + alpha) / ((n1 - c1) + alpha)
    )


def fp_contrast_scorer(alpha: float = 0.5):
    return lambda c0, c1, n0, n1: log_odds_contrast_score(c0, c1, n0, n1, alpha=alpha)


def coverage_risk_score(
    c0: pd.Series, c1: pd.Series, n0: int, n1: int, alpha: float = 0.5
) -> pd.Series:
    return 0


def split_metric_scorer(alpha: float = 0.5):
    return lambda c0, c1, n0, n1: coverage_risk_score(c0, c1, n0, n1, alpha=alpha)


def add_behavioral_features(df, time_col="timestamp", src_col="srcip", dst_col="dstip"):
    """
    Adds coarse-grained behavioral features to an alert dataframe based on
    source–destination interaction patterns within the current time window.
    So tokens resulting from here are relative to current window statistics.

    The function computes:
    - Source activity level (count and binned frequency)
    - Destination fan-in (unique sources per destination, binned)
    - Source fan-out (unique destinations per source, binned)

    Example: wazuh:src_freq_bin=high represents all alerts in the window whose source IP falls into the high activity bin.
    Then every alert from those IPs gets that token.

    Continuous counts are discretized to avoid high-cardinality tokens
    during symbolic mining

    Returns a modified copy of the dataframe with added feature columns.
    """
    df = df.sort_values(time_col).copy()

    # Source frequency in window
    src_counts = df[src_col].value_counts()
    df["count_src_window"] = df[src_col].map(src_counts)

    # Bin it
    df["src_freq_bin"] = pd.cut(
        df["count_src_window"],
        bins=[-1, 5, 20, 100, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )

    # Unique sources per destination (dst fan-in)
    fan_in = df.groupby(dst_col)[src_col].nunique()
    df["unique_src_per_dst"] = df[dst_col].map(fan_in)

    df["dst_fanin_bin"] = pd.cut(
        df["unique_src_per_dst"],
        bins=[-1, 3, 10, 50, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )

    # Unique destinations per source (src fan-out)
    fan_out = df.groupby(src_col)[dst_col].nunique(dropna=False)
    df["unique_dst_per_src"] = df[src_col].map(fan_out)

    df["src_fanout_bin"] = pd.cut(
        df["unique_dst_per_src"],
        bins=[-1, 3, 10, 50, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )

    # Burstiness: alerts per source in last X hours
    # TODO add this
    return df


def mine_token_counts(
    tokens: pd.Series,  # list-of-tokens per row
    y: pd.Series,  # 0 benign, 1 attack
    min_support: int = 100,  # only keep tokens that appear at least this many times overall
    alpha: float = 0.5,  # smoothing for log-odds
) -> pd.DataFrame:
    """
    Computes per-token counts for benign (c0) and attack (c1), plus totals n0 and n1
    over token occurrences
    """
    # Turn each alert into independent token occurences
    flat = tokens.explode()
    flat_y = y.loc[flat.index]

    # count token occurences for benign vs attack
    n0 = int((flat_y == 0).sum())
    n1 = int((flat_y == 1).sum())

    # Determine probability that an alert is benign given that a certain token X is present in the alert
    # Count token frequency per class
    c0 = flat[flat_y == 0].value_counts()  # counts single tokens only
    c1 = flat[flat_y == 1].value_counts()

    all_tokens = c0.index.union(c1.index)
    c0 = c0.reindex(all_tokens, fill_value=0)
    c1 = c1.reindex(all_tokens, fill_value=0)

    return c0, c1, n0, n1


def mine_itemset_counts(
    tokens: pd.Series, y: pd.Series, k: int = 2, min_support: int = 50
):
    """
    Mines frequent co-occuring token sets (fixed size k).
    Treats row alert as a transaction = one independent record containing a set of tokens.
    Generates unique k-combinations of tokens per alert.
    Function then checks in how many alerts this k-set of tokens appear together in other alerts.
    Computes per-class counts (benign vs attack).
    """

    # Ensure alignment
    y = y.loc[tokens.index]

    # Total transactions per class
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())

    itemset_counts_0 = {}
    itemset_counts_1 = {}

    for idx, row_tokens in tokens.items():
        if not row_tokens or len(row_tokens) < k:
            continue

        # unique tokens per transaction (avoid duplicates inside alert)
        unique_tokens = sorted(set(row_tokens))

        # generate k-itemsets
        itemsets = combinations(unique_tokens, k)

        target_dict = itemset_counts_0 if y.loc[idx] == 0 else itemset_counts_1

        for itemset in itemsets:
            target_dict[itemset] = target_dict.get(itemset, 0) + 1

    # Convert to Series
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


def mine_tokens(
    tokens: pd.Series,
    y: pd.Series,
    scorer: Callable[[pd.Series, pd.Series, int, int], pd.Series],
    score_name: str = "score",
    top_k: Optional[int] = None,
    min_support: int = 100,
):
    """
    Generic token miner:
    - counts token occurrences per class
    - filters by support
    - applies a scoring function
    - returns a ranked dataframe

    scorer signature: scorer(c0, c1, n0, n1) -> pd.Series aligned to c0.index
    """

    c0, c1, n0, n1 = mine_token_counts(tokens, y)
    total = c0 + c1
    keep = total[total >= min_support].index
    c0, c1, total = c0.loc[keep], c1.loc[keep], total.loc[keep]

    # compute token score based on input scorer
    score = scorer(c0, c1, n0, n1)

    out = pd.DataFrame(
        {
            "token": c0.index,
            "count_benign": c0.values,
            "count_attack": c1.values,
            "support_total": total.values,
            score_name: score.values,
            "p_benign_given_token": (c0 / (c0 + c1)).values,
        }
    ).sort_values(score_name, ascending=False)

    if top_k is not None:
        out = out.head(top_k)

    return out.reset_index(drop=True)


def window_based_mining(df):
    scenario_rankings = dict()
    scenario_attack_flags = dict()

    df_copy = add_behavioral_features(df)

    for scenario, df_s in df_copy.groupby("scenario", sort=False):
        print(f"running mining for scenario {scenario}....")
        rankings = []
        attack_flags = []
        t_s = df_s["timestamp"]
        windows = make_time_windows(
            df_copy["timestamp"], window_size="12H", step_size="12H", align_to="h"
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
                    "dst_fanin_bin" + "src_fanout_bin",
                ],  # added behavioral features here, add to base fields?
            )

            # TODO:urrently hardcodes contrast score, change to user input later
            ranking_k = mine_tokens(
                tokens=tokens_k,
                y=df_k["y"],
                scorer=fp_contrast_scorer(alpha=0.5),
                score_name="score_fp_contrast",
                min_support=100,
                top_k=None,
            )

            rankings.append(ranking_k)

        scenario_rankings[scenario] = rankings
        scenario_attack_flags[scenario] = attack_flags

    return scenario_rankings, scenario_attack_flags

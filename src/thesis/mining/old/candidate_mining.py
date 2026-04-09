import pandas as pd
import numpy as np
from typing import Callable, Optional, Any, Dict, Union
from pathlib import Path
import inspect
from scipy import sparse
import json

from thesis.mining.old.alert_tokenization import iter_precached_windows
from thesis.mining.old.build_features import (
    add_behavioral_features,
    behavioral_tokens_from_df,
)

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
    tokens and labels.

    No assumption of any temporal structure: the input
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

        # start from cached static tokens
        base_tokens = meta_k["tokens"].reset_index(drop=True).copy()

        # compute behavioral columns for this window slice
        meta_k = add_behavioral_features(meta_k, src_col="srcip", dst_col="dstip")

        # convert only behavioral feature columns into tokens
        beh_tokens = behavioral_tokens_from_df(meta_k).reset_index(drop=True)

        # merge behavioral tokens into cached static tokens
        meta_k["tokens"] = [
            sorted(set(base) | set(beh)) for base, beh in zip(base_tokens, beh_tokens)
        ]

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

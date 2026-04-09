import pandas as pd
import json
import ast
from typing import List, Optional
from utils.util import make_time_windows
from pathlib import Path
from scipy import sparse
import numpy as np
import os

# -----------------------------------
# Base token fields corresponding to labeled df columns, used for tokenization and candidate generation
# -----------------------------------
with open("../mining/base_fields.json", "r") as f:
    base_fields = json.load(f)


# -----------------------------------
# Tokenization
# -----------------------------------
def _expand_token_values(val):
    """
    Turn a cell value into a flat list of token values.

    Examples:
    - "abc" -> ["abc"]
    - '["a", "b"]' -> ["a", "b"]
    - ["a", "b"] -> ["a", "b"]
    """
    if val is None:
        return []

    # Handle numpy arrays
    if isinstance(val, np.ndarray):
        return [str(x).strip() for x in val if str(x).strip() and not pd.isna(x)]

    # missing scalar
    if pd.isna(val):
        return []

    s = str(val).strip()
    if not s:
        return []

    # try JSON / Python-literal list parsing
    if s.startswith("[") and s.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(s)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

    # fallback: single scalar value
    return [s]


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
    - Skips missing/empty values.
    - Expands list-like fields into separate tokens.
    """
    fields = [f for f in fields if f in df.columns]

    usable_fields = []
    for f in fields:
        if f in df.columns:
            # Convert to string to handle unhashable types like numpy arrays
            nunique = df[f].astype(str).nunique(dropna=True)
            if nunique <= max_unique_per_field:
                usable_fields.append(f)

    print(
        f"Tokenizing using {len(usable_fields)} fields (out of {len(fields)} requested). \nExcluded fields: {set(fields) - set(usable_fields)}"
    )

    tokens_per_row = []
    for _, row in df.iterrows():
        src_prefix = ""
        if (
            add_source_prefix
            and source_col in df.columns
            and not pd.isna(row[source_col])
        ):
            src_prefix = f"{row[source_col]}:"

        row_tokens = []
        for f in usable_fields:
            values = _expand_token_values(row[f])

            for v in values:
                row_tokens.append(f"{src_prefix}{f}={v}")

        tokens_per_row.append(row_tokens)

    return pd.Series(tokens_per_row, index=df.index)


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
    tokens = tokenize_alerts(
        df_k,
        base_fields + ["src_freq_bin", "dst_fanin_bin", "src_fanout_bin"],
    )
    tokens.index = df_k.index  # align indexes for later merging with labels
    return tokens


def build_token_cache_for_scenario(
    df: pd.DataFrame,
    scenario_name: str,
    time_col: str = "timestamp",
    sort_tokens: bool = True,
    extra_meta_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, sparse.csr_matrix, pd.Index]:
    """
    Build a scenario-level token cache with ONE TRANSACTION PER INPUT ROW.

    This function:
    1. filters the dataframe to one scenario
    2. tokenizes all rows in that scenario
    3. keeps one cache row per original dataframe row
    4. builds one sparse binary multi-hot matrix over all row-alerts

    Returns:
        meta_df:
            DataFrame with one row per original input row, containing:
            - timestamp
            - scenario
            - y
            - event_label
            - attack_type
            - alert_id (if present)
            - tokens (list[str])

            The row order of meta_df matches the row order of X_tokens_all.

        X_tokens_all:
            scipy.sparse.csr_matrix of shape (n_rows, n_tokens)

        vocab:
            pd.Index of token strings, aligned to the columns of X_tokens_all
    """
    print("Building token cache for scenario:", scenario_name)

    df_s = df[df["scenario"] == scenario_name].copy()
    if df_s.empty:
        raise ValueError(f"No rows found for scenario '{scenario_name}'")

    df_s[time_col] = pd.to_datetime(df_s[time_col])
    df_s = df_s.sort_values(time_col).reset_index(drop=True)

    # Tokenize each row independently
    tokens_s = tokenize_window(df_s)
    tokens_s.index = df_s.index

    # Make sure every row becomes exactly one transaction
    token_lists = []
    for toks in tokens_s:
        if isinstance(toks, list):
            # deduplicate within row so X is binary, not count-based
            toks = list(set(str(t) for t in toks if pd.notna(t)))
        elif pd.notna(toks):
            toks = [str(toks)]
        else:
            toks = []

        if sort_tokens:
            toks = sorted(toks)

        token_lists.append(toks)

    # Build row-level metadata
    meta_cols = [time_col, "scenario", "y", "event_label", "attack_type"]
    if "alert_id" in df_s.columns:
        meta_cols.append("alert_id")

    if extra_meta_cols is not None:
        meta_cols.extend(extra_meta_cols)

    meta_cols = [c for c in meta_cols if c in df_s.columns]
    meta_cols = list(dict.fromkeys(meta_cols))  # deduplicate, preserve order

    meta_df = df_s[meta_cols].copy()
    meta_df["tokens"] = token_lists
    meta_df = meta_df.reset_index(drop=True)

    # add transaction ID column so we know which cached transactions a candidate fires on
    meta_df.insert(0, "_tx_id", np.arange(len(meta_df), dtype=np.int64))

    # Build vocabulary
    vocab_set = set()
    for toks in meta_df["tokens"]:
        vocab_set.update(toks)

    vocab = pd.Index(sorted(vocab_set), name="token")
    token_to_col = {tok: j for j, tok in enumerate(vocab)}

    # Build sparse binary matrix
    row_idx = []
    col_idx = []

    for i, toks in enumerate(meta_df["tokens"]):
        for tok in toks:
            row_idx.append(i)
            col_idx.append(token_to_col[tok])

    data = np.ones(len(row_idx), dtype=np.uint8)

    X_tokens_all = sparse.csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(meta_df), len(vocab)),
        dtype=np.uint8,
    )

    print(
        "Token cache built: {} row-alerts, {} unique tokens".format(
            X_tokens_all.shape[0], X_tokens_all.shape[1]
        )
    )

    return meta_df, X_tokens_all, vocab


def save_token_cache(
    meta_df: pd.DataFrame,
    X_tokens_all: sparse.csr_matrix,
    vocab: pd.Index,
    scenario_name: str,
    run_name: str,
    out_base: Optional[str] = None,
    meta_filename: str = "meta.parquet",
    matrix_filename: str = "X_tokens.npz",
    vocab_filename: str = "vocab.json",
) -> str:
    """
    Save a scenario token cache to disk.

    Layout:
        {project_root}/out/{run_name}/tokens/{scenario_name}/...

    Files:
        ../out/{run_name}/tokens/{scenario_name}/meta.parquet
        ../out/{run_name}/tokens/{scenario_name}/X_tokens.npz
        ../out/{run_name}/tokens/{scenario_name}/vocab.json

    Returns:
        out_dir: directory where the cache was saved
    """
    if out_base is None:
        out_base = str(Path(__file__).parents[2] / "out")

    out_dir = os.path.join(out_base, run_name, "tokens", scenario_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving token cache for scenario '{scenario_name}' to disk...")

    meta_path = os.path.join(out_dir, meta_filename)
    matrix_path = os.path.join(out_dir, matrix_filename)
    vocab_path = os.path.join(out_dir, vocab_filename)

    meta_to_save = meta_df.copy()

    # store token lists as JSON strings
    if "tokens" in meta_to_save.columns:
        meta_to_save["tokens"] = meta_to_save["tokens"].apply(json.dumps)

    # Make sure tx ids are present
    if "_tx_id" not in meta_to_save.columns:
        meta_to_save.insert(0, "_tx_id", np.arange(len(meta_to_save), dtype=np.int64))

    meta_to_save.to_parquet(meta_path, index=False)
    sparse.save_npz(matrix_path, X_tokens_all)

    token_to_col = {str(tok): int(j) for j, tok in enumerate(vocab)}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(token_to_col, f, ensure_ascii=False)

    print(f"Token cache saved to: {out_dir}")


def load_token_cache(
    scenario_name: str,
    run_name: str,
    out_base: Optional[str] = None,
    meta_filename: str = "meta.parquet",
    matrix_filename: str = "X_tokens.npz",
    vocab_filename: str = "vocab.json",
) -> tuple[pd.DataFrame, sparse.csr_matrix, pd.Index]:
    """
    Load a previously saved scenario token cache.

    Returns:
        meta_df:
            one row per alert, same row order as X_tokens_all

        X_tokens_all:
            sparse CSR token matrix

        vocab:
            pd.Index aligned to matrix columns
    """
    if out_base is None:
        # Compute relative to the project root (msc-thesis)
        out_base = str(Path(__file__).parents[2] / "out")

    out_dir = os.path.join(out_base, run_name, "tokens", scenario_name)

    print(f"Loading token cache for scenario '{scenario_name}' from disk...")

    meta_path = os.path.join(out_dir, meta_filename)
    matrix_path = os.path.join(out_dir, matrix_filename)
    vocab_path = os.path.join(out_dir, vocab_filename)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    meta_df = pd.read_parquet(meta_path)

    if "tokens" in meta_df.columns:
        meta_df["tokens"] = meta_df["tokens"].apply(json.loads)

    X_tokens_all = sparse.load_npz(matrix_path).tocsr()

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = pd.Index(json.load(f), name="token")

    if len(meta_df) != X_tokens_all.shape[0]:
        raise ValueError(
            f"Row mismatch: meta_df has {len(meta_df)} rows, "
            f"but X_tokens_all has {X_tokens_all.shape[0]} rows"
        )

    if len(vocab) != X_tokens_all.shape[1]:
        raise ValueError(
            f"Column mismatch: vocab has {len(vocab)} entries, "
            f"but X_tokens_all has {X_tokens_all.shape[1]} columns"
        )

    print(f"Token cache loaded: {len(meta_df)} alerts, {len(vocab)} unique tokens")

    return meta_df, X_tokens_all, vocab


def slice_token_cache_by_time(
    meta_df: pd.DataFrame,
    X_tokens_all: sparse.csr_matrix,
    start_ts,
    end_ts,
    time_col: str = "timestamp",
    include_end: bool = False,
) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """
    Slice a scenario token cache by time window.

    Args:
        meta_df:
            alert-level metadata; row order must match X_tokens_all
        X_tokens_all:
            sparse matrix with one row per alert
        start_ts:
            inclusive lower bound
        end_ts:
            exclusive upper bound by default
        include_end:
            if True, uses <= end_ts instead of < end_ts

    Returns:
        meta_win:
            subset of meta_df for the requested time window

        X_win:
            sparse matrix slice aligned to meta_win rows
    """
    print(
        f"Slicing token cache by time: {start_ts} to {end_ts} (include_end={include_end})"
    )

    if len(meta_df) != X_tokens_all.shape[0]:
        raise ValueError("meta_df row count must match X_tokens_all row count")

    ts = pd.to_datetime(meta_df[time_col])
    start_ts = pd.to_datetime(start_ts)
    end_ts = pd.to_datetime(end_ts)

    if include_end:
        mask = (ts >= start_ts) & (ts <= end_ts)
    else:
        mask = (ts >= start_ts) & (ts < end_ts)

    row_idx = np.flatnonzero(mask.to_numpy())

    meta_win = meta_df.iloc[row_idx].reset_index(drop=True)
    X_win = X_tokens_all[row_idx]

    print(f"Sliced window: {len(meta_win)} alerts")

    return meta_win, X_win


def iter_precached_windows(
    scenario_name: str,
    run_name: str,
    out_base: Optional[str] = None,
    time_col: str = "timestamp",
    label_col: str = "y",
    window_size: str = "12H",
    step_size: str = "12H",
    align_to: str = "h",
):
    """
    Load one precached scenario and yield aligned window slices.

    Returns an iterator of tuples:
        start_k: window start timestamp
        end_k: window end timestamp
        meta_k: metadata slice for the window
        X_k: sparse token matrix slice for the window
        y_k: label Series aligned to meta_k / X_k
        window_has_attack: bool
    """
    if out_base is None:
        # Compute relative to the project root (msc-thesis)
        out_base = str(Path(__file__).parents[2] / "out")

    print(f"Iterating precached windows for scenario '{scenario_name}'...")

    meta_df, X_tokens_all, vocab = load_token_cache(
        scenario_name=scenario_name,
        run_name=run_name,
        out_base=out_base,
    )

    if meta_df.empty:
        raise ValueError(f"Loaded cache is empty for scenario '{scenario_name}'")
    if time_col not in meta_df.columns:
        raise ValueError(f"'{time_col}' not found in meta_df")
    if label_col not in meta_df.columns:
        raise ValueError(f"'{label_col}' not found in meta_df")

    meta_df = meta_df.sort_values(time_col).reset_index(drop=True)
    t_s = pd.to_datetime(meta_df[time_col])

    windows = make_time_windows(
        t_s,
        window_size=window_size,
        step_size=step_size,
        align_to=align_to,
    )

    for start_k, end_k in windows:
        meta_k, X_k = slice_token_cache_by_time(
            meta_df=meta_df,
            X_tokens_all=X_tokens_all,
            start_ts=start_k,
            end_ts=end_k,
            time_col=time_col,
        )

        if meta_k.empty:
            continue

        y_k = meta_k[label_col].reset_index(drop=True)
        window_has_attack = bool((y_k == 1).any())

        yield start_k, end_k, meta_k, X_k, y_k, window_has_attack, vocab

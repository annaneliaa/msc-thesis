import os
import ast
import json
from glob import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from scipy import sparse

def _candidate_to_tokens(candidate_str: str) -> list[str]:
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

def _load_vocab_token_to_col(vocab_path: str) -> dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)

    if isinstance(vocab_raw, dict):
        return {str(k): int(v) for k, v in vocab_raw.items()}
    if isinstance(vocab_raw, list):
        return {str(tok): i for i, tok in enumerate(vocab_raw)}

    raise TypeError(f"Unsupported vocab format: {type(vocab_raw)}")


def _ensure_meta_tx_id(meta_df: pd.DataFrame, tx_col: str) -> pd.DataFrame:
    meta_df = meta_df.copy()
    if tx_col not in meta_df.columns:
        meta_df = meta_df.reset_index(drop=True)
        meta_df.insert(0, tx_col, np.arange(len(meta_df), dtype=np.int64))
    return meta_df


def _attach_tx_id_from_meta(
    df_used: pd.DataFrame,
    meta_df: pd.DataFrame,
    id_col: str,
    tx_col: str,
) -> pd.DataFrame:
    if tx_col in df_used.columns:
        return df_used.copy()

    if id_col not in df_used.columns:
        raise ValueError(f"df_used must contain '{tx_col}' or '{id_col}'")
    if id_col not in meta_df.columns:
        raise ValueError(f"Cached meta must contain '{id_col}'")

    if meta_df[id_col].duplicated().any():
        raise ValueError(
            f"Cached meta has duplicate '{id_col}' values, cannot safely join tx ids"
        )

    out = df_used.merge(
        meta_df[[id_col, tx_col]],
        on=id_col,
        how="left",
        validate="many_to_one",
    )

    if out[tx_col].isna().any():
        n_missing = int(out[tx_col].isna().sum())
        raise ValueError(f"{n_missing} rows in df_used could not be matched to cached meta")

    out[tx_col] = out[tx_col].astype(np.int64)
    return out


def _parse_tidset_value(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=np.int64)
    if isinstance(x, float) and pd.isna(x):
        return np.array([], dtype=np.int64)
    if isinstance(x, str):
        return np.array(ast.literal_eval(x), dtype=np.int64)
    return np.asarray(x, dtype=np.int64)


def _candidate_token_ids(cand_str: str, token_to_col: dict[str, int]) -> list[int]:
    cand_tokens = [normalize_token(t) for t in _candidate_to_tokens(cand_str)]
    cand_tokens = list(dict.fromkeys(cand_tokens))
    token_ids = [token_to_col[t] for t in cand_tokens if t in token_to_col]

    if len(token_ids) != len(cand_tokens):
        return []

    return token_ids


def _feature_col_name(cand_str: str, max_len: int = 200) -> str:
    return "is_mined__" + cand_str.replace(" ", "").replace("&", "__AND__")[:max_len]


def _candidate_fire_rows(
    cand_row: pd.Series,
    X_sub: sparse.csr_matrix,
    tx_to_row: dict[int, int],
    token_to_col: dict[str, int],
) -> list[int]:
    if "tidset" in cand_row.index:
        tidset = _parse_tidset_value(cand_row["tidset"])
        if tidset.size > 0:
            return [tx_to_row[int(tx)] for tx in tidset if int(tx) in tx_to_row]

    cand_str = str(cand_row["candidate_str"])
    token_ids = _candidate_token_ids(cand_str, token_to_col)
    if not token_ids:
        return []

    nnz = X_sub[:, token_ids].getnnz(axis=1)
    return np.flatnonzero(nnz == len(token_ids)).tolist()


def build_symbolic_features_from_candidates_cached(
    df_used: pd.DataFrame,
    surviving_candidates_df: pd.DataFrame,
    scenario_name: str,
    run_name: str,
    tokens_base_dir: Optional[str] = None,
    id_col: str = "alert_id",
    tx_col: str = "_tx_id",
    min_fires: int = 1,
    meta_filename: str = "meta.parquet",
    matrix_filename: str = "X_tokens.npz",
    vocab_filename: str = "vocab.json",
) -> pd.DataFrame:
    """
    Build symbolic feature matrix from cached token files and survivors.

    - Matches df_used to cached transaction ids via id_col -> tx_col
    - Uses survivor tidsets if present
    - Else reconstruct fires from X_tokens
    - Returns X_sym indexed by id_col
    """
    if "candidate_str" not in surviving_candidates_df.columns:
        raise ValueError("surviving_candidates_df must contain 'candidate_str'")

    if tokens_base_dir is None:
        raise ValueError("tokens_base_dir must be provided")

    if id_col not in df_used.columns:
        raise ValueError(f"df_used must contain '{id_col}'")

    base_dir = os.path.join(tokens_base_dir, "tokens", scenario_name)
    meta_path = os.path.join(base_dir, meta_filename)
    matrix_path = os.path.join(base_dir, matrix_filename)
    vocab_path = os.path.join(base_dir, vocab_filename)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find meta file: {meta_path}")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Could not find matrix file: {matrix_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Could not find vocab file: {vocab_path}")

    meta_df = _ensure_meta_tx_id(pd.read_parquet(meta_path), tx_col)
    df_used = _attach_tx_id_from_meta(df_used.copy(), meta_df, id_col, tx_col)

    if df_used[id_col].duplicated().any():
        dupes = df_used.loc[df_used[id_col].duplicated(), id_col].head().tolist()
        raise ValueError(f"'{id_col}' must be unique. Example duplicates: {dupes}")

    X_tokens = sparse.load_npz(matrix_path).tocsr()
    token_to_col = _load_vocab_token_to_col(vocab_path)

    df_tx_ids = df_used[tx_col].to_numpy(dtype=np.int64)
    X_sub = X_tokens[df_tx_ids]
    tx_to_row = {int(tx): i for i, tx in enumerate(df_tx_ids)}

    candidates_df = (
        surviving_candidates_df
        .dropna(subset=["candidate_str"])
        .drop_duplicates(subset=["candidate_str"])
        .copy()
    )

    rows, cols, feature_names = [], [], []

    for _, cand_row in candidates_df.iterrows():
        fire_rows = _candidate_fire_rows(cand_row, X_sub, tx_to_row, token_to_col)
        if len(fire_rows) < min_fires:
            continue

        col_idx = len(feature_names)
        rows.extend(fire_rows)
        cols.extend([col_idx] * len(fire_rows))
        feature_names.append(_feature_col_name(str(cand_row["candidate_str"])))

    out_index = pd.Index(df_used[id_col].to_numpy(), name=id_col)

    if not feature_names:
        return pd.DataFrame(index=out_index)

    X_sym = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(df_used), len(feature_names)),
        dtype=np.uint8,
    )

    return pd.DataFrame.sparse.from_spmatrix(
        X_sym,
        index=out_index,
        columns=feature_names,
    )
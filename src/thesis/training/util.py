import numpy as np
import pandas as pd

from thesis.schemas.features import FeatureSchema


def find_valid_time_split(y, test_frac=0.3):
    n = len(y)
    split0 = int((1 - test_frac) * n)

    for delta in range(0, n):
        for s in (split0 - delta, split0 + delta):
            if s <= 1 or s >= n - 1:
                continue
            if np.unique(y[:s]).size == 2 and np.unique(y[s:]).size == 2:
                return s
    return None


def prepare_training_frame(
    X_full: pd.DataFrame,
    y,
    schema: FeatureSchema,
    time_col: str = "timestamp",
    random_split: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    if hasattr(y, "reset_index"):
        y = y.reset_index(drop=True)
    else:
        y = pd.Series(y)

    df = X_full.reset_index(drop=True).copy()
    df["__y__"] = y.values

    if not random_split and time_col in df.columns:
        df = df.sort_values(time_col, kind="stable").reset_index(drop=True)

    feature_names = schema.feature_names()
    X = df[feature_names].fillna(0)
    y_arr = df["__y__"].to_numpy()

    return X, y_arr


def make_holdout_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_frac: float = 0.3,
    train_start: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    n = len(X)
    split = int((1 - test_frac) * n)

    if split <= 0 or split >= n:
        raise ValueError(f"Invalid split index {split} for dataset of size {n}.")

    X_train, X_test = X.iloc[train_start:split], X.iloc[split:]
    y_train, y_test = y[train_start:split], y[split:]

    return X_train, X_test, y_train, y_test, split

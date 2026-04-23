import numpy as np
from thesis.schemas.features import FeatureSchema
import pandas as pd


def find_valid_time_split(y, test_frac=0.3):
    n = len(y)
    split0 = int((1 - test_frac) * n)

    # search nearby split points until both sides have both classes
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
) -> tuple[pd.DataFrame, np.ndarray]:
    if hasattr(y, "reset_index"):
        y = y.reset_index(drop=True)
    else:
        y = pd.Series(y)

    df = X_full.reset_index(drop=True).copy()
    df["__y__"] = y.values

    if time_col in df.columns:
        df = df.sort_values(time_col, kind="stable").reset_index(drop=True)

    X = df[schema.features].fillna(0)
    y_arr = df["__y__"].to_numpy()

    return X, y_arr


def make_holdout_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_frac: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    n = len(X)
    split = int((1 - test_frac) * n)

    if split <= 0 or split >= n:
        raise ValueError(f"Invalid split index {split} for dataset of size {n}.")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, split

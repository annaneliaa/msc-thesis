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


def effective_train_start(
    train_start: int, train_frac: float | None, n: int, split: int
) -> int:
    """
    Compose the mine_frac/no_overlap exclusion (train_start) with an explicit
    train_frac cap. train_frac is a fraction of the *full* dataset (n), same
    units as test_frac, so e.g. train_frac=0.1 + test_frac=0.9 reproduces a
    published "first N / rest" split (CSCAS's paper: 6 of 60 days train,
    remainder test) directly as fractions of the full timeline: it keeps only
    the last train_frac*n rows immediately preceding the test window.

    The two never conflict: train_frac can only push the start forward (more
    exclusion), never before train_start, and if train_frac is larger than
    what's available before the split it just saturates back to train_start
    with no error.
    """
    if train_frac is None:
        return train_start
    return max(train_start, split - int(train_frac * n))


def make_holdout_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_frac: float = 0.3,
    train_start: int = 0,
    train_frac: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    n = len(X)
    split = int((1 - test_frac) * n)

    if split <= 0 or split >= n:
        raise ValueError(f"Invalid split index {split} for dataset of size {n}.")

    start = effective_train_start(train_start, train_frac, n, split)
    if start >= split:
        raise ValueError(
            f"No training rows before the test split (train_start={train_start}, "
            f"train_frac={train_frac}, resolved start={start}, split={split})."
        )

    X_train, X_test = X.iloc[start:split], X.iloc[split:]
    y_train, y_test = y[start:split], y[split:]

    return X_train, X_test, y_train, y_test, split

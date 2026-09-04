"""
Shared training-pool construction for every trainable CSCAS baseline
(cscas.py, cscas_base.py, cscas_bert.py, and future cscas_logreg.py /
cscas_xgboost.py / cscas_securebert.py), plus the shared, frozen evaluation
subsample used by every non-replication baseline.

Three pool-construction strategies, one function each:
  - random_undersample_pool: naive random undersampling of the majority
    class down to len(important) rows (paper's Baseline 1).
  - guided_by_cscas_pool: CSCAS-guided undersampling using the SCAS inlier /
    outlier split (paper's Baseline 2 / "Imbalanced-0.5"). CSCAS-only by
    design -- AIT-ADS has no SCAS-equivalent column and generalizing this
    function is an explicit non-goal for now.
  - class_weighted_pool: natural-ratio pool (no undersampling) plus ready-
    to-use class-imbalance kwargs for sklearn/XGBoost. Dataset-agnostic.

All three return (pool, extra_kwargs) so every caller can loop over
conditions with one call pattern, even though random_undersample_pool and
guided_by_cscas_pool always return extra_kwargs={}.
"""

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from thesis.baselines._results import RESULTS_DIR


def random_undersample_pool(
    train: pd.DataFrame,
    important: pd.DataFrame,
    seed: int,
    label_col: str = "Label",
) -> tuple[pd.DataFrame, dict]:
    """Baseline 1: random undersampling.

    Extracted as-is from cscas.py's inline block, generalizing the
    hardcoded n=1_765 to len(important) -- same `.sample(random_state=seed)`
    semantics, no algorithm change, *when* the irrelevant (label==0) class
    is the majority -- true for every CSCAS split by construction, so this
    path is unchanged there.

    Some AIT-ADS (scenario, grouping_method) combinations flip that: e.g.
    deepcase grouping can concentrate far more/larger benign alerts into
    far fewer groups, leaving `important` (label==1) as the actual
    majority in `train`. Sampling `n=len(important)` irrelevant rows
    without replacement then raises ValueError ("Cannot take a larger
    sample than population") -- seen for real on 'fox' grouped with
    deepcase. Undersample whichever class is actually the majority down to
    the minority's count instead, keeping every minority-class row, so
    "random undersampling" means the same thing regardless of which label
    happens to be more numerous in a given split.
    """
    irrelevant = train[train[label_col] == 0]
    if len(irrelevant) >= len(important):
        irr_sample = irrelevant.sample(n=len(important), random_state=seed)
        return pd.concat([important, irr_sample]), {}
    important_sample = important.sample(n=len(irrelevant), random_state=seed)
    return pd.concat([important_sample, irrelevant]), {}


def guided_by_cscas_pool(
    train: pd.DataFrame,
    important: pd.DataFrame,
    seed: int,
    label_col: str = "Label",
    scas_col: str = "SCAS",
) -> tuple[pd.DataFrame, dict]:
    """Baseline 2: guided by CSCAS (Imbalanced-0.5, CSCAS-only).

    n_inliers/n_outliers generalized from hardcoded 882/883 to
    n_inliers = len(important) // 2; n_outliers = len(important) - n_inliers
    (reproduces 882/883 exactly for len(important) == 1_765). CSCAS-only
    intentionally -- do not generalize further; AIT-ADS analog is a
    separate, open decision.
    """
    irr_inliers = train[(train[label_col] == 0) & (train[scas_col] == 0)]
    irr_outliers = train[(train[label_col] == 0) & (train[scas_col] == 1)]

    n_inliers = len(important) // 2
    n_outliers = len(important) - n_inliers

    irr_inl_sample = irr_inliers.sample(n=n_inliers, random_state=seed)
    irr_out_sample = irr_outliers.sample(n=n_outliers, random_state=seed)
    return pd.concat([important, irr_inl_sample, irr_out_sample]), {}


def class_weighted_pool(
    train: pd.DataFrame,
    label_col: str = "Label",
    cap: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Third condition: natural-ratio pool, no undersampling.

    Returns the full `train` pool unchanged by default, plus ready-to-use
    imbalance-handling kwargs (class_weight='balanced' for sklearn,
    scale_pos_weight for XGBoost). When `cap` is supplied, samples `cap`
    rows via stratified proportional capping -- not rebalancing -- so the
    natural ratio is preserved and the class weights still do the
    correction. Dataset-agnostic: label_col is a parameter, reused later
    for AIT-ADS where the minority class flips between attack/benign.
    """
    n_positive = int((train[label_col] == 1).sum())
    n_negative = int((train[label_col] == 0).sum())
    extra_kwargs = {
        "class_weight": "balanced",
        "scale_pos_weight": n_negative / n_positive,
    }

    if cap is None or cap >= len(train):
        return train, extra_kwargs

    pool, _ = train_test_split(
        train,
        train_size=cap,
        stratify=train[label_col],
        random_state=seed,
    )
    return pool, extra_kwargs


EVAL_SUBSAMPLE_PATH = RESULTS_DIR / "cscas_eval_subsample.json"
EVAL_SUBSAMPLE_SEED = 0
EVAL_SUBSAMPLE_SIZE = 20_000


def get_cscas_eval_subsample(
    test: pd.DataFrame,
    label_col: str = "Label",
    n: int = EVAL_SUBSAMPLE_SIZE,
    seed: int = EVAL_SUBSAMPLE_SEED,
) -> pd.DataFrame:
    """Shared, frozen stratified-proportional subsample of the CSCAS test
    set, used by every non-replication CSCAS baseline and mandatorily by
    zero-shot. NOT used by cscas.py/cscas_base.py, which stay on the full
    test set forever (paper replication).

    Builds once with a fixed seed and persists row indices to disk on
    first call; every later call across every script loads the same rows
    rather than resampling.

    Persisted indices are `test`'s DataFrame index *labels* (not
    positions) -- stable only as long as `test` is rebuilt via the same
    recipe every script already uses (read_csv -> parse Timestamp ->
    sort_values("Timestamp") -> reset_index(drop=True) on the full
    dataset -> chronological split), so `test`'s labels are deterministic
    row identifiers even though they aren't 0..len(test)-1.
    """
    if EVAL_SUBSAMPLE_PATH.exists():
        indices = json.loads(EVAL_SUBSAMPLE_PATH.read_text(encoding="utf-8"))
        return test.loc[indices]

    subsample, _ = train_test_split(
        test,
        train_size=n,
        stratify=test[label_col],
        random_state=seed,
    )
    indices = sorted(subsample.index.tolist())

    RESULTS_DIR.mkdir(exist_ok=True)
    EVAL_SUBSAMPLE_PATH.write_text(json.dumps(indices, indent=2), encoding="utf-8")

    return test.loc[indices]

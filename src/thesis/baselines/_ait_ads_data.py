"""
Shared AIT-ADS data loader for the ait_ads_*.py baseline scripts -- plays
the same role CSCAS's `pd.read_csv(...)` load does in cscas.py/cscas_base.py/
cscas_bert.py, but AIT-ADS has no single raw CSV: alert_groups come from the
same ingest -> tokenize -> group -> encode pipeline experiments/baseline.py
already runs.

Returns (train, test) DataFrames shaped so _sampling.py's existing
random_undersample_pool/class_weighted_pool (already label_col-parameterized)
work completely unmodified:
  - Label: int 0/1, mapped from AlertGroup.group_label ("benign"/"attack");
    unlabelled/mixed rows dropped, same as experiments/baseline.py's own
    `mask = y.notna()` handling.
  - text: one string per alert_group, built by joining AlertGroup.raw_items
    tokens (human-readable sig:/host:/short: tags -- see
    preprocessing/tokenization.py) plus n_alerts/hour_of_day -- the AIT-ADS
    analog of CSCAS's SignatureText field, used by the BERT/SecureBERT/
    zero-shot scripts. All six ait_ads_*.py model scripts (rf, logreg,
    xgboost, bert, securebert, zeroshot) call this same function, so every
    model family scores the identical rows for a given (scenario,
    grouping_method) -- not just similarly-configured pipelines.

Reuses the exact same ingest/cache calls experiments/baseline.py runs (its
steps 1-5, schema_name="base") for the fixed_window/time_delta/
cscas_grouping methods, plus the same cache_dir convention
scripts/system_eval/run_model_comparison_attribute.py uses
(CACHE_DIR/<scenario>/groups/<grouping_method>), so alert_groups here are
the identical ones (same cache, same grouping) every other AIT-ADS
experiment already uses for this (scenario, grouping_method) pair -- not a
second, diverging ingestion. alertbert/deepcase groupings have no such
standard-pipeline path (see _ait_ads_grouping.py's module docstring for
why) -- materialize_learned_grouping() bridges them into the identical
cache format first, so everything downstream of that point is unified
across all 5 methods.

No hardcoded row-count assertions and no eval-subsample step (unlike CSCAS,
which needs a frozen 20k-row subsample because its test set is 1.25M rows):
there's no paper to replicate here, and AIT-ADS scenarios are far smaller.
Callers that need a size cap (e.g. for the class-weighted fine-tuning pool)
should do it themselves, same as CLASS_WEIGHTED_POOL_CAP in cscas_bert.py.

Known, accepted exclusion -- 'harrison' and 'santos' produce no baseline
result under ANY grouping method, and 'russellmitchell' produces one only
under 'deepcase': the split above is a fixed chronological
test_frac-from-the-end cut, no search for a "valid" boundary, and for
these scenarios every attack-labelled alert_group falls within that last
test_frac -- train ends up 100% benign regardless of grouping method,
since that's a property of *when* the attacks happen in the raw alert
timeline, not how the alerts get grouped. Callers detect this themselves
(train["Label"].nunique() < 2) and skip with a clear message rather than
this function raising -- deliberately not "fixed" by searching for a
different split point (a considered decision, not an oversight): doing
so would give these 3 scenarios a different, potentially far-from-70/30
effective train/test ratio than the other 5, which stay at a clean
test_frac=0.3 split throughout. Documented here, not just left to the
runtime skip message, the same way _ait_ads_grouping.py documents why
alertbert/deepcase exclude shaw/wardbeck/wheeler/wilson -- two structurally
different reasons (leakage vs. temporal attack concentration), same
"accepted exclusion, not a bug" treatment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from thesis.baselines._ait_ads_grouping import (
    LEARNED_METHODS,
    materialize_learned_grouping,
)
from thesis.config import GroupingConfig
from thesis.paths import CACHE_DIR
from thesis.pipeline.pipeline import (
    convert_ait_alerts_to_json,
    encode_and_cache_alert_groups,
    ensure_feature_manifest,
    ingest_ait_alert_batch,
    load_or_build_alert_groups,
)
from thesis.schemas.groups import AlertGroup

GROUPING_METHODS = [
    "fixed_window",
    "time_delta",
    "cscas_grouping",
    "alertbert",
    "deepcase",
]

_LABEL_MAP = {"benign": 0, "attack": 1}


def _build_text_column(alert_groups: list[AlertGroup]) -> list[str]:
    """One serialized string per alert_group: its raw_items tokens (the
    AIT-ADS analog of CSCAS's SignatureText -- see module docstring) plus
    n_alerts/hour_of_day, mirroring compute_ait_ads_baseline_features'
    own field choice (encoders/baseline.py)."""
    texts = []
    for tx in alert_groups:
        tokens = " ".join(sorted(tx.raw_items or []))
        hour_part = ""
        if tx.start_ts is not None:
            hour = datetime.fromtimestamp(int(tx.start_ts), tz=timezone.utc).hour
            hour_part = f" | Hour: {hour}"
        texts.append(f"Tokens: {tokens} | N_alerts: {tx.n_alerts}{hour_part}")
    return texts


def _load_encoded_ait_ads(
    scenario: str,
    grouping_method: str,
    cache_dir: Path | None,
) -> tuple[pd.DataFrame, list[AlertGroup]]:
    """Ingest (if not already cached) + encode `scenario` under the AIT-ADS
    base schema for `grouping_method`, returning the labeled df (`Label`/
    `text` columns added, unlabelled/mixed rows dropped) alongside the
    AlertGroup list filtered and ordered to match it 1:1 by position --
    the shared core both load_ait_ads_baseline_split and
    load_ait_ads_baseline_split_with_groups slice identically, so the two
    functions can never disagree about which rows a given split contains.
    """
    if grouping_method not in GROUPING_METHODS:
        raise ValueError(
            f"Unknown grouping_method {grouping_method!r}, expected one of {GROUPING_METHODS}"
        )

    cache_dir = cache_dir or (CACHE_DIR / scenario / "groups" / grouping_method)

    if grouping_method in LEARNED_METHODS:
        materialize_learned_grouping(scenario, grouping_method, cache_dir)
    else:
        grouping = GroupingConfig(mode=grouping_method)
        alerts_path = convert_ait_alerts_to_json(scenario)
        ingest_ait_alert_batch(
            scenario,
            alerts_path,
            cache_dir,
            grouping_mode=grouping.mode,
            grouping=grouping,
        )
    ensure_feature_manifest(scenario)
    alert_groups = load_or_build_alert_groups(scenario, cache_dir)

    df, _schema = encode_and_cache_alert_groups(
        scenario, alert_groups, "base", cache_dir
    )
    df = df.reset_index(drop=True)
    # encode_and_cache_alert_groups's df rows are in the same order as the
    # alert_groups list passed in (its meta_df/feature_df are both built
    # directly from it) -- safe to zip text back on by position.
    df["text"] = _build_text_column(alert_groups)
    df["Label"] = df["group_label"].map(_LABEL_MAP)

    n_unlabelled = int(df["Label"].isna().sum())
    if n_unlabelled:
        print(
            f"  [warn] Dropping {n_unlabelled} alert_groups with unlabelled/mixed group_label"
        )
        keep_mask = df["Label"].notna().to_numpy()
        df = df[keep_mask].reset_index(drop=True)
        alert_groups = [tx for tx, keep in zip(alert_groups, keep_mask) if keep]
    df["Label"] = df["Label"].astype(int)

    return df, alert_groups


def load_ait_ads_baseline_split(
    scenario: str,
    grouping_method: str = "fixed_window",
    test_frac: float = 0.3,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest (if not already cached) + encode `scenario` under the AIT-ADS
    base schema for the given `grouping_method`, chronologically split
    train/test, return (train, test) DataFrames with `Label` (0/1) and
    `text` columns.

    grouping_method must be one of GROUPING_METHODS. alertbert/deepcase
    raise ValueError for scenario in
    thesis.baselines._ait_ads_grouping.LEAKAGE_SCENARIOS -- see that
    module's docstring.

    Mirrors experiments/baseline.py's steps 1-5 (same ingest/cache calls,
    same schema_name="base") for fixed_window/time_delta/cscas_grouping, so
    this is scored on the identical alert_groups every other AIT-ADS
    experiment already uses for this (scenario, grouping_method) pair.
    """
    df, _alert_groups = _load_encoded_ait_ads(scenario, grouping_method, cache_dir)
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test


def load_ait_ads_baseline_split_with_groups(
    scenario: str,
    grouping_method: str = "fixed_window",
    test_frac: float = 0.3,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[AlertGroup], list[AlertGroup]]:
    """Same split as load_ait_ads_baseline_split (same helper, same
    test_frac cut), but also returns the (train_groups, test_groups)
    AlertGroup lists aligned 1:1 by position with (train, test) -- for
    ait_ads_mining.py, which needs the actual AlertGroup objects (not just
    the encoded base-schema columns) to run
    thesis.mining.attribute_mining_job.run_alert_group_attribute_mining_job
    on the train split, the same way cscas_mining.py does for CSCAS.
    """
    df, alert_groups = _load_encoded_ait_ads(scenario, grouping_method, cache_dir)
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    train_groups = alert_groups[:split_idx]
    test_groups = alert_groups[split_idx:]
    return train, test, train_groups, test_groups

"""
Same experimental setup as baselines/cscas_base.py (split, training-pool
sampling, seeds, eval set) -- the feature matrix is cscas_base.py's 5-feature
reduced base schema extended with symbolic features mined by the
attribute-mining pipeline (contrast-set + decision-tree rules, see
thesis.mining.attribute_mining_job) on the SAME train split.

The mined matrix is then fit with all three tabular classifiers -- the
mining counterparts of cscas_base.py / cscas_logreg.py / cscas_xgboost.py --
saved as `cscas_mining` (RF, name unchanged), `cscas_mining_logreg` and
`cscas_mining_xgboost`. LogReg gets the same median-impute-of-the--1-sentinel
+ StandardScaler Pipeline cscas_logreg.py applies (the mined symbolic columns
are binary indicators and pass through unchanged for every model). Each
result is skipped if its JSON already exists (set CSCAS_FORCE=1 to
recompute); the single attribute-mining pass still runs on every invocation,
since only its RF output was ever persisted.

Mining is deliberately restricted to exclude SCAS and every
Similarity-derived candidate field (scas, similarity,
signature_id_similarity, attr_value:*, attr_populated:*,
applicable_layer:*) -- the same fields cscas_base.py already excludes from
its own 5 features, for the same reason: none of these are things a real
deployment could compute for a fresh alert without already knowing the
answer or running CSCAS's offline similarity pipeline (see
Docs/Baselines.md). Excluding them keeps this baseline a fair "does mining
add value on top of the same realistic base schema" comparison rather than
partly winning by reintroducing information cscas_base.py ruled out.

The mined symbolic schema is built and used purely in-memory here (via
build_symbolic_feature_schema + SymbolicFeatureEncoder) rather than through
mine_or_reuse_attribute_schema's on-disk registry -- that registry is shared
with real experiments on scenario "cscas" and this is a standalone,
self-contained baseline script, same as cscas_base.py.

Two attribute-mining tree modes, via the CSCAS_MINING_MODE env var (default
"single_tree", the original config every existing *_mining*.json result was
produced with). "two_tree" is an add-on, not a replacement: it fits a
second, deeper tree for attack-leaning leaves specifically (see
_cscas_schema.mining_attribute_config), and writes to separate
"_twotree"-suffixed result files (cscas_mining_twotree,
cscas_mining_logreg_twotree, cscas_mining_xgboost_twotree, plus schema/eval
suffixes) so single-tree results are never overwritten.

Run:
    cd src/thesis/baselines
    python cscas_mining.py                              # single-tree (default)
    CSCAS_MINING_MODE=two_tree python cscas_mining.py    # two-tree add-on

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from thesis.baselines._cscas_schema import (
    MINING_MODE_SUFFIX,
    SCHEMAS_CLASSIFIER,
    active_mining_mode,
    active_schema,
    cscas_feature_cols,
    grid_outputs_done,
    mining_attribute_config,
    result_name,
    schema_blurb,
    sentinel_imputer,
)
from thesis.baselines._results import results_exist, save_baseline_results
from thesis.baselines._sampling import (
    class_weighted_pool,
    get_cscas_eval_subsample,
    guided_by_cscas_pool,
    random_undersample_pool,
)
from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.features.schema_builder import build_symbolic_feature_schema
from thesis.mining.attribute_mining_job import run_alert_group_attribute_mining_job
from thesis.paths import CACHE_DIR
from thesis.pipeline.pipeline import rows_to_cscas_alert_groups, save_alert_groups_json
from thesis.schemas.preprocessing import ATTR_SIMILARITY_COLUMNS

print("Using device: cpu")

# 1) Load and sort dataset

df = pd.read_csv("../../../data/cscas/dataset-labeled-anon-ip.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# 2) Verify dataset against papers numbers
assert len(df) == 1_395_324, f"got {len(df)}"
assert df["Label"].sum() == 20_952, f"got {df['Label'].sum()}"
assert df["SCAS"].sum() == 72_672, f"got {df['SCAS'].sum()}"

# 3) Split into train and test sets based on timestamp -- identical to
# cscas_base.py's split, so the two baselines are directly comparable.
split_time = pd.Timestamp("2022-01-26 06:23:21+02:00")

train = df[df["Timestamp"] <= split_time].copy()
test = df[df["Timestamp"] > split_time].copy()

assert len(train) == 139_532, f"got {len(train)}"
assert len(test) == 1_255_792, f"got {len(test)}"
assert train["Label"].sum() == 1_765, f"got {train['Label'].sum()}"
assert test["Label"].sum() == 19_187, f"got {test['Label'].sum()}"

# train is an unbroken 0-based prefix slice of df's own 0..N-1 RangeIndex
# (post reset_index(drop=True) above), so its index labels equal positional
# row order -- the pool -> symbolic-feature alignment below (via
# symbolic_train_df.iloc[pool.index]) depends on this invariant holding.
assert list(train.index) == list(range(len(train)))

# 4) Base feature schema -- CSCAS_SCHEMA env var picks "base" (5 cols) or
# "full" (the paper's 42). The mined symbolic features are added on top of
# whichever is active. See _cscas_schema.py.
SCHEMA = active_schema(SCHEMAS_CLASSIFIER)
FEATURE_COLS = cscas_feature_cols(df, schema=SCHEMA)
print(
    f"Base schema: {SCHEMA} -- {len(FEATURE_COLS)} feature columns (+ mined symbolic)"
)
print(FEATURE_COLS)

# 4b) Mining tree mode -- CSCAS_MINING_MODE env var picks "single_tree"
# (default, unchanged result files) or "two_tree" (add-on, "_twotree"-suffixed
# result files). See _cscas_schema.py.
MINING_MODE = active_mining_mode()
MODE_SUFFIX = MINING_MODE_SUFFIX[MINING_MODE]
print(f"Mining tree mode: {MINING_MODE}")

# Skip a model whose results/*.json already exists -- CSCAS_FORCE=1 to
# recompute. If ALL three models x both eval sets are already on disk for
# this schema, exit before the (minutes-long) attribute-mining pass.
FORCE = os.environ.get("CSCAS_FORCE", "0") == "1"
_MINING_STEMS = [
    f"cscas_mining{MODE_SUFFIX}",
    f"cscas_mining_logreg{MODE_SUFFIX}",
    f"cscas_mining_xgboost{MODE_SUFFIX}",
]
if grid_outputs_done(_MINING_STEMS, SCHEMA):
    print(
        f"[skip] all cscas_mining {SCHEMA} ({MINING_MODE}) outputs already exist "
        "(CSCAS_FORCE=1 to re-run)."
    )
    raise SystemExit(0)

# 5) Verify training pools against Table IV (pool construction itself lives
# in _sampling.py -- these are just the sanity-check counts).
important = train[train["Label"] == 1]
irr_inliers = train[(train["Label"] == 0) & (train["SCAS"] == 0)]
irr_outliers = train[(train["Label"] == 0) & (train["SCAS"] == 1)]

assert len(important) == 1_765, f"got {len(important)}"
assert len(irr_inliers) == 133_614, f"got {len(irr_inliers)}"
assert len(irr_outliers) == 4_153, f"got {len(irr_outliers)}"

# 6) Prepare eval sets -- both cells of the test-set axis.
#   subsample: shared, frozen 20k -- the grid every baseline lives in.
#   fulltest:  all 1.26M test rows -- the CSCAS paper's own protocol.
eval_df = get_cscas_eval_subsample(test)
EVAL_FRAMES = {"subsample": eval_df, "fulltest": test}
print(
    f"Evaluating on: subsample ({len(eval_df)} rows, {int(eval_df['Label'].sum())} pos)"
    f"  +  full test ({len(test)} rows, {int(test['Label'].sum())} pos)"
)

# 7) Build AlertGroups for train and eval, using the same per-row parser
# ingest_cscas_scenario() uses for the full dataset -- applied directly to
# these row subsets rather than re-deriving a global sort order, so there's
# no risk of pandas-vs-Python tie-breaking mismatches on duplicate
# timestamps. The full-test parse+encode is the slow step (~1.26M rows).
print("Building AlertGroups for train/eval splits...")
train_groups = rows_to_cscas_alert_groups(train.to_dict("records"))
eval_groups = {
    ek: rows_to_cscas_alert_groups(frame.to_dict("records"))
    for ek, frame in EVAL_FRAMES.items()
}
assert len(train_groups) == len(train), "row parsing dropped rows -- alignment broken"
for ek, groups in eval_groups.items():
    assert len(groups) == len(EVAL_FRAMES[ek]), f"row parsing dropped {ek} rows"

# 8) Persist train_groups to JSON -- run_alert_group_attribute_mining_job
# takes a file path, not an in-memory list.
train_alert_groups_path = (
    CACHE_DIR
    / "cscas"
    / "groups"
    / "cscas_pregrouped_baseline_mining"
    / "alert_groups"
    / "train_alert_groups.json"
)
train_alert_groups_path.parent.mkdir(parents=True, exist_ok=True)
save_alert_groups_json(train_groups, train_alert_groups_path)

# 9) Mine symbolic features on the train split, excluding SCAS/Similarity-
# derived candidate fields -- see module docstring for why.
LEAKY_ATTRIBUTE_FIELDS = {
    "scas",
    "similarity",
    "signature_id_similarity",
    *(f"attr_value:{n}" for n in ATTR_SIMILARITY_COLUMNS),
    *(f"attr_populated:{n}" for n in ATTR_SIMILARITY_COLUMNS),
    *(
        f"applicable_layer:{p.lower()}"
        for p in ("Dns", "Email", "Http", "Smtp", "Ssh", "Tls")
    ),
}

print(f"Mining attribute schema on train split ({MINING_MODE} mode)...")
mining_result = run_alert_group_attribute_mining_job(
    alert_groups_path=train_alert_groups_path,
    scenario_name="cscas",
    run_name=f"cscas_baseline_mining{MODE_SUFFIX}",
    config=mining_attribute_config(MINING_MODE),
    exclude_fields=LEAKY_ATTRIBUTE_FIELDS,
)
print(f"  Mined {len(mining_result.predicates)} predicates from train split.")

# 10) Build an in-memory symbolic schema (no on-disk registry writes -- that
# registry is shared with real experiments on scenario "cscas") and encode
# train/eval AlertGroups under it.
symbolic_schema = build_symbolic_feature_schema(
    df=mining_result.mined_df,
    source_label="attack",
    schema_name="cscas_mining_symbolic",
    schema_version="0.1.0",
    predicates=mining_result.predicates,
)
print(f"  Built {len(symbolic_schema.features)} symbolic features.")

encoder = SymbolicFeatureEncoder(feature_schema=symbolic_schema)
symbolic_train_df = encoder.transform(train_groups)
symbolic_eval_df = {ek: encoder.transform(groups) for ek, groups in eval_groups.items()}

# 11) Training pools (same 3 conditions as cscas_base.py) and per-model
# feature-matrix / classifier builders.
POOL_BUILDERS = {
    "random": lambda seed: random_undersample_pool(train, important, seed),
    "class_weighted": lambda seed: class_weighted_pool(train, seed=seed),
    "guided": lambda seed: guided_by_cscas_pool(train, important, seed),
}


def build_matrix(base_df: pd.DataFrame, symbolic_df: pd.DataFrame) -> np.ndarray:
    """base_df rows aligned 1:1 (by position) with symbolic_df rows. Raw base
    columns for every model -- the -1 sentinel is handled inside LogReg's own
    Pipeline (sentinel_imputer), and RF/XGBoost treat -1 as a low value. The
    mined symbolic columns are binary indicators, passed through unchanged."""
    return pd.concat(
        [
            base_df[FEATURE_COLS].reset_index(drop=True),
            symbolic_df.reset_index(drop=True),
        ],
        axis=1,
    ).values


def build_classifier(model: str, seed: int, extra_kwargs: dict):
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            class_weight=extra_kwargs.get("class_weight"),
        )
    if model == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            scale_pos_weight=extra_kwargs.get("scale_pos_weight"),
        )
    return Pipeline(
        [
            ("impute", sentinel_imputer()),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                    class_weight=extra_kwargs.get("class_weight"),
                ),
            ),
        ]
    )


# 12) Fit each of the three classifiers on the mined matrix. "cscas_mining"
# stays the RF result (name unchanged); the two new ones get a model suffix;
# MODE_SUFFIX additionally tags two_tree-mode results ("_twotree") so they
# never collide with single_tree's. A model whose JSON already exists is
# skipped (CSCAS_FORCE=1 to recompute).
MODELS = {
    "rf": (f"cscas_mining{MODE_SUFFIX}", "RandomForestClassifier(n_estimators=100)"),
    "logreg": (
        f"cscas_mining_logreg{MODE_SUFFIX}",
        "median-imputed -1 sentinel + StandardScaler + LogisticRegression",
    ),
    "xgboost": (
        f"cscas_mining_xgboost{MODE_SUFFIX}",
        "XGBClassifier(n_estimators=100)",
    ),
}

for model, (stem, model_desc) in MODELS.items():
    needed = {ek: result_name(stem, SCHEMA, ek) for ek in EVAL_FRAMES}
    if not FORCE and all(results_exist(n) for n in needed.values()):
        print(
            f"\n[skip] {model}: {list(needed.values())} already exist "
            "(set CSCAS_FORCE=1 to re-run)."
        )
        continue

    X_ev = {
        ek: build_matrix(EVAL_FRAMES[ek], symbolic_eval_df[ek]) for ek in EVAL_FRAMES
    }
    y_ev = {ek: EVAL_FRAMES[ek]["Label"].values for ek in EVAL_FRAMES}
    # results[eval_set][condition] -> per-seed metric dicts
    results: dict[str, dict[str, list[dict[str, float]]]] = {
        ek: {name: [] for name in POOL_BUILDERS} for ek in EVAL_FRAMES
    }

    for condition, build_pool in POOL_BUILDERS.items():
        print(f"\n=== {model} / {condition} ({SCHEMA} schema + mining) ===")

        for seed in range(5):
            pool, extra_kwargs = build_pool(seed)

            # pool.index gives positions into symbolic_train_df because train's
            # index labels equal positional row order (asserted in step 3).
            X_tr = build_matrix(pool, symbolic_train_df.iloc[pool.index])
            y_tr = pool["Label"].values

            clf = build_classifier(model, seed, extra_kwargs)
            clf.fit(X_tr, y_tr)

            row = []
            for ek in EVAL_FRAMES:
                y_pred = clf.predict(X_ev[ek])
                m = {
                    "precision": precision_score(y_ev[ek], y_pred),
                    "recall": recall_score(y_ev[ek], y_pred),
                    "f1": f1_score(y_ev[ek], y_pred),
                }
                results[ek][condition].append(m)
                row.append(f"{ek} F1={m['f1']:.3f}")
            print(f"  seed={seed}: " + "  |  ".join(row))

        for ek in EVAL_FRAMES:
            avg = pd.DataFrame(results[ek][condition]).mean()
            print(
                f"  AVERAGE [{ek}]: P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}"
            )

    for ek in EVAL_FRAMES:
        save_baseline_results(
            name=needed[ek],
            description=(
                f"{schema_blurb(SCHEMA, len(FEATURE_COLS))} + attribute-mined "
                f"symbolic features (contrast-set + decision-tree rules, {MINING_MODE} "
                "mode, mined on the same train split as cscas_base; SCAS/"
                f"Similarity-derived fields excluded from mining), {model_desc}, "
                "evaluated on the "
                f"{'shared 20k eval subsample' if ek == 'subsample' else 'full 1.26M-row test set'}"
            ),
            results=results[ek],
        )

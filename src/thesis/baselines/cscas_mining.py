"""
Same experimental setup as baselines/cscas_base.py (split, training-pool
sampling, classifier, seeds, eval set) -- the only thing that changes is the
feature matrix: cscas_base.py's 5-feature reduced base schema, extended with
symbolic features mined by the attribute-mining pipeline (contrast-set +
decision-tree rules, see thesis.mining.attribute_mining_job) on the SAME
train split.

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

Run:
    cd src/thesis/baselines
    python cscas_mining.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from thesis.baselines._results import save_baseline_results
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
from thesis.schemas.mining import AttributeMiningConfig
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

# 4) Define feature columns -- same reduced base schema as cscas_base.py.
DROP_COLS = [
    "Timestamp",
    "SignatureText",
    "Label",
    "ExtIP",
    "IntIP",
    "SignatureID",
    "SCAS",
]
FEATURE_COLS = [
    c for c in df.columns if c not in DROP_COLS and not c.endswith("Similarity")
]
assert len(FEATURE_COLS) == 5, f"got {len(FEATURE_COLS)}"
print(f"Base feature count: {len(FEATURE_COLS)}")
print(FEATURE_COLS)

# 5) Verify training pools against Table IV (pool construction itself lives
# in _sampling.py -- these are just the sanity-check counts).
important = train[train["Label"] == 1]
irr_inliers = train[(train["Label"] == 0) & (train["SCAS"] == 0)]
irr_outliers = train[(train["Label"] == 0) & (train["SCAS"] == 1)]

assert len(important) == 1_765, f"got {len(important)}"
assert len(irr_inliers) == 133_614, f"got {len(irr_inliers)}"
assert len(irr_outliers) == 4_153, f"got {len(irr_outliers)}"

# 6) Prepare eval set -- shared, frozen subsample (same as cscas_base.py).
eval_df = get_cscas_eval_subsample(test)
print(
    f"Evaluating on shared eval subsample: {len(eval_df)} rows, {int(eval_df['Label'].sum())} positive"
)

# 7) Build AlertGroups for train and eval, using the same per-row parser
# ingest_cscas_scenario() uses for the full dataset -- applied directly to
# these row subsets rather than re-deriving a global sort order, so there's
# no risk of pandas-vs-Python tie-breaking mismatches on duplicate
# timestamps.
print("Building AlertGroups for train/eval splits...")
train_groups = rows_to_cscas_alert_groups(train.to_dict("records"))
eval_groups = rows_to_cscas_alert_groups(eval_df.to_dict("records"))
assert len(train_groups) == len(train), "row parsing dropped rows -- alignment broken"
assert len(eval_groups) == len(eval_df), "row parsing dropped rows -- alignment broken"

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

print("Mining attribute schema on train split...")
mining_result = run_alert_group_attribute_mining_job(
    alert_groups_path=train_alert_groups_path,
    scenario_name="cscas",
    run_name="cscas_baseline_mining",
    config=AttributeMiningConfig(),
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
symbolic_eval_df = encoder.transform(eval_groups)

# 11) Build the eval feature matrix -- base + mined columns.
X_test = pd.concat(
    [
        eval_df[FEATURE_COLS].reset_index(drop=True),
        symbolic_eval_df.reset_index(drop=True),
    ],
    axis=1,
).values
y_test = eval_df["Label"].values

# 12) Three training-pool conditions -- same as cscas_base.py.
POOL_BUILDERS = {
    "random": lambda seed: random_undersample_pool(train, important, seed),
    "class_weighted": lambda seed: class_weighted_pool(train, seed=seed),
    "guided": lambda seed: guided_by_cscas_pool(train, important, seed),
}

results: dict[str, list[dict[str, float]]] = {name: [] for name in POOL_BUILDERS}

for condition, build_pool in POOL_BUILDERS.items():
    print(f"\n=== {condition} (base schema + mining) ===")

    for seed in range(5):
        pool, extra_kwargs = build_pool(seed)

        # pool.index gives positions into symbolic_train_df because train's
        # index labels equal positional row order (asserted in step 3).
        X_tr = np.hstack(
            [pool[FEATURE_COLS].values, symbolic_train_df.iloc[pool.index].values]
        )
        y_tr = pool["Label"].values

        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            class_weight=extra_kwargs.get("class_weight"),
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_test)

        p = precision_score(y_test, y_pred)
        r = recall_score(y_test, y_pred)
        f = f1_score(y_test, y_pred)
        results[condition].append({"precision": p, "recall": r, "f1": f})
        print(f"  seed={seed}: P={p:.3f} R={r:.3f} F1={f:.3f}")

    avg = pd.DataFrame(results[condition]).mean()
    print(f"  AVERAGE: P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}")


print("\n=== Summary: base schema + mining ===")
for condition in POOL_BUILDERS:
    avg = pd.DataFrame(results[condition]).mean()
    print(f"{condition:<16}P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}")

save_baseline_results(
    name="cscas_mining",
    description=(
        "Base schema (5 features) + attribute-mined symbolic features "
        "(contrast-set + decision-tree rules, mined on the same train split "
        "as cscas_base; SCAS/Similarity-derived fields excluded from "
        "mining), RandomForestClassifier(n_estimators=100), evaluated on "
        "the shared eval subsample"
    ),
    results=results,
)

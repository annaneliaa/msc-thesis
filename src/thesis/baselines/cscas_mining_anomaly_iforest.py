"""
IsolationForest counterpart to baselines/cscas_mining_anomaly.py: the same
5-feature reduced base schema extended with symbolic features mined by the
attribute-mining pipeline (contrast-set + decision-tree rules) on the train
split, but fit as a one-class IsolationForest on benign-only rows instead
of a OneClassSVM.

This is to cscas_mining_anomaly.py (OneClassSVM) what
cscas_anomaly_iforest.py is to cscas_anomaly.py -- it isolates model
choice within the anomaly-detector family for the *mining* scenario the
same way, rather than treating "the mining anomaly baseline" as a single
fixed model. Tree-based, so unlike OneClassSVM's model_factory entry
"iforest" isn't wrapped in a StandardScaler Pipeline.

Mining runs on the FULL train split (both classes) -- attack rows are still
needed to mine informative attack-vs-benign contrast predicates, even
though the model itself only ever fits on the benign subset of the
resulting (base + mined) feature matrix afterwards. Same
LEAKY_ATTRIBUTE_FIELDS exclusion, same train split, same in-memory schema,
same cache namespace as cscas_mining_anomaly.py, so the mined predicates
are shared between the two scripts.

No pool-condition loop, no seeds -- same "single deterministic run"
precedent as cscas_anomaly.py / cscas_mining_anomaly.py. IsolationForest's
own randomness is pinned by its fixed random_state=42 in model_factory.py.

Scoring convention (matches cscas_anomaly.py / cscas_mining_anomaly.py):
  scores = -model.decision_function(X_test)   # higher = more anomalous
  y_pred = (model.predict(X_test) == -1)      # 1 = anomaly = attack

Two attribute-mining tree modes, via the CSCAS_MINING_MODE env var (default
"single_tree", the original config every existing *_mining*.json result was
produced with). "two_tree" is an add-on, not a replacement, and uses the
IDENTICAL mining config cscas_mining.py does (see
_cscas_schema.mining_attribute_config) -- both the shallow benign-facing
tree and the deeper attack-facing one are fit during mining, same as the
classifier script. What's different here is what happens AFTER mining:
_cscas_schema.discard_attack_patterns() drops every attack-leaning mined
pattern (including, in two_tree mode, the whole attack-facing tree's leaves)
before the symbolic feature schema is built, so the anomaly detector's
(base + mined) training matrix never carries attack-derived information,
consistent with its benign-only .fit(). Writes "_twotree"-suffixed result
files so single-tree results are never overwritten.

Run:
    cd src/thesis/baselines
    python cscas_mining_anomaly_iforest.py                              # single-tree
    CSCAS_MINING_MODE=two_tree python cscas_mining_anomaly_iforest.py    # two-tree add-on

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from thesis.baselines._cscas_schema import (
    MINING_MODE_SUFFIX,
    SCHEMAS_ANOMALY,
    active_mining_mode,
    active_schema,
    cscas_feature_cols,
    discard_attack_patterns,
    grid_outputs_done,
    mining_attribute_config,
    result_name,
    schema_blurb,
)
from thesis.baselines._results import save_anomaly_results
from thesis.baselines._sampling import get_cscas_eval_subsample
from thesis.encoders.symbolic import SymbolicFeatureEncoder
from thesis.features.schema_builder import build_symbolic_feature_schema
from thesis.mining.attribute_mining_job import run_alert_group_attribute_mining_job
from thesis.paths import CACHE_DIR
from thesis.pipeline.pipeline import rows_to_cscas_alert_groups, save_alert_groups_json
from thesis.schemas.preprocessing import ATTR_SIMILARITY_COLUMNS
from thesis.training.workload import (
    average_workload_at_recall,
    compute_workload_at_recall,
)

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
# cscas_mining.py/cscas_base.py's split.
split_time = pd.Timestamp("2022-01-26 06:23:21+02:00")

train = df[df["Timestamp"] <= split_time].copy()
test = df[df["Timestamp"] > split_time].copy()

assert len(train) == 139_532, f"got {len(train)}"
assert len(test) == 1_255_792, f"got {len(test)}"
assert train["Label"].sum() == 1_765, f"got {train['Label'].sum()}"
assert test["Label"].sum() == 19_187, f"got {test['Label'].sum()}"

# train is an unbroken 0-based prefix slice of df's own 0..N-1 RangeIndex
# (post reset_index(drop=True) above), so its index labels equal positional
# row order -- train_benign.index below gives positions into
# symbolic_train_df because of this invariant.
assert list(train.index) == list(range(len(train)))

# 4) Feature schema -- CSCAS_SCHEMA env var picks "base" (5 cols),
# "full_noscas" (40) or "full_scas" (41, SCAS kept -- deliberately-circular
# diagnostic). IsolationForest is tree-based, no sentinel imputation. Mined
# symbolic features added on top. See _cscas_schema.py.
SCHEMA = active_schema(SCHEMAS_ANOMALY)
FEATURE_COLS = cscas_feature_cols(df, schema=SCHEMA)
print(
    f"Base schema: {SCHEMA} -- {len(FEATURE_COLS)} feature columns (+ mined symbolic)"
)
print(FEATURE_COLS)

# 4c) Mining tree mode -- CSCAS_MINING_MODE env var picks "single_tree"
# (default, unchanged result files) or "two_tree" (add-on, "_twotree"-suffixed
# result files). See _cscas_schema.py.
MINING_MODE = active_mining_mode()
MODE_SUFFIX = MINING_MODE_SUFFIX[MINING_MODE]
print(f"Mining tree mode: {MINING_MODE}")

# 4d) Skip early (before the minutes-long mining pass) if both result files
# for this schema already exist.
STEM = f"cscas_mining_anomaly_iforest{MODE_SUFFIX}"
NEEDED = {ek: result_name(STEM, SCHEMA, ek) for ek in ("subsample", "fulltest")}
if grid_outputs_done([STEM], SCHEMA):
    print(f"[skip] {list(NEEDED.values())} already exist (CSCAS_FORCE=1 to re-run).")
    raise SystemExit(0)

# 5) Eval sets -- both cells of the test-set axis.
eval_df = get_cscas_eval_subsample(test)
EVAL_FRAMES = {"subsample": eval_df, "fulltest": test}
print(
    f"Evaluating on: subsample ({len(eval_df)} rows, {int(eval_df['Label'].sum())} pos)"
    f"  +  full test ({len(test)} rows, {int(test['Label'].sum())} pos)"
)

# 6) Build AlertGroups for train and eval -- same per-row parser
# cscas_mining.py uses. The full-test parse+encode is the slow step.
print("Building AlertGroups for train/eval splits...")
train_groups = rows_to_cscas_alert_groups(train.to_dict("records"))
eval_groups = {
    ek: rows_to_cscas_alert_groups(frame.to_dict("records"))
    for ek, frame in EVAL_FRAMES.items()
}
assert len(train_groups) == len(train), "row parsing dropped rows -- alignment broken"
for ek, groups in eval_groups.items():
    assert len(groups) == len(EVAL_FRAMES[ek]), f"row parsing dropped {ek} rows"

train_alert_groups_path = (
    CACHE_DIR
    / "cscas"
    / "groups"
    / "cscas_pregrouped_baseline_mining_anomaly"
    / "alert_groups"
    / "train_alert_groups.json"
)
train_alert_groups_path.parent.mkdir(parents=True, exist_ok=True)
save_alert_groups_json(train_groups, train_alert_groups_path)

# 7) Mine symbolic features on the FULL train split (both classes -- attack
# rows are needed to mine informative contrast predicates even though the
# model below only ever fits on the benign subset), excluding SCAS/
# Similarity-derived candidate fields -- same reasoning as cscas_mining.py.
# Same run_name/config/exclude_fields as cscas_mining_anomaly.py, so the
# mined predicates are cache-shared between the two scripts.
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
    run_name=f"cscas_baseline_mining_anomaly{MODE_SUFFIX}",
    config=mining_attribute_config(MINING_MODE),
    exclude_fields=LEAKY_ATTRIBUTE_FIELDS,
)
print(f"  Mined {len(mining_result.predicates)} predicates from train split.")

# 7b) Discard every attack-leaning mined pattern (in two_tree mode, this
# drops the whole attack-facing tree's leaves) before building the symbolic
# schema -- the anomaly detector's own training is benign-only, so it should
# never be handed attack-derived features either, even though mining itself
# needed both classes' labels to find them. See discard_attack_patterns.
mined_df, mined_predicates = discard_attack_patterns(
    mining_result.mined_df, mining_result.predicates
)

symbolic_schema = build_symbolic_feature_schema(
    df=mined_df,
    source_label="attack",
    schema_name="cscas_mining_anomaly_symbolic",
    schema_version="0.1.0",
    predicates=mined_predicates,
)
print(f"  Built {len(symbolic_schema.features)} symbolic features.")

encoder = SymbolicFeatureEncoder(feature_schema=symbolic_schema)
symbolic_train_df = encoder.transform(train_groups)
symbolic_eval_df = {ek: encoder.transform(groups) for ek, groups in eval_groups.items()}

# 8) Benign-only training data -- no pool conditions, no undersampling.
train_benign = train[train["Label"] == 0]
print(f"Training on {len(train_benign)} benign-only rows (natural count)")


def _matrix(base_frame, symbolic_df):
    return pd.concat(
        [
            base_frame[FEATURE_COLS].reset_index(drop=True),
            symbolic_df.reset_index(drop=True),
        ],
        axis=1,
    ).values


# train_benign.index gives positions into symbolic_train_df (see step 3's
# invariant).
X_train = _matrix(train_benign, symbolic_train_df.iloc[train_benign.index])

# 9) Fit + score -- 5 seeds, IsolationForest(random_state=seed), identical
# (base + mined) benign training rows every seed. Each fitted model is scored
# on both eval sets; the tuned-operating-point view is collected per seed and
# seed-averaged before saving.
for ek, frame in EVAL_FRAMES.items():
    X_ev = _matrix(frame, symbolic_eval_df[ek])
    y_ev = frame["Label"].values

    seed_metrics: list[dict[str, float]] = []
    seed_workloads: list[dict] = []
    for seed in range(5):
        model = IsolationForest(
            n_estimators=100, contamination=0.05, random_state=seed, n_jobs=-1
        )
        model.fit(X_train)

        scores = -model.decision_function(X_ev)  # higher = more anomalous
        y_pred = (model.predict(X_ev) == -1).astype(int)  # 1 = anomaly = attack

        m = {
            "auc": roc_auc_score(y_ev, scores),
            "precision": precision_score(y_ev, y_pred, zero_division=0),
            "recall": recall_score(y_ev, y_pred, zero_division=0),
            "f1": f1_score(y_ev, y_pred, zero_division=0),
        }
        seed_metrics.append(m)
        seed_workloads.append(compute_workload_at_recall(y_ev, scores))
        print(
            f"  [{ek}] seed={seed}: AUC={m['auc']:.3f} P={m['precision']:.3f} "
            f"R={m['recall']:.3f} F1={m['f1']:.3f}"
        )

    workload = average_workload_at_recall(seed_workloads)
    avg = pd.DataFrame(seed_metrics).mean()
    print(f"\n=== cscas_mining_anomaly_iforest [{SCHEMA} / {ek}] (mean of 5 seeds) ===")
    print(
        f"AUC={avg.auc:.3f} P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}  (default cut)"
    )
    if workload.get("0.90"):
        w = workload["0.90"]
        print(
            f"  @recall>=0.90: P={w['precision']:.3f} FP={w['fp']:.0f} "
            f"workload_reduction={w['workload_reduction']:.3f}"
        )

    save_anomaly_results(
        name=NEEDED[ek],
        description=(
            "IsolationForest(n_estimators=100, contamination=0.05) fit on "
            f"benign-only rows of the {schema_blurb(SCHEMA, len(FEATURE_COLS))} + "
            f"attribute-mined symbolic features ({MINING_MODE} mode, mined on "
            "the same train split as cscas_mining, attack-leaning mined "
            "patterns discarded before schema-building; SCAS/Similarity-derived "
            "fields excluded from mining), evaluated on the "
            f"{'shared 20k eval subsample' if ek == 'subsample' else 'full 1.26M-row test set'}. "
            "No attack rows used in training. Mean over 5 seeds "
            "(random_state=0..4). precision/recall/f1 at the default "
            "contamination=0.05 cut; workload_at_recall is the tuned-threshold "
            "view (seed-averaged)."
        ),
        seeds=seed_metrics,
        workload=workload,
    )

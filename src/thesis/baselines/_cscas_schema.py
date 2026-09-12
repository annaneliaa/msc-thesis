"""
Shared feature-schema + eval-set selection for the CSCAS baseline grid.

Two axes every non-text CSCAS baseline now sweeps, so each model can be
placed in a full grid (feature schema x test set):

  - schema: which CSV columns become features. See SCHEMAS below.
  - eval:   "subsample" -- the shared, frozen 20k stratified subsample every
            non-replication baseline scores on (see
            _sampling.get_cscas_eval_subsample). "fulltest" -- all 1.26M
            test rows, the protocol cscas.py (the replication anchor) uses.

The schema is chosen per-process via the CSCAS_SCHEMA env var so
run_overnight.sh can invoke each script once per schema without an
in-script schema loop; the eval axis is swept inside each script (both
result files written per run).

Result files are named  <stem><SCHEMA_SUFFIX><EVAL_SUFFIX>.json  so the
comparison notebook can address any cell of the grid by suffix, e.g.:

    cscas_logreg                        base   / subsample
    cscas_logreg_fulltest               base   / full test
    cscas_logreg_fullfeat               full   / subsample
    cscas_logreg_fullfeat_fulltest      full   / full test
    cscas_anomaly_ocsvm_fullfeat_scas   full+SCAS / subsample   (anomaly only)

The RF replication pair (cscas.py / cscas_base.py) keeps its legacy names
instead -- that grid is {cscas, cscas_subsample, cscas_base,
cscas_base_fulltest}.
"""

from __future__ import annotations

import os

import pandas as pd

# Columns never used as features by any CSCAS baseline: the split key, the
# raw signature string (its own text-model axis), the label, and the two
# anonymised per-connection IPs.
_PAPER_DROP = ["Timestamp", "SignatureText", "Label", "ExtIP", "IntIP"]

# SignatureID is a nominal identifier (not a signal) -- excluded from every
# schema below except the literal paper replication "full".
_NOMINAL_ID = "SignatureID"
# SCAS is CSCAS's own offline outlier/inlier cluster flag.
_OUTLIER_FLAG = "SCAS"

#: schema -> (human label, result-name suffix)
_SCHEMA_META = {
    # 5 numeric columns: SignatureMatchesPerDay, AlertCount, Proto, ExtPort,
    # IntPort. SignatureID, SCAS and every *Similarity column dropped -- the
    # deployment-realistic floor (see cscas_base.py's docstring).
    "base": ("reduced base schema", ""),
    # The CSCAS paper's own 42 raw columns (only _PAPER_DROP removed):
    # 5 base + SignatureID + SCAS + Similarity + SignatureIDSimilarity + 33
    # per-attribute *Similarity columns. What cscas.py replicates.
    "full": ("full paper schema (42 cols, incl. SignatureID + SCAS)", "_fullfeat"),
    # full minus SignatureID and SCAS -> 40 cols. The "full" schema for
    # anomaly detectors: handing a one-class detector CSCAS's own precomputed
    # inlier/outlier verdict (SCAS) is circular, so it is dropped; the 35
    # *Similarity score columns (similarity signal, not an outlier verdict)
    # are kept.
    "full_noscas": ("full schema minus SignatureID + SCAS (40 cols)", "_fullfeat"),
    # full minus SignatureID only -> 41 cols. full_noscas + SCAS: the
    # deliberately-circular anomaly variant, for measuring exactly what the
    # precomputed outlier flag buys the detector (one-column ablation vs
    # full_noscas).
    "full_scas": (
        "full schema minus SignatureID, with SCAS (41 cols)",
        "_fullfeat_scas",
    ),
}

SCHEMAS = tuple(_SCHEMA_META)
EVAL_SETS = ("subsample", "fulltest")

EVAL_SUFFIX = {"subsample": "", "fulltest": "_fulltest"}

#: what each script type sweeps
SCHEMAS_CLASSIFIER = ("base", "full")
SCHEMAS_ANOMALY = ("base", "full_noscas", "full_scas")


#: attribute-mining tree config for the three mining baselines
#: (cscas_mining.py, cscas_mining_anomaly.py, cscas_mining_anomaly_iforest.py).
#: "single_tree" (default) is the original config every existing mining
#: result was produced with -- max_depth_attack unset, one shared tree.
#: "two_tree" is an add-on, not a replacement: fits a second, deeper tree
#: specifically for attack-leaning leaves alongside the shallow benign-facing
#: one. Selected the same way as CSCAS_SCHEMA (env var, additive result-name
#: suffix), so single-tree result files are untouched. All three scripts use
#: the identical config (mining_attribute_config below) -- the anomaly
#: scripts' "no important-class information in the model's training" principle
#: is enforced downstream instead, by discarding attack-leaning mined patterns
#: before they reach the symbolic feature schema; see
#: discard_attack_patterns in cscas_mining_anomaly.py /
#: cscas_mining_anomaly_iforest.py.
MINING_MODES = ("single_tree", "two_tree")
MINING_MODE_SUFFIX = {"single_tree": "", "two_tree": "_twotree"}


def active_mining_mode() -> str:
    """Which attribute-mining tree config this process should use, from
    CSCAS_MINING_MODE. Defaults to "single_tree" so a bare `python
    cscas_mining.py` reproduces the existing result files unchanged."""
    mode = os.environ.get("CSCAS_MINING_MODE", "single_tree").strip().lower()
    if mode not in MINING_MODES:
        raise ValueError(
            f"CSCAS_MINING_MODE={mode!r} not valid; expected one of {MINING_MODES}"
        )
    return mode


def mining_attribute_config(mode: str):
    """AttributeMiningConfig for the given mining mode -- shared by all three
    mining baselines (cscas_mining.py, cscas_mining_anomaly.py,
    cscas_mining_anomaly_iforest.py). The anomaly scripts additionally call
    discard_attack_patterns() after mining, before building the symbolic
    schema, so their model never trains on attack-derived features -- mining
    itself is identical across all three.

    "two_tree" is the gr3_md1_mda4 point from
    configs/screening_mining_settings.yaml (the project's own two-tree
    feasible grid, from attribute_mining_sweep_eda.ipynb section 5.3/6.3):
    max_depth=1 for the benign-facing tree (far more stable across windows
    than max_depth=2 per that notebook's addendum -- recall_benign mean 0.99,
    std 0.005 vs. mean 0.546, std 0.123), max_depth_attack=4 (matches
    single_tree's own max_depth exactly, so the attack-facing tree isn't
    additionally shallower than the baseline it's being compared against),
    min_samples_leaf=10 (that grid's own anchor; single_tree uses 20).
    Contrast-stage thresholds and class_weight are left at
    AttributeMiningConfig's defaults, identical between both modes."""
    from thesis.schemas.mining import AttributeMiningConfig, DecisionTreeRuleConfig

    if mode == "single_tree":
        return AttributeMiningConfig()
    if mode == "two_tree":
        return AttributeMiningConfig(
            tree=DecisionTreeRuleConfig(
                max_depth=1, max_depth_attack=4, min_samples_leaf=10
            )
        )
    raise ValueError(f"unknown mining mode {mode!r}; expected one of {MINING_MODES}")


def discard_attack_patterns(mined_df, predicates):
    """Drop every attack-leaning row from a mining job's mined_df (Step 1
    contrast-set survivors + Step 2 decision-tree leaf rules, concatenated,
    each already tagged with a "source_label" of "attack" or "benign"), and
    the predicates that only those rows referenced -- for
    cscas_mining_anomaly.py / cscas_mining_anomaly_iforest.py, called right
    after run_alert_group_attribute_mining_job and before
    build_symbolic_feature_schema.

    Mining itself (both the contrast-set stage and the decision tree's own
    fit, single-tree or two-tree) needs both classes' labels to run at all --
    that's unavoidable and unrelated to what reaches the anomaly detector
    afterwards. This is the actual enforcement point for "the anomaly model
    only ever sees benign-only training data": every attack-leaning mined
    pattern -- including, in two_tree mode, the entire attack-facing tree's
    leaves -- is discarded here, before a single symbolic feature is built
    from it, so the resulting (base + mined) feature matrix the OneClassSVM/
    IsolationForest fits on carries no attack-derived information at all.

    Predicate tokens are shared between Step 1's categorical-predicate
    itemsets and Step 2's AttributePredicate.token (both keyed off the same
    build_categorical_predicate_matrix column names), so filtering
    `predicates` to only tokens still present in a kept (benign) row's
    itemset is a safe, class-collision-free way to prune the alphabet down
    to what the surviving features actually use.

    Returns (benign_mined_df, benign_predicates)."""
    is_benign = mined_df["source_label"] == "benign"
    n_dropped = int((~is_benign).sum())
    benign_df = mined_df[is_benign].reset_index(drop=True)
    kept_tokens = {tok for itemset in benign_df["itemset"] for tok in itemset}
    benign_predicates = [p for p in predicates if p.token in kept_tokens]
    print(
        f"  Discarded {n_dropped} attack-leaning mined pattern(s) before "
        f"building the anomaly detector's symbolic schema; "
        f"{len(benign_df)} benign pattern(s) / {len(benign_predicates)} "
        "predicate(s) remain."
    )
    return benign_df, benign_predicates


def active_schema(allowed: tuple[str, ...] = SCHEMAS) -> str:
    """The schema this process should run, from CSCAS_SCHEMA.

    Defaults to the first entry of `allowed` (so a bare `python
    cscas_logreg.py` still runs the base schema, and a bare anomaly script
    also starts at base).
    """
    default = allowed[0]
    schema = os.environ.get("CSCAS_SCHEMA", default).strip().lower()
    if schema not in allowed:
        raise ValueError(
            f"CSCAS_SCHEMA={schema!r} not valid for this script; "
            f"expected one of {allowed}"
        )
    return schema


def cscas_feature_cols(df: pd.DataFrame, *, schema: str) -> list[str]:
    """Feature-column list for one schema (see _SCHEMA_META for definitions)."""
    if schema not in _SCHEMA_META:
        raise ValueError(f"unknown schema {schema!r}; expected one of {SCHEMAS}")

    drop = list(_PAPER_DROP)
    if schema == "base":
        drop += [_NOMINAL_ID, _OUTLIER_FLAG]
        return [c for c in df.columns if c not in drop and not c.endswith("Similarity")]
    if schema == "full":
        return [c for c in df.columns if c not in drop]
    if schema == "full_noscas":
        drop += [_NOMINAL_ID, _OUTLIER_FLAG]
        return [c for c in df.columns if c not in drop]
    if schema == "full_scas":
        drop += [_NOMINAL_ID]
        return [c for c in df.columns if c not in drop]
    raise AssertionError("unreachable")


def result_name(stem: str, schema: str, eval_set: str) -> str:
    """<stem><schema suffix><eval suffix>. Both suffixes empty for
    base/subsample, so legacy file names are unchanged."""
    if schema not in _SCHEMA_META:
        raise ValueError(f"unknown schema {schema!r}")
    if eval_set not in EVAL_SUFFIX:
        raise ValueError(f"unknown eval_set {eval_set!r}")
    return f"{stem}{_SCHEMA_META[schema][1]}{EVAL_SUFFIX[eval_set]}"


def schema_blurb(schema: str, n_features: int) -> str:
    """Short human description of the active schema, for result `description`."""
    label = _SCHEMA_META[schema][0]
    return f"{label} ({n_features} features)"


def force_recompute() -> bool:
    """True if CSCAS_FORCE=1 -- recompute even when result files already exist."""
    return os.environ.get("CSCAS_FORCE", "0") == "1"


def grid_outputs_done(
    stems: list[str], schema: str, eval_sets: tuple[str, ...] = EVAL_SETS
) -> bool:
    """True if every <stem> x <eval_set> result JSON for this schema already
    exists on disk (and CSCAS_FORCE is not set) -- lets a script exit before
    any expensive work on a resumed overnight run."""
    if force_recompute():
        return False
    from thesis.baselines._results import results_exist

    return all(
        results_exist(result_name(stem, schema, ek))
        for stem in stems
        for ek in eval_sets
    )


# CSCAS's "not applicable" sentinel: ExtPort is -1 for a protocol with no
# port; a *Similarity column is -1 for an attribute that protocol never
# populates. Tree models (RF, XGBoost, IsolationForest) handle it as-is -- a
# split just treats -1 as a low value. Scale-sensitive models (LogReg,
# OneClassSVM) prepend sentinel_imputer() to their sklearn Pipeline so -1
# doesn't skew StandardScaler's fitted mean/std.
SENTINEL_VALUE = -1


def sentinel_imputer():
    """First step for the scale-sensitive baselines' Pipelines: replace the
    -1 sentinel with the per-column, train-fitted median of the applicable
    values, before scaling. Columns with no -1 (e.g. mined binary symbolic
    features) pass through untouched.

    keep_empty_features=True: on the full schema, several *Similarity columns
    are -1 for *every* row of a given training pool (no benign SMTP/email
    alerts in the CSCAS train split, undersampled pools miss rare protocols
    entirely). Without this flag SimpleImputer silently *drops* those
    columns, making the fitted feature count depend on which protocols
    happened to land in the pool -- unstable across seeds. With it, an
    all-sentinel column is kept and filled with 0 (a constant column the
    downstream StandardScaler then neutralises)."""
    from sklearn.impute import SimpleImputer

    return SimpleImputer(
        strategy="median", missing_values=SENTINEL_VALUE, keep_empty_features=True
    )

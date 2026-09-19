"""
Attribute-mining tree-mode config, shared identically by every mining
baseline across both datasets: cscas_mining.py / cscas_mining_anomaly.py /
cscas_mining_anomaly_iforest.py (CSCAS) and ait_ads_mining.py /
ait_ads_mining_anomaly.py / ait_ads_mining_anomaly_iforest.py (AIT-ADS).
Extracted out of _cscas_schema.py (which still re-exports everything here
for its own three scripts, unchanged) so the two datasets' mining baselines
run through one definition of "single_tree" and "two_tree" instead of two
copies that could drift apart.

Two modes:
  - "single_tree" (default): AttributeMiningConfig()'s plain defaults
    (max_depth=4, no attack-facing tree, min_samples_leaf=20) -- the
    original config every existing *_mining*.json result was produced with.
  - "two_tree": an add-on, not a replacement -- writes to separate
    "_twotree"-suffixed result files (MINING_MODE_SUFFIX) so single_tree's
    results are never overwritten. mining_attribute_config() (classifier
    scripts) and mining_attribute_config_anomaly() (the two one-class
    scripts per dataset, which always call discard_attack_patterns()
    afterward) use different depths for it -- see each function's own
    docstring for why.

Selected per-process via an env var, one per dataset so a two-tree rerun on
one doesn't accidentally flip the other: CSCAS_MINING_MODE for the CSCAS
scripts, AIT_ADS_MINING_MODE for the AIT-ADS ones. Both default to
"single_tree", so a bare invocation of any of the six scripts reproduces
its existing result files unchanged.
"""

from __future__ import annotations

import os

MINING_MODES = ("single_tree", "two_tree")
MINING_MODE_SUFFIX = {"single_tree": "", "two_tree": "_twotree"}


def active_mining_mode(env_var: str) -> str:
    """Which attribute-mining tree config this process should use, from
    `env_var` (e.g. "CSCAS_MINING_MODE" or "AIT_ADS_MINING_MODE"). Defaults
    to "single_tree" so a bare invocation reproduces existing result files
    unchanged."""
    mode = os.environ.get(env_var, "single_tree").strip().lower()
    if mode not in MINING_MODES:
        raise ValueError(
            f"{env_var}={mode!r} not valid; expected one of {MINING_MODES}"
        )
    return mode


def mining_attribute_config(mode: str):
    """AttributeMiningConfig for the given mining mode -- shared by every
    classifier mining baseline (cscas_mining.py, ait_ads_mining.py). The
    anomaly-detector siblings (mining_attribute_config_anomaly below) always
    call discard_attack_patterns() after mining, before building the
    symbolic schema, so their model never trains on attack-derived features
    -- mining itself is identical between the classifier and anomaly paths
    within a dataset.

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
    AttributeMiningConfig's defaults, identical between both modes and both
    datasets (min_growth_rate=3.0, min_attack_coverage=min_benign_coverage=
    0.05).

    max_diverse_rounds=2 (round_min_coverage/round_min_growth_rate left at
    DecisionTreeRuleConfig's defaults, 0.05/3.0, matching Step 1's own
    ContrastSetFilterConfig defaults): after the depth=1 benign tree and the
    depth=4 attack tree above are fit, each is refit once more with its
    already-used feature column(s) excluded, on the same population, to
    surface a second independent characterization instead of just the
    dominant one (see decision_tree_rule_mining._fit_class_rounds). Swept
    max_diverse_rounds in {1, 2, 3} on CSCAS's guided-pool condition, 5-seed
    mean F1 on the eval subsample:
        rounds=1: rf base 0.7147, xgb base 0.7269, rf full 0.8966, xgb full 0.9264
        rounds=2: rf base 0.7936, xgb base 0.7594, rf full 0.9068, xgb full 0.9324
        rounds=3: rf base 0.7890, xgb base 0.7447, rf full 0.9015, xgb full 0.9280
    rounds=2 beats both rounds=1 and rounds=3 on all four cells -- rounds=3's
    added rule (proto=6-based) is individually significant but redundant in
    signal with rounds=2's winning rule (a NOT_proto=17-based rule; proto=6
    is a large chunk of "not proto=17"), so it adds correlated dimensionality
    rather than new information. Not swept beyond 3. Not independently
    re-validated on AIT-ADS -- applied there identically on the "exactly the
    same as CSCAS" premise, not because AIT-ADS's own rounds sweep agreed."""
    from thesis.schemas.mining import AttributeMiningConfig, DecisionTreeRuleConfig

    if mode == "single_tree":
        return AttributeMiningConfig()
    if mode == "two_tree":
        return AttributeMiningConfig(
            tree=DecisionTreeRuleConfig(
                max_depth=1,
                max_depth_attack=4,
                min_samples_leaf=10,
                max_diverse_rounds=2,
            )
        )
    raise ValueError(f"unknown mining mode {mode!r}; expected one of {MINING_MODES}")


def mining_attribute_config_anomaly(mode: str):
    """AttributeMiningConfig for the one-class anomaly mining scripts
    specifically (cscas_mining_anomaly.py / cscas_mining_anomaly_iforest.py /
    ait_ads_mining_anomaly.py / ait_ads_mining_anomaly_iforest.py) -- these
    always call discard_attack_patterns() on the mining output afterward
    (single_tree mode included, unlike an earlier version of this function),
    so the tree depth chosen here is really "how rich should the SURVIVING
    benign-only feature set be", not "match the classifier's two-tree
    config" (mining always needs both classes' labels to run at all
    regardless of depth -- that part is shared and unavoidable, same as for
    the classifier scripts).

    "single_tree": AttributeMiningConfig() defaults (max_depth=4,
    min_samples_leaf=20) -- kept for symmetry with mining_attribute_config,
    but note this is no longer literally reproducing the long-standing
    single_tree.json files for CSCAS, since those predate
    discard_attack_patterns existing at all (they were generated by code
    that never discarded anything). Re-running single_tree today will
    legitimately produce different, smaller (benign-only) results than
    what's on disk. AIT-ADS's anomaly-mining scripts never had a discard
    step before this addition either, so the same caveat applies there too.

    "two_tree": max_depth=4, min_samples_leaf=10, max_depth_attack left
    unset (no attack-facing tree fit at all -- its output would be discarded
    by discard_attack_patterns anyway, so fitting it is wasted compute).
    NOT mining_attribute_config's own two_tree point (max_depth=1) -- that
    was empirically tested and rejected on CSCAS. A depth sweep on CSCAS's
    Base(5) schema (OneClassSVM/IsolationForest AUC on the eval subsample,
    mining + discard_attack_patterns applied identically at each depth)
    gave:
        depth=1: OCSVM 0.706, IForest 0.967  (1 non-constant benign feature)
        depth=2: OCSVM 0.699, IForest 0.944  (dominated by depth=1, worse on both)
        depth=3: OCSVM 0.865, IForest 0.942
        depth=4: OCSVM 0.915, IForest 0.932  (IForest still >= the original
                 un-discarded single-tree baseline's 0.928)
    depth=1's benign-facing tree can express at most one split, so
    discard_attack_patterns leaves OneClassSVM's StandardScaler+RBF pipeline
    with essentially one informative dimension -- catastrophic for a
    distance-based kernel even though tree-based IsolationForest barely
    notices. depth=4 recovers most of OneClassSVM's performance at a small
    IsolationForest cost, so it's the better aggregate choice despite not
    being IsolationForest's own individual optimum (depth=1). Not
    independently re-swept on AIT-ADS -- applied there identically on the
    "exactly the same as CSCAS" premise."""
    from thesis.schemas.mining import AttributeMiningConfig, DecisionTreeRuleConfig

    if mode == "single_tree":
        return AttributeMiningConfig()
    if mode == "two_tree":
        return AttributeMiningConfig(
            tree=DecisionTreeRuleConfig(max_depth=4, min_samples_leaf=10)
        )
    raise ValueError(f"unknown mining mode {mode!r}; expected one of {MINING_MODES}")


def discard_attack_patterns(mined_df, predicates):
    """Drop every attack-leaning row from a mining job's mined_df (Step 1
    contrast-set survivors + Step 2 decision-tree leaf rules, concatenated,
    each already tagged with a "source_label" of "attack" or "benign"), and
    the predicates that only those rows referenced -- called by every
    one-class anomaly mining script (CSCAS and AIT-ADS alike), right after
    run_alert_group_attribute_mining_job and before
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

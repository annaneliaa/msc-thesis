#!/usr/bin/env bash
# Overnight batch runner for the AIT-ADS mining baselines (ait_ads_mining.py,
# ait_ads_mining_anomaly.py -- OneClassSVM, ait_ads_mining_anomaly_iforest.py
# -- IsolationForest, 5-seeded) -- the AIT-ADS counterpart to
# baselines/run_overnight.sh's cscas_mining.py/cscas_mining_anomaly.py
# steps. Each script already loops every AIT-ADS scenario x grouping method
# internally (AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS below to run a
# subset of either), sharing the exact same split
# (_ait_ads_data.load_ait_ads_baseline_split_with_groups) that every other
# ait_ads_*.py script's plain split is built from -- these results are
# directly comparable to those, not just similarly configured.
#
# ait_ads_mining.py now fits all three tabular models per combo on the mined
# matrix -- ait_ads_mining_<run_tag>.json (RF), plus
# ait_ads_mining_logreg_<run_tag>.json / ait_ads_mining_xgboost_<run_tag>.json.
# It is per-model resumable: a combo already carrying all three model JSONs
# is skipped with no mining pass; one still missing a model re-mines once
# (the mining output was never cached) and fits only the missing models, so
# the existing RF results are neither recomputed nor overwritten. Running
# this script as-is is the intended way to backfill LogReg/XGBoost onto a
# tree that already has the RF mining results.
#
# ait_ads_mining.py has the same train-side single-class guard as
# ait_ads_rf.py (so still excludes harrison/santos/russellmitchell);
# ait_ads_mining_anomaly.py has the test-side-only guard ait_ads_anomaly.py
# uses instead, so it's the only mining-based script that also produces a
# result for those 3 scenarios -- see each script's own module docstring.
#
# MINING TREE MODE. Also sweeps AIT_ADS_MINING_MODE -- exactly the same two
# modes, same config values, as run_cscas_mining.sh's CSCAS_MINING_MODE
# sweep (see baselines/_mining_modes.py):
#   single_tree  (default) -- the original config every existing
#                ait_ads_mining*.json result was produced with; unchanged
#                filenames.
#   two_tree     -- add-on: a second, deeper tree for attack-leaning leaves
#                specifically (max_depth=1 / max_depth_attack=4 /
#                min_samples_leaf=10 for the classifier script; max_depth=4 /
#                min_samples_leaf=10, no attack tree, for the two anomaly
#                scripts, which also start discarding attack-leaning mined
#                patterns in this mode), written to separate
#                "_twotree"-suffixed result files so single_tree's results
#                are never overwritten. Roughly doubles this script's total
#                runtime (a full second sweep, not a cheap addition).
#
# Cheap-ish (CPU, one attribute-mining pass + RF/LogReg/XGBoost/OneClassSVM
# fit per combo, no GPU needed) but each mining pass costs more than a plain
# tabular fit -- budget more time than run_ait_ads_tabular.sh/
# run_ait_ads_anomaly.sh, not as much as run_ait_ads_bert_securebert.sh.
#
# alertbert grouping still needs the thesis-alertbert conda env (graph-tool
# -- see _ait_ads_grouping.py's module docstring), same split as
# run_ait_ads_tabular.sh. Needs sklearn + xgboost installed in
# thesis-alertbert too (usually already there from the tabular scripts'
# setup -- see setup_container.sh's thesis-alertbert branch).
#
# Does not abort on a single script's failure (no `set -e`) -- if one
# script errors out, its exit code is logged and the run continues to the
# next step (each script's own run_scenario() already skips single-class/
# leakage combos gracefully on its own).
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_ait_ads_mining.sh > /dev/null 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/src/thesis/baselines"
cd "$BASELINES_DIR"

PYTHON="python3"
LOG_DIR="$BASELINES_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ait_ads_mining_$(date +%Y%m%d_%H%M%S).log"

export AIT_ADS_SCENARIOS="${AIT_ADS_SCENARIOS:-}"  # empty = every AIT-ADS scenario
NON_ALERTBERT_METHODS="${AIT_ADS_GROUPING_METHODS:-fixed_window,time_delta,cscas_grouping,deepcase}"
NON_ALERTBERT_METHODS="${NON_ALERTBERT_METHODS//alertbert/}"
NON_ALERTBERT_METHODS="${NON_ALERTBERT_METHODS//,,/,}"

MINING_MODES=(single_tree two_tree)

run_step() {
    local label="$1"
    shift
    echo ""
    echo "--- $label started at $(date) ---"
    local start end status
    start=$(date +%s)
    "$@"
    status=$?
    end=$(date +%s)
    echo "--- $label finished at $(date) (exit=$status, $((end - start))s) ---"
    return 0  # never abort the batch on a single step's failure
}

{
    echo "=== AIT-ADS mining baseline run started at $(date) ==="
    echo "AIT_ADS_SCENARIOS=${AIT_ADS_SCENARIOS:-<all>}"
    echo "Non-alertbert grouping methods (plain venv): $NON_ALERTBERT_METHODS"
    echo "Mining tree modes: ${MINING_MODES[*]}"

    for mode in "${MINING_MODES[@]}"; do
        export AIT_ADS_MINING_MODE="$mode"
        for script in ait_ads_mining.py ait_ads_mining_anomaly.py ait_ads_mining_anomaly_iforest.py; do
            AIT_ADS_GROUPING_METHODS="$NON_ALERTBERT_METHODS" \
                run_step "$script (AIT_ADS_MINING_MODE=$mode, fixed_window/time_delta/cscas_grouping/deepcase)" "$PYTHON" "$script"
            run_step "$script (AIT_ADS_MINING_MODE=$mode, alertbert)" \
                env AIT_ADS_MINING_MODE="$mode" AIT_ADS_GROUPING_METHODS=alertbert conda run -n thesis-alertbert --no-capture-output \
                "$PYTHON" "$script"
        done
    done

    echo ""
    echo "=== AIT-ADS mining baseline run finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

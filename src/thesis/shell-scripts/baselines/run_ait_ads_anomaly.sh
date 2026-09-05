#!/usr/bin/env bash
# Overnight batch runner for the AIT-ADS anomaly baselines
# (ait_ads_anomaly.py -- OneClassSVM, ait_ads_anomaly_iforest.py --
# IsolationForest) -- the AIT-ADS counterpart to run_ait_ads_tabular.sh's
# multi-script loop pattern. Each script loops every AIT-ADS scenario x
# grouping method internally (AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS
# below to run a subset of either), sharing the exact same split
# (_ait_ads_data.load_ait_ads_baseline_split) as every other ait_ads_*.py
# script -- these results are directly comparable to those, not just
# similarly configured.
#
# Unlike the six classifier/LLM baselines, both of these are expected to
# also produce a result for harrison/santos/russellmitchell -- see
# ait_ads_anomaly.py's module docstring for why their single-class guard is
# test-side only, not train-side.
#
# alertbert grouping still needs the thesis-alertbert conda env (graph-tool
# -- see _ait_ads_grouping.py's module docstring), same split as
# run_ait_ads_tabular.sh. Needs sklearn installed in thesis-alertbert too
# (usually already there from the tabular scripts' setup -- see
# setup_container.sh's thesis-alertbert branch).
#
# Cheap (CPU, no pool conditions to sweep -- OneClassSVM is one
# deterministic fit per combo; IsolationForest is 5 fits per combo, one per
# seed, still fast) -- comparable cost to run_ait_ads_tabular.sh, not
# run_ait_ads_bert_securebert.sh.
#
# Does not abort on a single script's failure (no `set -e`) -- if the
# script errors out, its exit code is logged and the run continues to the
# next step (ait_ads_anomaly.py's own run_scenario() already skips
# single-class-test/leakage combos gracefully on its own).
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_ait_ads_anomaly.sh > /dev/null 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/src/thesis/baselines"
cd "$BASELINES_DIR"

PYTHON="python3"
LOG_DIR="$BASELINES_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ait_ads_anomaly_$(date +%Y%m%d_%H%M%S).log"

export AIT_ADS_SCENARIOS="${AIT_ADS_SCENARIOS:-}"  # empty = every AIT-ADS scenario
NON_ALERTBERT_METHODS="${AIT_ADS_GROUPING_METHODS:-fixed_window,time_delta,cscas_grouping,deepcase}"
NON_ALERTBERT_METHODS="${NON_ALERTBERT_METHODS//alertbert/}"
NON_ALERTBERT_METHODS="${NON_ALERTBERT_METHODS//,,/,}"

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
    echo "=== AIT-ADS anomaly baseline run started at $(date) ==="
    echo "AIT_ADS_SCENARIOS=${AIT_ADS_SCENARIOS:-<all>}"
    echo "Non-alertbert grouping methods (plain venv): $NON_ALERTBERT_METHODS"

    for script in ait_ads_anomaly.py ait_ads_anomaly_iforest.py; do
        AIT_ADS_GROUPING_METHODS="$NON_ALERTBERT_METHODS" \
            run_step "$script (fixed_window/time_delta/cscas_grouping/deepcase)" "$PYTHON" "$script"
        run_step "$script (alertbert)" \
            env AIT_ADS_GROUPING_METHODS=alertbert conda run -n thesis-alertbert --no-capture-output \
            "$PYTHON" "$script"
    done

    echo ""
    echo "=== AIT-ADS anomaly baseline run finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

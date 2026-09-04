#!/usr/bin/env bash
# Overnight batch runner for the AIT-ADS mining baselines (ait_ads_mining.py,
# ait_ads_mining_anomaly.py) -- the AIT-ADS counterpart to
# baselines/run_overnight.sh's cscas_mining.py/cscas_mining_anomaly.py
# steps. Each script already loops every AIT-ADS scenario x grouping method
# internally (AIT_ADS_SCENARIOS / AIT_ADS_GROUPING_METHODS below to run a
# subset of either), sharing the exact same split
# (_ait_ads_data.load_ait_ads_baseline_split_with_groups) that every other
# ait_ads_*.py script's plain split is built from -- these results are
# directly comparable to those, not just similarly configured.
#
# ait_ads_mining.py has the same train-side single-class guard as
# ait_ads_rf.py (so still excludes harrison/santos/russellmitchell);
# ait_ads_mining_anomaly.py has the test-side-only guard ait_ads_anomaly.py
# uses instead, so it's the only mining-based script that also produces a
# result for those 3 scenarios -- see each script's own module docstring.
#
# Cheap-ish (CPU, one attribute-mining pass + RF/OneClassSVM fit per combo,
# no GPU needed) but each mining pass costs more than a plain tabular fit --
# budget more time than run_ait_ads_tabular.sh/run_ait_ads_anomaly.sh, not
# as much as run_ait_ads_bert_securebert.sh.
#
# alertbert grouping still needs the thesis-alertbert conda env (graph-tool
# -- see _ait_ads_grouping.py's module docstring), same split as
# run_ait_ads_tabular.sh. Needs sklearn installed in thesis-alertbert too
# (usually already there from the tabular scripts' setup -- see
# setup_container.sh's thesis-alertbert branch).
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

    for script in ait_ads_mining.py ait_ads_mining_anomaly.py; do
        AIT_ADS_GROUPING_METHODS="$NON_ALERTBERT_METHODS" \
            run_step "$script (fixed_window/time_delta/cscas_grouping/deepcase)" "$PYTHON" "$script"
        run_step "$script (alertbert)" \
            env AIT_ADS_GROUPING_METHODS=alertbert conda run -n thesis-alertbert --no-capture-output \
            "$PYTHON" "$script"
    done

    echo ""
    echo "=== AIT-ADS mining baseline run finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

#!/usr/bin/env bash
# Overnight batch runner for the AIT-ADS tabular baselines (ait_ads_rf.py,
# ait_ads_logreg.py, ait_ads_xgboost.py) -- the AIT-ADS counterpart to
# baselines/run_overnight.sh's tabular steps. Each script already loops
# every AIT-ADS scenario x grouping method internally (AIT_ADS_SCENARIOS /
# AIT_ADS_GROUPING_METHODS below to run a subset of either), sharing the
# exact same split (_ait_ads_data.load_ait_ads_baseline_split) that
# ait_ads_bert.py/ait_ads_securebert.py/ait_ads_zeroshot.py use -- these
# results are directly comparable to those, not just similarly configured.
#
# These are cheap (CPU, seconds per seed) compared to run_ait_ads_bert_
# securebert.sh's fine-tuning, but alertbert grouping still needs the
# thesis-alertbert conda env (graph-tool -- see _ait_ads_grouping.py's
# module docstring) regardless of which downstream model is being trained,
# so this script still runs the alertbert grouping method separately via
# `conda run`, same reasoning as run_ait_ads_bert_securebert.sh's own
# split -- see that script's header for the fuller explanation. Needs
# sklearn/xgboost/pandas installed in thesis-alertbert too (usually already
# there, but not guaranteed -- see setup_container.sh's thesis-alertbert
# branch).
#
# Does not abort on a single script's failure (no `set -e`) -- if one
# script errors out (e.g. a single-class split or a leakage scenario,
# both already skipped gracefully by each script's own run_scenario()),
# its exit code is logged and the run continues to the next step.
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_ait_ads_tabular.sh > /dev/null 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/src/thesis/baselines"
cd "$BASELINES_DIR"

PYTHON="python3"
LOG_DIR="$BASELINES_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ait_ads_tabular_$(date +%Y%m%d_%H%M%S).log"

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
    echo "=== AIT-ADS tabular baseline run started at $(date) ==="
    echo "AIT_ADS_SCENARIOS=${AIT_ADS_SCENARIOS:-<all>}"
    echo "Non-alertbert grouping methods (plain venv): $NON_ALERTBERT_METHODS"

    for script in ait_ads_rf.py ait_ads_logreg.py ait_ads_xgboost.py; do
        AIT_ADS_GROUPING_METHODS="$NON_ALERTBERT_METHODS" \
            run_step "$script (fixed_window/time_delta/cscas_grouping/deepcase)" "$PYTHON" "$script"
        run_step "$script (alertbert)" \
            env AIT_ADS_GROUPING_METHODS=alertbert conda run -n thesis-alertbert --no-capture-output \
            "$PYTHON" "$script"
    done

    echo ""
    echo "=== AIT-ADS tabular baseline run finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

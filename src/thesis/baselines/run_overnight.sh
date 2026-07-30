#!/usr/bin/env bash
# Overnight batch runner: real (non-quick) sweeps for cscas_bert.py and
# cscas_securebert.py, followed by re-executing the comparison notebook so
# every plot is baked in and ready to look at in the morning.
#
# Does NOT touch cscas.py/cscas_base.py/cscas_logreg.py/cscas_xgboost.py --
# those already have real saved results. Does NOT run cscas_zeroshot.py --
# it needs a gated model + your own Hugging Face auth, left out of this
# batch deliberately (run it yourself separately once you've confirmed
# access is set up).
#
# class_weighted is capped at 15,000 rows (stratified, proportional -- see
# _sampling.class_weighted_pool) for both scripts, via
# CSCAS_CLASS_WEIGHTED_POOL_CAP, so both scripts' full N_SEEDS=3 x 3-
# condition sweeps fit comfortably in one overnight run -- uncapped, a
# single SecureBERT smoke test's class_weighted stage alone measured at
# 2+ hours remaining at only 16% progress for ONE seed.
#
# Does not abort on a single script's failure (no `set -e`) -- if one
# script errors out, its exit code is logged and the run continues to the
# next step, including the final notebook execution, so partial results
# still get visualized.
#
# Run:
#   cd src/thesis/baselines
#   nohup ./run_overnight.sh > /dev/null 2>&1 &
# (or just `./run_overnight.sh &` in a terminal you're about to close --
# nohup is the safer bet so a closed terminal/SSH session doesn't kill it)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="../../../venv/bin/python3"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_$(date +%Y%m%d_%H%M%S).log"

export CSCAS_QUICK_SANITY_CHECK=0
export CSCAS_CLASS_WEIGHTED_POOL_CAP=15000

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
    echo "=== Overnight baseline run started at $(date) ==="
    echo "CSCAS_QUICK_SANITY_CHECK=$CSCAS_QUICK_SANITY_CHECK CSCAS_CLASS_WEIGHTED_POOL_CAP=$CSCAS_CLASS_WEIGHTED_POOL_CAP"

    run_step "cscas_bert.py" "$PYTHON" cscas_bert.py
    run_step "cscas_securebert.py" "$PYTHON" cscas_securebert.py
    run_step "notebook execution" "$PYTHON" -m jupyter nbconvert \
        --to notebook --execute --inplace \
        ../notebooks/baselines/cscas_baseline_comparison.ipynb

    echo ""
    echo "=== Overnight baseline run finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

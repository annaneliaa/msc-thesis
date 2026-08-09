#!/usr/bin/env bash
# Overnight batch runner: every implemented baseline (cscas.py,
# cscas_base.py, cscas_mining.py, cscas_logreg.py, cscas_xgboost.py,
# cscas_bert.py, cscas_securebert.py), followed by re-executing the
# comparison notebook so every plot is baked in and ready to look at in the
# morning.
#
# Does NOT run cscas_zeroshot.py -- gated (needs a Llama 3.1 license
# acceptance + your own Hugging Face auth) and not yet run for real here.
# The notebook already tolerates its results file being absent (prints a
# "[skip] ... not found" note and skips the zero-shot plot), so leaving it
# out of this batch doesn't break the notebook step. Run cscas_zeroshot.py
# yourself separately once you've confirmed access is set up.
#
# The 4 tabular scripts (cscas/cscas_base/cscas_logreg/cscas_xgboost) are
# RF/LogReg/XGBoost -- CPU, seconds per seed regardless of condition count,
# so re-running all three conditions x 5 seeds every time is cheap; this
# always re-generates their saved results fresh rather than assuming
# they're already there.
#
# class_weighted is capped at 15,000 rows (stratified, proportional -- see
# _sampling.class_weighted_pool) for the two fine-tuned scripts, via
# CSCAS_CLASS_WEIGHTED_POOL_CAP, so their full N_SEEDS=5 x 3-condition
# sweeps (15 fine-tune runs each) fit in one overnight run -- uncapped, a
# single SecureBERT smoke test's class_weighted stage alone measured at
# 2+ hours remaining at only 16% progress for ONE seed.
#
# Does not abort on a single script's failure (no `set -e`) -- if one
# script errors out, its exit code is logged and the run continues to the
# next step, including the final notebook execution, so partial results
# still get visualized.
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_overnight.sh > /dev/null 2>&1 &
# (or just `./run_overnight.sh &` in a terminal you're about to close --
# nohup is the safer bet so a closed terminal/SSH session doesn't kill it)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/src/thesis/baselines"
cd "$BASELINES_DIR"

PYTHON="python3"
LOG_DIR="$BASELINES_DIR/logs"
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

    run_step "cscas.py" "$PYTHON" cscas.py
    run_step "cscas_base.py" "$PYTHON" cscas_base.py
    run_step "cscas_mining.py" "$PYTHON" cscas_mining.py
    run_step "cscas_logreg.py" "$PYTHON" cscas_logreg.py
    run_step "cscas_xgboost.py" "$PYTHON" cscas_xgboost.py
    run_step "cscas_bert.py" "$PYTHON" cscas_bert.py
    run_step "cscas_securebert.py" "$PYTHON" cscas_securebert.py
    run_step "notebook execution" "$PYTHON" -m jupyter nbconvert \
        --to notebook --execute --inplace \
        ../notebooks/baselines/cscas_baseline_comparison.ipynb

    echo ""
    echo "=== Overnight baseline run finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

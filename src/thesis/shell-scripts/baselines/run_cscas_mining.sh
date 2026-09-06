#!/usr/bin/env bash
# Dedicated runner for the CSCAS "internal system + mining" classifier
# baseline (cscas_mining.py), which now fits all three tabular models -- RF,
# LogReg, XGBoost -- on the same mined feature matrix and saves one result
# each: cscas_mining.json (RF, name unchanged), cscas_mining_logreg.json,
# cscas_mining_xgboost.json.
#
# Per-model resumable: a model whose results/*.json already exists is
# skipped, so running this after RF is already done only computes the two
# new ones. Set CSCAS_FORCE=1 to recompute all three. The attribute-mining
# pass itself still runs once per invocation (only its RF output was ever
# persisted) -- that's a single pass, minutes not hours.
#
# Separate from run_overnight.sh, which re-runs every CSCAS baseline
# including the multi-hour BERT/SecureBERT fine-tuning. cscas_mining.py is
# still listed there too (and is now resumable, so it won't redo an existing
# model), but if all you want is the mining-classifier models, run THIS.
#
# The anomaly-mining baselines (cscas_mining_anomaly.py /
# cscas_mining_anomaly_iforest.py) are unchanged by this and stay in
# run_overnight.sh.
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_cscas_mining.sh > /dev/null 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/src/thesis/baselines"
cd "$BASELINES_DIR"

PYTHON="python3"
LOG_DIR="$BASELINES_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cscas_mining_$(date +%Y%m%d_%H%M%S).log"

{
    echo "=== CSCAS mining-classifier baseline run started at $(date) ==="
    echo "CSCAS_FORCE=${CSCAS_FORCE:-0}"

    start=$(date +%s)
    "$PYTHON" cscas_mining.py
    status=$?
    end=$(date +%s)

    echo ""
    echo "=== CSCAS mining-classifier baseline run finished at $(date) (exit=$status, $((end - start))s) ==="
} 2>&1 | tee "$LOG_FILE"

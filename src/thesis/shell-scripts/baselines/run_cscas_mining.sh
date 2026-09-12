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
# MINING TREE MODE. Also sweeps CSCAS_MINING_MODE:
#   single_tree  (default) -- the original config every existing
#                *_mining*.json result was produced with; unchanged filenames.
#   two_tree     -- add-on: a second, deeper tree for attack-leaning leaves
#                specifically (max_depth=1 / max_depth_attack=4 /
#                min_samples_leaf=10 -- see
#                baselines/_cscas_schema.mining_attribute_config), written to
#                separate "_twotree"-suffixed result files
#                (cscas_mining_twotree, cscas_mining_logreg_twotree,
#                cscas_mining_xgboost_twotree, plus schema/eval suffixes) so
#                single_tree's results are never overwritten. This roughly
#                doubles the total runtime of this script (a full second
#                sweep, not a cheap addition).
#
# Separate from run_overnight.sh, which re-runs every CSCAS baseline
# including the multi-hour BERT/SecureBERT fine-tuning. cscas_mining.py is
# still listed there too (and is now resumable, so it won't redo an existing
# model/mode combo), but if all you want is the mining-classifier models,
# run THIS.
#
# The anomaly-mining baselines (cscas_mining_anomaly.py /
# cscas_mining_anomaly_iforest.py) also sweep CSCAS_MINING_MODE now, but stay
# in run_overnight.sh, unchanged by this script.
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

export CSCAS_FORCE="${CSCAS_FORCE:-0}"

MINING_MODES=(single_tree two_tree)

{
    echo "=== CSCAS mining-classifier baseline run started at $(date) ==="
    echo "CSCAS_FORCE=$CSCAS_FORCE"

    # Both feature schemas (base, full) x both mining tree modes
    # (single_tree, two_tree). Each run sweeps both eval sets internally
    # (subsample + full test) and is resumable -- cscas_mining.py exits
    # before the mining pass when all its result files for that (schema,
    # mode) combo already exist (CSCAS_FORCE=1 overrides).
    overall_start=$(date +%s)
    for mode in "${MINING_MODES[@]}"; do
        for schema in base full; do
            echo ""
            echo "--- CSCAS_MINING_MODE=$mode, CSCAS_SCHEMA=$schema started at $(date) ---"
            start=$(date +%s)
            env CSCAS_MINING_MODE="$mode" CSCAS_SCHEMA="$schema" "$PYTHON" cscas_mining.py
            status=$?
            end=$(date +%s)
            echo "--- CSCAS_MINING_MODE=$mode, CSCAS_SCHEMA=$schema finished (exit=$status, $((end - start))s) ---"
        done
    done
    overall_end=$(date +%s)

    echo ""
    echo "=== CSCAS mining-classifier baseline run finished at $(date) ($((overall_end - overall_start))s) ==="
} 2>&1 | tee "$LOG_FILE"

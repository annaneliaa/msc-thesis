#!/usr/bin/env bash
# Overnight batch runner for the full CSCAS baseline grid, followed by
# re-executing the comparison notebook so every plot is baked in and ready
# to look at in the morning.
#
# THE GRID. Two axes are swept for every non-text baseline:
#   * feature schema -- CSCAS_SCHEMA env var:
#       base         5 cols  (deployment-realistic floor)
#       full         42 cols (the CSCAS paper's own schema) -- classifiers
#       full_noscas  40 cols (full minus SignatureID + SCAS)  -- anomaly
#       full_scas    41 cols (full_noscas + SCAS, deliberately circular) -- anomaly
#   * test set -- swept *inside* each script: the shared 20k eval subsample
#     AND the full 1.26M-row test set, one result file each.
# Result files: <stem>[_fullfeat|_fullfeat_scas][_fulltest].json  (a bare
# stem == base schema / subsample, i.e. the legacy names are unchanged).
#
# The RF replication pair keeps its legacy layout: cscas.py emits
# {cscas, cscas_subsample} (42 cols), cscas_base.py emits
# {cscas_base, cscas_base_fulltest} (5 cols) -- no CSCAS_SCHEMA sweep, those
# two ARE the RF schema axis.
#
# Text models (cscas_bert.py / cscas_securebert.py) have no schema axis (the
# similarity scores are non-textual) -- they only add the full-test eval
# cell: {cscas_bert, cscas_bert_fulltest}. Does NOT run cscas_zeroshot.py
# (gated on Ollama; run run_zeroshot.sh separately). The notebook tolerates
# any missing result file.
#
# RESUMABLE. Every script exits early (before any fit / mining pass) when all
# of its result files for the active CSCAS_SCHEMA already exist, so re-running
# this after an interrupted night only computes what's missing. Set
# CSCAS_FORCE=1 to recompute everything regardless.
#
# MINING TREE MODE. The three attribute-mining scripts (cscas_mining.py,
# cscas_mining_anomaly.py, cscas_mining_anomaly_iforest.py) additionally sweep
# CSCAS_MINING_MODE:
#   single_tree  (default) -- the original config every existing
#                *_mining*.json result was produced with; unchanged filenames.
#   two_tree     -- add-on: a second, deeper tree fit for attack-leaning
#                leaves specifically (max_depth=1 / max_depth_attack=4 /
#                min_samples_leaf=10 -- see
#                baselines/_cscas_schema.mining_attribute_config), written to
#                separate "_twotree"-suffixed result files
#                (cscas_mining_twotree, cscas_mining_logreg_twotree,
#                cscas_mining_xgboost_twotree, cscas_mining_anomaly_ocsvm_twotree,
#                cscas_mining_anomaly_iforest_twotree, plus schema/eval
#                suffixes) so single_tree's results are never overwritten.
#
# COST. The base/full-test cells are free (extra .predict() on already-fitted
# models). The full-schema cells are quick refits (CPU-seconds/seed for the
# tabular + anomaly models). cscas_mining* run the attribute-mining pass +
# symbolic encoding of the full 1.26M-row test set once per (schema, mining
# mode) combo (minutes) -- two_tree roughly doubles the mining-script total
# since it's a second full sweep, not a cheap addition. cscas_bert /
# cscas_securebert add ~15-30 min of full-test inference each (no extra
# fine-tuning). class_weighted is capped at 15,000 rows for the two
# fine-tuned scripts via CSCAS_CLASS_WEIGHTED_POOL_CAP.
#
# Does not abort on a single script's failure (no `set -e`).
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_overnight.sh > /dev/null 2>&1 &

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
CSCAS_FORCE="${CSCAS_FORCE:-0}"
export CSCAS_FORCE

# Schemas swept per script family.
CLASSIFIER_SCHEMAS=(base full)
ANOMALY_SCHEMAS=(base full_noscas full_scas)

# Mining tree modes swept for the three attribute-mining scripts only.
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

# Run one script once per schema in the given list (CSCAS_SCHEMA scoped to
# that invocation only). Each script's own early-exit guard skips schemas
# whose result files already exist.
run_sweep() {
    local script="$1"
    shift
    local schema
    for schema in "$@"; do
        run_step "$script (CSCAS_SCHEMA=$schema)" \
            env CSCAS_SCHEMA="$schema" "$PYTHON" "$script"
    done
}

# Same as run_sweep, but also sweeps CSCAS_MINING_MODE (single_tree then
# two_tree) -- for the three attribute-mining scripts only. Each script's
# early-exit guard skips (mode, schema) combos whose "_twotree"-suffixed (or
# unsuffixed, for single_tree) result files already exist.
run_mining_sweep() {
    local script="$1"
    shift
    local mode schema
    for mode in "${MINING_MODES[@]}"; do
        for schema in "$@"; do
            run_step "$script (CSCAS_MINING_MODE=$mode, CSCAS_SCHEMA=$schema)" \
                env CSCAS_MINING_MODE="$mode" CSCAS_SCHEMA="$schema" "$PYTHON" "$script"
        done
    done
}

{
    echo "=== Overnight baseline grid run started at $(date) ==="
    echo "CSCAS_QUICK_SANITY_CHECK=$CSCAS_QUICK_SANITY_CHECK" \
         "CSCAS_CLASS_WEIGHTED_POOL_CAP=$CSCAS_CLASS_WEIGHTED_POOL_CAP" \
         "CSCAS_FORCE=$CSCAS_FORCE"

    # RF replication pair -- own schema axis, no CSCAS_SCHEMA sweep.
    run_step "cscas.py" "$PYTHON" cscas.py
    run_step "cscas_base.py" "$PYTHON" cscas_base.py

    # Tabular classifiers -- base + full schema.
    run_sweep cscas_logreg.py "${CLASSIFIER_SCHEMAS[@]}"
    run_sweep cscas_xgboost.py "${CLASSIFIER_SCHEMAS[@]}"
    run_mining_sweep cscas_mining.py "${CLASSIFIER_SCHEMAS[@]}"

    # Anomaly detectors -- base + full_noscas + full_scas.
    run_sweep cscas_anomaly.py "${ANOMALY_SCHEMAS[@]}"
    run_sweep cscas_anomaly_iforest.py "${ANOMALY_SCHEMAS[@]}"
    run_mining_sweep cscas_mining_anomaly.py "${ANOMALY_SCHEMAS[@]}"
    run_mining_sweep cscas_mining_anomaly_iforest.py "${ANOMALY_SCHEMAS[@]}"

    # Text models -- no schema axis, full-test eval added inside the script.
    run_step "cscas_bert.py" "$PYTHON" cscas_bert.py
    run_step "cscas_securebert.py" "$PYTHON" cscas_securebert.py

    run_step "notebook execution" "$PYTHON" -m jupyter nbconvert \
        --to notebook --execute --inplace \
        ../notebooks/baselines/01_cscas_baseline.ipynb

    echo ""
    echo "=== Overnight baseline grid run finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

#!/usr/bin/env bash
# One-shot cleanup + rerun for exactly the mining-dependent baseline results
# that went stale from this round of mining fixes/improvements:
#   - CSCAS two_tree mining: depth=1->4 fix for the anomaly scripts, plus
#     today's max_diverse_rounds=2 addition to the classifier's own
#     two_tree config (_cscas_schema.mining_attribute_config /
#     mining_attribute_config_anomaly).
#   - AIT-ADS mining (classifier + both anomaly scripts, every scenario x
#     grouping-method combo): the new short/sig/host set-valued candidate
#     fields, host normalization, and the two supporting bug fixes
#     (attribute_contrast_mining._mutually_exclusive,
#     monitor.predicate_eval.evaluate_condition) changed AIT-ADS's mining
#     candidate space globally, so every existing AIT-ADS mining result
#     predates the fix.
#
# NOT touched (still valid, left alone):
#   - CSCAS single_tree mining (classifier + both anomaly scripts) --
#     verified byte-reproducible, unaffected by any of the above.
#   - Every non-mining CSCAS/AIT-ADS baseline (cscas.py, cscas_base.py,
#     cscas_logreg.py, cscas_xgboost.py, cscas_anomaly*.py, cscas_bert.py,
#     cscas_securebert.py, ait_ads_rf.py/logreg.py/xgboost.py/anomaly*.py/
#     bert/securebert/zeroshot).
#   - CSCAS's cached/fingerprinted mining path (mine_or_reuse_attribute_
#     schema, used by temporal_decay/monitor_drift/screening_sweep, NOT by
#     any baseline script) -- that cache self-invalidates on next use via a
#     fingerprint miss, nothing to clear here.
#
# Two phases:
#   1) Delete exactly the stale artifacts/mining/ run directories and
#      results/*.json files identified above (nothing else).
#   2) Rerun cscas_mining.py / cscas_mining_anomaly.py /
#      cscas_mining_anomaly_iforest.py (both CSCAS_MINING_MODE values --
#      single_tree is included for parity with run_cscas_mining.sh/
#      run_overnight.sh's own sweep idiom, but is a fast no-op here since
#      its result files were never deleted) and the three AIT-ADS mining
#      scripts (AIT_ADS_FORCE=1, every scenario x grouping-method combo,
#      alertbert included). Since phase 1 already deleted every stale
#      result file, each script's own resumability guard means this only
#      ever computes what's actually missing -- interrupting and
#      re-running this script is safe.
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_stale_mining_rerun.sh > /dev/null 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/src/thesis/baselines"
RESULTS_DIR="$BASELINES_DIR/results"
MINING_ARTIFACTS_DIR="$REPO_ROOT/artifacts/mining"

PYTHON="python3"
LOG_DIR="$BASELINES_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/stale_mining_rerun_$(date +%Y%m%d_%H%M%S).log"

CSCAS_MINING_MODES=(single_tree two_tree)
CSCAS_CLASSIFIER_SCHEMAS=(base full)
CSCAS_ANOMALY_SCHEMAS=(base full_noscas full_scas)

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
    echo "=== Stale-mining cleanup + rerun started at $(date) ==="

    echo ""
    echo "--- Phase 1a: clearing stale artifacts/mining/ run directories ---"
    # CSCAS two_tree only -- single_tree mining artifacts are untouched.
    find "$MINING_ARTIFACTS_DIR" -maxdepth 1 -type d -name '*cscas_baseline_mining_twotree*' -print -exec rm -rf {} +
    find "$MINING_ARTIFACTS_DIR" -maxdepth 1 -type d -name '*cscas_baseline_mining_anomaly_twotree*' -print -exec rm -rf {} +
    # AIT-ADS: every scenario x grouping combo, classifier + anomaly.
    find "$MINING_ARTIFACTS_DIR" -maxdepth 1 -type d -name '*ait_ads_mining_*' -print -exec rm -rf {} +

    echo ""
    echo "--- Phase 1b: clearing stale results/*.json ---"
    cd "$RESULTS_DIR"
    # CSCAS: only "_twotree"-suffixed mining results (classifier RF/LogReg/
    # XGBoost + both anomaly detectors, every schema/eval-set variant) --
    # this substring never appears in a single_tree result's filename.
    for f in cscas_mining*_twotree*.json; do
        [ -e "$f" ] || continue
        echo "  rm $f"
        rm -f -- "$f"
    done
    # AIT-ADS: every mining classifier + anomaly result, every scenario x
    # grouping-method combo -- the candidate feature space itself changed,
    # so nothing here is still valid.
    for f in ait_ads_mining_*.json; do
        [ -e "$f" ] || continue
        echo "  rm $f"
        rm -f -- "$f"
    done
    cd "$BASELINES_DIR"

    echo ""
    echo "=== Phase 2: CSCAS mining rerun (classifier + both anomaly detectors) ==="
    for mode in "${CSCAS_MINING_MODES[@]}"; do
        for schema in "${CSCAS_CLASSIFIER_SCHEMAS[@]}"; do
            run_step "cscas_mining.py (CSCAS_MINING_MODE=$mode, CSCAS_SCHEMA=$schema)" \
                env CSCAS_MINING_MODE="$mode" CSCAS_SCHEMA="$schema" "$PYTHON" cscas_mining.py
        done
        for schema in "${CSCAS_ANOMALY_SCHEMAS[@]}"; do
            run_step "cscas_mining_anomaly.py (CSCAS_MINING_MODE=$mode, CSCAS_SCHEMA=$schema)" \
                env CSCAS_MINING_MODE="$mode" CSCAS_SCHEMA="$schema" "$PYTHON" cscas_mining_anomaly.py
            run_step "cscas_mining_anomaly_iforest.py (CSCAS_MINING_MODE=$mode, CSCAS_SCHEMA=$schema)" \
                env CSCAS_MINING_MODE="$mode" CSCAS_SCHEMA="$schema" "$PYTHON" cscas_mining_anomaly_iforest.py
        done
    done

    echo ""
    echo "=== Phase 3: AIT-ADS mining rerun (classifier + both anomaly detectors, every scenario x grouping combo) ==="
    for script in ait_ads_mining.py ait_ads_mining_anomaly.py ait_ads_mining_anomaly_iforest.py; do
        AIT_ADS_GROUPING_METHODS="$NON_ALERTBERT_METHODS" \
            run_step "$script (fixed_window/time_delta/cscas_grouping/deepcase)" \
            env AIT_ADS_FORCE=1 "$PYTHON" "$script"
        run_step "$script (alertbert)" \
            env AIT_ADS_FORCE=1 AIT_ADS_GROUPING_METHODS=alertbert conda run -n thesis-alertbert --no-capture-output \
            "$PYTHON" "$script"
    done

    echo ""
    echo "=== Stale-mining cleanup + rerun finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

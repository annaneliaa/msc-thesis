#!/usr/bin/env bash
# Repairs the AIT-ADS baseline grid's cscas_grouping data for fox/harrison/
# russellmitchell, then fills every cell that leaves genuinely open -- see
# the "AIT-ADS Baseline Grid" artifact / MEMORY.md for the full 640-cell
# breakdown. Everything else in that grid is either already present,
# excluded by design (harrison/santos/russellmitchell's chronological
# train-side single-class split, or alertbert/deepcase leakage on
# shaw/wardbeck/wheeler/wilson), or a structural DeepCASE-on-santos failure
# -- none of that is touched here.
#
# STEP 0 -- cache repair (found 2026-09-12). cscas_grouping's alert_groups
# cache for exactly fox/harrison/russellmitchell was stale/truncated: 113,
# 566, and 44 alert_groups respectively, versus the correct 6093, 16024,
# 4504 (verified against a from-scratch local rebuild off the same,
# byte-identical alerts.json -- fixed_window/time_delta and alerts.json
# itself all matched local exactly, so this was isolated to cscas_grouping
# for these 3 scenarios only, not a general data problem). That tiny cache
# made fox's train/test splits single-class (0 attacks each), so every
# classifier/BERT/zero-shot combo silently skipped instead of erroring --
# looked exactly like an ordinary "not yet run" gap. It also meant the 12
# anomaly-family results below (fox/harrison/russellmitchell x
# {anomaly_ocsvm, anomaly_iforest, mining_anomaly_ocsvm,
# mining_anomaly_iforest}) WERE computed and saved, just on the wrong data
# -- anomaly's guard is test-side-only, and 34/566ths and 44-ish rows were
# still enough to pass that check. Deleting the cache dir forces a full,
# correct rebuild from alerts.json (still present, still correct) on the
# next ingest; AIT_ADS_FORCE=1 makes the anomaly scripts recompute instead
# of skipping their now-stale-but-still-existing result files.
#
# STEP 1+ -- the 20 cells with no exclusion that were simply never run:
#   cscas_grouping / fox:
#     ait_ads_rf.py, ait_ads_logreg.py, ait_ads_xgboost.py         (3)
#     ait_ads_mining.py -- one mining pass, 3 result files         (RF/LogReg/XGBoost)
#     ait_ads_bert.py, ait_ads_securebert.py                        (2)
#     ait_ads_zeroshot.py x 4 models                                (4)
#   cscas_grouping / {harrison, russellmitchell} -- zero-shot only, since
#   only the classifiers are excluded there (train-side guard), not
#   zero-shot (test-side only, and both scenarios' test splits carry both
#   classes under cscas_grouping, real data or corrupted):
#     ait_ads_zeroshot.py x 4 models x 2 scenarios                  (8)
#
# Every script past step 0 is resumable (skips a combo whose result JSON
# already exists), so this is safe to re-run after an interruption, and
# safe to run alongside/after the general run_ait_ads_*.sh scripts without
# redoing their work -- AIT_ADS_SCENARIOS/AIT_ADS_GROUPING_METHODS below
# scope every step to exactly these cells. Re-running this script a second
# time is also safe: step 0's rm -rf is a no-op once the cache is already
# rebuilt, and AIT_ADS_FORCE on the anomaly scripts only matters the first
# time (they'll have fresh, correct results after that).
#
# No alertbert/deepcase involved (cscas_grouping only), so unlike
# run_ait_ads_tabular.sh/run_ait_ads_mining.sh/run_ait_ads_bert_securebert.sh
# this script never needs the thesis-alertbert conda env -- one env for the
# whole thing.
#
# BERT/SecureBERT default to a non-saving QUICK_SANITY_CHECK pass -- forced
# off below. Zero-shot needs Ollama reachable at OLLAMA_HOST with each model
# pulled (done automatically here via /api/pull) -- defaults to the Docker
# bridge gateway (172.17.0.1), not localhost, since Ollama runs on the DGX
# host, not in this container (override OLLAMA_HOST if that's not your
# setup).
#
# Does not abort on a single script's failure (no `set -e`).
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_ait_ads_gap_fill.sh > /dev/null 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/src/thesis/baselines"
cd "$BASELINES_DIR"

PYTHON="python3"
LOG_DIR="$BASELINES_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ait_ads_gap_fill_$(date +%Y%m%d_%H%M%S).log"

export AIT_ADS_QUICK_SANITY_CHECK=0
export AIT_ADS_CLASS_WEIGHTED_POOL_CAP="${AIT_ADS_CLASS_WEIGHTED_POOL_CAP:-15000}"
export AIT_ADS_REQUIRE_GPU="${AIT_ADS_REQUIRE_GPU:-0}"

FOX_ONLY="fox"
FOX_PLUS_ZEROSHOT_EXTRAS="fox,harrison,russellmitchell"
GROUPING="cscas_grouping"
CORRUPTED_SCENARIOS=(fox harrison russellmitchell)

# Ollama runs on the DGX host, not in this container -- 172.17.0.1 is the
# default Docker bridge gateway, reachable regardless of --network mode
# (unlike localhost, which only works with --network host). Override if
# your setup differs.
OLLAMA_HOST="${OLLAMA_HOST:-http://172.17.0.1:11434}"
MODELS=(
    "llama3.1:8b"
    "llama3.1:70b"
    "qwen2.5:7b"
    "qwen2.5:72b"
)

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

ollama_pull() {
    local model="$1"
    curl -sf -X POST "$OLLAMA_HOST/api/pull" \
        -d "{\"name\": \"$model\", \"stream\": false}"
}

{
    echo "=== AIT-ADS gap-fill run started at $(date) ==="
    echo "Grouping: $GROUPING | classifiers/mining/bert scoped to: $FOX_ONLY | zero-shot scoped to: $FOX_PLUS_ZEROSHOT_EXTRAS"
    echo "AIT_ADS_QUICK_SANITY_CHECK=$AIT_ADS_QUICK_SANITY_CHECK AIT_ADS_CLASS_WEIGHTED_POOL_CAP=$AIT_ADS_CLASS_WEIGHTED_POOL_CAP AIT_ADS_REQUIRE_GPU=$AIT_ADS_REQUIRE_GPU"

    # -- Step 0: wipe the truncated cscas_grouping cache for the 3 affected
    # scenarios so the next ingest rebuilds it correctly from alerts.json.
    # Scoped to exactly these 3 scenarios' cscas_grouping subtree -- never
    # touches fixed_window/time_delta or any other scenario.
    for s in "${CORRUPTED_SCENARIOS[@]}"; do
        cache_dir="$REPO_ROOT/artifacts/cache/$s/groups/$GROUPING"
        run_step "wipe corrupted cache ($s/$GROUPING)" rm -rf "$cache_dir"
    done

    # -- Step 0b: force-recompute the 4 anomaly-family results per scenario
    # that WERE saved on the corrupted data (test-side-only guard let them
    # through). AIT_ADS_FORCE=1 makes each script recompute instead of
    # trusting its now-stale existing result file.
    for script in ait_ads_anomaly.py ait_ads_anomaly_iforest.py \
                  ait_ads_mining_anomaly.py ait_ads_mining_anomaly_iforest.py; do
        AIT_ADS_SCENARIOS="$FOX_PLUS_ZEROSHOT_EXTRAS" AIT_ADS_GROUPING_METHODS="$GROUPING" \
            AIT_ADS_FORCE=1 \
            run_step "$script ($GROUPING/$FOX_PLUS_ZEROSHOT_EXTRAS, forced recompute)" "$PYTHON" "$script"
    done

    # -- 3 tabular classifiers: cscas_grouping / fox --
    for script in ait_ads_rf.py ait_ads_logreg.py ait_ads_xgboost.py; do
        AIT_ADS_SCENARIOS="$FOX_ONLY" AIT_ADS_GROUPING_METHODS="$GROUPING" \
            run_step "$script ($GROUPING/$FOX_ONLY)" "$PYTHON" "$script"
    done

    # -- mining (RF + LogReg + XGBoost in one pass): cscas_grouping / fox --
    AIT_ADS_SCENARIOS="$FOX_ONLY" AIT_ADS_GROUPING_METHODS="$GROUPING" \
        run_step "ait_ads_mining.py ($GROUPING/$FOX_ONLY)" "$PYTHON" ait_ads_mining.py

    # -- BERT / SecureBERT: cscas_grouping / fox --
    AIT_ADS_SCENARIOS="$FOX_ONLY" AIT_ADS_GROUPING_METHODS="$GROUPING" \
        run_step "ait_ads_bert.py ($GROUPING/$FOX_ONLY)" "$PYTHON" ait_ads_bert.py
    AIT_ADS_SCENARIOS="$FOX_ONLY" AIT_ADS_GROUPING_METHODS="$GROUPING" \
        run_step "ait_ads_securebert.py ($GROUPING/$FOX_ONLY)" "$PYTHON" ait_ads_securebert.py

    # -- zero-shot x 4 models: cscas_grouping / {fox, harrison, russellmitchell} --
    for model in "${MODELS[@]}"; do
        run_step "ollama pull $model" ollama_pull "$model"
        AIT_ADS_SCENARIOS="$FOX_PLUS_ZEROSHOT_EXTRAS" AIT_ADS_GROUPING_METHODS="$GROUPING" \
            run_step "ait_ads_zeroshot.py ($model, $GROUPING/$FOX_PLUS_ZEROSHOT_EXTRAS)" \
            env OLLAMA_MODEL="$model" "$PYTHON" ait_ads_zeroshot.py
    done

    echo ""
    echo "=== AIT-ADS gap-fill run finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

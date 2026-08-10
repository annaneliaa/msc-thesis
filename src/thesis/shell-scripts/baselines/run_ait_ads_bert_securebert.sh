#!/usr/bin/env bash
# Overnight batch runner for the AIT-ADS BERT/SecureBERT baselines
# (ait_ads_bert.py, ait_ads_securebert.py) -- the AIT-ADS counterpart to
# baselines/run_overnight.sh. Each script already loops every AIT-ADS
# scenario x grouping method internally (AIT_ADS_SCENARIOS /
# AIT_ADS_GROUPING_METHODS below to run a subset of either), so this
# script only needs run_step calls per model x env, not loops of its own.
#
# alertbert grouping needs the `thesis-alertbert` conda env (graph-tool --
# see _ait_ads_grouping.py's module docstring), same as run_grouping.sh's
# own alertbert_sweep.py step -- but BERT/SecureBERT fine-tuning itself
# (transformers/torch) needs the plain venv's package set. Since
# ait_ads_bert.py/ait_ads_securebert.py do grouping AND fine-tuning in one
# process, this script runs each script TWICE: once for
# fixed_window/time_delta/cscas_grouping/deepcase in the plain venv, once
# for alertbert alone via `conda run -n thesis-alertbert` -- so whichever
# env this script itself is launched from, both halves still work. This
# only works if thesis-alertbert also has this project's other deps
# (transformers, datasets, torch) installed -- see setup_container.sh's
# thesis-alertbert branch, or install them there yourself if running
# locally without that script.
#
# Does NOT run ait_ads_zeroshot.py (separate Ollama dependency -- see
# run_ait_ads_zeroshot.sh) or re-execute the comparison notebook: unlike
# cscas_baseline_comparison.ipynb (single dataset),
# ait_ads_baseline_comparison.ipynb is SCENARIO/GROUPING_METHOD-
# parameterized (its own Settings cell) -- there's no single "the" notebook
# run across every combination to automate here. Open it manually per
# (scenario, grouping method) once these scripts finish.
#
# class_weighted is capped at AIT_ADS_CLASS_WEIGHTED_POOL_CAP rows (see
# ait_ads_bert.py's module docstring / _sampling.class_weighted_pool) --
# unset by default (uncapped); set it below if a scenario's natural-ratio
# pool turns out too large to fine-tune on directly.
#
# Does not abort on a single script's failure (no `set -e`) -- if one
# script errors out (e.g. one scenario has a single-class split, or an
# alertbert/deepcase leakage scenario, both of which each script's own
# run_scenario() already skips gracefully), its exit code is logged and
# the run continues to the next step.
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_ait_ads_bert_securebert.sh > /dev/null 2>&1 &
# (or just `./run_ait_ads_bert_securebert.sh &` in a terminal you're about
# to close -- nohup is the safer bet so a closed terminal/SSH session
# doesn't kill it)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASELINES_DIR="$REPO_ROOT/src/thesis/baselines"
cd "$BASELINES_DIR"

PYTHON="python3"
LOG_DIR="$BASELINES_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ait_ads_bert_securebert_$(date +%Y%m%d_%H%M%S).log"

export AIT_ADS_QUICK_SANITY_CHECK=0
export AIT_ADS_CLASS_WEIGHTED_POOL_CAP="${AIT_ADS_CLASS_WEIGHTED_POOL_CAP:-}"
export AIT_ADS_SCENARIOS="${AIT_ADS_SCENARIOS:-}"  # empty = every AIT-ADS scenario
# Non-alertbert methods run in whatever env this script itself is invoked
# with; alertbert always runs separately via conda run below, so it's
# excluded here regardless of what AIT_ADS_GROUPING_METHODS the caller set.
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
    echo "=== AIT-ADS BERT/SecureBERT run started at $(date) ==="
    echo "AIT_ADS_QUICK_SANITY_CHECK=$AIT_ADS_QUICK_SANITY_CHECK AIT_ADS_CLASS_WEIGHTED_POOL_CAP=$AIT_ADS_CLASS_WEIGHTED_POOL_CAP AIT_ADS_SCENARIOS=${AIT_ADS_SCENARIOS:-<all>}"
    echo "Non-alertbert grouping methods (plain venv): $NON_ALERTBERT_METHODS"

    AIT_ADS_GROUPING_METHODS="$NON_ALERTBERT_METHODS" \
        run_step "ait_ads_bert.py (fixed_window/time_delta/cscas_grouping/deepcase)" "$PYTHON" ait_ads_bert.py
    AIT_ADS_GROUPING_METHODS="$NON_ALERTBERT_METHODS" \
        run_step "ait_ads_securebert.py (fixed_window/time_delta/cscas_grouping/deepcase)" "$PYTHON" ait_ads_securebert.py

    run_step "ait_ads_bert.py (alertbert)" \
        env AIT_ADS_GROUPING_METHODS=alertbert conda run -n thesis-alertbert --no-capture-output \
        "$PYTHON" ait_ads_bert.py
    run_step "ait_ads_securebert.py (alertbert)" \
        env AIT_ADS_GROUPING_METHODS=alertbert conda run -n thesis-alertbert --no-capture-output \
        "$PYTHON" ait_ads_securebert.py

    echo ""
    echo "=== AIT-ADS BERT/SecureBERT run finished at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

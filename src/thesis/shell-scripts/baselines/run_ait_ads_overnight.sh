#!/usr/bin/env bash
# Full AIT-ADS system-task baseline comparison, unattended, for the DGX --
# chains every piece needed to populate ait_ads_baseline_comparison.ipynb
# for all 8 AIT-ADS scenarios: tabular models, then the two fine-tuned
# LLM baselines, then zero-shot, then the anomaly baseline, then the two
# mining baselines. Mirrors
# shell-scripts/system_eval/run_overnight.sh's "label:script" chaining
# pattern -- every step runs regardless of earlier failures, failures are
# reported at the end, not abandoned mid-batch.
#
# Stages:
#   1. run_ait_ads_tabular.sh -- RF/LogReg/XGBoost, all 5 grouping methods
#      x 8 scenarios (minus alertbert/deepcase leakage skips) x 2 pool
#      conditions x 5 seeds. Shares the exact same split
#      (_ait_ads_data.load_ait_ads_baseline_split) as stages 2-4 below, so
#      results are directly comparable across all four model families --
#      see ait_ads_rf.py's module docstring for why this replaced the
#      earlier run_model_comparison_attribute.py-based approach (that
#      pipeline is fixed_window-only and a separate codepath, so it could
#      only ever guarantee *similar* splits, not identical ones).
#   2. run_ait_ads_bert_securebert.sh -- BERT/SecureBERT, same grouping
#      method x scenario x pool condition x seed grid as stage 1. Handles
#      the thesis-alertbert conda env split internally -- see that
#      script's own header.
#   3. run_ait_ads_zeroshot.sh -- zero-shot, all 5 grouping methods x 8
#      scenarios x 4 Ollama models. Requires Ollama reachable at
#      OLLAMA_HOST (default http://localhost:11434) with --network host if
#      containerized -- see that script's own header.
#   4. run_ait_ads_anomaly.sh -- OneClassSVM fit on benign-only rows, same
#      grouping method x scenario grid as stage 1 but no pool
#      conditions/seeds (single deterministic run per combo). Produces a
#      result for harrison/santos/russellmitchell -- see
#      ait_ads_anomaly.py's module docstring.
#   5. run_ait_ads_mining.sh -- ait_ads_mining.py (RF on base schema +
#      attribute-mined symbolic features, same pool conditions x seeds as
#      stage 1) and ait_ads_mining_anomaly.py (OneClassSVM sibling of
#      stage 4, same test-side-only guard -- the only mining-based script
#      that also produces a harrison/santos/russellmitchell result).
#
# Every stage is resumable: each ait_ads_*.py script skips a
# (grouping_method, scenario) combo whose results/*.json already exists,
# so restarting this script after a crash or an interrupted run only
# redoes what's still missing. Set AIT_ADS_FORCE=1 (exported before
# calling this script, or edited into the stage scripts directly) to force
# a full re-run from scratch instead.
#
# /!\ Total runtime: expect this to run for many hours to well over a day
# depending on GPU/CPU -- every stage is N_SEEDS x 2-condition training (or
# 4-model prompting for zero-shot) repeated per grouping method per
# scenario. Budget real DGX time, don't expect this back before the next
# work session.
#
# Prerequisite: this container/machine must already be set up --
# shell-scripts/baselines/grouping/setup_container.sh once (installs this
# project + the thesis-alertbert conda env with graph-tool; despite living
# under grouping/, it's not grouping-specific -- see that script's own
# header). Every stage's alertbert steps additionally need this project's
# other deps (transformers/datasets/torch for stage 2; sklearn/xgboost/
# pandas for stage 1) installed in thesis-alertbert too -- see
# setup_container.sh's thesis-alertbert branch.
#
# Run:
#   nohup src/thesis/shell-scripts/baselines/run_ait_ads_overnight.sh > /dev/null 2>&1 &
# (or, for a `docker exec -it`/SSH session you're about to close, see
# shell-scripts/system_eval/run_model_comparison_attribute.sh's header for
# the nohup+disown / docker exec -d patterns -- same considerations apply
# here.)
#
# Each stage still writes its own log(s) under artifacts/logs/ or
# src/thesis/baselines/logs/ as usual -- this script's own stdout is just
# per-stage start/end timestamps and exit codes.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"

LOG_DIR="$REPO_ROOT/artifacts/logs/ait_ads_overnight"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ait_ads_overnight_$(date -u +%Y%m%d_%H%M%S).log"

# Plain array of "label:script" pairs rather than an associative array --
# macOS's default /usr/bin/bash is 3.2, which doesn't have those.
steps=(
  "1. Tabular models (run_ait_ads_tabular.sh):$HERE/run_ait_ads_tabular.sh"
  "2. BERT/SecureBERT (run_ait_ads_bert_securebert.sh):$HERE/run_ait_ads_bert_securebert.sh"
  "3. Zero-shot (run_ait_ads_zeroshot.sh):$HERE/run_ait_ads_zeroshot.sh"
  "4. Anomaly (run_ait_ads_anomaly.sh):$HERE/run_ait_ads_anomaly.sh"
  "5. Mining (run_ait_ads_mining.sh):$HERE/run_ait_ads_mining.sh"
)

overall_failed=0

{
  echo "=== AIT-ADS overnight baseline run started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

  for step in "${steps[@]}"; do
    label="${step%%:*}"
    script="${step#*:}"
    echo
    echo "============================================================"
    echo "  START: $label  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "============================================================"
    "$script"
    status=$?
    echo "  END:   $label  ($(date -u +%Y-%m-%dT%H:%M:%SZ))  exit=$status"
    if [[ $status -ne 0 ]]; then
      overall_failed=1
      echo "  [warn] $label failed (exit $status) -- continuing to the next stage anyway"
    fi
  done

  echo
  echo "============================================================"
  if [[ $overall_failed -eq 0 ]]; then
    echo "  AIT-ADS OVERNIGHT RUN: all stages succeeded  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  else
    echo "  AIT-ADS OVERNIGHT RUN: at least one stage failed -- check logs above  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  fi
  echo "============================================================"
  # exit here (inside the piped block), not after it -- `{ ... } | tee` runs
  # this block in a subshell, so a plain `exit $overall_failed` placed after
  # the pipe would see the *outer* shell's never-updated copy of the
  # variable, not this one. ${PIPESTATUS[0]} below picks up this exit code.
  exit $overall_failed
} 2>&1 | tee "$LOG_FILE"

exit "${PIPESTATUS[0]}"

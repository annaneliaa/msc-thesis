#!/usr/bin/env bash
#
# Monitor Attached: for each scenario, runs run_monitor_attached.py over the
# parameter grid -- MINING_SETTINGS below crossed with GRANULARITIES x
# MODELS. Locked to the same "monitor EDA" scope as run_temporal_decay.sh /
# run_rolling_walk_forward.sh / run_monitor_drift.sh (see those scripts' own
# comments) so all four experiments' horizons/configs line up and can be
# overlaid directly: single mining setting gr3_md1_mda4_rounds2 (="two_tree"),
# models in {xgboost, rf}, granularities in {0.1, 0.05}, source window
# anchored to the CSCAS baseline's own train/test boundary
# (source_split_mode=baseline_split). Only feature_set='symbolic' configs
# ever produce a DynamicSchema, so the derived shortlist is filtered to
# those (see run_monitor_attached.py).
#
# For each config, mines/fits once on W_src's train split -- identical setup
# to run_monitor_drift.py -- then walks the schema/model/Vk forward one
# window at a time, but here a sustained monitor signal actually retrains or
# remines in place instead of only being logged. Every horizon is timed
# (fast-route serving cost, workload funnel) and every retrain/remine event
# is timed and diffed against the schema it replaced.
#
# PSI_THRESHOLD/CAL_THRESHOLD/CONSECUTIVE_WINDOWS/MIN_SAMPLES_SIGNAL_2 are
# required arguments, not defaults baked into this script: PSI_THRESHOLD and
# CAL_THRESHOLD gate real retrain/remine actions here (unlike
# run_monitor_drift.py's purely diagnostic `elevated` column), so they must
# be the values Drift Signal EDA's threshold sweep actually selected
# (03_monitor_signal_drift.ipynb, Analysis 3) -- there is no "untuned
# default" that's safe to silently fall back to for a real run.
#
# Usage:
#   src/thesis/shell-scripts/system_eval/run_monitor_attached.sh \
#     <psi_threshold> <cal_threshold> <consecutive_windows> <min_samples_signal_2>
#
# Example:
#   src/thesis/shell-scripts/system_eval/run_monitor_attached.sh 0.15 0.12 3 30
#
# TRIGGER (env, default "feedback") picks what acts: analyst-feedback
# precision (FEEDBACK_CONSECUTIVE_WINDOWS in a row -> FEEDBACK_ACTION) or
# "thesis" (the Signal 1/2 table, gated by the positional thresholds). The
# other decision is still logged every window. "combined": pattern validity
# (PATTERN_LABEL_MODE labels, default audit) -> REMINE_AND_RETRAIN, else
# feedback precision -> RETRAIN_ONLY.
#
# Edit SCENARIOS/MINING_SETTINGS/MODELS below, or set GRANULARITIES
# (space-separated) in the environment, to change what else runs.

set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <psi_threshold> <cal_threshold> <consecutive_windows> <min_samples_signal_2>" >&2
  echo "  e.g. $0 0.15 0.12 3 30   (psi/cal from Drift Signal EDA's Analysis 3 sweep)" >&2
  exit 1
fi
PSI_THRESHOLD="$1"
CAL_THRESHOLD="$2"
MONITOR_CONSECUTIVE_WINDOWS="$3"
MONITOR_MIN_SAMPLES_SIGNAL_2="$4"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis

SCENARIOS=(cscas)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/monitor_eda_mining_setting.yaml"
read -r -a GRANULARITIES <<< "${GRANULARITIES:-0.1 0.05 0.02}"
TRIGGER="${TRIGGER:-feedback}"
FEEDBACK_CONSECUTIVE_WINDOWS="${FEEDBACK_CONSECUTIVE_WINDOWS:-1}"
FEEDBACK_ACTION="${FEEDBACK_ACTION:-REMINE_AND_RETRAIN}"
PATTERN_LABEL_MODE="${PATTERN_LABEL_MODE:-audit}"
SCHEDULE_ACTION="${SCHEDULE_ACTION:-REMINE_AND_RETRAIN}"
# Label budget (DeepCASE comparison only): UPDATE_LABELS=cumulative_budget,
# INITIAL_BUDGET_MODE=full|stratified|deepcase_items|deepcase_cluster_samples,
# INITIAL_BUDGET_SIZE (stratified), DEEPCASE_ITEMS (parquet), AUDIT_RATE.
UPDATE_LABELS="${UPDATE_LABELS:-window_full}"
INITIAL_BUDGET_MODE="${INITIAL_BUDGET_MODE:-full}"
AUDIT_RATE="${AUDIT_RATE:-0.05}"
MODELS=(xgboost rf)
SOURCE_SPLIT_MODE="baseline_split"  # anchors W_src to the CSCAS baseline's own train/test boundary
THRESHOLD_MODE="fixed"  # or "calibrated_recall" -- keep in sync with the other experiment scripts
CALIBRATED_RECALL_TARGET="0.90"  # only used when THRESHOLD_MODE=calibrated_recall
LATENCY_SAMPLE_N="200"  # total individually-timed single-alert-group calls per config; 0 disables

LOG_DIR="$REPO_ROOT/artifacts/logs/monitor_attached"
mkdir -p "$LOG_DIR"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"

total=0
failed=()

for scenario in "${SCENARIOS[@]}"; do
  total=$((total + 1))
  log_file="$LOG_DIR/${RUN_TS}_${scenario}.log"

  echo "[$total] $scenario"

  if [[ ! -f "$MINING_SETTINGS" ]]; then
    echo "    FAILED — no mining-settings grid at $MINING_SETTINGS"
    failed+=("$scenario")
    continue
  fi

  # -u: unbuffered stdout -- see run_monitor_drift.sh for why this matters
  # when redirecting to a log file.
  cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_monitor_attached.py" \
    "$scenario" \
    --mining-settings "$MINING_SETTINGS" \
    --granularities "${GRANULARITIES[@]}" \
    --models "${MODELS[@]}" \
    --source-split-mode "$SOURCE_SPLIT_MODE" \
    --threshold-mode "$THRESHOLD_MODE" \
    --psi-threshold "$PSI_THRESHOLD" \
    --cal-threshold "$CAL_THRESHOLD" \
    --monitor-consecutive-windows "$MONITOR_CONSECUTIVE_WINDOWS" \
    --monitor-min-samples-signal-2 "$MONITOR_MIN_SAMPLES_SIGNAL_2" \
    --trigger "$TRIGGER" \
    --feedback-consecutive-windows "$FEEDBACK_CONSECUTIVE_WINDOWS" \
    --feedback-action "$FEEDBACK_ACTION" \
    --pattern-label-mode "$PATTERN_LABEL_MODE" \
    --schedule-action "$SCHEDULE_ACTION" \
    --update-labels "$UPDATE_LABELS" \
    --initial-budget-mode "$INITIAL_BUDGET_MODE" \
    --audit-rate "$AUDIT_RATE" \
    --latency-sample-n "$LATENCY_SAMPLE_N")
  if [[ -n "${INITIAL_BUDGET_SIZE:-}" ]]; then
    cmd+=(--initial-budget-size "$INITIAL_BUDGET_SIZE")
  fi
  if [[ -n "${DEEPCASE_ITEMS:-}" ]]; then
    cmd+=(--deepcase-items "$DEEPCASE_ITEMS")
  fi
  if [[ "$THRESHOLD_MODE" == "calibrated_recall" ]]; then
    cmd+=(--calibrated-recall-target "$CALIBRATED_RECALL_TARGET")
  fi

  "${cmd[@]}" >"$log_file" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$scenario")
  else
    grep -E "Monitor Attached results|Saved →" "$log_file" | tail -n 5 | sed 's/^/    /'
  fi
done

echo
echo "============================================================"
echo "  MONITOR ATTACHED SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed scenarios:"
  for scenario in "${failed[@]}"; do
    echo "  - $scenario"
  done
  exit 1
fi

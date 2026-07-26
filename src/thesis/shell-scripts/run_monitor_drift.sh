#!/usr/bin/env bash
#
# Drift-Monitor Evaluation (Experiment 4, observe-only): for each scenario,
# runs run_monitor_drift.py directly against the mining notebook's
# structural shortlist (attribute_mining_sweep_eda.ipynb's
# feasible_configs_all.csv -- every config that clears the mining-only
# precision/recall floors), crossed with GRANULARITIES below. No separate
# real-evaluation ranking step (notebooks/config_selection.ipynb is no
# longer part of this pipeline) -- STRUCTURAL_CONFIGS is the single source
# of truth, same as run_temporal_decay.sh/run_rolling_walk_forward.sh. For
# each resulting config, mines/fits once on window 0's train split -- plus,
# for symbolic configs, builds a deployment-scoped DynamicSchema (Vk) from
# that same mining pass -- then walks the frozen schema/model/Vk forward one
# window at a time, running the drift monitor at every horizon and logging
# every signal/alarm it raises. The monitor never triggers an actual
# re-mine/retrain here -- it only observes and records what it would have
# done. Unlike the other two experiments, mining is never cached here (see
# experiments/monitor_drift.py's module docstring), so every run mines fresh
# for symbolic configs.
#
# Usage:
#   src/thesis/shell-scripts/run_monitor_drift.sh
#
# Edit the variables below to change the scenario(s), granularities,
# threshold mode, or monitor sensitivity.

set -uo pipefail

# Don't rely on the caller's shell already having `thesis` active -- activate
# it explicitly so this script works the same from a cron job, CI, or a
# terminal that's sitting in base/another env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis

SCENARIOS=(cscas)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/screening_mining_settings.yaml"
STRUCTURAL_CONFIGS="$REPO_ROOT/artifacts/experiments/attribute_mining_parameter_grid/feasible_configs_all.csv"
GRANULARITIES=(0.1)
THRESHOLD_MODE="fixed"  # or "calibrated_recall" -- keep in sync with the other experiment scripts
CALIBRATED_RECALL_TARGET="0.90"  # only used when THRESHOLD_MODE=calibrated_recall
MONITOR_CONSECUTIVE_WINDOWS="3"  # consecutive elevated horizons before a soft alert hard-triggers
MONITOR_MIN_SAMPLES_SIGNAL_2="30"  # min labeled matching rows before a rule's calibration drift is evaluated

LOG_DIR="$REPO_ROOT/artifacts/logs/monitor_drift"
mkdir -p "$LOG_DIR"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"

total=0
failed=()

for scenario in "${SCENARIOS[@]}"; do
  total=$((total + 1))
  log_file="$LOG_DIR/${RUN_TS}_${scenario}.log"

  echo "[$total] $scenario"

  if [[ ! -f "$STRUCTURAL_CONFIGS" ]]; then
    echo "    FAILED — no structural shortlist at $STRUCTURAL_CONFIGS (run attribute_mining_sweep_eda.ipynb's section 5.3 first)"
    failed+=("$scenario")
    continue
  fi

  # Built up incrementally (rather than expanding a possibly-empty array)
  # since "${empty_array[@]}" errors under `set -u` on bash <4.4 -- macOS's
  # default /usr/bin/bash is 3.2.
  # -u: unbuffered stdout -- without it, redirecting to $log_file makes
  # Python fully block-buffer stdout (prints only flush every ~8KB or at
  # exit) while warnings.warn() writes straight to unbuffered stderr, so the
  # log looks stuck spewing only warnings for the whole run with none of the
  # "[n/4] ..."/"Saved →" progress prints showing up until process exit.
  cmd=(python -u "$REPO_ROOT/src/thesis/scripts/mining/run_monitor_drift.py" \
    "$scenario" \
    --structural-configs "$STRUCTURAL_CONFIGS" \
    --granularities "${GRANULARITIES[@]}" \
    --mining-settings "$MINING_SETTINGS" \
    --threshold-mode "$THRESHOLD_MODE" \
    --monitor-consecutive-windows "$MONITOR_CONSECUTIVE_WINDOWS" \
    --monitor-min-samples-signal-2 "$MONITOR_MIN_SAMPLES_SIGNAL_2")
  if [[ "$THRESHOLD_MODE" == "calibrated_recall" ]]; then
    cmd+=(--calibrated-recall-target "$CALIBRATED_RECALL_TARGET")
  fi

  "${cmd[@]}" >"$log_file" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$scenario")
  else
    grep -E "Monitor drift results|Saved →" "$log_file" | tail -n 5 | sed 's/^/    /'
  fi
done

echo
echo "============================================================"
echo "  MONITOR DRIFT SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed scenarios:"
  for scenario in "${failed[@]}"; do
    echo "  - $scenario"
  done
  exit 1
fi

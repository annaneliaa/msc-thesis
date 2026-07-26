#!/usr/bin/env bash
#
# Rolling / Walk-Forward Evaluation (Experiment 3): for each scenario, runs
# run_rolling_walk_forward.py directly against the mining notebook's
# structural shortlist (attribute_mining_sweep_eda.ipynb's
# feasible_configs_all.csv -- every config that clears the mining-only
# precision/recall floors), crossed with GRANULARITIES below. No separate
# real-evaluation ranking step (notebooks/config_selection.ipynb is no
# longer part of this pipeline) -- STRUCTURAL_CONFIGS is the single source
# of truth, same as run_temporal_decay.sh. For each resulting config, walks
# the timeline one window at a time, mining+fitting from scratch on the full
# window Wi and evaluating on W(i+1) at every step -- the "always retrain"
# anchor, contrasted against run_temporal_decay.sh's frozen-model decay
# curve. Mined schemas are cached the same way as the other experiments, so
# rerunning this script only (re)mines whatever isn't already cached.
#
# Usage:
#   src/thesis/shell-scripts/run_rolling_walk_forward.sh
#
# Edit the variables below to change the scenario(s), granularities, or
# threshold mode.

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
THRESHOLD_MODE="fixed"  # or "calibrated_recall" -- keep in sync with run_temporal_decay.sh
CALIBRATED_RECALL_TARGET="0.90"  # only used when THRESHOLD_MODE=calibrated_recall

LOG_DIR="$REPO_ROOT/artifacts/logs/rolling_walk_forward"
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
  cmd=(python -u "$REPO_ROOT/src/thesis/scripts/mining/run_rolling_walk_forward.py" \
    "$scenario" \
    --structural-configs "$STRUCTURAL_CONFIGS" \
    --granularities "${GRANULARITIES[@]}" \
    --mining-settings "$MINING_SETTINGS" \
    --threshold-mode "$THRESHOLD_MODE")
  if [[ "$THRESHOLD_MODE" == "calibrated_recall" ]]; then
    cmd+=(--calibrated-recall-target "$CALIBRATED_RECALL_TARGET")
  fi

  "${cmd[@]}" >"$log_file" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$scenario")
  else
    grep -E "Rolling walk-forward results|Saved →" "$log_file" | tail -n 5 | sed 's/^/    /'
  fi
done

echo
echo "============================================================"
echo "  ROLLING WALK-FORWARD SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed scenarios:"
  for scenario in "${failed[@]}"; do
    echo "  - $scenario"
  done
  exit 1
fi

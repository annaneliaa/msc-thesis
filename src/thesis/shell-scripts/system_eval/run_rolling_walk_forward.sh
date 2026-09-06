#!/usr/bin/env bash
#
# Rolling / Walk-Forward Evaluation (Experiment 3): for each scenario, runs
# run_rolling_walk_forward.py over the parameter grid -- every entry in
# MINING_SETTINGS (configs/screening_mining_settings.yaml) crossed with
# GRANULARITIES below, plus a baseline row per granularity. That YAML is the
# single input: no feasible-config CSV, no notebook export step, no
# real-evaluation ranking. Edit the YAML to change what runs. For each
# resulting config, walks the timeline one window at a time, mining+fitting
# from scratch on the full window Wi and evaluating on W(i+1) at every step
# -- the "always retrain" anchor, contrasted against run_temporal_decay.sh's
# frozen-model decay curve. Mined schemas are cached, so rerunning this
# script only (re)mines whatever isn't already cached.
#
# Usage:
#   src/thesis/shell-scripts/system_eval/run_rolling_walk_forward.sh
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
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/screening_mining_settings.yaml"
GRANULARITIES=(0.1)  # 0.1 gives 10 windows; add 0.25 0.5 for the coarser (thinner) walks
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

  if [[ ! -f "$MINING_SETTINGS" ]]; then
    echo "    FAILED — no mining-settings grid at $MINING_SETTINGS"
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
  cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_rolling_walk_forward.py" \
    "$scenario" \
    --mining-settings "$MINING_SETTINGS" \
    --granularities "${GRANULARITIES[@]}" \
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

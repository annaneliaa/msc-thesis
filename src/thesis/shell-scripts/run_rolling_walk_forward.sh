#!/usr/bin/env bash
#
# Rolling / Walk-Forward Evaluation (Experiment 3): for each scenario, runs
# run_rolling_walk_forward.py against that scenario's shortlist (as produced
# by notebooks/config_selection.ipynb from a screening-sweep run, the same
# shortlist run_temporal_decay.sh uses). For each shortlisted config, walks
# the timeline one window at a time, mining+fitting from scratch on the full
# window Wi and evaluating on W(i+1) at every step -- the "always retrain"
# anchor, contrasted against run_temporal_decay.sh's frozen-model decay
# curve. Mined schemas are cached the same way as the other experiments, so
# rerunning this script only (re)mines whatever isn't already cached.
#
# Usage:
#   src/thesis/shell-scripts/run_rolling_walk_forward.sh
#
# Edit the variables below to change the scenario(s) or threshold mode. Each
# scenario needs a shortlist.csv already saved at
# artifacts/experiments/screening_sweep/<scenario>/shortlist.csv.

set -uo pipefail

# Don't rely on the caller's shell already having `thesis` active -- activate
# it explicitly so this script works the same from a cron job, CI, or a
# terminal that's sitting in base/another env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis

SCENARIOS=(cscas)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/screening_mining_settings.yaml"
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
  shortlist="$REPO_ROOT/artifacts/experiments/screening_sweep/$scenario/shortlist.csv"

  echo "[$total] $scenario"

  if [[ ! -f "$shortlist" ]]; then
    echo "    FAILED — no shortlist at $shortlist (run config_selection.ipynb first)"
    failed+=("$scenario")
    continue
  fi

  # Built up incrementally (rather than expanding a possibly-empty array)
  # since "${empty_array[@]}" errors under `set -u` on bash <4.4 -- macOS's
  # default /usr/bin/bash is 3.2.
  cmd=(python "$REPO_ROOT/src/thesis/scripts/mining/run_rolling_walk_forward.py" \
    "$scenario" \
    --shortlist "$shortlist" \
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

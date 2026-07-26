#!/usr/bin/env bash
#
# Temporal Generalization (Rolling-Horizon Decay): for each scenario, runs
# run_temporal_decay.py directly against the mining notebook's structural
# shortlist (attribute_mining_sweep_eda.ipynb's feasible_configs_all.csv --
# every config that clears the mining-only precision/recall floors), crossed
# with GRANULARITIES below. No separate real-evaluation ranking step
# (notebooks/config_selection.ipynb is no longer part of this pipeline) --
# STRUCTURAL_CONFIGS is the single source of truth. For each resulting
# config, run_temporal_decay.py mines/fits once on window 0's train split and
# walks the frozen schema/model/threshold forward one window at a time to
# the end of the timeline, tracking SHAP/LIME importances alongside the
# metric decay -- mined schemas are cached the same way as the screening
# sweep, so rerunning this script only (re)mines whatever isn't already
# cached.
#
# Usage:
#   src/thesis/shell-scripts/run_temporal_decay.sh
#
# Edit the variables below to change the scenario(s), granularities,
# threshold mode, or explanation sampling.

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
GRANULARITIES=(0.1 0.2 0.25)
THRESHOLD_MODE="fixed"  # or "calibrated_recall"
CALIBRATED_RECALL_TARGET="0.90"  # only used when THRESHOLD_MODE=calibrated_recall
COMPUTE_EXPLANATIONS=1  # 0 to skip SHAP/LIME (metrics only, much faster)
EXPLAIN_SAMPLE_N=50
LIME_NUM_SAMPLES=1000

LOG_DIR="$REPO_ROOT/artifacts/logs/temporal_decay"
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
  # log looks like it's stuck spewing only sklearn warnings for the whole
  # run with none of the "[n/4] ..."/"Saved →" progress prints showing up
  # until the process exits.
  cmd=(python -u "$REPO_ROOT/src/thesis/scripts/mining/run_temporal_decay.py" \
    "$scenario" \
    --structural-configs "$STRUCTURAL_CONFIGS" \
    --granularities "${GRANULARITIES[@]}" \
    --mining-settings "$MINING_SETTINGS" \
    --threshold-mode "$THRESHOLD_MODE" \
    --explain-sample-n "$EXPLAIN_SAMPLE_N" \
    --lime-num-samples "$LIME_NUM_SAMPLES")
  if [[ "$THRESHOLD_MODE" == "calibrated_recall" ]]; then
    cmd+=(--calibrated-recall-target "$CALIBRATED_RECALL_TARGET")
  fi
  if [[ "$COMPUTE_EXPLANATIONS" -eq 0 ]]; then
    cmd+=(--no-explanations)
  fi

  "${cmd[@]}" >"$log_file" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$scenario")
  else
    grep -E "Temporal decay results|Saved →" "$log_file" | tail -n 5 | sed 's/^/    /'
  fi
done

echo
echo "============================================================"
echo "  TEMPORAL DECAY SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed scenarios:"
  for scenario in "${failed[@]}"; do
    echo "  - $scenario"
  done
  exit 1
fi

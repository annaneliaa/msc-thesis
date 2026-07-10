#!/usr/bin/env bash
#
# Pre-mine attribute schemas for cscas across a grid of mining parameters,
# via mine_attribute_schema.py --windowed. Each MINE_FRACS value is treated as
# a granularity: instead of mining once on the first mine_frac fraction of the
# timeline, --windowed mines one schema per chronological window of that size,
# covering the *entire* timeline (mine_frac=0.1 -> 10 windows, mine_frac=1.0 ->
# 1 window). Every (growth_rate, max_depth, granularity, window, class_weight,
# min_samples_leaf) combination is cached on disk (see
# thesis.mining.attribute_schema_cache) keyed by its own fingerprint -- which
# hashes the *entire* AttributeMiningConfig, not just growth_rate/max_depth --
# so this sweep is idempotent -- rerunning it only mines whatever isn't
# already cached, regardless of which block below produced the existing entry.
#
# Two grids:
#   1. The original growth_rate x max_depth x granularity grid (unchanged --
#      still 9*21=189 schemas, same cache entries as before; see the sizing
#      note in that block below).
#   2. A second, additive grid testing class_weight/min_samples_leaf --
#      neither has ever been varied anywhere in this codebase's mining
#      pipeline before. One axis at a time (not crossed with each other), at
#      MIN_GROWTH_RATE_FIXED and every MAX_DEPTHS x MINE_FRACS point, to stay
#      a cheap first look rather than a combinatorial blow-up. growth_rate is
#      held fixed here (not swept) since config_selection.ipynb's
#      parameter_importance already found it negligible for model precision
#      (eta^2~=0.0002) -- see notebooks/inspect_cscas_dataset.ipynb section 17
#      for the data-level distribution behind that finding and for whether
#      MIN_GROWTH_RATE_FIXED itself is still the right value to hold constant.
#
# Usage:
#   src/thesis/shell-scripts/sweep_attribute_schema.sh
#
# Edit the arrays below to change either grid or the scenario(s).

set -uo pipefail

# Don't rely on the caller's shell already having `thesis` active -- activate
# it explicitly so this script works the same from a cron job, CI, or a
# terminal that's sitting in base/another env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis

SCENARIOS=(cscas)
MIN_GROWTH_RATES=(3.0 4.0 5.0)
MAX_DEPTHS=(3 4 5)
MINE_FRACS=(0.1 0.2 0.33 0.5 1.0)

# Grid 2 (class_weight / min_samples_leaf) axis values and the growth_rate
# they're held fixed at -- see the module docstring above.
MIN_GROWTH_RATE_FIXED=3.0
CLASS_WEIGHTS_NEW=(none)
MIN_SAMPLES_LEAVES_NEW=(10 40)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="$REPO_ROOT/artifacts/logs/attribute_schema_sweep"
mkdir -p "$LOG_DIR"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"

total=0
failed=()

# One mining call, tagged/logged by every axis it varies (growth_rate,
# max_depth, mine_frac, class_weight, min_samples_leaf) so two combinations
# that differ only in the newer axes never share a log filename and silently
# clobber each other's log via the `>` redirect below.
mine_one() {
  local scenario="$1" growth_rate="$2" max_depth="$3" mine_frac="$4" class_weight="$5" min_samples_leaf="$6"
  total=$((total + 1))
  local tag="${scenario}_gr${growth_rate}_md${max_depth}_cw${class_weight}_msl${min_samples_leaf}_mf${mine_frac}"
  local log_file="$LOG_DIR/${RUN_TS}_${tag}.log"

  echo "[$total] $tag"
  python "$REPO_ROOT/src/thesis/scripts/mining/mine_attribute_schema.py" \
    "$scenario" \
    --min-growth-rate "$growth_rate" \
    --max-depth "$max_depth" \
    --mine-frac "$mine_frac" \
    --class-weight "$class_weight" \
    --min-samples-leaf "$min_samples_leaf" \
    --windowed \
    >"$log_file" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$tag")
  else
    grep -E "windowed|window .*/.*\]" "$log_file" | tail -n 3 | sed 's/^/    /'
  fi
}

# Grid 1: the original growth_rate x max_depth x granularity sweep, at the
# default class_weight=balanced / min_samples_leaf=20 -- unchanged from
# before, so every one of these ~189 schemas stays a cache hit.
for scenario in "${SCENARIOS[@]}"; do
  for growth_rate in "${MIN_GROWTH_RATES[@]}"; do
    for max_depth in "${MAX_DEPTHS[@]}"; do
      for mine_frac in "${MINE_FRACS[@]}"; do
        mine_one "$scenario" "$growth_rate" "$max_depth" "$mine_frac" balanced 20
      done
    done
  done
done

# Grid 2: class_weight / min_samples_leaf, one axis at a time, growth_rate
# fixed -- see the module docstring above.
for scenario in "${SCENARIOS[@]}"; do
  for max_depth in "${MAX_DEPTHS[@]}"; do
    for mine_frac in "${MINE_FRACS[@]}"; do
      for class_weight in "${CLASS_WEIGHTS_NEW[@]}"; do
        mine_one "$scenario" "$MIN_GROWTH_RATE_FIXED" "$max_depth" "$mine_frac" "$class_weight" 20
      done
      for min_samples_leaf in "${MIN_SAMPLES_LEAVES_NEW[@]}"; do
        mine_one "$scenario" "$MIN_GROWTH_RATE_FIXED" "$max_depth" "$mine_frac" balanced "$min_samples_leaf"
      done
    done
  done
done

echo
echo "============================================================"
echo "  SWEEP SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed combinations:"
  for tag in "${failed[@]}"; do
    echo "  - $tag"
  done
  exit 1
fi

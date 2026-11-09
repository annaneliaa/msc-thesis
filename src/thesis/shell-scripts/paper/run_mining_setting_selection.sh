#!/usr/bin/env bash
#
# Paper Check 1 (mining settings picked with knowledge of the test data?):
# re-select the two-tree settings on W0 only -- 80/20 time split of the CSCAS
# baseline train period, 54-point grid (max_depth {1,2} x max_depth_attack
# {2,3,4} x min_samples_leaf {5,10,20} x max_diverse_rounds {1,2,3}), RF x 5
# seeds per point, scored on tuning-val. Mined schemas are cached, so a rerun
# only refits the RFs. stage2_grid.csv is rewritten after every grid point.
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_mining_setting_selection.sh
# Smoke test (2 grid points, 1 seed):
#   SMOKE=1 SEEDS="0" src/thesis/shell-scripts/paper/run_mining_setting_selection.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV="${THESIS_CONDA_ENV:-thesis}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null \
    || echo "  [warn] 'conda activate $CONDA_ENV' failed -- using $(command -v python)" >&2
fi

if ! python -c "import thesis, sklearn, numpy, pandas" 2>/dev/null; then
  echo "FATAL: active python ($(command -v python)) can't import the core deps." \
       "Set THESIS_CONDA_ENV to the right env." >&2
  exit 1
fi

GRAN="${GRAN:-0.05}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STAGE1_FULL="${STAGE1_FULL:-1}"  # 1 = also the full-timeline Stage-1 curve (read-only, a few minutes)
SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"

LOG_DIR="$REPO_ROOT/artifacts/logs/mining_setting_selection"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S).log"

# shellcheck disable=SC2206
cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_mining_setting_selection.py"
  --gran "$GRAN" --seeds $SEEDS)
[[ "$STAGE1_FULL" -eq 1 ]] && cmd+=(--stage1-full-timeline)
[[ "$SMOKE" -eq 1 ]] && cmd+=(--smoke)
[[ "$FORCE" -eq 1 ]] && cmd+=(--force)

echo "Running: ${cmd[*]}"
echo "Log: $log_file"
if "${cmd[@]}" >"$log_file" 2>&1; then
  grep -E "Saved →|Mirrored →" "$log_file" | sed 's/^/  /'
else
  echo "FAILED — see $log_file"
  exit 1
fi

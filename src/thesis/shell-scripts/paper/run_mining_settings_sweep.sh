#!/usr/bin/env bash
#
# Settings-sensitivity check for the mining comparison, CSCAS (HANDOFF
# "Deferred" 2): Eclat / contrast tree / RIPPER at the low, default and high
# value of their main knob (mining_settings_sweep.SETTINGS_GRID), walk-forward
# as in the main comparison, then the spread and the best-vs-worst stress test.
# One process; mining is cached per setting.
#
# AIT-ADS uses the parallel launcher instead:
#   SETTINGS_VARIANT=low  src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh
#   SETTINGS_VARIANT=high src/thesis/shell-scripts/paper/run_mining_comparison_ait_ads.sh
# then run_mining_settings_sweep.py --summarize (see its docstring).
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_mining_settings_sweep.sh
# Smoke test:
#   MAX_STEPS=2 SEEDS=0 src/thesis/shell-scripts/paper/run_mining_settings_sweep.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

CONDA_ENV="${THESIS_CONDA_ENV:-thesis}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null \
    || echo "  [warn] 'conda activate $CONDA_ENV' failed -- using $(command -v python)" >&2
fi
if ! python -c "import thesis, sklearn, numpy, pandas, wittgenstein" 2>/dev/null; then
  echo "FATAL: active python ($(command -v python)) can't import the core deps." >&2
  exit 1
fi

VARIANTS="${VARIANTS:-low default high}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MAX_STEPS="${MAX_STEPS:-}"

LOG_DIR="$REPO_ROOT/artifacts/logs/mining_settings_sweep"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/cscas_$(date -u +%Y%m%d_%H%M%S).log"

# shellcheck disable=SC2206
cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_mining_settings_sweep.py"
  --dataset cscas --variants $VARIANTS --seeds $SEEDS)
[[ -n "$MAX_STEPS" ]] && cmd+=(--max-steps "$MAX_STEPS")

echo "Running: ${cmd[*]}"
echo "Log: $log_file"
if "${cmd[@]}" >"$log_file" 2>&1; then
  grep -E "Saved →" "$log_file" | sed 's/^/  /'
else
  echo "FAILED — see $log_file"
  exit 1
fi

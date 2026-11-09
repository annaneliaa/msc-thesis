#!/usr/bin/env bash
#
# Paper Check 3 (signature-derived item leakage), steps 2-3: walk-forward
# ablation on CSCAS -- (a) Base(5), (b) Base(5)+mining on all items,
# (c) Base(5)+mining without signature-derived items -- RF x 5 seeds,
# baseline_split windows at gran 0.05 (mine+fit on Wi, test on Wi+1).
#
# Arm (b)'s schemas share the rolling-walk-forward mining cache (same
# setting, same slice tags); arm (c) mines fresh at every step on its first
# run (its exclude set is part of the cache fingerprint), cached after.
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_signature_leakage.sh
# Smoke test:
#   MAX_STEPS=1 SEEDS="0" src/thesis/shell-scripts/paper/run_signature_leakage.sh

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
ARMS="${ARMS:-a b c}"
MODEL="${MODEL:-rf}"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/monitor_eda_mining_setting.yaml"
MAX_STEPS="${MAX_STEPS:-}"   # empty = full walk
FORCE="${FORCE:-0}"

LOG_DIR="$REPO_ROOT/artifacts/logs/signature_leakage"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S).log"

# shellcheck disable=SC2206
cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_signature_leakage.py"
  --gran "$GRAN" --seeds $SEEDS --arms $ARMS --model "$MODEL"
  --mining-settings "$MINING_SETTINGS")
[[ -n "$MAX_STEPS" ]] && cmd+=(--max-steps "$MAX_STEPS")
[[ "$FORCE" -eq 1 ]] && cmd+=(--force)

echo "Running: ${cmd[*]}"
echo "Log: $log_file"
if "${cmd[@]}" >"$log_file" 2>&1; then
  grep -E "Saved →|Mirrored →" "$log_file" | sed 's/^/  /'
else
  echo "FAILED — see $log_file"
  exit 1
fi

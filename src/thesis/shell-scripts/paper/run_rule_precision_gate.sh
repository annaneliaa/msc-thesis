#!/usr/bin/env bash
#
# Paper Check 2 (precision gate on mined rules): walk-forward gate off
# (arm b) vs. gate on at g = 1.5 / 2 / 3 (arms b_g1.5, b_g2, b_g3) on the
# same mined schema per step -- RF x 5 seeds, baseline_split windows at
# gran 0.05 (mine+fit on Wi, test on Wi+1). Every arm shares one mining per
# step (and the cache with Check 3's arm b), so the gate arms only add RF
# fits. rules.csv logs each mined rule's attack rate on Wi and Wi+1.
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_rule_precision_gate.sh
# Smoke test:
#   MAX_STEPS=1 SEEDS="0" src/thesis/shell-scripts/paper/run_rule_precision_gate.sh

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
ARMS="${ARMS:-b b_g1.5 b_g2 b_g3}"
MODEL="${MODEL:-rf}"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/monitor_eda_mining_setting.yaml"
MAX_STEPS="${MAX_STEPS:-}"   # empty = full walk
FORCE="${FORCE:-0}"

LOG_DIR="$REPO_ROOT/artifacts/logs/rule_precision_gate"
mkdir -p "$LOG_DIR"
log_file="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S).log"

# shellcheck disable=SC2206
cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_rule_precision_gate.py"
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

#!/usr/bin/env bash
#
# Paper Check 1, step 5: the W0-selected mining setting differs from Table
# E.13 (attack-tree depth 2 instead of 4, 1 round, leaf 20), so run the full
# walk-forward with each and compare F1 / PR-AUC / FP/day / #features.
# Settings: src/thesis/configs/check1_step5_mining_settings.yaml. Per setting,
# one run_signature_leakage.py walk-forward (baseline_split, gran 0.05, RF x 5
# seeds) with arms b (gate off) and b_g2 (Check 2's precision gate, g = 2),
# which share each step's mining. Output: artifacts/experiments/check1_step5/
# <ts>/ per setting (config.json names the setting), mirrored to
# results/paper/check1-step5/. Analysis: notebooks/paper/01_mining_setting_
# selection.ipynb, step-5 section.
#
# Usage (on the DGX):
#   src/thesis/shell-scripts/paper/run_check1_step5.sh
# Smoke test:
#   MAX_STEPS=1 SEEDS="0" src/thesis/shell-scripts/paper/run_check1_step5.sh

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
ARMS="${ARMS:-b b_g2}"
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/check1_step5_mining_settings.yaml"
SETTINGS=(${SETTINGS:-gr3_md1_mda4_rounds2 gr3_md1_mda2_msl20_rounds1})
MAX_STEPS="${MAX_STEPS:-}"
FORCE="${FORCE:-0}"

LOG_DIR="$REPO_ROOT/artifacts/logs/check1_step5"
mkdir -p "$LOG_DIR"
failed=()

for setting in "${SETTINGS[@]}"; do
  log_file="$LOG_DIR/$(date -u +%Y%m%d_%H%M%S)_${setting}.log"
  # shellcheck disable=SC2206
  cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_signature_leakage.py"
    --gran "$GRAN" --seeds $SEEDS --arms $ARMS --model rf
    --mining-settings "$MINING_SETTINGS" --mining-setting "$setting"
    --experiment check1_step5)
  [[ -n "$MAX_STEPS" ]] && cmd+=(--max-steps "$MAX_STEPS")
  [[ "$FORCE" -eq 1 ]] && cmd+=(--force)

  echo "[$setting] ${cmd[*]}"
  echo "  log: $log_file"
  if "${cmd[@]}" >"$log_file" 2>&1; then
    grep -E "Saved →|Mirrored →" "$log_file" | sed 's/^/  /'
  else
    echo "  FAILED — see $log_file"
    failed+=("$setting")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED settings: ${failed[*]}"
  exit 1
fi
echo "STEP5_DONE"

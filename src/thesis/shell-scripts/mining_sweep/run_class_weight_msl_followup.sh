#!/usr/bin/env bash
#
# Follow-up sweep: class_weight x depth + min_samples_leaf, added to
# sweep_attribute_schema.py's grid to close the last independence gaps in
# attribute_mining_sweep_eda.ipynb (§3.6 / §5.3 / §5.4).
#
# Runs a --dry-run --repair-missing check first and ABORTS unless it reports
# "0 existing run_dir but stale fingerprint" -- i.e. every existing result is
# left untouched and only the genuinely-new (config, window) combos mine.
# Then runs the real sweep with --repair-missing (which skips anything that
# already has a run_dir, independent of the fingerprint cache).
#
# Usage:  src/thesis/shell-scripts/mining_sweep/run_class_weight_msl_followup.sh [--workers N]

set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT="$REPO_ROOT/src/thesis/scripts/mining_sweep/sweep_attribute_schema.py"
LOG_DIR="$REPO_ROOT/artifacts/logs/attribute_schema_sweep"
mkdir -p "$LOG_DIR"
TS="$(date -u +%Y%m%d_%H%M%S)"
DRY_LOG="$LOG_DIR/${TS}_followup_dryrun.log"
RUN_LOG="$LOG_DIR/${TS}_followup_run.log"

echo "[1/2] dry-run safety check -> $DRY_LOG"
python "$SCRIPT" --dry-run --repair-missing "$@" 2>&1 | tee "$DRY_LOG"

if ! grep -qE "genuinely new \(no run_dir\), 0 existing run_dir but stale" "$DRY_LOG"; then
  echo "ABORT: dry-run did not confirm existing results are untouched. See $DRY_LOG"
  exit 1
fi

echo
echo "[2/2] real sweep (--repair-missing) -> $RUN_LOG"
python "$SCRIPT" --repair-missing "$@" 2>&1 | tee "$RUN_LOG"

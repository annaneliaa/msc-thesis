#!/usr/bin/env bash
#
# Rolling / Walk-Forward Evaluation (Experiment 3): for each scenario, runs
# run_rolling_walk_forward.py over the parameter grid -- every entry in
# MINING_SETTINGS below (a mining-grid YAML in the screening_mining_settings.yaml
# format) crossed with GRANULARITIES x MODELS, plus a baseline row per
# (granularity, model). That YAML is the single input: no feasible-config
# CSV, no notebook export step, no real-evaluation ranking. Edit the YAML
# (or point MINING_SETTINGS at a different one) to change what runs. For
# each resulting config, walks the timeline one window at a time,
# mining+fitting from scratch on the full window Wi and evaluating on W(i+1)
# at every step -- the "always retrain" anchor, contrasted against
# run_temporal_decay.sh's frozen-model decay curve. Mined schemas are
# cached, so rerunning this script only (re)mines whatever isn't already
# cached. Tracks SHAP/LIME importances at every step too (background from
# Wi, explained sample from W(i+1) -- both step-local, redrawn every step
# since nothing is frozen across steps the way Exp2's W_src is).
#
# Usage:
#   src/thesis/shell-scripts/system_eval/run_rolling_walk_forward.sh
#
# Edit the variables below to change the scenario(s), models, granularities,
# mining grid, or threshold mode.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Best-effort conda activation. Default env "thesis" (a dev box); override
# with THESIS_CONDA_ENV for a context where it's named differently -- e.g.
# inside a container whose deps live in base:
#   THESIS_CONDA_ENV=base bash run_rolling_walk_forward.sh
# A missing conda or missing env is only a warning: fall through to whatever
# `python` is already active. PYTHONPATH above means the package needn't be
# pip-installed (run_rolling_walk_forward.py also self-inserts src/ as a
# backstop).
CONDA_ENV="${THESIS_CONDA_ENV:-thesis}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null \
    || echo "  [warn] 'conda activate $CONDA_ENV' failed -- using $(command -v python)" >&2
fi

if ! python -c "import thesis, sklearn, numpy, pandas" 2>/dev/null; then
  echo "FATAL: active python ($(command -v python)) can't import the core deps" \
       "(thesis/sklearn/numpy/pandas). Set THESIS_CONDA_ENV to the right env." >&2
  exit 1
fi

SCENARIOS=(cscas)
# Trimmed to the "two_tree" mining point (md1/mda4) at diverse-rounds 2 and 3
# only -- see that file's header. Kept in sync with run_temporal_decay.sh so
# both experiments score the exact same schemas. Both entries share their
# contrast/tree values (and names) with screening_mining_settings.yaml's
# full 10-point grid, so this doesn't invalidate or bypass the mining cache.
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/temporal_decay_two_tree_grid.yaml"
GRANULARITIES=(0.1 0.05)  # kept in sync with run_temporal_decay.sh: 0.1
                          # matches the CSCAS dataset's own train-window
                          # size, 0.05 zooms in. Unlike Exp2's baseline_split
                          # mode, mining here happens on every window Wi, so
                          # a finer granularity does cost more mining (no
                          # free lunch the way it is in run_temporal_decay.sh).
# Every model is crossed with every (grid setting x granularity). xgboost and
# rf are both supervised tree ensembles, fit fresh on each window Wi's full
# encoding, class_weight/scale_pos_weight balanced -- see
# experiments._shared.fit_scored_model. No one-class models in this setup,
# so THRESHOLD_MODE="fixed" is just the flat 0.5 operating point for both.
MODELS=(xgboost rf)
THRESHOLD_MODE="fixed"  # or "calibrated_recall" -- keep in sync with run_temporal_decay.sh
CALIBRATED_RECALL_TARGET="0.90"  # only used when THRESHOLD_MODE=calibrated_recall
# CSCAS_FULL=1 adds one cscas_full feature-set row per (granularity, model):
# the CSCAS paper's own full feature set (5 base cols + SCAS + Similarity +
# SignatureIDSimilarity + 33 attr-similarity columns). A non-deployable
# *reference ceiling* -- SCAS and the offline *Similarity scores can't be
# computed for a fresh alert. SCAS is dropped for one-class models (none in
# MODELS above, but the flag matters if you add one back). CSCAS only.
CSCAS_FULL=1  # 0 to skip the cscas_full arm
# CSCAS_FULL_SYMBOLIC=1 adds a cscas_full_symbolic arm: the full CSCAS columns
# PLUS a schema mined on window Wi (shared base columns encoded once). One
# row per (mining setting, granularity, model) -- as many configs as the
# symbolic arm.
CSCAS_FULL_SYMBOLIC=1  # 0 to skip
# SHAP/LIME per step, for both models (xgboost and rf are both classifiers
# with analytic TreeExplainer SHAP -- no one-class model in MODELS above, so
# ONECLASS_SHAP is a no-op here; left in place for when one is added back).
# Background is drawn from Wi (what the model was just fit on), the
# explained sample from W(i+1) (what the metrics are scored on) -- both
# redrawn fresh every step.
COMPUTE_EXPLANATIONS=1  # 0 to skip SHAP/LIME entirely (metrics only)
ONECLASS_SHAP=0         # 1 to also compute (slow) SHAP for any one-class model in MODELS
EXPLAIN_SAMPLE_N=50
LIME_NUM_SAMPLES=1000
# FORCE=1 ignores any cached mined schema and re-mines from scratch for every
# (mining setting, feature_set, window) in this run -- use when you want a
# clean mine rather than trusting whatever's already in the cache (e.g.
# after a mining code change, or to rule out a stale/corrupted cache entry).
# Slower: every symbolic/cscas_full_symbolic config pays the full mining
# cost again instead of a cache hit, at every window (not just once, unlike
# Exp2's single W_src mine).
FORCE=0

# If explanations are on but shap/lime aren't importable in this env, drop
# to metrics-only rather than failing the whole run partway through.
if [[ "$COMPUTE_EXPLANATIONS" -eq 1 ]] && ! python -c "import shap, lime" 2>/dev/null; then
  echo "  [warn] shap/lime not importable -- running metrics only" \
       "(COMPUTE_EXPLANATIONS=0)" >&2
  COMPUTE_EXPLANATIONS=0
fi

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
    --models "${MODELS[@]}" \
    --threshold-mode "$THRESHOLD_MODE" \
    --explain-sample-n "$EXPLAIN_SAMPLE_N" \
    --lime-num-samples "$LIME_NUM_SAMPLES")
  if [[ "$THRESHOLD_MODE" == "calibrated_recall" ]]; then
    cmd+=(--calibrated-recall-target "$CALIBRATED_RECALL_TARGET")
  fi
  if [[ "$COMPUTE_EXPLANATIONS" -eq 0 ]]; then
    cmd+=(--no-explanations)
  fi
  if [[ "${ONECLASS_SHAP:-0}" -eq 1 ]]; then
    cmd+=(--oneclass-shap)
  fi
  if [[ "${CSCAS_FULL:-0}" -eq 1 ]]; then
    cmd+=(--cscas-full)
  fi
  if [[ "${CSCAS_FULL_SYMBOLIC:-0}" -eq 1 ]]; then
    cmd+=(--cscas-full-symbolic)
  fi
  if [[ "${FORCE:-0}" -eq 1 ]]; then
    cmd+=(--force)
  fi

  "${cmd[@]}" >"$log_file" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$scenario")
  else
    grep -E "Rolling walk-forward results|Saved →|Mirrored →" "$log_file" | tail -n 7 | sed 's/^/    /'
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

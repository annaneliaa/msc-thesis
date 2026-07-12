#!/usr/bin/env bash
#
# Entry point for the cscas attribute-mining parameter sweep. The actual
# grids (growth_rate x max_depth x granularity, class_weight/
# min_samples_leaf, min_attack_coverage/min_benign_coverage) and sweep logic
# live in scripts/mining/sweep_attribute_schema.py -- this file used to
# duplicate them in bash and shell out to mine_attribute_schema.py once per
# combination (1176 subprocess invocations, each reloading the full
# ~1.4M-row/~7GB cscas_alert_groups cache from scratch for no benefit -- the
# schema-cache lookup itself is a cheap file `.stat()`, not a data load).
# That's gone: the Python driver loads alert_groups once and runs the whole
# grid through a thread pool instead. Keeping this .sh as a thin wrapper so
# the existing invocation path (terminal, cron, CI) still works, without two
# copies of the grids to keep in sync.
#
# Usage:
#   src/thesis/shell-scripts/sweep_attribute_schema.sh
#   src/thesis/shell-scripts/sweep_attribute_schema.sh --workers 3
#
# Edit the grids in scripts/mining/sweep_attribute_schema.py, not here.

set -uo pipefail

# Don't rely on the caller's shell already having `thesis` active -- activate
# it explicitly so this script works the same from a cron job, CI, or a
# terminal that's sitting in base/another env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

exec python "$REPO_ROOT/src/thesis/scripts/mining/sweep_attribute_schema.py" "$@"

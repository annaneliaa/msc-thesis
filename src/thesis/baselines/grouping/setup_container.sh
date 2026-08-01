#!/usr/bin/env bash
# One-time setup for a fresh container/machine before running
# run_grouping.sh -- installs this project (and, in a separate conda env,
# graph-tool + the alertbert package) into whatever Python environments
# run_grouping.sh's steps expect to already have it. Written for the
# Dockerfile.dgx workflow (see that file's header for the full docker
# build/run/exec sequence) but has no Docker-specific assumptions itself --
# works on any machine with the same base env (torch already correctly
# installed for the target GPU) plus, optionally, a `thesis-alertbert`
# conda env containing graph-tool.
#
# Deliberately does NOT `pip install torch`/`pip install numpy` in the base
# env: this project's pyproject.toml lists "torch" unpinned and
# "numpy>=1.0,<2.0", and a plain `pip install -e .` would let pip try to
# satisfy/replace both -- risking clobbering whatever GPU-matched torch
# build the base image already has (this matters a lot on very recent
# hardware, where a generic PyPI torch wheel may not even exist yet or may
# not be built for this exact CUDA/driver combination) and force-
# downgrading numpy to 1.x against a base image that ships 2.x. Instead:
# `pip install --no-deps -e .` for the editable install itself, then every
# OTHER dependency from pyproject.toml (parsed directly from it, so this
# never drifts out of sync by hand) installed explicitly, skipping torch
# and numpy specifically. Verify torch still works after this script runs
# (it prints torch.cuda.is_available() itself) -- if pyproject.toml's own
# dependency list ever changes to need a newer numpy than the base image
# ships, that's a real conflict to resolve deliberately, not paper over.
#
# The `thesis-alertbert` conda env (if present) is different: it starts
# with no torch at all, so installing one there is expected, not a clobber
# risk -- but since it's a fresh env on possibly-very-new hardware, there's
# no guarantee pip finds a CUDA-enabled wheel for it either; this script
# verifies and reports what it got rather than assuming success.
#
# Run once per fresh container (idempotent -- safe to re-run):
#   bash src/thesis/baselines/grouping/setup_container.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "=== repo root: $REPO_ROOT ==="

echo ""
echo "--- git submodules (external/AlertBERT) ---"
git submodule update --init --recursive

# Every pyproject.toml dependency except torch/numpy -- see header comment
# for why those two are deliberately left alone here. Parsed directly from
# pyproject.toml (not hand-copied) so this can't silently drift out of sync
# with it.
non_torch_numpy_deps() {
    python3 - <<'PYEOF'
import re

try:
    import tomllib  # stdlib, Python >=3.11
except ImportError:
    import tomli as tomllib  # pip install tomli first on Python 3.10

with open("pyproject.toml", "rb") as f:
    deps = tomllib.load(f)["project"]["dependencies"]

skip = re.compile(r"^(torch|numpy)([<>=!~\s]|$)")
for dep in deps:
    if not skip.match(dep):
        print(dep)
PYEOF
}

echo ""
echo "--- base env: installing thesis (editable, --no-deps) ---"
pip install --no-cache-dir --no-deps -e .

echo ""
echo "--- base env: installing every other pyproject.toml dependency ---"
mapfile -t deps < <(non_torch_numpy_deps)
pip install --no-cache-dir "${deps[@]}"

echo ""
echo "--- base env: torch/CUDA check ---"
python3 -c "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"

if command -v conda >/dev/null 2>&1 && conda env list | grep -q "^thesis-alertbert "; then
    echo ""
    echo "--- thesis-alertbert conda env: installing torch + alertbert + thesis ---"
    # Fresh env, no existing torch to protect -- a normal (non --no-deps)
    # install so its own CUDA-runtime companion packages come along too.
    conda run -n thesis-alertbert --no-capture-output \
        pip install --no-cache-dir torch tensordict

    # Our own packages: --no-deps, same reasoning as the base env above --
    # dependency resolution is handled explicitly via non_torch_numpy_deps
    # instead, so numpy (already provided by graph-tool's own conda install
    # in this env) is never touched by pip here either.
    conda run -n thesis-alertbert --no-capture-output \
        pip install --no-cache-dir --no-deps -e external/AlertBERT
    conda run -n thesis-alertbert --no-capture-output \
        pip install --no-cache-dir --no-deps -e .

    mapfile -t deps < <(non_torch_numpy_deps)
    conda run -n thesis-alertbert --no-capture-output \
        pip install --no-cache-dir "${deps[@]}"

    echo ""
    echo "--- thesis-alertbert conda env: torch/CUDA + graph-tool check ---"
    conda run -n thesis-alertbert --no-capture-output python3 -c \
        "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"
    conda run -n thesis-alertbert --no-capture-output python3 -c \
        "import graph_tool; print(f'graph_tool {graph_tool.__version__}')"
else
    echo ""
    echo "[skip] no 'thesis-alertbert' conda env found -- alertbert_sweep.py won't run." \
        "Build from Dockerfile.dgx, or create it yourself:" \
        "mamba create -n thesis-alertbert -c conda-forge python=3.11 graph-tool"
fi

echo ""
echo "=== setup complete -- see src/thesis/baselines/grouping/run_grouping.sh ==="

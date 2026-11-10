"""Load the authors' MCTSExtent code (Mathonat et al., KAIS 2021) at a pinned
commit, without copying it into this repository.

Why not a normal dependency: the upstream repo has no license (so its code
is never committed here; external/MCTSExtent/ is gitignored), its setup.py
installs a package name that doesn't exist ("mctsextend"), and it pins
numpy 1.15 / pandas 0.23. Instead ensure_mctsextent() clones the repo at
COMMIT into external/MCTSExtent, and load_launch_mcts() puts it on sys.path
(its modules are top-level: general, mctsextent, seqscout) and imports
mctsextent.main.launch_mcts, the in-memory entry point.

The one compatibility patch, applied in memory (upstream files untouched):
the code calls random.sample() on sets, which Python >= 3.11 rejects. The
shim converts a set population to a sorted list: items sort directly, MCTS
nodes sort by their pattern. Sorting (rather than list(set)) also makes a
seeded run reproducible; with list(set) the node order follows memory
addresses and two identical runs gave different patterns. Upstream's random
choices are otherwise unchanged.
"""

from __future__ import annotations

import random as _random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_URL = "https://github.com/Romathonat/MCTSExtent.git"
COMMIT = "7fe135d81416ce8cc242bf3f696657e511031e1c"
VENDOR_DIR = Path(__file__).resolve().parents[5] / "external" / "MCTSExtent"


def ensure_mctsextent(path: Path = VENDOR_DIR, fetch: bool = True) -> Path:
    """Return the checkout at COMMIT, cloning it first if missing (fetch=True)."""
    if not (path / "mctsextent" / "main.py").exists():
        if not fetch:
            raise FileNotFoundError(
                f"MCTSExtent not found at {path}; run ensure_mctsextent() "
                "(needs network) or clone REPO_URL at COMMIT there"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO_URL, str(path)], check=True)
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(path), "checkout", "-q", COMMIT], check=True)
    return path


def _intent_key(node) -> tuple:
    """Order MCTS nodes by their pattern (unique per node: upstream keeps one
    node per pattern in node_hashmap); the root's intent is None."""
    intent = getattr(node, "intent", None)
    return tuple(tuple(sorted(step)) for step in intent) if intent else ()


def _sample(population, k):
    if isinstance(population, (set, frozenset)):
        try:
            population = sorted(population)
        except TypeError:
            # Sets of Node objects (update()): iteration order follows id(),
            # and the order nodes are visited changes the result, so sort
            # by pattern to make a seeded run reproducible.
            population = sorted(population, key=_intent_key)
    return _random.sample(population, k)


# A stand-in for the `random` module inside upstream's modules: every
# attribute is the real module's (same global RNG state, so random.seed()
# still controls it), except sample().
_RANDOM_SHIM = SimpleNamespace(
    **{n: getattr(_random, n) for n in dir(_random) if not n.startswith("_")}
)
_RANDOM_SHIM.sample = _sample


def load_launch_mcts(path: Path = VENDOR_DIR, fetch: bool = True):
    """Import upstream's launch_mcts with the random.sample shim applied."""
    root = str(ensure_mctsextent(path, fetch=fetch))
    if root not in sys.path:
        sys.path.insert(0, root)
    import mctsextent.main as mcts_main  # noqa: E402  (upstream, on sys.path)
    import mctsextent.node as mcts_node  # noqa: E402

    mcts_main.random = _RANDOM_SHIM
    mcts_node.random = _RANDOM_SHIM
    return mcts_main.launch_mcts

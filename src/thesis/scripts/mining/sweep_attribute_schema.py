"""
Pre-mine attribute schemas for cscas across a grid of mining parameters --
single-process replacement for shell-scripts/sweep_attribute_schema.sh's old
approach of shelling out to mine_attribute_schema.py once per combination.

That approach paid the full ~1.4M-row/~7GB cscas_alert_groups load (parsing
alert_groups_raw.json into AlertGroup objects) on *every* invocation,
regardless of whether the resulting schema was a cache hit -- the schema
cache lookup itself (mine_or_reuse_attribute_schema) is a cheap file
`.stat()`, not a data load, so at this sweep's scale (1176 combinations) that
reload dominated wall-clock time for essentially no benefit. This script
loads alert_groups once and runs the full parameter grid through a thread
pool, calling mine_or_reuse_attribute_schema directly per (config, window)
pair.

Window-slice files are materialized and deleted per (granularity, win_idx)
GROUP, not once for the whole sweep. An earlier version of this script
materialized all 25 window-slice files up front and never deleted them --
several of MINE_FRACS's granularities each cover the *entire* timeline
across their windows, so that piled up ~6x the dataset's size in redundant
JSON on disk (worst case: mine_frac=1.0's single window duplicates the
entire ~3GB raw file by itself) and exhausted disk space mid-run. This is
exactly the failure mode thesis.mining.window_schema_cache's docstring
describes -- fixed the same way here: jobs are grouped by (mine_frac,
win_idx), the group's slice file is materialized once, every config for
that window runs through the thread pool against it, and the slice file is
deleted before moving to the next group -- so at most one window's slice
file exists on disk at a time (worst case ~3GB transient), not all 25 at
once. (window_schema_cache.py's own per-call materialize/delete wasn't used
directly here because its slice-file path is keyed only by (gran, win_idx,
tag), not by config -- with this sweep's per-window fan-out across dozens of
configs, concurrent callers sharing one window would race on that file's
delete. Grouping and deleting once per window, after every config sharing
it has finished, avoids that race entirely.)

Threads, not processes: a process pool would reintroduce the reload problem
(each worker process needs its own copy of alert_groups unless using
multiprocessing's "fork" start method for copy-on-write sharing -- and
CPython's docs flag fork as unsafe on macOS since 3.8, a real risk here, not
hypothetical, once matplotlib or other Cocoa-touching libraries are loaded
in the parent). Threads share the loaded alert_groups for free and are safe
here specifically because sklearn's tree fit and most numpy/pandas array
ops release the GIL, and attribute_schema_cache._INDEX_LOCK already exists
for exactly this "mining concurrently from a thread pool" scenario (see its
docstring, added for temporal_decay.py).

The one thing that is NOT thread-safe and needed a real fix rather than
just wrapping the old per-call code in a pool: contextlib.redirect_stdout
mutates sys.stdout globally, so two threads redirecting to two different
per-combination log files at once would stomp each other's target. Fixed
below with a small thread-local stdout/stderr proxy: each worker thread
points its own threading.local() slot at its log file for the duration of
its mining call, and print() (which always targets whatever sys.stdout
currently is) transparently dispatches to the calling thread's slot.

Organized by mining stage, matching attribute_mining_sweep_eda.ipynb's section
2 finding that Step 1 (contrast-set filter: growth_rate, coverage) and Step 2
(decision tree) are independent -- a Step-1 axis's effect on Step-1 metrics
doesn't depend on which Step-2 settings are in force, and vice versa,
confirmed at the raw per-window grain, not just on an averaged view. That
means each axis's own effect is fully characterized varying it alone against
a single anchor point in the *other* stage, without paying for a cross
product with that stage's own axes.

Step 2 fits *two* trees now, not one (thesis.mining.decision_tree_rule_mining
.fit_and_extract_rules, via DecisionTreeRuleConfig.max_depth /
.max_depth_attack): a single shared max_depth couldn't simultaneously
maximize attack-leaf precision and benign-leaf recall, since they move in
opposite directions as depth increases -- attribute_mining_sweep_eda.ipynb's
section 5.3 feasible-region plot found *zero* configs in the old single-tree
grid clearing both a precision floor and a recall floor at once. max_depth
is the benign-facing tree's depth (kept as the pre-existing field so a
benign-only caller, e.g. a future anomaly-detector mining pass, never needs
to touch max_depth_attack at all); max_depth_attack is the attack-facing
tree's depth, swept more densely below since precision_attack was found
non-monotonic in the old single-tree data (peaking at depth 3, worse at both
1 and 5) while recall_benign was cleanly monotonic (best shallow) -- the
benign axis needs confirming, not searching for a hidden peak.

This is a full re-mine, not an incremental addition: fitting two trees
instead of one is a different Step-2 algorithm, so none of the old
single-tree run directories are reused by anything below (they're still on
disk and still what attribute_mining_sweep_eda.ipynb's historical sections
read, just not part of what this script generates going forward).

Two grids:
  1. Step 1 (contrast-set filter): growth_rate swept alone, and
     min_attack_coverage/min_benign_coverage swept one axis at a time
     (holding the other at its 0.05 default) -- each x granularity, anchored
     at STEP2_DEFAULT_MAX_DEPTH / STEP2_DEFAULT_MAX_DEPTH_ATTACK /
     STEP2_DEFAULT_CLASS_WEIGHT.
  2. Step 2 (decision tree): max_depth swept alone, max_depth_attack swept
     alone (each holding the other tree's depth at its default), and
     class_weight swept alone -- each x granularity, anchored at
     STEP1_DEFAULT_GROWTH_RATE / STEP1_DEFAULT_ATTACK_COVERAGE /
     STEP1_DEFAULT_BENIGN_COVERAGE. min_samples_leaf is not swept at all --
     negligible on every structural metric tracked in the single-tree
     analysis (attribute_mining_sweep_eda.ipynb section 4.1), no reason to
     expect that changes under two-tree fitting.

A small growth_rate x max_depth_attack cross-check IS kept this time
(CROSS_CHECK_GROWTH_RATES x CROSS_CHECK_MAX_DEPTHS_ATTACK, 4 combos) --
unlike the single-tree script's later revision, which dropped its
cross-check as redundant with an already-exhaustively-verified grid. Here
the orthogonality argument is being carried over to a genuinely new
algorithm variant (two-tree fitting) rather than re-litigated on the same
one, so a cheap spot-check is worth it before fully trusting the one-axis-
per-stage design against it.

Every (alert_groups window file, AttributeMiningConfig) combination is
cached on disk (thesis.mining.attribute_schema_cache) exactly as before, so
this script is idempotent the same way the old one was -- rerunning it only
mines whatever isn't already cached.

`total`/the per-combination log files are counted per (config, window) pair
rather than per CLI invocation (which used to bundle every window for a
given mine_frac into one log). The grids were originally sized at 4900 raw
mining attempts (11 growth_rates x 4 depths x 6 granularities, plus 16-point
coverage grids swept per-axis) but that proved infeasible, and ~400 of those
4900 were exact duplicates anyway -- every axis's sweep holds every other
axis at a fixed anchor value, and whenever a swept axis's own grid happens
to still contain that anchor value (e.g. 0.05 in the coverage grid,
"balanced" in class_weight), the call produces a (mine_frac, win_idx,
config) triple another axis's sweep already queued, which is why the
same-looking tag could show up twice with a "cache hit" the second time --
not a fingerprinting bug, just a redundant job. `build_jobs` dedupes on that
key regardless of grid sizing.

This script has been re-scoped three times since: first coarsened to ~816
attempts while still crossing growth_rate x max_depth directly plus two
anchored sensitivity sweeps; then reorganized into the single-tree two-stage
design; now moved to the two-tree Step 2 described above -- see
STEP1_GROWTH_RATES/STEP1_COVERAGE_GRID/STEP2_MAX_DEPTHS/
STEP2_MAX_DEPTHS_ATTACK/STEP2_CLASS_WEIGHTS/CROSS_CHECK_GROWTH_RATES/
CROSS_CHECK_MAX_DEPTHS_ATTACK/MINE_FRACS below for the current grids.

Usage:
  python src/thesis/scripts/mining/sweep_attribute_schema.py
  python src/thesis/scripts/mining/sweep_attribute_schema.py --workers 8
  python src/thesis/scripts/mining/sweep_attribute_schema.py --workers 1  # serial, old behavior
  python src/thesis/scripts/mining/sweep_attribute_schema.py --repair-missing

--repair-missing mines only the (config, window) combos that don't already
have a valid run_dir on disk, and forces those past the fingerprint cache --
see scan_existing_job_keys and run_job's docstring for why: the run-name
collision this mode repairs left the *schema* correctly cached under
attribute_schema_cache (each racing job still fingerprints and registers
independently, protected by _INDEX_LOCK) even for combos whose diagnostic
run_dir got clobbered, so a normal rerun would cache-hit those combos and
never regenerate the missing config.yaml/mined_attribute_features.csv/
contrast_stats_all.csv this repo's EDA notebooks read. force=True bypasses
that cache hit for exactly the combos that need their run_dir rebuilt,
leaving the other ~95% that were never affected untouched.

Edit the grids below to change what gets swept. MAX_WORKERS's default is a
starting point, not a measured optimum -- benchmark on your own machine
before assuming a bigger number is strictly better (each concurrent
full-population (mine_frac=1.0) job briefly allocates its own X_cat/X_num,
so worker count trades wall-clock time against peak memory, on top of the
~7GB the shared alert_groups list already holds).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.mining.attribute_schema_cache import (
    compute_fingerprint_from_identity,
    mine_or_reuse_attribute_schema,
)
from thesis.paths import CACHE_DIR
from thesis.pipeline.pipeline import (
    compute_window_bounds,
    ensure_feature_manifest,
    ingest_cscas_scenario,
    load_or_build_alert_groups,
    resolve_window_alert_groups_path,
)
from thesis.schemas.mining import AttributeMiningConfig

SCENARIO = "cscas"

MINE_FRACS = [0.1, 0.25, 0.5]

# Anchors: each stage's fixed default while the *other* stage's axes vary --
# see the module docstring above for why one anchor point per stage is
# sufficient rather than a cross product. STEP1_DEFAULT_GROWTH_RATE/coverage/
# STEP2_DEFAULT_CLASS_WEIGHT/min_samples_leaf match the freeze/vary analysis
# in summaries/ATTRIBUTE_MINING_PARAMETER_ANALYSIS.md (built on the old
# single-tree data, but nothing there depended on Step 2 being one tree
# rather than two -- growth_rate/coverage never touched Step 2 either way).
# STEP2_DEFAULT_MAX_DEPTH_ATTACK=3 and STEP2_DEFAULT_MAX_DEPTH=1 are the
# single-tree analysis's best-known points for each class (precision_attack
# peaked at depth 3; recall_benign was best at the shallowest depth tested,
# 1) -- starting points for the two-tree search below, not assumed final.
STEP1_DEFAULT_GROWTH_RATE = 3.0
STEP1_DEFAULT_ATTACK_COVERAGE = 0.05
STEP1_DEFAULT_BENIGN_COVERAGE = 0.05
STEP2_DEFAULT_MAX_DEPTH = 1
STEP2_DEFAULT_MAX_DEPTH_ATTACK = 3
STEP2_DEFAULT_CLASS_WEIGHT = "balanced"
STEP2_DEFAULT_MIN_SAMPLES_LEAF = 20

# Step 1 grid (contrast-set filter).
STEP1_GROWTH_RATES = [2.0, 3.0, 5.0, 10.0]
STEP1_COVERAGE_GRID = [0.0, 0.05, 0.15, 0.30]
# min_growth_rate_attack, swept alone at the min_growth_rate (benign-facing)
# anchor -- see ContrastSetFilterConfig for why this split exists (mirrors
# max_depth/max_depth_attack: min_growth_rate stays the benign-facing
# threshold, min_growth_rate_attack is the optional attack-facing override).
# 3.0 deliberately excluded: identical filtering behavior to leaving
# min_growth_rate_attack unset (None), which every other job in this grid
# already does, but fingerprints differently -- including it would mine a
# hash-distinct duplicate of data already cached under the growth_rate sweep.
STEP1_GROWTH_RATES_ATTACK = [2.0, 5.0, 10.0]

# Step 2 grid (decision tree, two trees now -- see module docstring).
# max_depth (benign-facing) is a narrow search: recall_benign was cleanly
# monotonic in the old single-tree data, so this confirms shallow-is-best
# rather than hunting for a peak. max_depth_attack (attack-facing) is a
# denser search: precision_attack was non-monotonic (peaked at 3, worse at
# both 1 and 5), and 2/4 were never tested, so this axis has real unknowns.
STEP2_MAX_DEPTHS = [1, 2, 3]
STEP2_MAX_DEPTHS_ATTACK = [1, 2, 3, 4, 5]
STEP2_CLASS_WEIGHTS = ["balanced", "none"]
# min_samples_leaf is not swept at all here -- negligible on every
# structural metric in the single-tree analysis (attribute_mining_sweep_eda
# .ipynb section 4.1), fixed at STEP2_DEFAULT_MIN_SAMPLES_LEAF everywhere.

# Small growth_rate x max_depth_attack cross-check -- see module docstring
# for why this one script keeps a cross-check where the single-tree version
# dropped it.
CROSS_CHECK_GROWTH_RATES = [2.0, 10.0]
CROSS_CHECK_MAX_DEPTHS_ATTACK = [1, 5]

MAX_WORKERS = 6

REPO_ROOT = Path(__file__).resolve().parents[4]
LOG_DIR = REPO_ROOT / "artifacts" / "logs" / "attribute_schema_sweep"
MINING_ROOT = REPO_ROOT / "artifacts" / "mining"

# Matches both the old "symbolic_cscas_gran{g}_win{w}" run names and the
# current run_job-produced ones, which insert the full config (gr/md/cw/msl/
# ac/bc) between the scenario and "gran{g}_win{w}" -- see run_job's
# docstring. Kept in sync with attribute_mining_sweep_eda.ipynb's copy.
RUN_NAME_RE = re.compile(
    rf"^\d{{8}}_\d{{6}}_symbolic_{SCENARIO}_.*gran([\d.]+)_win(\d+)$"
)


class _ThreadLocalStream:
    """Proxy that makes print()/traceback output dispatch per-thread instead
    of globally -- see the module docstring for why plain
    contextlib.redirect_stdout is unsafe once multiple worker threads are
    each redirecting to their own log file at the same time. Install once
    (as sys.stdout/sys.stderr) before starting the pool; each worker sets
    its own `.target` for the duration of one mining call and clears it
    after, in a try/finally.
    """

    def __init__(self, fallback):
        self._local = threading.local()
        self._fallback = fallback

    def _target(self):
        return getattr(self._local, "target", None) or self._fallback

    def write(self, s: str) -> int:
        return self._target().write(s)

    def flush(self) -> None:
        self._target().flush()

    def set(self, target) -> None:
        self._local.target = target

    def clear(self) -> None:
        self._local.target = None


def build_config(
    growth_rate: float,
    max_depth: int,
    max_depth_attack: int,
    class_weight: str,
    min_samples_leaf: int,
    attack_coverage: float,
    benign_coverage: float,
    growth_rate_attack: float | None = None,
) -> AttributeMiningConfig:
    config = AttributeMiningConfig()
    config.contrast.min_attack_coverage = attack_coverage
    config.contrast.min_benign_coverage = benign_coverage
    config.contrast.min_growth_rate = growth_rate
    config.contrast.min_growth_rate_attack = growth_rate_attack
    config.tree.max_depth = max_depth
    config.tree.max_depth_attack = max_depth_attack
    config.tree.min_samples_leaf = min_samples_leaf
    config.tree.class_weight = None if class_weight == "none" else class_weight
    return config


@dataclass
class MineJob:
    tag: str
    mine_frac: float
    win_idx: int
    config: AttributeMiningConfig


def window_fingerprint(
    alert_groups_path: Path,
    mine_frac: float,
    win_idx: int,
    slice_start: int,
    slice_end: int,
    config: AttributeMiningConfig,
) -> str:
    """Fingerprint a (window, config) mining input against the *raw*,
    never-deleted alert_groups file's stat plus the slicing parameters --
    not against the per-run materialized window slice file, whose mtime
    changes every time this script re-creates it after a prior run deleted
    it (see the module docstring's disk-exhaustion fix). Letting
    mine_or_reuse_attribute_schema derive its own fingerprint from that
    disposable file defeats the cache across separate script runs even
    when nothing about the mining inputs changed -- the exact failure mode
    thesis.mining.window_schema_cache's module docstring describes, and
    solves the same way: same identity fields (tag="full" -- this script
    mines each window in full, no held-out split, like
    get_or_mine_full_window_attribute_schema), so cache entries even end up
    shared with that entry point where inputs coincide.
    """
    stat = alert_groups_path.stat()
    identity = {
        "raw_alert_groups_path": str(alert_groups_path.resolve()),
        "raw_alert_groups_size": stat.st_size,
        "raw_alert_groups_mtime": stat.st_mtime,
        "gran": mine_frac,
        "win_idx": win_idx,
        "slice_start": slice_start,
        "slice_end": slice_end,
        "tag": "full",
    }
    return compute_fingerprint_from_identity(identity, config)


def build_jobs(mine_fracs_windows: dict[float, int]) -> list[MineJob]:
    jobs: list[MineJob] = []
    # Both stages' grids hold every axis but one at the *other* stage's
    # anchor value -- whenever a swept axis's own grid also contains its
    # stage's anchor value (e.g. 0.05 sitting in STEP1_COVERAGE_GRID, or
    # STEP2_DEFAULT_MAX_DEPTH sitting in STEP2_MAX_DEPTHS), that call
    # produces the exact (mine_frac, win_idx, config) combination another
    # axis's sweep already queued. Tracked here instead of hand-tuning each
    # grid to dodge the anchor, since that's what actually caused duplicate
    # "cache hit" log lines for two different-looking tags in an earlier
    # version of this script -- same config, same window, mined/cached
    # twice for no reason.
    seen: set[tuple] = set()

    def add(
        growth_rate,
        max_depth,
        max_depth_attack,
        mine_frac,
        class_weight,
        min_samples_leaf=STEP2_DEFAULT_MIN_SAMPLES_LEAF,
        attack_coverage=0.05,
        benign_coverage=0.05,
        growth_rate_attack=None,
    ):
        key = (
            growth_rate,
            max_depth,
            max_depth_attack,
            mine_frac,
            class_weight,
            min_samples_leaf,
            attack_coverage,
            benign_coverage,
            growth_rate_attack,
        )
        if key in seen:
            return
        seen.add(key)

        config = build_config(
            growth_rate,
            max_depth,
            max_depth_attack,
            class_weight,
            min_samples_leaf,
            attack_coverage,
            benign_coverage,
            growth_rate_attack=growth_rate_attack,
        )
        # Only jobs that actually set growth_rate_attack get the extra tag
        # segment -- keeps every other job's tag (and thus its log file name)
        # unchanged from before this axis existed.
        gra_tag = f"_gra{growth_rate_attack}" if growth_rate_attack is not None else ""
        base_tag = (
            f"{SCENARIO}_gr{growth_rate}{gra_tag}_md{max_depth}_mda{max_depth_attack}_cw{class_weight}"
            f"_msl{min_samples_leaf}_ac{attack_coverage}_bc{benign_coverage}_gran{mine_frac:g}"
        )
        for win_idx in range(mine_fracs_windows[mine_frac]):
            jobs.append(
                MineJob(
                    tag=f"{base_tag}_win{win_idx}",
                    mine_frac=mine_frac,
                    win_idx=win_idx,
                    config=config,
                )
            )

    # Step 1 grid: growth_rate x granularity, at Step 2's anchor -- growth_rate's
    # own effect on Step-1 metrics doesn't depend on either tree's depth/
    # class_weight (see module docstring), so one Step-2 anchor point is
    # enough; no cross product with STEP2_MAX_DEPTHS/STEP2_MAX_DEPTHS_ATTACK
    # needed here (see the small cross-check below instead).
    for growth_rate in STEP1_GROWTH_RATES:
        for mine_frac in MINE_FRACS:
            add(
                growth_rate,
                STEP2_DEFAULT_MAX_DEPTH,
                STEP2_DEFAULT_MAX_DEPTH_ATTACK,
                mine_frac,
                STEP2_DEFAULT_CLASS_WEIGHT,
            )

    # Step 1 grid: min_attack_coverage / min_benign_coverage, one axis at a
    # time (holding the other at its 0.05 default), same Step-2 anchor.
    for mine_frac in MINE_FRACS:
        for attack_coverage in STEP1_COVERAGE_GRID:
            add(
                STEP1_DEFAULT_GROWTH_RATE,
                STEP2_DEFAULT_MAX_DEPTH,
                STEP2_DEFAULT_MAX_DEPTH_ATTACK,
                mine_frac,
                STEP2_DEFAULT_CLASS_WEIGHT,
                attack_coverage=attack_coverage,
                benign_coverage=STEP1_DEFAULT_BENIGN_COVERAGE,
            )
        for benign_coverage in STEP1_COVERAGE_GRID:
            add(
                STEP1_DEFAULT_GROWTH_RATE,
                STEP2_DEFAULT_MAX_DEPTH,
                STEP2_DEFAULT_MAX_DEPTH_ATTACK,
                mine_frac,
                STEP2_DEFAULT_CLASS_WEIGHT,
                attack_coverage=STEP1_DEFAULT_ATTACK_COVERAGE,
                benign_coverage=benign_coverage,
            )

    # Step 1 grid: min_growth_rate_attack, swept alone at the min_growth_rate
    # (benign-facing) anchor and the Step-2 anchor -- see
    # STEP1_GROWTH_RATES_ATTACK's own comment for why 3.0 isn't in this list.
    for growth_rate_attack in STEP1_GROWTH_RATES_ATTACK:
        for mine_frac in MINE_FRACS:
            add(
                STEP1_DEFAULT_GROWTH_RATE,
                STEP2_DEFAULT_MAX_DEPTH,
                STEP2_DEFAULT_MAX_DEPTH_ATTACK,
                mine_frac,
                STEP2_DEFAULT_CLASS_WEIGHT,
                growth_rate_attack=growth_rate_attack,
            )

    # Step 2 grid: max_depth (benign-facing tree) x granularity, at Step 1's
    # anchor and the attack tree's default depth -- narrow search, see
    # module docstring (recall_benign was monotonic in the old data).
    for max_depth in STEP2_MAX_DEPTHS:
        for mine_frac in MINE_FRACS:
            add(
                STEP1_DEFAULT_GROWTH_RATE,
                max_depth,
                STEP2_DEFAULT_MAX_DEPTH_ATTACK,
                mine_frac,
                STEP2_DEFAULT_CLASS_WEIGHT,
                attack_coverage=STEP1_DEFAULT_ATTACK_COVERAGE,
                benign_coverage=STEP1_DEFAULT_BENIGN_COVERAGE,
            )

    # Step 2 grid: max_depth_attack (attack-facing tree) x granularity, at
    # Step 1's anchor and the benign tree's default depth -- denser search,
    # see module docstring (precision_attack was non-monotonic).
    for max_depth_attack in STEP2_MAX_DEPTHS_ATTACK:
        for mine_frac in MINE_FRACS:
            add(
                STEP1_DEFAULT_GROWTH_RATE,
                STEP2_DEFAULT_MAX_DEPTH,
                max_depth_attack,
                mine_frac,
                STEP2_DEFAULT_CLASS_WEIGHT,
                attack_coverage=STEP1_DEFAULT_ATTACK_COVERAGE,
                benign_coverage=STEP1_DEFAULT_BENIGN_COVERAGE,
            )

    # Step 2 grid: class_weight, one axis at a time, same Step-1 anchor and
    # both trees' default depths. min_samples_leaf is not swept -- see
    # module docstring.
    for mine_frac in MINE_FRACS:
        for class_weight in STEP2_CLASS_WEIGHTS:
            add(
                STEP1_DEFAULT_GROWTH_RATE,
                STEP2_DEFAULT_MAX_DEPTH,
                STEP2_DEFAULT_MAX_DEPTH_ATTACK,
                mine_frac,
                class_weight,
                attack_coverage=STEP1_DEFAULT_ATTACK_COVERAGE,
                benign_coverage=STEP1_DEFAULT_BENIGN_COVERAGE,
            )

    # Small growth_rate x max_depth_attack cross-check -- see module
    # docstring for why this script keeps one where the single-tree version
    # dropped it (the algorithm changed; the orthogonality argument hasn't
    # been re-verified under two-tree fitting yet).
    for growth_rate in CROSS_CHECK_GROWTH_RATES:
        for max_depth_attack in CROSS_CHECK_MAX_DEPTHS_ATTACK:
            for mine_frac in MINE_FRACS:
                add(
                    growth_rate,
                    STEP2_DEFAULT_MAX_DEPTH,
                    max_depth_attack,
                    mine_frac,
                    STEP2_DEFAULT_CLASS_WEIGHT,
                )

    return jobs


def job_key(config: AttributeMiningConfig, mine_frac: float, win_idx: int) -> tuple:
    """Identity of a (config, window) mining input, independent of run_name/
    timestamp -- the same fields --repair-missing reads back out of an
    existing run_dir's config.yaml in scan_existing_job_keys, so the two are
    directly comparable."""
    return (
        config.contrast.min_growth_rate,
        config.contrast.min_growth_rate_attack,
        config.tree.max_depth,
        config.tree.max_depth_attack,
        config.tree.class_weight,
        config.tree.min_samples_leaf,
        config.contrast.min_attack_coverage,
        config.contrast.min_benign_coverage,
        mine_frac,
        win_idx,
    )


# CSVs attribute_mining_sweep_eda.ipynb reads out of every run_dir -- kept in
# sync with that notebook's cells 4/5 (mined_attribute_features.csv,
# contrast_stats_all.csv) and its Step-1-vs-survivors sections
# (contrast_survivors.csv).
_RUN_DIR_CSVS = [
    "mined_attribute_features.csv",
    "contrast_stats_all.csv",
    "contrast_survivors.csv",
]


def _run_dir_is_valid(mining_dir: Path) -> bool:
    """True if every CSV the EDA notebook reads out of this run_dir parses
    cleanly. config.yaml/metadata.json surviving the run-name collision
    (see run_job's docstring) doesn't imply the CSVs did too: two threads'
    `df.to_csv()` calls racing on the *same path* both open it in write mode
    (truncating) and write from their own buffer independently, so their
    byte streams can interleave into a single file that's neither config's
    output -- e.g. one job's row bytes spliced into another's, producing a
    line with the wrong field count or an `itemset` string that's truncated
    mid-token. pandas' C parser raises on the former; ast.literal_eval on
    the latter. Both were observed in the wild after the run-name fix
    still left already-corrupted files behind from before it existed.
    """
    for fname in _RUN_DIR_CSVS:
        path = mining_dir / fname
        if not path.exists():
            return False
        try:
            df = pd.read_csv(path)
        except Exception:
            return False
        if "itemset" in df.columns:
            try:
                df["itemset"].apply(ast.literal_eval)
            except Exception:
                return False
    return True


def scan_existing_job_keys() -> set[tuple]:
    """(config, window) combos that already have a valid run_dir on disk,
    read from each run_dir's own config.yaml -- not inferred from the
    run_name -- since the collision run_job's docstring describes could
    leave a directory whose *contents* belong to a different config than an
    old-style name would imply. Used by --repair-missing to skip everything
    that's already fine and only force-remine what isn't.
    """
    keys: set[tuple] = set()
    if not MINING_ROOT.exists():
        return keys
    for run_dir in MINING_ROOT.iterdir():
        m = RUN_NAME_RE.match(run_dir.name)
        if m is None:
            continue
        mining_dir = run_dir / "attribute_mining"
        cfg_path, meta_path = mining_dir / "config.yaml", mining_dir / "metadata.json"
        if not (cfg_path.exists() and meta_path.exists()):
            continue
        if not _run_dir_is_valid(mining_dir):
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        keys.add(
            (
                cfg["contrast"]["min_growth_rate"],
                cfg["contrast"].get("min_growth_rate_attack"),
                cfg["tree"]["max_depth"],
                cfg["tree"].get("max_depth_attack"),
                cfg["tree"]["class_weight"],
                cfg["tree"]["min_samples_leaf"],
                cfg["contrast"]["min_attack_coverage"],
                cfg["contrast"]["min_benign_coverage"],
                float(m.group(1)),
                int(m.group(2)),
            )
        )
    return keys


def run_job(
    job: MineJob, window_path: Path, run_ts: str, fingerprint: str, force: bool = False
) -> tuple[str, str | None]:
    """Run one mining job against an already-materialized window_path, with
    output routed to its own log file via the calling thread's
    _ThreadLocalStream slot. Returns (status_line, error) -- error is None
    on success, the exception text on failure. Does NOT touch window_path's
    lifecycle -- the caller (main's group loop) owns materializing and
    deleting it, since many jobs share one window_path. `fingerprint` is
    precomputed by the caller from the stable raw alert_groups file (see
    window_fingerprint) rather than left for mine_or_reuse_attribute_schema
    to derive from window_path itself, since window_path's mtime is not
    stable across script runs.

    run_name is job.tag (fully unique per config+window), not a generic
    "gran{g}_win{w}" -- create_run_dir's run_id is `{timestamp}_{run_name}`
    at *second* precision via `mkdir(exist_ok=True)`, so every job sharing a
    window (up to `--workers` of them, run concurrently against the same
    window_path by main's group loop) used to collide on one directory
    whenever two finished mining within the same wall-clock second --
    silently, since exist_ok=True raises nothing. Whichever thread's
    config.yaml/mined_attribute_features.csv/etc. wrote last won; every
    other config sharing that window+second vanished with no error and no
    log line, which is what produced attribute_mining_sweep_eda.ipynb's
    "inconsistent window coverage across configs" assertion -- some
    (growth_rate, max_depth) grid points were missing whole windows that had
    in fact been mined and then clobbered.

    `force`, set by --repair-missing, bypasses the fingerprint cache even
    though it would otherwise report a cache hit here -- each racing job
    above still registered its own schema under its own fingerprint (see
    attribute_schema_cache._INDEX_LOCK), so the schema was never actually
    lost, only its diagnostic run_dir was. A cache hit returns before
    calling create_run_dir at all, so without `force` this job would report
    success without ever rebuilding the missing run_dir.
    """
    log_file = LOG_DIR / f"{run_ts}_{job.tag}.log"
    lf = log_file.open("w")
    sys.stdout.set(lf)
    sys.stderr.set(lf)
    try:
        schema_path, _run_dir, mining_stats = mine_or_reuse_attribute_schema(
            scenario=SCENARIO,
            alert_groups_path=window_path,
            run_name=f"symbolic_{job.tag}",
            attribute_mining_config=job.config,
            fingerprint=fingerprint,
            force=force,
        )
        status = "cache hit" if mining_stats.get("cache_hit") else "mined fresh"
        return f"{status} → {schema_path}", None
    except Exception:
        error = traceback.format_exc()
        print(error)
        return "", error
    finally:
        sys.stdout.clear()
        sys.stderr.clear()
        lf.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Thread pool size (default {MAX_WORKERS}; use 1 to run serially).",
    )
    parser.add_argument(
        "--repair-missing",
        action="store_true",
        help=(
            "Only (re)mine (config, window) combos that don't already have a "
            "valid run_dir on disk, forcing past the fingerprint cache -- see "
            "the module docstring and run_job's docstring."
        ),
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Installed once, single-threaded, before the pool starts -- see
    # _ThreadLocalStream's docstring for why this is safe under threads
    # while contextlib.redirect_stdout per-call is not.
    sys.stdout = _ThreadLocalStream(sys.stdout)
    sys.stderr = _ThreadLocalStream(sys.stderr)

    cache_dir = CACHE_DIR / SCENARIO / "groups" / CSCAS_PREGROUPED_METHOD
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}\n  Loading {SCENARIO} alert_groups (once)\n{'=' * 60}")
    ingest_cscas_scenario(cache_dir=cache_dir)
    ensure_feature_manifest(SCENARIO)
    alert_groups = load_or_build_alert_groups(SCENARIO, cache_dir)
    alert_groups.sort(key=lambda t: t.start_ts or "")
    alert_groups_path = cache_dir / "alert_groups" / "alert_groups_raw.json"
    print(f"  Loaded {len(alert_groups)} alert_groups")

    mine_fracs_windows: dict[float, int] = {}
    print(
        "\nWindow counts per granularity (slice files are materialized "
        "per-window during the run below, not up front -- see module docstring):"
    )
    for gran in MINE_FRACS:
        _, _, n_windows = compute_window_bounds(len(alert_groups), gran, 0)
        mine_fracs_windows[gran] = n_windows
        print(f"  granularity={gran:g} -> {n_windows} window(s)")

    jobs = build_jobs(mine_fracs_windows)

    if args.repair_missing:
        existing = scan_existing_job_keys()
        n_grid = len(jobs)
        jobs = [
            j for j in jobs if job_key(j.config, j.mine_frac, j.win_idx) not in existing
        ]
        print(
            f"\n--repair-missing: {n_grid - len(jobs)}/{n_grid} combos already have a "
            f"valid run_dir, {len(jobs)} to force-remine"
        )

    total = len(jobs)
    print(f"\n{total} mining attempts queued, {args.workers} worker thread(s)\n")

    if total == 0:
        print("Nothing to do.")
        return

    # Grouped by (mine_frac, win_idx): materialize that window's slice file
    # once, run every config sharing it through the pool, delete it, move on
    # -- at most one window's slice file exists on disk at a time. See the
    # module docstring for why this replaced materializing all windows
    # up front (disk exhaustion).
    groups: dict[tuple[float, int], list[MineJob]] = defaultdict(list)
    for job in jobs:
        groups[(job.mine_frac, job.win_idx)].append(job)

    failed: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (mine_frac, win_idx), group_jobs in groups.items():
            window_path = resolve_window_alert_groups_path(
                alert_groups, alert_groups_path, gran=mine_frac, win_idx=win_idx
            )
            slice_start, slice_end, _ = compute_window_bounds(
                len(alert_groups), mine_frac, win_idx
            )
            try:
                futures = {
                    pool.submit(
                        run_job,
                        job,
                        window_path,
                        run_ts,
                        window_fingerprint(
                            alert_groups_path,
                            mine_frac,
                            win_idx,
                            slice_start,
                            slice_end,
                            job.config,
                        ),
                        args.repair_missing,
                    ): job
                    for job in group_jobs
                }
                for future in as_completed(futures):
                    job = futures[future]
                    done += 1
                    status_line, error = future.result()
                    if error is None:
                        print(f"[{done}/{total}] {job.tag}\n    {status_line}")
                    else:
                        log_file = LOG_DIR / f"{run_ts}_{job.tag}.log"
                        print(
                            f"[{done}/{total}] {job.tag}\n    FAILED — see {log_file}"
                        )
                        failed.append(job.tag)
            finally:
                window_path.unlink(missing_ok=True)

    print()
    print("=" * 60)
    print(f"  SWEEP SUMMARY: {total - len(failed)}/{total} succeeded")
    print("=" * 60)
    if failed:
        print("Failed combinations:")
        for tag in failed:
            print(f"  - {tag}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

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

Same three grids as before (see git history of sweep_attribute_schema.sh
and this file for the full rationale on each range):
  1. growth_rate x max_depth x granularity, at the default
     class_weight=balanced / min_samples_leaf=20.
  2. class_weight / min_samples_leaf, one axis at a time, growth_rate held
     at MIN_GROWTH_RATE_FIXED.
  3. min_attack_coverage / min_benign_coverage, one axis at a time (holding
     the other at its 0.05 default), same anchor as grid 2.

Every (alert_groups window file, AttributeMiningConfig) combination is
cached on disk (thesis.mining.attribute_schema_cache) exactly as before, so
this script is idempotent the same way the old one was -- rerunning it only
mines whatever isn't already cached.

`total`/the per-combination log files are counted per (config, window) pair
rather than per CLI invocation (which used to bundle every window for a
given mine_frac into one log). The grids were originally sized at 4900 raw
mining attempts (11 growth_rates x 4 depths x 6 granularities, plus 16-point
coverage grids swept per-axis) but that proved infeasible, and ~400 of those
4900 were exact duplicates anyway -- grids 2/3 hold every axis but one at
the grid-1 anchor value, and whenever a swept axis's own grid happens to
still contain that anchor value (e.g. 0.05 in the coverage grid, "balanced"
in class_weight), the call produces a (mine_frac, win_idx, config) triple
grid 1 already queued, which is why the same-looking tag could show up
twice with a "cache hit" the second time -- not a fingerprinting bug, just a
redundant job. `build_jobs` now dedupes on that key regardless of grid
sizing, and the grids themselves were trimmed to a coarser, still-covering
set of points (see MIN_GROWTH_RATES/MAX_DEPTHS/MINE_FRACS/COVERAGE_GRID_NEW
below), bringing this down to ~816 unique mining attempts.

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

MIN_GROWTH_RATES = [2.0, 3.0, 5.0, 10.0]
MAX_DEPTHS = [1, 3, 5]
MINE_FRACS = [0.1, 0.25, 0.5]

# Grid 2/3 anchor see the module docstring above.
MIN_GROWTH_RATE_FIXED = 3.0
CLASS_WEIGHTS_NEW = ["balanced"]
MIN_SAMPLES_LEAVES_NEW = [5, 10, 20]
COVERAGE_GRID_NEW = [0.0, 0.05, 0.15, 0.30]

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
    class_weight: str,
    min_samples_leaf: int,
    attack_coverage: float,
    benign_coverage: float,
) -> AttributeMiningConfig:
    config = AttributeMiningConfig()
    config.contrast.min_attack_coverage = attack_coverage
    config.contrast.min_benign_coverage = benign_coverage
    config.contrast.min_growth_rate = growth_rate
    config.tree.max_depth = max_depth
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
    # Grids 2 and 3 each hold every axis but one at the grid-1 anchor value
    # (growth_rate=MIN_GROWTH_RATE_FIXED, class_weight="balanced",
    # min_samples_leaf=20, attack/benign coverage=0.05) -- whenever the swept
    # axis's own grid *also* contains that anchor value (e.g. 0.05 sitting in
    # COVERAGE_GRID_NEW, or "balanced" in CLASS_WEIGHTS_NEW), that call
    # produces the exact (mine_frac, win_idx, config) combination grid 1
    # already queued. Tracked here instead of hand-tuning each grid to dodge
    # the anchor, since that's what actually caused the duplicate "cache hit"
    # log lines for two different-looking tags -- same config, same window,
    # mined/cached twice for no reason.
    seen: set[tuple] = set()

    def add(
        growth_rate,
        max_depth,
        mine_frac,
        class_weight,
        min_samples_leaf,
        attack_coverage=0.05,
        benign_coverage=0.05,
    ):
        key = (
            growth_rate,
            max_depth,
            mine_frac,
            class_weight,
            min_samples_leaf,
            attack_coverage,
            benign_coverage,
        )
        if key in seen:
            return
        seen.add(key)

        config = build_config(
            growth_rate,
            max_depth,
            class_weight,
            min_samples_leaf,
            attack_coverage,
            benign_coverage,
        )
        base_tag = (
            f"{SCENARIO}_gr{growth_rate}_md{max_depth}_cw{class_weight}"
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

    # Grid 1: growth_rate x max_depth x granularity, at the default
    # class_weight=balanced / min_samples_leaf=20.
    for growth_rate in MIN_GROWTH_RATES:
        for max_depth in MAX_DEPTHS:
            for mine_frac in MINE_FRACS:
                add(growth_rate, max_depth, mine_frac, "balanced", 20)

    # Grid 2: class_weight / min_samples_leaf, one axis at a time,
    # growth_rate fixed -- see the module docstring above.
    for max_depth in MAX_DEPTHS:
        for mine_frac in MINE_FRACS:
            for class_weight in CLASS_WEIGHTS_NEW:
                add(MIN_GROWTH_RATE_FIXED, max_depth, mine_frac, class_weight, 20)
            for min_samples_leaf in MIN_SAMPLES_LEAVES_NEW:
                add(
                    MIN_GROWTH_RATE_FIXED,
                    max_depth,
                    mine_frac,
                    "balanced",
                    min_samples_leaf,
                )

    # Grid 3: min_attack_coverage / min_benign_coverage, one axis at a time
    # (holding the other at its 0.05 default), same anchor as grid 2.
    for max_depth in MAX_DEPTHS:
        for mine_frac in MINE_FRACS:
            for attack_coverage in COVERAGE_GRID_NEW:
                add(
                    MIN_GROWTH_RATE_FIXED,
                    max_depth,
                    mine_frac,
                    "balanced",
                    20,
                    attack_coverage=attack_coverage,
                    benign_coverage=0.05,
                )
            for benign_coverage in COVERAGE_GRID_NEW:
                add(
                    MIN_GROWTH_RATE_FIXED,
                    max_depth,
                    mine_frac,
                    "balanced",
                    20,
                    attack_coverage=0.05,
                    benign_coverage=benign_coverage,
                )

    return jobs


def job_key(config: AttributeMiningConfig, mine_frac: float, win_idx: int) -> tuple:
    """Identity of a (config, window) mining input, independent of run_name/
    timestamp -- the same fields --repair-missing reads back out of an
    existing run_dir's config.yaml in scan_existing_job_keys, so the two are
    directly comparable."""
    return (
        config.contrast.min_growth_rate,
        config.tree.max_depth,
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
                cfg["tree"]["max_depth"],
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

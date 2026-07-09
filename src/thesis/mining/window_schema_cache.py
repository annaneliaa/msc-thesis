"""Per-window, train-split-only attribute mining cache.

Sibling to mining/attribute_schema_cache.py, which fingerprints and caches a
schema mined from an arbitrary alert_groups file. This module adds the
windowing layer on top: given a full (chronologically sorted) alert_groups
list, a granularity, and a window index, it resolves that window's train
split (see pipeline.compute_window_bounds / compute_window_train_end),
serializes just that slice, and delegates to
mine_or_reuse_attribute_schema for the actual mine-or-reuse.

This is the "pass a precached mined schema for window W, granularity g, and
mining parameters -- if not existing, mine it" entry point used by the
in-window screening sweep (experiments/screening_sweep.py). Mining only on
the window's train split (not the whole window) keeps the window's test
split unseen by the miner, matching the sweep's per-window train/test
protocol.

get_or_mine_full_window_attribute_schema is the sibling entry point for
experiments (e.g. experiments/temporal_decay.py) that mine and train on a
source window in full, with no held-out split -- the frozen result is then
evaluated on other windows entirely, so no per-window test split is needed.

Cache identity note: a window's mining input (train split or full window) is
a deterministic slice of the raw, already-on-disk alert_groups file -- fully
pinned down by that raw file's identity plus (gran, win_idx, split bounds).
So cache lookups here fingerprint against the *raw* file's stat plus those
slicing parameters (see attribute_schema_cache.compute_fingerprint_from_identity),
not against the materialized per-window slice file. That slice file is only
ever written lazily, on an actual cache miss, right before mining reads it --
a cache hit never requires it to exist on disk at all -- and is deleted again
immediately after that one mining call reads it (see _mine_and_discard_slice):
the mining job loads it into memory once and the fingerprint no longer
depends on it persisting, so keeping it around afterward is pure disk cost
with no further benefit. A sweep across many (setting x granularity x
window) combos would otherwise pile up every window's slice file
(hundreds of MB to multiple GB each) simultaneously, which is exactly what
exhausted disk space before this module's cache identity was decoupled from
the slice file's existence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thesis.paths import FEATURE_DIR
from thesis.pipeline.pipeline import (
    alert_group_to_dict,
    compute_window_bounds,
    compute_window_train_end,
)
from thesis.mining.attribute_schema_cache import (
    compute_fingerprint_from_identity,
    lookup as lookup_cached_schema,
    mine_or_reuse_attribute_schema,
)
from thesis.schemas.groups import AlertGroup
from thesis.schemas.mining import AttributeMiningConfig


@dataclass(slots=True)
class WindowSchemaResult:
    schema_path: Path
    mining_run_dir: Path | None
    mining_stats: dict
    win_start: int
    win_end: int
    win_train_end: int
    cache_hit: bool


def _resolve_window_slice_alert_groups_path(
    alert_groups: list[AlertGroup],
    alert_groups_path: Path,
    gran: float,
    win_idx: int,
    slice_start: int,
    slice_end: int,
    tag: str,
) -> Path:
    """Serialize alert_groups[slice_start:slice_end] to its own file, named
    by (gran, win_idx, tag).

    Only writes if the file doesn't already exist -- content is a
    deterministic function of (alert_groups, gran, win_idx, tag), and
    mine_or_reuse_attribute_schema's cache fingerprint is sensitive to this
    file's mtime (see resolve_window_alert_groups_path for the same issue).
    """
    import json

    gran_tag = f"{gran:.6f}".rstrip("0").rstrip(".")
    win_path = (
        alert_groups_path.parent
        / f"alert_groups_gran{gran_tag}_win{win_idx}_{tag}.json"
    )
    if not win_path.exists():
        win_path.write_text(
            json.dumps(
                [alert_group_to_dict(t) for t in alert_groups[slice_start:slice_end]]
            )
        )
    return win_path


def _resolve_window_train_alert_groups_path(
    alert_groups: list[AlertGroup],
    alert_groups_path: Path,
    gran: float,
    win_idx: int,
    win_start: int,
    win_train_end: int,
    train_frac: float,
) -> Path:
    """Serialize alert_groups[win_start:win_train_end] (the window's train
    split) to its own file -- see _resolve_window_slice_alert_groups_path."""
    tf_tag = f"{train_frac:.6f}".rstrip("0").rstrip(".")
    return _resolve_window_slice_alert_groups_path(
        alert_groups,
        alert_groups_path,
        gran=gran,
        win_idx=win_idx,
        slice_start=win_start,
        slice_end=win_train_end,
        tag=f"train{tf_tag}",
    )


def _window_slice_identity(
    alert_groups_path: Path,
    gran: float,
    win_idx: int,
    slice_start: int,
    slice_end: int,
    tag: str,
) -> dict:
    """Identity for a window's mining input, derived from the raw
    (unsliced) alert_groups file's stat plus the slicing parameters --
    instead of a materialized per-window slice file's own stat. See the
    module docstring for why this is sufficient: the slice's content is a
    deterministic function of (raw file, gran, win_idx, slice bounds)."""
    stat = alert_groups_path.stat()
    return {
        "raw_alert_groups_path": str(alert_groups_path.resolve()),
        "raw_alert_groups_size": stat.st_size,
        "raw_alert_groups_mtime": stat.st_mtime,
        "gran": gran,
        "win_idx": win_idx,
        "slice_start": slice_start,
        "slice_end": slice_end,
        "tag": tag,
    }


def _mine_and_discard_slice(
    scenario: str,
    slice_path: Path,
    run_name: str,
    attribute_mining_config: AttributeMiningConfig,
    root_dir: Path,
    force: bool,
    fingerprint: str,
) -> tuple[Path, Path | None, dict]:
    """Mine using `slice_path` (already materialized by the caller because
    the cache lookup missed), then delete it -- the mining job only ever
    reads it once, and the cache no longer needs it to exist. Deletes on
    failure too, so a crashed sweep doesn't leave the slice behind either."""
    try:
        return mine_or_reuse_attribute_schema(
            scenario=scenario,
            alert_groups_path=slice_path,
            run_name=run_name,
            attribute_mining_config=attribute_mining_config,
            root_dir=root_dir,
            force=force,
            fingerprint=fingerprint,
        )
    finally:
        slice_path.unlink(missing_ok=True)


def get_or_mine_window_attribute_schema(
    scenario: str,
    alert_groups: list[AlertGroup],
    alert_groups_path: Path,
    gran: float,
    win_idx: int,
    attribute_mining_config: AttributeMiningConfig,
    train_frac: float = 0.7,
    root_dir: Path = FEATURE_DIR,
    force: bool = False,
) -> WindowSchemaResult:
    """Mine (or reuse a cached) attribute schema for window `win_idx` at
    granularity `gran`, using only that window's train split.

    `alert_groups` must already be the full scenario, chronologically
    sorted; `alert_groups_path` is the base alert_groups_raw.json path used
    to derive a sibling filename for the window's train-split file.
    """
    win_start, win_end, _ = compute_window_bounds(len(alert_groups), gran, win_idx)
    win_train_end = compute_window_train_end(win_start, win_end, train_frac)

    tf_tag = f"{train_frac:.6f}".rstrip("0").rstrip(".")
    identity = _window_slice_identity(
        alert_groups_path,
        gran=gran,
        win_idx=win_idx,
        slice_start=win_start,
        slice_end=win_train_end,
        tag=f"train{tf_tag}",
    )
    fingerprint = compute_fingerprint_from_identity(identity, attribute_mining_config)

    cached_schema_path = (
        None if force else lookup_cached_schema(scenario, fingerprint, root_dir)
    )
    if cached_schema_path is not None:
        return WindowSchemaResult(
            schema_path=cached_schema_path,
            mining_run_dir=None,
            mining_stats={"cache_hit": True, "fingerprint": fingerprint},
            win_start=win_start,
            win_end=win_end,
            win_train_end=win_train_end,
            cache_hit=True,
        )

    window_train_path = _resolve_window_train_alert_groups_path(
        alert_groups,
        alert_groups_path,
        gran=gran,
        win_idx=win_idx,
        win_start=win_start,
        win_train_end=win_train_end,
        train_frac=train_frac,
    )

    run_name = f"screening_{scenario}_gran{gran:g}_win{win_idx}"
    schema_path, mining_run_dir, mining_stats = _mine_and_discard_slice(
        scenario=scenario,
        slice_path=window_train_path,
        run_name=run_name,
        attribute_mining_config=attribute_mining_config,
        root_dir=root_dir,
        force=force,
        fingerprint=fingerprint,
    )

    return WindowSchemaResult(
        schema_path=schema_path,
        mining_run_dir=mining_run_dir,
        mining_stats=mining_stats,
        win_start=win_start,
        win_end=win_end,
        win_train_end=win_train_end,
        cache_hit=bool(mining_stats.get("cache_hit")),
    )


def get_or_mine_full_window_attribute_schema(
    scenario: str,
    alert_groups: list[AlertGroup],
    alert_groups_path: Path,
    gran: float,
    win_idx: int,
    attribute_mining_config: AttributeMiningConfig,
    root_dir: Path = FEATURE_DIR,
    force: bool = False,
) -> WindowSchemaResult:
    """Mine (or reuse a cached) attribute schema using the *entire* window
    `win_idx` at granularity `gran` -- no train/test split within the window.

    Sibling to get_or_mine_window_attribute_schema, for experiments (e.g.
    temporal_decay.py) that mine and train on a source window in full, then
    freeze the result for evaluation on other windows. Returns the same
    WindowSchemaResult shape, with win_train_end == win_end as a
    self-documenting "no held-out split" sentinel.
    """
    win_start, win_end, _ = compute_window_bounds(len(alert_groups), gran, win_idx)

    identity = _window_slice_identity(
        alert_groups_path,
        gran=gran,
        win_idx=win_idx,
        slice_start=win_start,
        slice_end=win_end,
        tag="full",
    )
    fingerprint = compute_fingerprint_from_identity(identity, attribute_mining_config)

    cached_schema_path = (
        None if force else lookup_cached_schema(scenario, fingerprint, root_dir)
    )
    if cached_schema_path is not None:
        return WindowSchemaResult(
            schema_path=cached_schema_path,
            mining_run_dir=None,
            mining_stats={"cache_hit": True, "fingerprint": fingerprint},
            win_start=win_start,
            win_end=win_end,
            win_train_end=win_end,
            cache_hit=True,
        )

    window_full_path = _resolve_window_slice_alert_groups_path(
        alert_groups,
        alert_groups_path,
        gran=gran,
        win_idx=win_idx,
        slice_start=win_start,
        slice_end=win_end,
        tag="full",
    )

    run_name = f"temporal_decay_{scenario}_gran{gran:g}_win{win_idx}_full"
    schema_path, mining_run_dir, mining_stats = _mine_and_discard_slice(
        scenario=scenario,
        slice_path=window_full_path,
        run_name=run_name,
        attribute_mining_config=attribute_mining_config,
        root_dir=root_dir,
        force=force,
        fingerprint=fingerprint,
    )

    return WindowSchemaResult(
        schema_path=schema_path,
        mining_run_dir=mining_run_dir,
        mining_stats=mining_stats,
        win_start=win_start,
        win_end=win_end,
        win_train_end=win_end,
        cache_hit=bool(mining_stats.get("cache_hit")),
    )

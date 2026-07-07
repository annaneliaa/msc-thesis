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
from thesis.mining.attribute_schema_cache import mine_or_reuse_attribute_schema
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


def _resolve_window_train_alert_groups_path(
    alert_groups: list[AlertGroup],
    alert_groups_path: Path,
    gran: float,
    win_idx: int,
    win_start: int,
    win_train_end: int,
    train_frac: float,
) -> Path:
    """Serialize alert_groups[win_start:win_train_end] to its own file.

    Only writes if the file doesn't already exist -- content is a
    deterministic function of (alert_groups, gran, win_idx, train_frac), and
    mine_or_reuse_attribute_schema's cache fingerprint is sensitive to this
    file's mtime (see resolve_window_alert_groups_path for the same issue).
    """
    import json

    gran_tag = f"{gran:.6f}".rstrip("0").rstrip(".")
    tf_tag = f"{train_frac:.6f}".rstrip("0").rstrip(".")
    win_path = (
        alert_groups_path.parent
        / f"alert_groups_gran{gran_tag}_win{win_idx}_train{tf_tag}.json"
    )
    if not win_path.exists():
        win_path.write_text(
            json.dumps(
                [alert_group_to_dict(t) for t in alert_groups[win_start:win_train_end]]
            )
        )
    return win_path


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
    schema_path, mining_run_dir, mining_stats = mine_or_reuse_attribute_schema(
        scenario=scenario,
        alert_groups_path=window_train_path,
        run_name=run_name,
        attribute_mining_config=attribute_mining_config,
        root_dir=root_dir,
        force=force,
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

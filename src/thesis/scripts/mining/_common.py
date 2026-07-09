"""Shared helpers for scripts/mining CLIs (run_screening_sweep.py,
run_temporal_decay.py, ...)."""

from __future__ import annotations

from pathlib import Path

from thesis.configs import dataset_for_scenario
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.paths import CACHE_DIR


def cache_dir_for(
    scenario: str, filtered: bool, method: str | None, window_size: int
) -> Path:
    if dataset_for_scenario(scenario) == "cscas":
        method_tag = CSCAS_PREGROUPED_METHOD
    elif filtered:
        window_tag = f"_w{window_size}" if window_size != 2 else ""
        method_tag = (
            f"filtered_{method}{window_tag}" if method else f"filtered{window_tag}"
        )
    elif window_size != 2:
        method_tag = f"w{window_size}"
    else:
        method_tag = "fixed_window"
    cache_dir = CACHE_DIR / scenario / "groups" / method_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

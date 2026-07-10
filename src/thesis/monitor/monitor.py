from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from thesis.monitor.signals import (
    DEFAULT_MIN_SAMPLES_SIGNAL_2,
    PredicateSignal,
    RuleSignal,
    compute_signal_1,
    compute_signal_2,
)
from thesis.monitor.state import MonitorState
from thesis.monitor.triggers import DEFAULT_CONSECUTIVE_WINDOWS, Action, classify_action
from thesis.schemas.dynamic_schema import DynamicSchema
from thesis.schemas.groups import AlertGroup


@dataclass(slots=True)
class MonitorSnapshot:
    scenario_name: str
    schema_version: int
    window_start: datetime
    window_end: datetime
    n_incoming_groups: int
    n_labeled_groups: int
    signal_1_results: list[PredicateSignal]
    signal_2_results: list[RuleSignal]
    signal_1_elevated: bool
    signal_2_elevated: bool
    n_elevated: int
    trigger_remine: bool
    action: Action
    state_after: MonitorState


def run_monitor_window(
    schema: DynamicSchema,
    state: MonitorState,
    incoming_groups: Sequence[AlertGroup],
    window_start: datetime,
    window_end: datetime,
    consecutive_windows: int = DEFAULT_CONSECUTIVE_WINDOWS,
    min_samples_signal_2: int = DEFAULT_MIN_SAMPLES_SIGNAL_2,
) -> MonitorSnapshot:
    """
    Pure decision function: computes both signals, advances `state` in
    place, classifies the action, returns a full snapshot. Never calls
    DynamicSchemaRegistry.deploy() or any mining job -- a future Experiment 4
    calls this once per rolling window (same compute_window_bounds walk
    pattern as experiments/rolling_walk_forward.py) and decides, externally,
    whether to act on `action`.
    """
    if state.deployed_schema_version != schema.version:
        raise ValueError(
            f"MonitorState is scoped to schema version "
            f"{state.deployed_schema_version}, but schema version "
            f"{schema.version} was passed -- construct a fresh MonitorState "
            "when a new Vk is deployed instead of reusing a stale one."
        )

    labeled_groups = [
        tx for tx in incoming_groups if tx.group_label in ("benign", "attack")
    ]

    signal_1_results = compute_signal_1(schema, incoming_groups)
    signal_2_results = compute_signal_2(
        schema, labeled_groups, min_samples=min_samples_signal_2
    )

    signal_1_elevated = any(r.elevated for r in signal_1_results)
    signal_2_elevated = any(r.elevated for r in signal_2_results)
    n_elevated = int(signal_1_elevated) + int(signal_2_elevated)

    state.advance(signal_1_elevated, signal_2_elevated)
    trigger_remine, action = classify_action(
        signal_1_elevated,
        signal_2_elevated,
        state,
        consecutive_windows=consecutive_windows,
    )

    return MonitorSnapshot(
        scenario_name=state.scenario_name,
        schema_version=schema.version,
        window_start=window_start,
        window_end=window_end,
        n_incoming_groups=len(incoming_groups),
        n_labeled_groups=len(labeled_groups),
        signal_1_results=signal_1_results,
        signal_2_results=signal_2_results,
        signal_1_elevated=signal_1_elevated,
        signal_2_elevated=signal_2_elevated,
        n_elevated=n_elevated,
        trigger_remine=trigger_remine,
        action=action,
        state_after=state,
    )

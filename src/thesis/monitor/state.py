from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MonitorState:
    """
    Consecutive-window elevation counters, scoped to one deployed Vk's
    lifetime -- a fresh schema deploy means a fresh MonitorState, never an
    implicit carryover of stale drift history. run_monitor_window
    (monitor.monitor) asserts deployed_schema_version matches the schema
    being evaluated before advancing this state.
    """

    scenario_name: str
    deployed_schema_version: int
    consecutive_signal_1_elevated: int = 0
    consecutive_signal_2_elevated: int = 0
    windows_observed: int = 0

    def advance(self, signal_1_elevated: bool, signal_2_elevated: bool) -> None:
        """
        Increments the relevant counter(s) if elevated this window, resets
        to 0 otherwise. windows_observed always increments.
        """
        self.consecutive_signal_1_elevated = (
            self.consecutive_signal_1_elevated + 1 if signal_1_elevated else 0
        )
        self.consecutive_signal_2_elevated = (
            self.consecutive_signal_2_elevated + 1 if signal_2_elevated else 0
        )
        self.windows_observed += 1

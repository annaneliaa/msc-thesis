from __future__ import annotations

from typing import Literal

from thesis.monitor.state import MonitorState

Action = Literal["NO_ACTION", "SOFT_ALERT", "RETRAIN_ONLY", "REMINE_AND_RETRAIN"]

DEFAULT_CONSECUTIVE_WINDOWS = 3


def classify_action(
    signal_1_elevated: bool,
    signal_2_elevated: bool,
    state: MonitorState,
    consecutive_windows: int = DEFAULT_CONSECUTIVE_WINDOWS,
) -> tuple[bool, Action]:
    """
    `state` must already be advanced (state.advance(...) called) with this
    window's elevated flags, so the consecutive_* counters reflect this
    window's contribution.

    | n_elevated | consecutive streak | trigger_remine | action             |
    |-----------:|:-------------------|:----------------|:-------------------|
    |          0 | impossible          | False           | NO_ACTION          |
    |  1 (sig 1) | No                  | False           | SOFT_ALERT         |
    |  1 (sig 1) | Yes                 | True            | RETRAIN_ONLY       |
    |  1 (sig 2) | No                  | False           | SOFT_ALERT         |
    |  1 (sig 2) | Yes                 | True            | REMINE_AND_RETRAIN |
    |          2 | any                 | True            | REMINE_AND_RETRAIN |

    n_elevated < 2 and nonzero implies exactly one of signal_1/signal_2 is
    both elevated and the one driving any consecutive streak, so this table
    is exhaustive -- there is no ambiguous "hard trigger with no concrete
    action" state.
    """
    n_elevated = int(signal_1_elevated) + int(signal_2_elevated)
    consecutive_hit = (
        state.consecutive_signal_1_elevated >= consecutive_windows
        or state.consecutive_signal_2_elevated >= consecutive_windows
    )
    trigger_remine = n_elevated >= 2 or consecutive_hit

    if n_elevated == 0:
        return False, "NO_ACTION"
    if not trigger_remine:
        return False, "SOFT_ALERT"
    if signal_2_elevated and not signal_1_elevated:
        return True, "REMINE_AND_RETRAIN"
    if signal_1_elevated and not signal_2_elevated:
        return True, "RETRAIN_ONLY"
    return True, "REMINE_AND_RETRAIN"

from thesis.monitor.state import MonitorState


def _make_state() -> MonitorState:
    return MonitorState(scenario_name="cscas", deployed_schema_version=1)


def test_advance_increments_when_elevated():
    state = _make_state()
    state.advance(True, False)
    assert state.consecutive_signal_1_elevated == 1
    assert state.consecutive_signal_2_elevated == 0
    assert state.windows_observed == 1

    state.advance(True, True)
    assert state.consecutive_signal_1_elevated == 2
    assert state.consecutive_signal_2_elevated == 1
    assert state.windows_observed == 2


def test_advance_resets_when_not_elevated():
    state = _make_state()
    state.advance(True, True)
    state.advance(True, True)
    state.advance(False, True)

    assert state.consecutive_signal_1_elevated == 0
    assert state.consecutive_signal_2_elevated == 3
    assert state.windows_observed == 3


def test_windows_observed_always_increments():
    state = _make_state()
    for _ in range(5):
        state.advance(False, False)
    assert state.windows_observed == 5
    assert state.consecutive_signal_1_elevated == 0
    assert state.consecutive_signal_2_elevated == 0

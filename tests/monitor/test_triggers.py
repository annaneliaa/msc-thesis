from thesis.monitor.state import MonitorState
from thesis.monitor.triggers import classify_action


def _state(consecutive_1: int = 0, consecutive_2: int = 0) -> MonitorState:
    return MonitorState(
        scenario_name="cscas",
        deployed_schema_version=1,
        consecutive_signal_1_elevated=consecutive_1,
        consecutive_signal_2_elevated=consecutive_2,
    )


def test_no_signals_elevated():
    trigger, action = classify_action(False, False, _state())
    assert trigger is False
    assert action == "NO_ACTION"


def test_signal_1_elevated_below_consecutive_threshold():
    trigger, action = classify_action(True, False, _state(consecutive_1=1))
    assert trigger is False
    assert action == "SOFT_ALERT"


def test_signal_1_elevated_hits_consecutive_threshold():
    trigger, action = classify_action(True, False, _state(consecutive_1=3))
    assert trigger is True
    assert action == "RETRAIN_ONLY"


def test_signal_2_elevated_below_consecutive_threshold():
    trigger, action = classify_action(False, True, _state(consecutive_2=1))
    assert trigger is False
    assert action == "SOFT_ALERT"


def test_signal_2_elevated_hits_consecutive_threshold():
    trigger, action = classify_action(False, True, _state(consecutive_2=3))
    assert trigger is True
    assert action == "REMINE_AND_RETRAIN"


def test_both_signals_elevated_triggers_regardless_of_consecutive_count():
    trigger, action = classify_action(True, True, _state())
    assert trigger is True
    assert action == "REMINE_AND_RETRAIN"


def test_custom_consecutive_windows_threshold():
    trigger, action = classify_action(
        True, False, _state(consecutive_1=2), consecutive_windows=2
    )
    assert trigger is True
    assert action == "RETRAIN_ONLY"

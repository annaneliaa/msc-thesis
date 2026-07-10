from thesis.monitor.predicate_eval import evaluate_all_conditions, evaluate_condition


def test_evaluate_condition_equals():
    feats = {"category": "EXPLOIT"}
    assert evaluate_condition(feats, "category", "==", "EXPLOIT") is True
    assert evaluate_condition(feats, "category", "==", "SNMP") is False


def test_evaluate_condition_not_equals():
    feats = {"category": "EXPLOIT"}
    assert evaluate_condition(feats, "category", "!=", "SNMP") is True
    assert evaluate_condition(feats, "category", "!=", "EXPLOIT") is False


def test_evaluate_condition_greater_than():
    feats = {"signature_matches_per_day": 150.0}
    assert evaluate_condition(feats, "signature_matches_per_day", ">", 100.0) is True
    assert evaluate_condition(feats, "signature_matches_per_day", ">", 200.0) is False


def test_evaluate_condition_less_equal():
    feats = {"signature_matches_per_day": 150.0}
    assert evaluate_condition(feats, "signature_matches_per_day", "<=", 150.0) is True
    assert evaluate_condition(feats, "signature_matches_per_day", "<=", 100.0) is False


def test_evaluate_condition_missing_field_is_false():
    feats = {"category": "EXPLOIT"}
    assert evaluate_condition(feats, "missing_field", "==", "x") is False


def test_evaluate_condition_unknown_operator_is_false():
    feats = {"category": "EXPLOIT"}
    assert evaluate_condition(feats, "category", "~=", "EXPLOIT") is False


def test_evaluate_all_conditions_is_and():
    feats = {"category": "EXPLOIT", "signature_matches_per_day": 50.0}
    conditions = [
        ("category", "==", "EXPLOIT"),
        ("signature_matches_per_day", "<=", 100.0),
    ]
    assert evaluate_all_conditions(feats, conditions) is True

    conditions_with_failure = conditions + [("signature_matches_per_day", ">", 1000.0)]
    assert evaluate_all_conditions(feats, conditions_with_failure) is False

def label_window_from_alert_labels(
    alert_labels: set[str],
    benign_label: str = "false_positive",
) -> str:
    labels = {str(lbl) for lbl in alert_labels if lbl is not None}

    has_benign = benign_label in labels
    has_attack = any(lbl != benign_label for lbl in labels)

    if has_attack and has_benign:
        return "mixed"
    if has_attack:
        return "attack"
    return "benign"

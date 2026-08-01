"""
Metric functions shared by every grouping baseline script -- relocated
verbatim from grouping_comparison.ipynb's old inline "Metric functions"
section, no logic changes.

`is_outlier` (thesis.schemas.groups.GroupingRecord) lets `coverage`
distinguish "this alert is the sole member of a real group" from "this
alert was rejected and never grouped at all" -- DeepCASE specifically has a
genuine rejection path (DBSCAN noise, or the Context Builder's confidence
never clearing `threshold`), and the two cases are indistinguishable after
the fact if both just look like a group of size 1. The four
deterministic/rule-based methods (fixed window, time-delta, CSCAS-style)
and AlertBERT have no equivalent rejection step, so their records leave
is_outlier at its default False -- the functions below read
`getattr(r, "is_outlier", False)` defensively.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

LABEL_FIELD = "label"  # IncomingAlert/TokenizedAlert rename the raw "time_label" CSV column to "label"
BENIGN_LABELS = {
    "false_positive"
}  # only value marking benign in this dataset's label vocabulary


def alert_volume_reduction(records: list, num_raw_alerts: int) -> float:
    num_groups = len({r.group_id for r in records})
    return 1 - (num_groups / num_raw_alerts)


def coverage(records: list, num_raw_alerts: int) -> float:
    """
    Fraction of alerts that ended up in a real group, as opposed to being
    rejected outright (DeepCASE's DBSCAN noise / low-confidence sequences).
    Reads as 1.0 for every method that has no rejection step, since
    is_outlier defaults to False -- see the module docstring.

    Counts non-outlier records directly (not distinct alert_ids): alert_id
    is a content hash (thesis.preprocessing.parsing.make_alert_id), not a
    uniqueness guarantee -- repeated identical alerts (e.g. a burst of
    identical scan-traffic signatures within the same second) legitimately
    share an alert_id. GroupingRecord is one-per-alert-instance (verified:
    len(records) == len(alerts)), so num_raw_alerts is an instance count
    too -- deduplicating the numerator by alert_id while leaving the
    denominator as a raw instance count would silently understate coverage
    whenever alert_id collisions exist, which they do in this dataset.
    """
    covered = sum(1 for r in records if not getattr(r, "is_outlier", False))
    return covered / num_raw_alerts if num_raw_alerts else 0.0


def ungroupable_fraction(records: list, num_raw_alerts: int) -> float:
    """
    1 - coverage: the fraction of alerts a method rejected outright rather
    than grouping. Nearly free given coverage already exists -- reported
    separately since "coverage" and "ungroupable_fraction" answer different
    questions (how much got grouped vs. how much got explicitly deferred).
    Expected to be exactly 0.0 for every method except DeepCASE, which is
    the only one with a rejection/deferral path in this comparison.
    """
    return 1.0 - coverage(records, num_raw_alerts)


def group_size_stats(records: list) -> dict:
    sizes = Counter(r.group_id for r in records)
    values = np.array(list(sizes.values()))
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": int(values.max()),
        "n_groups": int(len(values)),
        "sizes": values,  # kept for boxplot; saved separately, dropped from the row dict
    }


def group_purity(
    records: list, alert_index: dict, label_field: str = LABEL_FIELD
) -> dict:
    groups: dict[str, set[str]] = {}
    for r in records:
        groups.setdefault(r.group_id, set()).add(r.alert_id)

    pure_benign = pure_attack = mixed = 0
    for alert_ids in groups.values():
        labels = set()
        for aid in alert_ids:
            alert = alert_index.get(aid)
            if alert is None:
                continue
            label = getattr(alert, label_field, None)
            labels.add("benign" if label in BENIGN_LABELS else "attack")
        if labels == {"benign"}:
            pure_benign += 1
        elif labels == {"attack"}:
            pure_attack += 1
        else:
            mixed += 1

    total = max(pure_benign + pure_attack + mixed, 1)
    return {
        "pure_benign_frac": pure_benign / total,
        "pure_attack_frac": pure_attack / total,
        "mixed_frac": mixed / total,
    }


def host_purity(records: list, alert_index: dict) -> float:
    """
    Fraction of groups whose alerts all come from a single host. Trivially
    1.0 for the per-host variants and CSCAS-style (host-split by
    construction) -- meaningful mainly for the global variants of fixed
    window and time-delta, as a second, orthogonal purity axis alongside
    benign/attack purity.
    """
    groups: dict[str, set[str]] = defaultdict(set)
    for r in records:
        groups[r.group_id].add(r.alert_id)
    if not groups:
        return 1.0

    pure = 0
    for alert_ids in groups.values():
        hosts = {getattr(alert_index.get(aid), "host", None) for aid in alert_ids}
        hosts.discard(None)
        if len(hosts) <= 1:
            pure += 1
    return pure / len(groups)


def signature_diversity_stats(records: list, alert_index: dict) -> dict:
    """
    Number of distinct detector signatures per group (mean/median/max).
    Complements CSCAS-style's structural single-signature-per-group
    constraint with a number that's actually comparable across all
    five methods, which don't all enforce that constraint.
    """
    groups: dict[str, set[str]] = defaultdict(set)
    for r in records:
        groups[r.group_id].add(r.alert_id)

    counts = []
    for alert_ids in groups.values():
        sigs = {getattr(alert_index.get(aid), "signature", None) for aid in alert_ids}
        sigs.discard(None)
        counts.append(len(sigs) if sigs else 1)
    values = np.array(counts) if counts else np.array([0])
    return {
        "signature_diversity_mean": float(values.mean()),
        "signature_diversity_median": float(np.median(values)),
        "signature_diversity_max": int(values.max()),
    }


def temporal_span_stats(records: list, alert_index: dict) -> dict:
    """
    Wall-clock duration of each group (max ts - min ts among its alerts),
    mean/median/max across groups. A different failure mode than group
    size: a group can be small but span hours, or large but span
    milliseconds -- neither reduction nor group_size surfaces this alone.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        alert = alert_index.get(r.alert_id)
        if alert is None:
            continue
        groups[r.group_id].append(alert.ts)

    spans = [max(ts_list) - min(ts_list) for ts_list in groups.values() if ts_list]
    values = np.array(spans) if spans else np.array([0.0])
    return {
        "temporal_span_mean": float(values.mean()),
        "temporal_span_median": float(np.median(values)),
        "temporal_span_max": float(values.max()),
    }


def minority_label_exposure(
    records: list, alert_index: dict, label_field: str = LABEL_FIELD
) -> float:
    """
    Of alerts sitting in a mixed (benign+attack) group, the fraction
    belonging to the minority label within their own group. Approximates
    what an analyst sampling only the majority label from a mixed group
    would miss -- the closest analogue here to DeepCASE's own paper's
    "underestimation" metric. 0.0 if there are no mixed groups.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for r in records:
        alert = alert_index.get(r.alert_id)
        if alert is None:
            continue
        label = getattr(alert, label_field, None)
        groups[r.group_id].append("benign" if label in BENIGN_LABELS else "attack")

    minority_alerts = 0
    mixed_group_alerts = 0
    for labels in groups.values():
        counts = Counter(labels)
        if len(counts) <= 1:
            continue
        mixed_group_alerts += len(labels)
        minority_alerts += min(counts.values())

    return minority_alerts / mixed_group_alerts if mixed_group_alerts else 0.0


def evaluate(
    method: str,
    param_label: str,
    records: list,
    alerts: list,
    alert_index: dict,
    train_time_seconds: float | None = None,
    inference_time_seconds: float | None = None,
) -> tuple[dict, np.ndarray]:
    size_stats = group_size_stats(records)
    row = {
        "method": method,
        "param": param_label,
        "n_raw_alerts": len(alerts),
        "reduction": alert_volume_reduction(records, len(alerts)),
        "coverage": coverage(records, len(alerts)),
        "ungroupable_fraction": ungroupable_fraction(records, len(alerts)),
        "group_size_mean": size_stats["mean"],
        "group_size_median": size_stats["median"],
        "group_size_max": size_stats["max"],
        "n_groups": size_stats["n_groups"],
        "host_purity": host_purity(records, alert_index),
        "train_time_seconds": train_time_seconds,
        "inference_time_seconds": inference_time_seconds,
    }
    row.update(group_purity(records, alert_index))
    row.update(signature_diversity_stats(records, alert_index))
    row.update(temporal_span_stats(records, alert_index))
    row["minority_label_exposure"] = minority_label_exposure(records, alert_index)
    return row, size_stats["sizes"]

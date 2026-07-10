from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from thesis.experiments._shared import LABEL_MAP
from thesis.mining.attribute_features import compute_candidate_attribute_features
from thesis.monitor.predicate_eval import evaluate_all_conditions, evaluate_condition
from thesis.schemas.dynamic_schema import DynamicSchema
from thesis.schemas.groups import AlertGroup

PSI_ELEVATED_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.2
CALIBRATION_DRIFT_THRESHOLD = 0.10
DEFAULT_MIN_SAMPLES_SIGNAL_2 = 30


@dataclass(slots=True)
class PredicateSignal:
    predicate_id: str
    p_expected: float
    p_observed: float
    psi: float
    elevated: bool  # psi > PSI_ELEVATED_THRESHOLD
    significant: bool  # psi > PSI_SIGNIFICANT_THRESHOLD
    n_observed: int


@dataclass(slots=True)
class RuleSignal:
    rule_id: str
    mined_confidence: float
    observed_confidence: float | None  # None if below min_samples
    drift: float | None
    elevated: bool  # drift is not None and drift > CALIBRATION_DRIFT_THRESHOLD
    n_matching: int


def compute_psi(p_expected: float, p_observed: float, eps: float = 1e-6) -> float:
    """
    Two-bucket (fired / not-fired) Population Stability Index for one
    predicate's activation rate. A predicate's activation is Bernoulli, not
    a continuous score, so 2 buckets is the correct formulation here, not
    the usual textbook multi-bin PSI.
    """
    p_exp = min(max(p_expected, eps), 1 - eps)
    p_obs = min(max(p_observed, eps), 1 - eps)

    fired = (p_obs - p_exp) * math.log(p_obs / p_exp)
    not_fired = ((1 - p_obs) - (1 - p_exp)) * math.log((1 - p_obs) / (1 - p_exp))
    return fired + not_fired


def compute_signal_1(
    schema: DynamicSchema,
    incoming_groups: Sequence[AlertGroup],
) -> list[PredicateSignal]:
    """
    p_expected = attack_support*base_attack_rate +
    benign_support*(1-base_attack_rate) (spec formula); p_observed =
    fraction of incoming_groups where the predicate's condition holds.

    compute_candidate_attribute_features(tx) is called once per group (it
    does real per-group computation, not a cheap lookup -- see
    mining/attribute_features.py), not once per predicate.
    """
    feats_by_group = [
        compute_candidate_attribute_features(tx) for tx in incoming_groups
    ]
    n_observed = len(feats_by_group)

    results: list[PredicateSignal] = []
    for pred in schema.single_predicates:
        p_expected = (
            pred.attack_support * schema.base_attack_rate
            + pred.benign_support * (1 - schema.base_attack_rate)
        )
        if n_observed:
            n_fired = sum(
                1
                for feats in feats_by_group
                if evaluate_condition(feats, pred.field, pred.operator, pred.value)
            )
            p_observed = n_fired / n_observed
        else:
            p_observed = 0.0

        psi = compute_psi(p_expected, p_observed)
        results.append(
            PredicateSignal(
                predicate_id=pred.predicate_id,
                p_expected=p_expected,
                p_observed=p_observed,
                psi=psi,
                elevated=psi > PSI_ELEVATED_THRESHOLD,
                significant=psi > PSI_SIGNIFICANT_THRESHOLD,
                n_observed=n_observed,
            )
        )
    return results


def compute_signal_2(
    schema: DynamicSchema,
    labeled_incoming_groups: Sequence[AlertGroup],
    min_samples: int = DEFAULT_MIN_SAMPLES_SIGNAL_2,
) -> list[RuleSignal]:
    """
    For every compound rule, filter labeled_incoming_groups (group_label in
    {"benign","attack"}) to those matching all of the rule's conditions; if
    fewer than min_samples match, emit a RuleSignal with
    observed_confidence=None/drift=None/elevated=False. Otherwise
    observed_confidence is the fraction of matching rows whose true label
    equals rule.prediction, and drift is the absolute difference from the
    mined confidence.
    """
    feats_by_group = [
        (tx, compute_candidate_attribute_features(tx)) for tx in labeled_incoming_groups
    ]

    results: list[RuleSignal] = []
    for rule in schema.compound_rules:
        matching_labels = [
            LABEL_MAP[tx.group_label]
            for tx, feats in feats_by_group
            if evaluate_all_conditions(feats, rule.conditions)
        ]
        n_matching = len(matching_labels)

        if n_matching < min_samples:
            results.append(
                RuleSignal(
                    rule_id=rule.rule_id,
                    mined_confidence=rule.confidence,
                    observed_confidence=None,
                    drift=None,
                    elevated=False,
                    n_matching=n_matching,
                )
            )
            continue

        predicted_label = 1.0 if rule.prediction == "attack" else 0.0
        observed_confidence = (
            sum(1 for label in matching_labels if label == predicted_label) / n_matching
        )
        drift = abs(observed_confidence - rule.confidence)

        results.append(
            RuleSignal(
                rule_id=rule.rule_id,
                mined_confidence=rule.confidence,
                observed_confidence=observed_confidence,
                drift=drift,
                elevated=drift > CALIBRATION_DRIFT_THRESHOLD,
                n_matching=n_matching,
            )
        )
    return results

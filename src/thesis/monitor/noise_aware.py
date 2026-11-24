"""
Noise-aware monitor signals with a day-by-day baseline (observe-only; not
wired into classify_action).

The thesis signals compare a window against mining-time values with fixed
cutoffs (PSI > 0.10, calibration drift > 0.05). Two problems: the
mining-time values are in-sample (Signal 2 was already "drifting" on W_src's
own held-out split), and a fixed cutoff says nothing about how much a signal
normally moves. A bootstrap null was tried first and failed on CSCAS: with
~125k groups per window, sampling noise is ~0, while real day-to-day rhythm
is much larger, so every signal fired in every window.

So "normal" is measured from real time blocks of the training week:
1. Each statistic has a reference period, cut into chronological blocks:
   * input statistics (s1_max_psi): all of W_src (train + held-out split;
     predicate firing rates are not fitted, so in-sample rows are fine),
     in 24h blocks;
   * output statistics (score, s2, novelty): only W_src's held-out split
     (the model and the mined patterns have seen the train split), in 12h
     blocks.
   Blocks smaller than MIN_BLOCK_FRACTION of the median block are dropped
   (a trailing partial block).
2. Threshold = the largest leave-one-out value over the blocks: each block's
   statistic against the other blocks pooled, i.e. "bigger than the biggest
   normal block-to-block difference in the training week".
3. A window's statistic is computed against the whole reference period.

Only training-week data and fixed choices (block lengths, max rule) set the
thresholds, so nothing from the test period leaks in. Caveats: few blocks
(5 input, 3 output on CSCAS), and blocks are shorter than a window, so they
carry more noise than a window does -- the thresholds err on the
conservative (fewer-alarms) side.

Statistics (all label-free except s2):
* s1_max_psi:      max over single predicates of the two-bucket PSI of the
                   firing rate vs the reference rate (thesis Signal 1).
* s2_max_drift:    max over compound rules with >= min_samples labelled
                   matches of |precision - reference precision| (thesis
                   Signal 2; needs labels).
* score_psi:       10-bin PSI of the model's predicted attack probability.
* flag_rate_shift: |share of groups at/above the model threshold - ref|.
* margin_shift:    |share of groups with |p - threshold| < MARGIN - ref|.
* novelty_shift:   share of groups whose predicate-firing pattern is unseen
                   at mining time, minus the reference share (one-sided).
* decision_psi:    3-bucket PSI over the model's decision regions, set by
                   its own threshold t: confident benign (p < t - MARGIN),
                   uncertain (|p - t| < MARGIN), confident attack
                   (p >= t + MARGIN). Combines the two output signals that
                   tracked F1 in the 2026-11-24 run (score_psi for RF,
                   margin_shift for XGBoost) without new tuning: the buckets
                   come from the model's threshold and the same MARGIN.
                   DeepCASE counterpart: auto-benign / sent to analyst /
                   auto-attack shares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from thesis.monitor.predicate_eval import evaluate_all_conditions, evaluate_condition
from thesis.schemas.dynamic_schema import DynamicSchema
from thesis.schemas.groups import AlertGroup

SCORE_BIN_EDGES = np.linspace(0.0, 1.0, 11)
MARGIN = 0.2
INPUT_BLOCK_SECONDS = 24 * 3600
OUTPUT_BLOCK_SECONDS = 12 * 3600
MIN_BLOCK_FRACTION = 0.25
_EPS = 1e-6

INPUT_STATISTICS = ("s1_max_psi",)
OUTPUT_STATISTICS = (
    "s2_max_drift",
    "score_psi",
    "flag_rate_shift",
    "margin_shift",
    "novelty_shift",
    "decision_psi",
)
STATISTICS = INPUT_STATISTICS + OUTPUT_STATISTICS
REGIONS = ("confident_benign", "uncertain", "confident_attack")


@dataclass(slots=True)
class WindowRows:
    """Per-group indicator arrays for one set of rows (float32 0/1 so a
    weight vector/matrix aggregates them with one matmul)."""

    ts: np.ndarray  # (n,) group start_ts, epoch seconds
    fires: np.ndarray  # (n, P) predicate holds
    rule_match_labeled: np.ndarray  # (n, R) rule matches AND row labelled
    rule_correct: np.ndarray  # (n, R) matches, labelled, label == prediction
    score_bins: np.ndarray | None  # (n, 10) one-hot score bin; None if no scores
    flagged: np.ndarray | None  # (n,) score >= threshold
    near_margin: np.ndarray | None  # (n,) |score - threshold| < MARGIN
    novel: np.ndarray  # (n,) firing pattern unseen at mining time
    patterns: list[frozenset[str]] = field(default_factory=list)
    regions: np.ndarray | None = None  # (n, 3) one-hot decision region (REGIONS)
    labeled: np.ndarray | None = None  # (n,) group_label in {attack, benign}
    is_attack: np.ndarray | None = None  # (n,) group_label == attack

    def __len__(self) -> int:
        return self.fires.shape[0]

    def concat(self, other: "WindowRows") -> "WindowRows":
        def cat(a, b):
            return None if a is None or b is None else np.concatenate([a, b])

        return WindowRows(
            ts=np.concatenate([self.ts, other.ts]),
            fires=np.concatenate([self.fires, other.fires]),
            rule_match_labeled=np.concatenate(
                [self.rule_match_labeled, other.rule_match_labeled]
            ),
            rule_correct=np.concatenate([self.rule_correct, other.rule_correct]),
            score_bins=cat(self.score_bins, other.score_bins),
            flagged=cat(self.flagged, other.flagged),
            near_margin=cat(self.near_margin, other.near_margin),
            novel=np.concatenate([self.novel, other.novel]),
            patterns=self.patterns + other.patterns,
            regions=cat(self.regions, other.regions),
            labeled=cat(self.labeled, other.labeled),
            is_attack=cat(self.is_attack, other.is_attack),
        )


def build_window_rows(
    schema: DynamicSchema,
    groups: Sequence[AlertGroup],
    feats: Sequence[dict[str, Any]],
    reference_patterns: frozenset[frozenset[str]],
    scores: np.ndarray | None,
    threshold: float,
) -> WindowRows:
    preds = schema.single_predicates
    rules = schema.compound_rules
    n = len(groups)
    fires = np.zeros((n, len(preds)), dtype=np.float32)
    match = np.zeros((n, len(rules)), dtype=np.float32)
    correct = np.zeros((n, len(rules)), dtype=np.float32)
    patterns: list[frozenset[str]] = []
    for i, (g, f) in enumerate(zip(groups, feats)):
        hold = [evaluate_condition(f, p.field, p.operator, p.value) for p in preds]
        fires[i] = hold
        patterns.append(frozenset(p.predicate_id for p, h in zip(preds, hold) if h))
        if g.group_label not in ("attack", "benign"):
            continue
        for j, r in enumerate(rules):
            if evaluate_all_conditions(f, r.conditions):
                match[i, j] = 1.0
                correct[i, j] = float(g.group_label == r.prediction)
    novel = np.array([p not in reference_patterns for p in patterns], dtype=np.float32)
    labeled = np.array(
        [g.group_label in ("attack", "benign") for g in groups], dtype=np.float32
    )
    is_attack = np.array([g.group_label == "attack" for g in groups], dtype=np.float32)
    ts = np.array([g.start_ts or 0 for g in groups], dtype=np.int64)

    if scores is not None and len(scores) == n:
        s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        b = np.clip(np.digitize(s, SCORE_BIN_EDGES[1:-1]), 0, len(SCORE_BIN_EDGES) - 2)
        score_bins = np.eye(len(SCORE_BIN_EDGES) - 1, dtype=np.float32)[b]
        flagged = (s >= threshold).astype(np.float32)
        near_margin = (np.abs(s - threshold) < MARGIN).astype(np.float32)
        region = np.where(
            s < threshold - MARGIN, 0, np.where(s >= threshold + MARGIN, 2, 1)
        )
        regions = np.eye(len(REGIONS), dtype=np.float32)[region]
    else:
        score_bins = flagged = near_margin = regions = None
    return WindowRows(
        ts,
        fires,
        match,
        correct,
        score_bins,
        flagged,
        near_margin,
        novel,
        patterns,
        regions,
        labeled,
        is_attack,
    )


def _psi_two_bucket(p_ref: np.ndarray, p_obs: np.ndarray) -> np.ndarray:
    a = np.clip(p_ref, _EPS, 1 - _EPS)
    b = np.clip(p_obs, _EPS, 1 - _EPS)
    return (b - a) * np.log(b / a) + ((1 - b) - (1 - a)) * np.log((1 - b) / (1 - a))


def _psi_bins(h_ref: np.ndarray, h_obs: np.ndarray) -> np.ndarray:
    a = np.clip(h_ref, _EPS, None)
    b = np.clip(h_obs, _EPS, None)
    return ((b - a) * np.log(b / a)).sum(axis=-1)


def _aggregate(rows: WindowRows, w: np.ndarray) -> dict[str, Any]:
    """Weighted aggregates over rows (w: (n,) 0/1 selection weights)."""
    tot = float(w.sum())
    out: dict[str, Any] = {
        "rates": (w @ rows.fires) / tot,
        "match": w @ rows.rule_match_labeled,
        "correct": w @ rows.rule_correct,
        "novel": float(w @ rows.novel) / tot,
    }
    if rows.score_bins is not None:
        out["hist"] = (w @ rows.score_bins) / tot
        out["flag"] = float(w @ rows.flagged) / tot
        out["margin"] = float(w @ rows.near_margin) / tot
    if rows.regions is not None:
        out["regions"] = (w @ rows.regions) / tot
    return out


def _statistics(
    obs: dict[str, Any], ref: dict[str, Any], min_samples: int
) -> dict[str, float]:
    out: dict[str, float] = {
        "s1_max_psi": float(_psi_two_bucket(ref["rates"], obs["rates"]).max())
        if len(obs["rates"])
        else np.nan,
        "novelty_shift": obs["novel"] - ref["novel"],
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        p_obs = obs["correct"] / obs["match"]
        p_ref = ref["correct"] / ref["match"]
        ok = (obs["match"] >= min_samples) & (ref["match"] >= min_samples)
        drift = np.abs(p_obs - p_ref)[ok]
    out["s2_max_drift"] = float(drift.max()) if drift.size else np.nan
    if "hist" in obs and "hist" in ref:
        out["score_psi"] = float(_psi_bins(ref["hist"], obs["hist"]))
        out["flag_rate_shift"] = abs(obs["flag"] - ref["flag"])
        out["margin_shift"] = abs(obs["margin"] - ref["margin"])
    else:
        out["score_psi"] = out["flag_rate_shift"] = out["margin_shift"] = np.nan
    if "regions" in obs and "regions" in ref:
        out["decision_psi"] = float(_psi_bins(ref["regions"], obs["regions"]))
    else:
        out["decision_psi"] = np.nan
    return out


def time_blocks(ts: np.ndarray, block_seconds: int) -> list[np.ndarray]:
    """Chronological block masks over `ts`, dropping blocks smaller than
    MIN_BLOCK_FRACTION of the median block size (trailing partial blocks)."""
    if len(ts) == 0:
        return []
    ids = (ts - ts.min()) // block_seconds
    blocks = [ids == b for b in np.unique(ids)]
    sizes = np.array([m.sum() for m in blocks])
    keep = sizes >= MIN_BLOCK_FRACTION * np.median(sizes)
    return [m for m, k in zip(blocks, keep) if k]


@dataclass(slots=True)
class _Baseline:
    rows: WindowRows
    pooled: dict[str, Any]
    thresholds: dict[str, float]
    n_blocks: int
    block_values: dict[str, list[float]]


def _baseline(
    rows: WindowRows, block_seconds: int, names: tuple[str, ...], min_samples: int
) -> _Baseline:
    pooled = _aggregate(rows, np.ones(len(rows), dtype=np.float32))
    blocks = time_blocks(rows.ts, block_seconds)
    values: dict[str, list[float]] = {k: [] for k in names}
    if len(blocks) >= 2:
        for m in blocks:
            w_in = m.astype(np.float32)
            stats = _statistics(
                _aggregate(rows, w_in), _aggregate(rows, 1.0 - w_in), min_samples
            )
            for k in names:
                values[k].append(stats[k])
    thresholds = {
        k: float(np.nanmax(v)) if v and np.isfinite(v).any() else np.nan
        for k, v in values.items()
    }
    return _Baseline(rows, pooled, thresholds, len(blocks), values)


class NoiseAwareMonitor:
    """Day-by-day baseline: input statistics from all of W_src in 24h
    blocks, output statistics from W_src's held-out split in 12h blocks."""

    def __init__(
        self,
        input_reference: WindowRows,
        output_reference: WindowRows,
        min_samples: int,
        input_block_seconds: int = INPUT_BLOCK_SECONDS,
        output_block_seconds: int = OUTPUT_BLOCK_SECONDS,
    ) -> None:
        if len(input_reference) == 0 or len(output_reference) == 0:
            raise ValueError("reference rows are empty")
        self.min_samples = min_samples
        self.input = _baseline(
            input_reference, input_block_seconds, INPUT_STATISTICS, min_samples
        )
        self.output = _baseline(
            output_reference, output_block_seconds, OUTPUT_STATISTICS, min_samples
        )

    @property
    def thresholds(self) -> dict[str, float]:
        return {**self.input.thresholds, **self.output.thresholds}

    def evaluate(self, rows: WindowRows) -> dict[str, float | bool | None]:
        """Per statistic: value (window vs the whole reference period),
        threshold, elevated flag -- flat na_* columns."""
        if len(rows) == 0:
            return {}
        obs = _aggregate(rows, np.ones(len(rows), dtype=np.float32))
        out: dict[str, float | bool | None] = {}
        for base, names in (
            (self.input, INPUT_STATISTICS),
            (self.output, OUTPUT_STATISTICS),
        ):
            stats = _statistics(obs, base.pooled, self.min_samples)
            for k in names:
                v, t = stats[k], base.thresholds[k]
                out[f"na_{k}"] = v
                out[f"na_{k}_thr"] = t
                out[f"na_{k}_elevated"] = (
                    bool(v > t) if np.isfinite(v) and np.isfinite(t) else None
                )
        # Raw decision-region shares, window vs reference, for inspection.
        for i, name in enumerate(REGIONS):
            out[f"na_share_{name}"] = (
                float(obs["regions"][i]) if "regions" in obs else None
            )
            out[f"na_share_{name}_ref"] = (
                float(self.output.pooled["regions"][i])
                if "regions" in self.output.pooled
                else None
            )
        return out

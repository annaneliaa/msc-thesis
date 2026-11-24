"""
Logs every monitor signal ever considered (novelty + the noise-aware
statistics with their day-by-day baselines), per window, for any monitor
experiment -- observe-only: nothing here feeds classify_action. Shared by
system_eval.monitor_drift (one deployment for the whole walk) and
system_eval.monitor_attached (redeploys on trigger), so both log exactly the
same columns under the same baseline rules.

Also logs pattern validity (thesis.monitor.pattern_validity: are the mined
patterns still discriminative / precise / covering, under oracle, escalated
and escalated+audit labels), per-feature PSI over every model input (thesis.monitor.feature_psi)
when the caller passes the encoded model inputs, and analyst-feedback
precision (thesis.monitor.feedback): precision
of the escalated, labelled groups vs the reference rows' precision.

Baselines per deployment:
* Initial deployment (mined/fit on W_src's train split): novelty reference =
  the train split's firing patterns; noise-aware baseline = train + held-out
  split for input statistics, held-out split for output statistics (see
  thesis.monitor.noise_aware). The held-out split is horizon 0.
* After a redeploy at window h (the new model/schema was fit on window h):
  novelty reference = window h's groups; the noise-aware baseline is taken
  from the NEXT window, the first data the new deployment hasn't seen
  (leak-free warm-up). That window is logged against itself and flagged
  signals_reference_window=True.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import pandas as pd

from thesis.mining.common.attribute_features import compute_candidate_attribute_features
from thesis.monitor.feature_psi import FEATURE_PSI_COLUMNS, FeaturePsiMonitor
from thesis.monitor.feedback import (
    FEEDBACK_COLUMNS,
    FeedbackCounts,
    feedback_counts,
    feedback_signal,
)
from thesis.monitor.noise_aware import (
    REGIONS,
    STATISTICS,
    NoiseAwareMonitor,
    WindowRows,
    build_window_rows,
)
from thesis.monitor.pattern_validity import (
    PATTERN_VALIDITY_COLUMNS,
    PatternValidityMonitor,
)
from thesis.monitor.novelty import (
    NoveltyReference,
    build_novelty_reference,
    novelty_from_patterns,
)
from thesis.schemas.dynamic_schema import DynamicSchema
from thesis.schemas.groups import AlertGroup

NOVELTY_FIELDS = (
    "pattern_novelty_rate",
    "no_fire_rate",
    "reference_no_fire_rate",
    "n_reference_patterns",
    "n_novel_patterns",
    "pattern_novelty_rate_attack",
    "pattern_novelty_rate_benign",
)
LOGGED_COLUMNS = (
    ("signals_reference_window",)
    + tuple(f"novelty_{k}" for k in NOVELTY_FIELDS)
    + tuple(f"na_{k}{s}" for k in STATISTICS for s in ("", "_thr", "_elevated"))
    + tuple(f"na_share_{r}{s}" for r in REGIONS for s in ("", "_ref"))
    + FEEDBACK_COLUMNS
    + FEATURE_PSI_COLUMNS
    + PATTERN_VALIDITY_COLUMNS
)


def _counts(rows: WindowRows) -> FeedbackCounts | None:
    return feedback_counts(rows.flagged, rows.labeled, rows.is_attack)


def rows_for_groups(
    schema: DynamicSchema,
    groups: Sequence[AlertGroup],
    reference_patterns: frozenset[frozenset[str]],
    scores: np.ndarray | None,
    threshold: float,
) -> WindowRows:
    """One candidate-feature pass over `groups` -> every per-group indicator
    the logged signals need. `scores` must align with `groups` (or be None:
    score-based statistics are then NaN)."""
    if scores is not None and len(scores) != len(groups):
        scores = None
    feats = [compute_candidate_attribute_features(g) for g in groups]
    return build_window_rows(
        schema, groups, feats, reference_patterns, scores, threshold
    )


class SignalLog:
    def __init__(
        self,
        schema: DynamicSchema,
        train_groups: Sequence[AlertGroup],
        heldout_groups: Sequence[AlertGroup],
        heldout_scores: np.ndarray | None,
        threshold: float,
        min_samples: int,
        train_X: pd.DataFrame | None = None,
        heldout_X: pd.DataFrame | None = None,
    ) -> None:
        """train_X/heldout_X: encoded model inputs aligned row-for-row with
        train_groups/heldout_groups (skipped if they don't align). Reference
        for per-feature PSI: all of W_src (train + held-out), 24h blocks."""
        self.min_samples = min_samples
        self.feature_monitor: FeaturePsiMonitor | None = None
        if (
            train_X is not None
            and heldout_X is not None
            and len(train_X) == len(train_groups)
            and len(heldout_X) == len(heldout_groups)
            and len(train_X) + len(heldout_X) > 0
        ):
            ts = np.array(
                [g.start_ts or 0 for g in list(train_groups) + list(heldout_groups)]
            )
            self.feature_monitor = FeaturePsiMonitor(
                pd.concat([train_X, heldout_X], ignore_index=True), ts
            )
        self.novelty_ref: NoveltyReference = build_novelty_reference(
            schema, train_groups
        )
        self.monitor: NoiseAwareMonitor | None = None
        self.feedback_ref: FeedbackCounts | None = None
        self.pattern_monitor: PatternValidityMonitor | None = None
        self._reference_groups = heldout_groups
        self.baseline_note = "no held-out rows"
        if train_groups and heldout_groups:
            heldout = rows_for_groups(
                schema,
                heldout_groups,
                self.novelty_ref.patterns,
                heldout_scores,
                threshold,
            )
            train = rows_for_groups(
                schema, train_groups, self.novelty_ref.patterns, None, threshold
            )
            self.monitor = NoiseAwareMonitor(
                input_reference=train.concat(heldout),
                output_reference=heldout,
                min_samples=min_samples,
            )
            self.feedback_ref = _counts(heldout)
            self.pattern_monitor = PatternValidityMonitor(
                schema, heldout, heldout_groups, min_samples
            )
            self.baseline_note = self._describe()

    def _describe(self) -> str:
        m = self.monitor
        fb = self.feedback_ref
        fp = self.feature_monitor
        fp_note = (
            f"; feature PSI: {fp.n_blocks} blocks, {len(fp.columns)} features, "
            f"per-feature thr median {fp.thresholds.median():.4f} max "
            f"{fp.thresholds.max():.4f} ({fp.thresholds.idxmax()})"
            if fp is not None and fp.thresholds is not None
            else ""
        )
        fb_note = (
            f"; feedback precision ref {fb.precision:.3f} (n={fb.n_escalated})"
            if fb is not None and fb.n_escalated
            else ""
        )
        pv = self.pattern_monitor
        pv_note = (
            f"; pattern validity: {pv.n_valid_ref('oracle')}/{len(pv.ids)} "
            "patterns valid on reference (oracle labels)"
            if pv is not None
            else ""
        )
        return (
            f"baseline blocks: input {m.input.n_blocks}, output {m.output.n_blocks}; "
            + ", ".join(f"{k}={v:.4f}" for k, v in m.thresholds.items())
            + fb_note
            + fp_note
            + pv_note
        )

    def redeploy(
        self, schema: DynamicSchema, training_groups: Sequence[AlertGroup]
    ) -> None:
        """New model and/or schema fit on `training_groups`: reset the
        novelty reference to them and take the noise-aware baseline from the
        next logged window."""
        self.novelty_ref = build_novelty_reference(schema, training_groups)
        self.monitor = None
        self.feedback_ref = None
        self.feature_monitor = None
        self.pattern_monitor = None
        self._reference_groups = None

    def log_window(
        self,
        schema: DynamicSchema,
        groups: Sequence[AlertGroup],
        scores: np.ndarray | None,
        threshold: float,
        X: pd.DataFrame | None = None,
    ) -> dict:
        """All logged signal columns for one window (LOGGED_COLUMNS keys).
        X: the window's encoded model inputs, aligned with `groups` (optional;
        per-feature PSI is skipped without it)."""
        out: dict = dict.fromkeys(LOGGED_COLUMNS)
        if not groups:
            return out
        rows = rows_for_groups(
            schema, groups, self.novelty_ref.patterns, scores, threshold
        )
        is_reference = groups is self._reference_groups
        if X is not None and len(X) != len(groups):
            X = None
        if self.monitor is None and X is not None:
            ts = np.array([g.start_ts or 0 for g in groups])
            self.feature_monitor = FeaturePsiMonitor(X, ts)
        if self.monitor is None:
            # Warm-up after a redeploy: this window is the new baseline.
            self.monitor = NoiseAwareMonitor(rows, rows, min_samples=self.min_samples)
            self.feedback_ref = _counts(rows)
            self.pattern_monitor = PatternValidityMonitor(
                schema, rows, groups, self.min_samples
            )
            self._reference_groups = groups
            self.baseline_note = self._describe()
            is_reference = True
        out["signals_reference_window"] = is_reference
        novelty = novelty_from_patterns(self.novelty_ref, rows.patterns, groups)
        out.update({f"novelty_{k}": getattr(novelty, k) for k in NOVELTY_FIELDS})
        out.update(self.monitor.evaluate(rows))
        out.update(feedback_signal(_counts(rows), self.feedback_ref))
        if self.feature_monitor is not None and X is not None:
            out.update(self.feature_monitor.evaluate(X))
        if self.pattern_monitor is not None:
            out.update(self.pattern_monitor.evaluate(rows, groups))
        return out

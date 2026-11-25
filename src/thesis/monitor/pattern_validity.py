"""
Pattern validity: are the deployed mined patterns still USEFUL? (observe-only;
logged by SignalLog.)

Signal 1 asks "does each predicate fire as often as at mining time?", which
reacts to harmless traffic-mix changes and misses a pattern that keeps
firing but now fires on the other class (synthetic drift test, 2026-11-24).
This signal asks the question the monitor is meant to answer -- "has the
input changed such that the mined patterns stop working?" -- for every
pattern in Vk (single predicates AND compound rules), in its mined direction:

* contrast validity: would the miner still keep it? Growth rate in the
  pattern's direction (support in its class / support in the other class)
  >= MIN_GROWTH_RATE and coverage of its class >= MIN_COVERAGE -- the
  contrast miner's own keep rule (filter_contrast_survivors, mining setting
  gr3_md1_mda4: min_growth_rate 3, coverage 0.05), failed only when broken
  with 99% confidence (PatternStats.invalid_confident; the point-estimate
  version is still logged as *_point). Only patterns that pass
  on the reference rows are checked, so a pattern that never met the rule
  outside its training rows can't fire the signal;
* precision drift: share of the pattern's firings that are its class,
  significantly lower than on the reference (one-sided two-proportion test
  at 99%, as feedback precision) AND at least PRECISION_DROP lower (Signal
  2's calibration threshold) -- large windows make tiny drops significant;
* coverage: share of attacks on which at least one attack-direction pattern
  fires (and the same for benign), lower than on the reference by the same
  test. Catches traffic the patterns no longer describe.

Reference = the deployment's reference rows (W_src's held-out split, or the
warm-up window after a redeploy), evaluated in the same label mode, so a
label mode's own bias cancels.

Needs labels. Three label modes, all logged:
* oracle: every labelled group (upper bound; what Signal 2 uses);
* escalated: only escalated groups (score >= threshold) -- what analysts
  review anyway. Missed attacks are never labelled, so the benign class is
  only the false positives;
* audit: escalated groups + a random AUDIT_RATE sample of the suppressed
  ones, weighted 1/AUDIT_RATE (inverse-probability weights), so class
  supports are unbiased estimates. The draw is a hash of alert_group_id, so
  it is the same across runs and injections. Sample sizes use Kish's
  effective n.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from thesis.monitor.feedback import Z_ONE_SIDED_99
from thesis.monitor.noise_aware import WindowRows
from thesis.monitor.signals import CALIBRATION_DRIFT_THRESHOLD
from thesis.schemas.dynamic_schema import DynamicSchema
from thesis.schemas.groups import AlertGroup

MIN_GROWTH_RATE = 3.0
MIN_COVERAGE = 0.05
PRECISION_DROP = CALIBRATION_DRIFT_THRESHOLD
AUDIT_RATE = 0.05
LABEL_MODES = ("oracle", "escalated", "audit")
_EPS = 1e-6

_FIELDS = (
    "n_checked",
    "n_invalid",
    "invalid_patterns",
    "worst_pattern",
    "worst_gr_ratio",
    "n_prec_drop",
    "prec_drop_patterns",
    "worst_prec_drop",
    "cov_attack",
    "cov_attack_ref",
    "cov_benign",
    "cov_benign_ref",
    "n_eff_attack",
    "n_eff_benign",
    "n_invalid_point",
    "invalid_patterns_point",
    "contrast_elevated",
    "contrast_point_elevated",
    "precision_elevated",
    "coverage_elevated",
    "elevated",
)
PATTERN_VALIDITY_COLUMNS = tuple(f"pv_{m}_{f}" for m in LABEL_MODES for f in _FIELDS)


def audit_mask(groups: Sequence[AlertGroup], rate: float = AUDIT_RATE) -> np.ndarray:
    """Deterministic random audit draw per group (same group -> same draw)."""
    return np.array(
        [zlib.crc32(str(g.alert_group_id).encode()) / 2**32 < rate for g in groups],
        dtype=np.float32,
    )


def pattern_meta(schema: DynamicSchema) -> tuple[list[str], np.ndarray]:
    """Pattern ids and direction (True = attack) in WindowRows column order:
    single predicates (rows.fires), then compound rules
    (rows.rule_match_labeled)."""
    ids = [p.predicate_id for p in schema.single_predicates] + [
        r.rule_id for r in schema.compound_rules
    ]
    to_attack = np.array(
        [p.direction == "attack" for p in schema.single_predicates]
        + [r.prediction == "attack" for r in schema.compound_rules],
        dtype=bool,
    )
    return ids, to_attack


def _n_eff(w: np.ndarray) -> float:
    s2 = float((w * w).sum())
    return float(w.sum()) ** 2 / s2 if s2 > 0 else 0.0


@dataclass(frozen=True, slots=True)
class PatternStats:
    """Weighted per-pattern statistics of one set of rows in one label mode."""

    gr: np.ndarray  # (P,) growth rate in the pattern's direction
    supp_dir: np.ndarray  # (P,) support in the pattern's own class
    supp_oth: np.ndarray  # (P,) support in the other class
    n_dir: np.ndarray  # (P,) effective n of the pattern's own class
    n_oth: np.ndarray  # (P,) effective n of the other class
    checkable: bool  # both classes have >= min_samples effective rows
    prec: np.ndarray  # (P,) share of the pattern's firings in its class
    n_fire: np.ndarray  # (P,) effective n of the pattern's firings
    cov_attack: float
    cov_benign: float
    n_eff_attack: float
    n_eff_benign: float
    min_samples: int

    @property
    def valid(self) -> np.ndarray:
        return (
            self.checkable
            & (self.gr >= MIN_GROWTH_RATE)
            & (self.supp_dir >= MIN_COVERAGE)
        )

    @property
    def invalid_confident(self) -> np.ndarray:
        """The keep rule is broken with 99% confidence (one-sided): the
        upper bound of the growth rate is < MIN_GROWTH_RATE (delta method on
        log GR, proportions smoothed by half a count), or the upper bound of
        the coverage is < MIN_COVERAGE (normal approximation under p =
        MIN_COVERAGE). Point estimates alone let borderline patterns cross
        the rule by sampling noise."""
        z = Z_ONE_SIDED_99
        nd, no = self.n_dir, self.n_oth
        pd_ = (self.supp_dir * nd + 0.5) / (nd + 1)
        po = (self.supp_oth * no + 0.5) / (no + 1)
        se_log = np.sqrt((1 - pd_) / (pd_ * (nd + 1)) + (1 - po) / (po * (no + 1)))
        gr_bad = np.log(pd_ / po) + z * se_log < math.log(MIN_GROWTH_RATE)
        se_cov = math.sqrt(MIN_COVERAGE * (1 - MIN_COVERAGE)) / np.sqrt(
            np.maximum(nd, 1e-12)
        )
        cov_bad = self.supp_dir + z * se_cov < MIN_COVERAGE
        return self.checkable & (gr_bad | cov_bad)


def pattern_stats(
    rows: WindowRows,
    to_attack: np.ndarray,
    mode: str,
    audit: np.ndarray | None,
    min_samples: int,
    audit_rate: float = AUDIT_RATE,
) -> PatternStats | None:
    if rows.labeled is None or rows.is_attack is None or len(rows) == 0:
        return None
    lab = rows.labeled.astype(float)
    if mode == "oracle":
        w = lab
    else:
        if rows.flagged is None:
            return None
        flg = rows.flagged.astype(float)
        if mode == "escalated":
            w = lab * flg
        elif mode == "audit":
            if audit is None or len(audit) != len(rows):
                return None
            w = lab * (flg + (1 - flg) * audit.astype(float) / audit_rate)
        else:
            raise ValueError(f"unknown label mode {mode!r}")

    fires = np.concatenate([rows.fires, rows.rule_match_labeled], axis=1).astype(float)
    att = rows.is_attack.astype(float)
    wa, wb = w * att, w * (1 - att)
    A, B = wa.sum(), wb.sum()
    fa, fb = fires.T @ wa, fires.T @ wb
    sa = fa / A if A > 0 else np.zeros_like(fa)
    sb = fb / B if B > 0 else np.zeros_like(fb)
    supp_dir = np.where(to_attack, sa, sb)
    supp_oth = np.where(to_attack, sb, sa)
    n_a, n_b = _n_eff(wa), _n_eff(wb)

    f_tot = fa + fb
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(f_tot > 0, np.where(to_attack, fa, fb) / f_tot, np.nan)
        n_fire = np.where(f_tot > 0, f_tot**2 / (fires.T @ (w * w)), 0.0)

    def coverage(cls_w: np.ndarray, total: float, cols: np.ndarray) -> float:
        if total <= 0 or not cols.any():
            return math.nan
        covered = fires[:, cols].sum(axis=1) > 0
        return float((cls_w * covered).sum() / total)

    return PatternStats(
        gr=supp_dir / np.maximum(supp_oth, _EPS),
        supp_dir=supp_dir,
        supp_oth=supp_oth,
        n_dir=np.where(to_attack, n_a, n_b),
        n_oth=np.where(to_attack, n_b, n_a),
        checkable=bool(n_a >= min_samples and n_b >= min_samples),
        prec=prec,
        n_fire=n_fire,
        cov_attack=coverage(wa, A, to_attack),
        cov_benign=coverage(wb, B, ~to_attack),
        n_eff_attack=n_a,
        n_eff_benign=n_b,
        min_samples=min_samples,
    )


def _drop_significant(
    p_win: float, n_win: float, p_ref: float, n_ref: float, min_samples: int
) -> bool:
    if (
        not (np.isfinite(p_win) and np.isfinite(p_ref))
        or n_win < min_samples
        or n_ref < min_samples
    ):
        return False
    se = math.sqrt(max(p_ref * (1 - p_ref), 1e-12) * (1 / n_win + 1 / n_ref))
    return bool(p_win < p_ref - Z_ONE_SIDED_99 * se and p_ref - p_win >= PRECISION_DROP)


def compare(
    win: PatternStats | None, ref: PatternStats | None, ids: list[str], mode: str
) -> dict:
    out: dict = {f"pv_{mode}_{f}": None for f in _FIELDS}
    if win is None or ref is None:
        return out
    k = f"pv_{mode}_"
    out[k + "cov_attack"] = win.cov_attack
    out[k + "cov_attack_ref"] = ref.cov_attack
    out[k + "cov_benign"] = win.cov_benign
    out[k + "cov_benign_ref"] = ref.cov_benign
    out[k + "n_eff_attack"] = win.n_eff_attack
    out[k + "n_eff_benign"] = win.n_eff_benign

    check = ref.valid & win.checkable
    invalid = check & win.invalid_confident
    invalid_point = check & ~win.valid
    out[k + "n_checked"] = int(check.sum())
    out[k + "n_invalid"] = int(invalid.sum())
    out[k + "invalid_patterns"] = ";".join(i for i, b in zip(ids, invalid) if b)
    out[k + "n_invalid_point"] = int(invalid_point.sum())
    out[k + "invalid_patterns_point"] = ";".join(
        i for i, b in zip(ids, invalid_point) if b
    )
    if check.any():
        ratio = np.where(
            check,
            np.minimum(win.gr / MIN_GROWTH_RATE, win.supp_dir / MIN_COVERAGE),
            np.inf,
        )
        j = int(np.argmin(ratio))
        out[k + "worst_pattern"] = ids[j]
        out[k + "worst_gr_ratio"] = float(min(ratio[j], 1e6))

    n = win.min_samples
    drop = np.array(
        [
            _drop_significant(win.prec[j], win.n_fire[j], ref.prec[j], ref.n_fire[j], n)
            for j in range(len(ids))
        ]
    )
    out[k + "n_prec_drop"] = int(drop.sum())
    out[k + "prec_drop_patterns"] = ";".join(i for i, b in zip(ids, drop) if b)
    d = ref.prec - win.prec
    if np.isfinite(d).any():
        out[k + "worst_prec_drop"] = float(np.nanmax(d))

    cov = _drop_significant(
        win.cov_attack, win.n_eff_attack, ref.cov_attack, ref.n_eff_attack, n
    ) or _drop_significant(
        win.cov_benign, win.n_eff_benign, ref.cov_benign, ref.n_eff_benign, n
    )
    out[k + "contrast_elevated"] = bool(invalid.any())
    out[k + "contrast_point_elevated"] = bool(invalid_point.any())
    out[k + "precision_elevated"] = bool(drop.any())
    out[k + "coverage_elevated"] = bool(cov)
    out[k + "elevated"] = bool(invalid.any() or drop.any() or cov)
    return out


class PatternValidityMonitor:
    """Reference statistics per label mode for one deployment."""

    def __init__(
        self,
        schema: DynamicSchema,
        ref_rows: WindowRows,
        ref_groups: Sequence[AlertGroup],
        min_samples: int,
        audit_rate: float = AUDIT_RATE,
    ) -> None:
        self.ids, self.to_attack = pattern_meta(schema)
        self.min_samples = min_samples
        self.audit_rate = audit_rate
        audit = audit_mask(ref_groups, audit_rate)
        self.ref = {
            m: pattern_stats(
                ref_rows, self.to_attack, m, audit, min_samples, audit_rate
            )
            for m in LABEL_MODES
        }

    def n_valid_ref(self, mode: str = "oracle") -> int:
        r = self.ref.get(mode)
        return int(r.valid.sum()) if r is not None else 0

    def evaluate(self, rows: WindowRows, groups: Sequence[AlertGroup]) -> dict:
        out: dict = dict.fromkeys(PATTERN_VALIDITY_COLUMNS)
        if rows.fires.shape[1] + rows.rule_match_labeled.shape[1] != len(self.ids):
            return out  # schema changed without a new reference
        audit = audit_mask(groups, self.audit_rate)
        for m in LABEL_MODES:
            win = pattern_stats(
                rows, self.to_attack, m, audit, self.min_samples, self.audit_rate
            )
            out.update(compare(win, self.ref[m], self.ids, m))
        return out

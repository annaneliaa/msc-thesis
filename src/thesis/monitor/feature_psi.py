"""
Per-feature PSI over every model input (observe-only; logged by SignalLog).

Signal 1 and novelty only see the deployed Vk's predicates (~30 on CSCAS),
while the model reads every encoded feature (~290: all mined symbolic
features + base features). The synthetic drift test (2026-11-24) showed that
drift in model features outside Vk hurts F1 without any input signal seeing
it. This signal closes that gap: PSI of each encoded model feature, window
vs reference.

Binning (fixed from the reference rows only):
* <= MAX_CATEGORIES distinct reference values -> one bin per value, plus one
  bin for values never seen in the reference;
* otherwise -> N_BINS quantile bins from the reference (values beyond the
  reference range land in the outer bins, so a scale shift is visible).
NaN is binned as its own value (sentinel).

Decisions (two variants, both logged):
* fpsi_elevated: any feature's PSI above ITS OWN day-by-day threshold, same
  rule as thesis.monitor.noise_aware: the largest leave-one-out PSI of that
  feature over 24h blocks of the reference period, floored at MIN_THRESHOLD
  (so a feature that is constant in the reference doesn't fire on a handful
  of new values). Per feature, because on CSCAS a few features swing
  hugely from day to day (a single max-over-features threshold came out at
  5.7 and could never fire);
* fpsi_fixed_elevated: any feature PSI > FIXED_PSI (0.2, the common "major
  shift" rule of thumb), no baseline.
fpsi_max_ratio = max over features of PSI / own threshold (> 1 = elevated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thesis.monitor.noise_aware import INPUT_BLOCK_SECONDS, time_blocks

N_BINS = 10
MAX_CATEGORIES = 12
FIXED_PSI = 0.2
MIN_THRESHOLD = 0.01
_NAN = -9.87654321e12
_EPS = 1e-6

FEATURE_PSI_COLUMNS = (
    "fpsi_max",
    "fpsi_top_feature",
    "fpsi_n_features",
    "fpsi_n_over_fixed",
    "fpsi_max_ratio",
    "fpsi_top_ratio_feature",
    "fpsi_n_elevated",
    "fpsi_elevated",
    "fpsi_fixed_elevated",
)


@dataclass(frozen=True, slots=True)
class _Binner:
    categorical: bool
    points: np.ndarray  # sorted category values, or quantile edges

    @property
    def n_bins(self) -> int:
        return len(self.points) + 1  # +1: unseen category / top numeric bin

    def codes(self, x: np.ndarray) -> np.ndarray:
        if self.categorical:
            idx = np.searchsorted(self.points, x)
            idx_c = np.clip(idx, 0, len(self.points) - 1)
            seen = self.points[idx_c] == x
            return np.where(seen, idx_c, len(self.points))
        return np.searchsorted(self.points, x, side="right")


def _values(X: pd.DataFrame) -> np.ndarray:
    v = X.to_numpy(dtype=float, na_value=np.nan)
    return np.where(np.isnan(v), _NAN, v)


def _fit_binner(col: np.ndarray) -> _Binner:
    uniq = np.unique(col)
    if len(uniq) <= MAX_CATEGORIES:
        return _Binner(True, uniq)
    edges = np.unique(np.quantile(col, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    return _Binner(False, edges)


def _psi(h_ref: np.ndarray, h_obs: np.ndarray) -> float:
    a = np.clip(h_ref, _EPS, None)
    b = np.clip(h_obs, _EPS, None)
    return float(((b - a) * np.log(b / a)).sum())


class FeaturePsiMonitor:
    def __init__(
        self,
        X_ref: pd.DataFrame,
        ts_ref: np.ndarray,
        block_seconds: int = INPUT_BLOCK_SECONDS,
        fixed_psi: float = FIXED_PSI,
    ) -> None:
        if len(X_ref) == 0:
            raise ValueError("empty reference")
        self.columns = list(X_ref.columns)
        self.fixed_psi = fixed_psi
        v = _values(X_ref)
        self.binners = [_fit_binner(v[:, j]) for j in range(v.shape[1])]
        codes = self._codes(v)
        self.ref_counts = self._counts(codes, np.ones(len(v), dtype=bool))
        self.ref_hist = [c / c.sum() for c in self.ref_counts]

        blocks = time_blocks(np.asarray(ts_ref), block_seconds)
        block_psi = np.full((len(blocks), len(self.binners)), np.nan)
        if len(blocks) >= 2:
            for b, m in enumerate(blocks):
                inside = self._counts(codes, m)
                block_psi[b] = [
                    _psi(_norm(r - i), _norm(i))
                    for r, i in zip(self.ref_counts, inside)
                ]
        self.n_blocks = len(blocks)
        self.block_psi = block_psi
        self.thresholds = (
            pd.Series(
                np.maximum(np.nanmax(block_psi, axis=0), MIN_THRESHOLD),
                index=self.columns,
            )
            if len(blocks) >= 2
            else None
        )

    def _codes(self, v: np.ndarray) -> np.ndarray:
        return np.stack(
            [b.codes(v[:, j]) for j, b in enumerate(self.binners)], axis=1
        ).astype(np.int32)

    def _counts(self, codes: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
        sel = codes[mask]
        return [
            np.bincount(sel[:, j], minlength=b.n_bins).astype(float)
            for j, b in enumerate(self.binners)
        ]

    def psi_per_feature(self, X: pd.DataFrame) -> pd.Series:
        X = X.reindex(columns=self.columns, fill_value=0)
        codes = self._codes(_values(X))
        obs = self._counts(codes, np.ones(len(X), dtype=bool))
        return pd.Series(
            [_psi(r, _norm(o)) for r, o in zip(self.ref_hist, obs)],
            index=self.columns,
        )

    def evaluate(self, X: pd.DataFrame) -> dict:
        out: dict = dict.fromkeys(FEATURE_PSI_COLUMNS)
        if len(X) == 0 or not self.columns:
            return out
        psi = self.psi_per_feature(X)
        top = float(psi.max())
        out.update(
            fpsi_max=top,
            fpsi_top_feature=str(psi.idxmax()),
            fpsi_n_features=len(psi),
            fpsi_n_over_fixed=int((psi > self.fixed_psi).sum()),
            fpsi_fixed_elevated=bool(top > self.fixed_psi),
        )
        if self.thresholds is not None:
            ratio = psi / self.thresholds
            out.update(
                fpsi_max_ratio=float(ratio.max()),
                fpsi_top_ratio_feature=str(ratio.idxmax()),
                fpsi_n_elevated=int((ratio > 1).sum()),
                fpsi_elevated=bool((ratio > 1).any()),
            )
        return out


def _norm(c: np.ndarray) -> np.ndarray:
    s = c.sum()
    return c / s if s > 0 else c

"""
On-demand, single-instance SHAP/LIME case studies.

Sibling to experiments/temporal_decay.py: reuses its fit_source_window
(mine+fit window 0's train split) and encode_target_window (encode any
later window under that frozen schema) so a case study mines/fits *exactly*
the same way the sweep did, without re-running the sweep or persisting the
fitted model anywhere. Meant for the "why does this specific prediction
look the way it does at this horizon" question the aggregate drift plots in
temporal_decay_eda.ipynb can't answer -- e.g. "explain the model's most
confident false positive at horizon 5" -- picked and explained fresh, each
time it's asked, rather than added to every sweep run's cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.experiments._shared import sample_rows
from thesis.system_eval.temporal_decay import (
    SourceWindowFit,
    encode_target_window,
    fit_source_window,
)
from thesis.metrics.shortlist import ShortlistedConfig
from thesis.schemas.features import FeatureSchema
from thesis.training.explain import (
    compute_lime_signed_importances,
    compute_shap_signed_importances,
)


def select_error_instances(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    kind: str = "both",
    top_n: int = 5,
) -> list[int]:
    """Row indices (positional, into whatever X/y these came from) of the
    model's most confidently *wrong* predictions -- ranked by
    |proba - threshold| descending, so "confidently wrong" means far past
    the threshold on the wrong side, not just barely misclassified.

    kind: "fp" (false positives), "fn" (false negatives), or "both" (up to
    top_n of each, fp first)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    confidence = np.abs(proba - threshold)

    indices: list[int] = []
    if kind in ("fp", "both"):
        fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
        indices.extend(
            sorted(fp_idx.tolist(), key=lambda i: confidence[i], reverse=True)[:top_n]
        )
    if kind in ("fn", "both"):
        fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
        indices.extend(
            sorted(fn_idx.tolist(), key=lambda i: confidence[i], reverse=True)[:top_n]
        )
    return indices


@dataclass(slots=True)
class InstanceExplanation:
    horizon_window_index: int
    row_index: int
    error_kind: str  # "fp" | "fn"
    y_true: int
    proba: float
    threshold: float
    shap_importances: dict[str, float]
    lime_importances: dict[str, float]
    lime_fidelity: float
    feature_values: dict[str, float]


def explain_instances_for_config(
    cfg: ShortlistedConfig,
    scenario: str,
    alert_groups: list,
    alert_groups_path: Path,
    n_total: int,
    base_schema: FeatureSchema,
    mining_settings_by_name: dict,
    mining_settings_path: Path,
    horizon_window_index: int,
    kind: str = "both",
    top_n: int = 5,
    train_frac_within_window: float = 0.7,
    threshold_mode: str = "fixed",
    calibrated_recall_target: float = 0.90,
    explain_background_n: int = 100,
    lime_num_samples: int = 1000,
    top_n_importances: int = 30,
    random_seed: int = 42,
    force_remine: bool = False,
) -> list[InstanceExplanation]:
    """Mine+fit `cfg` on window 0 (exactly as the sweep does), score the
    requested horizon window, pick its most confident FP/FN instances, and
    explain each one individually with SHAP + LIME. Raises if the config
    can't be fit at all (bad mining setting, single-class W_src) -- unlike
    the sweep, there's no larger batch of configs to keep going for, so a
    hard failure here should surface immediately instead of being logged
    and skipped."""
    fit: SourceWindowFit | None = fit_source_window(
        cfg=cfg,
        scenario=scenario,
        alert_groups=alert_groups,
        alert_groups_path=alert_groups_path,
        n_total=n_total,
        base_schema=base_schema,
        mining_settings_by_name=mining_settings_by_name,
        mining_settings_path=mining_settings_path,
        train_frac_within_window=train_frac_within_window,
        threshold_mode=threshold_mode,
        calibrated_recall_target=calibrated_recall_target,
        force_remine=force_remine,
    )
    if fit is None:
        raise ValueError(
            f"Could not mine/fit {cfg} on window 0 -- see printed warnings above."
        )

    if horizon_window_index == 0:
        X_h, y_h = fit.X_test, fit.y_test
    elif horizon_window_index < fit.n_windows:
        X_h, y_h, _ = encode_target_window(
            alert_groups, n_total, fit.gran, horizon_window_index, fit.schema
        )
    else:
        raise ValueError(
            f"horizon_window_index={horizon_window_index} >= n_windows={fit.n_windows}"
        )

    if len(y_h) == 0:
        print(f"    [warn] horizon {horizon_window_index} has no labeled rows")
        return []

    proba_h = fit.model.predict_proba(X_h)[:, 1]
    indices = select_error_instances(
        y_h, proba_h, fit.threshold, kind=kind, top_n=top_n
    )
    if not indices:
        print(
            f"    [info] no {kind} instances at horizon {horizon_window_index} "
            f"(threshold={fit.threshold:.3f})"
        )
        return []

    explain_background = sample_rows(fit.X_train, explain_background_n, random_seed)

    results: list[InstanceExplanation] = []
    for idx in indices:
        row = X_h.iloc[[idx]]
        y_true = int(y_h[idx])
        error_kind = "fp" if y_true == 0 else "fn"

        try:
            shap_importances = compute_shap_signed_importances(
                fit.model,
                explain_background,
                row,
                fit.feature_names,
                top_n=top_n_importances,
            )
        except Exception as exc:
            print(f"    [warn] SHAP failed for row {idx}: {exc}")
            shap_importances = {}

        lime_importances: dict[str, float] = {}
        lime_fidelity = float("nan")
        try:
            lime_result = compute_lime_signed_importances(
                fit.model,
                explain_background,
                row,
                fit.feature_names,
                top_n=top_n_importances,
                num_samples=lime_num_samples,
                random_state=random_seed,
            )
            lime_importances = lime_result.importances
            lime_fidelity = lime_result.mean_fidelity
        except Exception as exc:
            print(f"    [warn] LIME failed for row {idx}: {exc}")

        results.append(
            InstanceExplanation(
                horizon_window_index=horizon_window_index,
                row_index=int(idx),
                error_kind=error_kind,
                y_true=y_true,
                proba=float(proba_h[idx]),
                threshold=fit.threshold,
                shap_importances=shap_importances,
                lime_importances=lime_importances,
                lime_fidelity=lime_fidelity,
                feature_values=row.iloc[0].to_dict(),
            )
        )

    return results


def explanations_to_long_dataframe(
    explanations: list[InstanceExplanation], base_row: dict
) -> pd.DataFrame:
    """Long format (one row per instance x method x feature) -- mirrors
    temporal_decay.py's explanations.csv shape so the same kind of
    per-feature plotting code works on either."""
    rows: list[dict] = []
    for exp in explanations:
        meta = {
            **base_row,
            "horizon_window_index": exp.horizon_window_index,
            "row_index": exp.row_index,
            "error_kind": exp.error_kind,
            "y_true": exp.y_true,
            "proba": exp.proba,
            "threshold": exp.threshold,
            "lime_fidelity": exp.lime_fidelity,
        }
        rows.extend(
            {**meta, "method": "shap", "feature": feat, "importance": val, "rank": rank}
            for rank, (feat, val) in enumerate(exp.shap_importances.items())
        )
        rows.extend(
            {**meta, "method": "lime", "feature": feat, "importance": val, "rank": rank}
            for rank, (feat, val) in enumerate(exp.lime_importances.items())
        )
    return pd.DataFrame(rows)

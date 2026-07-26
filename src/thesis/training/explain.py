"""Shared SHAP/LIME importance helpers.

Used by train.train_eval_holdout (single-window importances) and
experiments.temporal_decay (importances tracked across horizon steps, to see
how they drift) -- both want "mean signed per-feature importance over a
sample of rows, given a fitted model and a background set", dispatched on
model type the same way, so the dispatch logic lives here once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thesis.training.model_factory import preprocess_for_estimator, unwrap_estimator


@dataclass(slots=True)
class LimeResult:
    importances: dict[str, float]
    # mean of LIME's own local-surrogate R^2 (Explanation.score) over every
    # explained row -- how well a *linear* model fits the (frozen) model's
    # behavior near each point. Not a measure of predictive accuracy; a drop
    # means the decision surface is becoming harder to approximate locally,
    # independent of whether the signed weights above are also drifting.
    mean_fidelity: float


def compute_shap_signed_importances(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 30,
) -> dict[str, float]:
    """Mean signed SHAP value per feature over `X_explain`, top `top_n` by
    |value|. Dispatches on model type: a model's own get_shap_values (e.g.
    GradientExplainer for LSTM), TreeExplainer for tree ensembles,
    LinearExplainer (through the fitted preprocessing) for linear models
    exposing coef_, else PermutationExplainer via predict_proba."""
    import shap

    estimator = unwrap_estimator(model)

    if hasattr(model, "get_shap_values"):
        bg_arr = (
            X_background.values
            if hasattr(X_background, "values")
            else np.asarray(X_background)
        )
        x_arr = (
            X_explain.values if hasattr(X_explain, "values") else np.asarray(X_explain)
        )
        vals = model.get_shap_values(bg_arr, x_arr)
    elif getattr(model, "_skip_shap", False):
        raise RuntimeError("SHAP skipped: model flagged as too expensive")
    elif hasattr(estimator, "feature_importances_"):
        sv = shap.TreeExplainer(estimator).shap_values(X_explain)
        vals = (
            sv[:, :, 1]
            if isinstance(sv, np.ndarray) and sv.ndim == 3
            else (sv[1] if isinstance(sv, list) else sv)
        )
    elif hasattr(estimator, "coef_"):
        # Linear models: LinearExplainer needs data in the same space the
        # estimator was actually fit on -- if `model` is a scaled Pipeline,
        # that's post-scaler, not raw X.
        bg_transformed = preprocess_for_estimator(model, X_background)
        x_explain_transformed = preprocess_for_estimator(model, X_explain)
        vals = shap.LinearExplainer(estimator, bg_transformed).shap_values(
            x_explain_transformed
        )
    else:
        sv = shap.Explainer(model.predict_proba, X_background)(X_explain)
        vals = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values

    mean_signed = np.asarray(vals).mean(axis=0)
    pairs = sorted(
        zip(feature_names, mean_signed), key=lambda x: abs(x[1]), reverse=True
    )
    return {name: float(v) for name, v in pairs[:top_n]}


def compute_lime_signed_importances(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 30,
    num_samples: int = 1000,
    random_state: int = 42,
) -> LimeResult:
    """Mean signed LIME local-surrogate weight per feature, averaged over
    every row in `X_explain` (one LimeTabularExplainer fit on
    `X_background`, one explain_instance call per row), plus the mean local
    fidelity (R^2) of those per-row surrogates.

    `num_features=len(feature_names)` is requested for every instance (not
    LIME's default top-k) so every instance's weight vector covers the same
    features and can be averaged directly -- letting LIME pick its own
    per-instance subset would make different instances contribute different
    features to the average. discretize_continuous=False so
    `exp.as_list()` keys are literal feature names, not binned range
    descriptions, and line up with `feature_names` directly.
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_background),
        feature_names=feature_names,
        class_names=["benign", "attack"],
        mode="classification",
        discretize_continuous=False,
        random_state=random_state,
    )

    # LIME perturbs `row` into a raw ndarray batch and calls this predict_fn
    # directly -- re-wrap as a DataFrame before handing it to `model` so a
    # scaled Pipeline's StandardScaler (fit on a DataFrame) doesn't re-warn
    # "X does not have valid feature names" on every single explain_instance
    # call (num_samples perturbations per row, batched into one predict_proba
    # call, but still once per row -- floods the log across every
    # explained row x horizon x config).
    def _predict_fn(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(x, columns=feature_names))

    n_features = len(feature_names)
    signed_sum = np.zeros(n_features)
    fidelities: list[float] = []
    x_arr = np.asarray(X_explain)
    for row in x_arr:
        exp = explainer.explain_instance(
            row,
            _predict_fn,
            labels=(1,),
            num_features=n_features,
            num_samples=num_samples,
        )
        weights = dict(exp.as_list(label=1))
        signed_sum += np.array([weights.get(name, 0.0) for name in feature_names])
        fidelities.append(float(exp.score))

    mean_signed = signed_sum / len(x_arr)
    pairs = sorted(
        zip(feature_names, mean_signed), key=lambda x: abs(x[1]), reverse=True
    )
    return LimeResult(
        importances={name: float(v) for name, v in pairs[:top_n]},
        mean_fidelity=float(np.mean(fidelities)),
    )

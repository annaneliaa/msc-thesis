from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
from typing import Callable, Any

from thesis.grouping._device import resolve_device
from thesis.training.anomaly_models import BernoulliOneClass, BinaryAutoencoder
from thesis.training.gpu_models import (
    CuMLRandomForestClassifierWrapper,
    TorchMLPClassifier,
)


def _xgb_device() -> str:
    """resolve_device()'s mps->cuda->cpu chain, collapsed to what xgboost's
    own `device` param understands (no mps concept there)."""
    d = resolve_device()
    return "cuda" if d.type == "cuda" else "cpu"


def _warn_unweighted(model_name: str, extra_kwargs: dict) -> None:
    if extra_kwargs.get("class_weight") or extra_kwargs.get("scale_pos_weight"):
        print(
            f"  [warn] '{model_name}' has no class_weight/sample_weight support -- "
            f"training on the natural-ratio pool unweighted despite "
            f"pool_condition='class_weighted'."
        )


def _build_mlp(**extra) -> MLPClassifier:
    _warn_unweighted("mlp", extra)
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )


def _build_rf_gpu(**extra) -> CuMLRandomForestClassifierWrapper:
    _warn_unweighted("rf_gpu", extra)
    return CuMLRandomForestClassifierWrapper(n_estimators=200, random_state=42)


def _build_xgboost(**extra) -> XGBClassifier:
    device = _xgb_device()
    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        device=device,
        scale_pos_weight=extra.get("scale_pos_weight"),
    )
    if device == "cuda":
        # Same multi-process GPU contention concern as torch_nn/rf_gpu (see
        # train.py's _n_jobs line) -- only when actually GPU-backed; the
        # CPU case is unaffected and keeps using n_jobs=-1 for permutation
        # importance.
        model._gpu_model = True
    return model


# Each entry is a builder taking **extra_kwargs (the pool-sampling condition's
# ready-to-use imbalance kwargs, e.g. {"class_weight": "balanced",
# "scale_pos_weight": ...} from pool_sampling.class_weighted_extra_kwargs) --
# see get_model_factory below. Most models ignore extra_kwargs entirely
# (logreg/rf are unconditionally class_weight="balanced" already; mlp/rf_gpu
# have no such knob at all and print a warning instead of silently ignoring
# it).
MODEL_FACTORIES = {
    # Base features mix wildly different scales (raw ports up to 65535, alert
    # counts, alongside 0/1 symbolic indicators and [0,1] similarity scores).
    # Unscaled, that's exactly what makes lbfgs/liblinear fail to converge and
    # burn their full iteration budget on every fit (observed empirically:
    # hundreds of ConvergenceWarnings across both the screening-sweep and
    # temporal-decay experiments). StandardScaler is standard practice for
    # gradient-based linear model fitting regardless of the convergence angle.
    # See training/model_factory.unwrap_estimator for how callers that
    # introspect `.coef_` (train.py, visualization/plots.py) reach the
    # underlying LogisticRegression through this wrapper.
    "logreg": lambda **_: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    ),
    "logreg_l1": lambda **_: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    penalty="l1",
                    C=1.0,
                ),
            ),
        ]
    ),
    "rf": lambda **_: RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "mlp": _build_mlp,
    "xgboost": _build_xgboost,
    "torch_nn": lambda **extra: TorchMLPClassifier(
        hidden_sizes=(128, 64),
        random_state=42,
        pos_weight=extra.get("scale_pos_weight"),
    ),
    "rf_gpu": _build_rf_gpu,
    "bernoulli_oc": lambda **_: BernoulliOneClass(contamination=0.05, alpha=1.0),
    "autoencoder_oc": lambda **_: BinaryAutoencoder(
        hidden_layer_sizes=(64, 32, 64),
        max_iter=500,
        contamination=0.05,
        random_state=42,
    ),
    # Unlike bernoulli_oc/autoencoder_oc (both built for binary feature
    # vectors -- see anomaly_models.py's module docstring), OneClassSVM
    # handles raw numeric features directly, so it's scaled the same way
    # "logreg" above is (distance-based, not scale-invariant). nu=0.05
    # matches the other two anomaly models' contamination=0.05 convention.
    "ocsvm": lambda **_: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", OneClassSVM(kernel="rbf", nu=0.05)),
        ]
    ),
}


def get_model_factory(model_name: str, **extra_kwargs) -> Callable[[], Any]:
    """Returns a zero-arg factory for model_name, with extra_kwargs (e.g. the
    active pool-sampling condition's class_weight/scale_pos_weight) baked in.
    Every existing call site that doesn't pass extra_kwargs is unaffected."""
    if model_name not in MODEL_FACTORIES:
        raise KeyError(
            f"Unknown model_name '{model_name}'. "
            f"Available models: {list(MODEL_FACTORIES.keys())}"
        )

    builder = MODEL_FACTORIES[model_name]
    return lambda: builder(**extra_kwargs)


def unwrap_estimator(model: Any) -> Any:
    """Return the final fitted estimator inside a Pipeline (e.g. the scaled
    "logreg"/"logreg_l1" factories above), or `model` itself if it isn't a
    Pipeline. Use this before introspecting fitted attributes like
    `.coef_`/`.feature_importances_`, which live on the estimator, not on the
    Pipeline wrapping it -- `hasattr(pipeline, "coef_")` is always False even
    though the wrapped estimator has one."""
    return model.steps[-1][1] if isinstance(model, Pipeline) else model


def preprocess_for_estimator(model: Any, X: Any) -> Any:
    """Apply every step in `model` except the final estimator (e.g. the
    scaler) to `X`, or return `X` unchanged if `model` isn't a Pipeline.

    Needed when handing raw X directly to something built against the
    *unwrapped* estimator (e.g. shap.LinearExplainer(unwrap_estimator(model),
    ...)) -- that estimator was fit on scaled data, so it needs to see data
    in that same space, not the raw X the Pipeline as a whole accepts."""
    if isinstance(model, Pipeline):
        for _, step in model.steps[:-1]:
            X = step.transform(X)
    return X

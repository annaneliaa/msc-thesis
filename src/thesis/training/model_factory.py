from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Callable, Any

from thesis.training.anomaly_models import BernoulliOneClass, BinaryAutoencoder

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
    "logreg": lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    ),
    "logreg_l1": lambda: Pipeline(
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
    "rf": lambda: RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "mlp": lambda: MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    ),
    "bernoulli_oc": lambda: BernoulliOneClass(contamination=0.05, alpha=1.0),
    "autoencoder_oc": lambda: BinaryAutoencoder(
        hidden_layer_sizes=(64, 32, 64),
        max_iter=500,
        contamination=0.05,
        random_state=42,
    ),
}


def get_model_factory(model_name: str) -> Callable[[], Any]:
    if model_name not in MODEL_FACTORIES:
        raise KeyError(
            f"Unknown model_name '{model_name}'. "
            f"Available models: {list(MODEL_FACTORIES.keys())}"
        )

    model_factory = MODEL_FACTORIES[model_name]
    return model_factory


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

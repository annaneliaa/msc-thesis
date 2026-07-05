from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from typing import Callable, Any

from thesis.training.anomaly_models import BernoulliOneClass, BinaryAutoencoder

MODEL_FACTORIES = {
    "logreg": lambda: LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
    "logreg_l1": lambda: LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        penalty="l1",
        C=1.0,
    ),
    "random_forest": lambda: RandomForestClassifier(
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

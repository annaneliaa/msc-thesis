from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from typing import Callable, Any

from thesis.training.lstm_classifier import LSTMClassifier

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
    "logreg_sweep": lambda: LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
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
    "lstm": lambda: LSTMClassifier(),
    "iforest": lambda: IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    ),
    "ocsvm": lambda: OneClassSVM(
        kernel="rbf",
        nu=0.1,
        gamma="scale",
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

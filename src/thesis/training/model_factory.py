from sklearn.linear_model import LogisticRegression
from typing import Callable, Any

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
}


def get_model_factory(model_name: str) -> Callable[[], Any]:
    if model_name not in MODEL_FACTORIES:
        raise KeyError(
            f"Unknown model_name '{model_name}'. "
            f"Available models: {list(MODEL_FACTORIES.keys())}"
        )

    model_factory = MODEL_FACTORIES[model_name]
    return model_factory

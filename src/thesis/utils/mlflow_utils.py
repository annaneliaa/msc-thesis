import mlflow
from thesis.paths import ARTIFACTS_DIR

MLFLOW_DB = ARTIFACTS_DIR / "mlflow.db"


def start_run(run_name: str):
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.resolve()}")
    mlflow.set_experiment("thesis")
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    mlflow.log_params(params)


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def log_artifact(path: str):
    mlflow.log_artifact(path)


def set_tags(tags: dict) -> None:
    mlflow.set_tags(tags)

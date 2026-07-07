"""Mlflow tracking is disabled -- these are no-ops so every mining job's
`with start_run(...): log_params(...); log_metrics(...); ...` calls keep
working unchanged without touching mlflow or its sqlite store at all."""

import contextlib


@contextlib.contextmanager
def start_run(run_name: str):
    yield None


def log_params(params: dict) -> None:
    pass


def log_metrics(metrics: dict) -> None:
    pass


def log_artifact(path: str) -> None:
    pass


def set_tags(tags: dict) -> None:
    pass

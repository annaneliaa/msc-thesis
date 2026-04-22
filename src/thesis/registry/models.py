from pathlib import Path

from thesis.paths import MODELS_DIR

# registry layer for listing available models, resolving paths, validating existence
# file system focused


def list_all_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]


def get_model_path(name: str, version: str) -> Path:
    return MODELS_DIR / name / version


def load_trained_model(name: str, version: str):
    return 0


# should return a full loaded model object so that inference code doesnt have to deal with raw paths

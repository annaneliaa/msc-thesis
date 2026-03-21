from pathlib import Path

from thesis.paths import MODELS_DIR


def list_all_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]


def get_model_path(name: str, version: str) -> Path:
    return MODELS_DIR / name / version

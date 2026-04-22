from pathlib import Path

from thesis.paths import MODELS_DIR, MODEL_FILENAME, METADATA_FILENAME


# registry layer for listing available models, resolving paths, validating existence
# file system focused
def list_all_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]


def get_model_path(name: str, version: str) -> Path:
    return MODELS_DIR / name / version


def resolve_model_paths(name: str, version: str) -> tuple[Path, Path]:
    model_dir = get_model_path(name, version)
    model_path = model_dir / MODEL_FILENAME
    metadata_path = model_dir / METADATA_FILENAME

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    return model_path, metadata_path

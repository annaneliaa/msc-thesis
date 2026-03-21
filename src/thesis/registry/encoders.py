from pathlib import Path

from thesis.paths import ENCODERS_DIR


def list_all_encoders() -> list[str]:
    if not ENCODERS_DIR.exists():
        return []
    return [p.name for p in ENCODERS_DIR.iterdir() if p.is_dir()]


def get_encoder_path(name: str, version: str) -> Path:
    return ENCODERS_DIR / name / version

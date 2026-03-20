from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
ARTIFACTS_DIR = ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
MINING_DIR = ARTIFACTS_DIR / "mining"
LOGS_DIR = ARTIFACTS_DIR / "logs"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CACHE_DIR = ARTIFACTS_DIR / "cache"


def ensure_artifact_dirs() -> None:
    for path in [ARTIFACTS_DIR, MODELS_DIR, MINING_DIR, LOGS_DIR, RUNS_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)

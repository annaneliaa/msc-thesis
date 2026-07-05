from pathlib import Path
import datetime

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
ARTIFACTS_DIR = ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"
MINING_DIR = ARTIFACTS_DIR / "mining"
LOGS_DIR = ARTIFACTS_DIR / "logs"
RUNS_DIR = ARTIFACTS_DIR / "runs"
CACHE_DIR = ARTIFACTS_DIR / "cache"
FEATURE_DIR = ARTIFACTS_DIR / "features"

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
ABSTRACTION_MAP_PATH = ROOT / "src/thesis/configs/abstraction_map.json"


def ensure_artifact_dirs() -> None:
    for path in [
        ARTIFACTS_DIR,
        MODELS_DIR,
        ENCODERS_DIR,
        MINING_DIR,
        LOGS_DIR,
        RUNS_DIR,
        CACHE_DIR,
        FEATURE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_mining_run_dir(run_name: str) -> Path:
    run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MINING_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_feature_run_dir(run_name: str) -> Path:
    run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = FEATURE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

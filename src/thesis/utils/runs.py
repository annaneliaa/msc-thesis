from datetime import datetime
from pathlib import Path
import json
import yaml
import pandas as pd

from thesis.paths import MINING_DIR


def make_run_id(run_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{run_name}"


def create_run_dir(run_name: str) -> Path:
    run_id = make_run_id(run_name)
    run_dir = MINING_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_dataframe_artifact(df: pd.DataFrame, run_dir: Path, name: str) -> Path:
    path = run_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def write_manifest(run_dir: Path, config: dict, metadata: dict) -> None:
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

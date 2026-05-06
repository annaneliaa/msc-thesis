from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from thesis.schemas.mining import MiningFilterConfig

from thesis.paths import CONFIG_DIR


class AppConfig(BaseModel):
    name: str = "msc-thesis"
    env: str = "dev"
    host: str = "127.0.0.1"
    port: int = 8000


class ArtifactConfig(BaseModel):
    base_dir: str = "artifacts"
    model_dir: str = "artifacts/models"
    mining_dir: str = "artifacts/mining"
    runs_dir: str = "artifacts/runs"


class ModelConfig(BaseModel):
    scenario: str = "default"
    model_name: str = "dummy-model"
    model_version: str = "0.1.0"


class EncoderConfig(BaseModel):
    encoder_name: str = "dummy-encoder"
    encoder_version: str = "0.1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="THESIS_", extra="ignore")

    app: AppConfig = AppConfig()
    artifacts: ArtifactConfig = ArtifactConfig()
    model: ModelConfig = ModelConfig()
    encoder: EncoderConfig = EncoderConfig()


def load_yaml_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(config_name: str = "base.yaml") -> Settings:
    cfg = load_yaml_config(CONFIG_DIR / config_name)
    return Settings(**cfg)


def load_mining_filter_config(path: Path) -> "MiningFilterConfig":
    data = load_yaml_config(path)
    return MiningFilterConfig(**data)

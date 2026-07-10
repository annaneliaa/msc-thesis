from datetime import datetime, timezone
from pathlib import Path

import pytest

from thesis.features.dynamic_schema_registry import DynamicSchemaRegistry
from thesis.schemas.dynamic_schema import DynamicSchema

_WINDOW_START = datetime(2022, 1, 20, tzinfo=timezone.utc)
_WINDOW_END = datetime(2022, 1, 21, tzinfo=timezone.utc)


def _build_fn(version: int) -> DynamicSchema:
    return DynamicSchema(
        version=version,
        mined_at=datetime(2022, 1, 26, tzinfo=timezone.utc),
        mining_window_start=_WINDOW_START,
        mining_window_end=_WINDOW_END,
        base_attack_rate=0.5,
        single_predicates=[],
        compound_rules=[],
    )


def test_deploy_first_version(tmp_path: Path):
    registry = DynamicSchemaRegistry(root_dir=tmp_path)
    path = registry.deploy("cscas", _build_fn)

    assert path.exists()
    schema = registry.load_deployed("cscas")
    assert schema.version == 1
    assert schema.deployed_at is not None
    assert schema.superseded_at is None


def test_deploy_second_version_supersedes_first(tmp_path: Path):
    registry = DynamicSchemaRegistry(root_dir=tmp_path)
    registry.deploy("cscas", _build_fn)
    registry.deploy("cscas", _build_fn)

    history = registry.history("cscas")
    assert [h["version"] for h in history] == [1, 2]
    assert history[0]["superseded_at"] is not None
    assert history[1]["superseded_at"] is None

    deployed = registry.load_deployed("cscas")
    assert deployed.version == 2
    assert deployed.superseded_at is None

    v1 = registry.load_version("cscas", 1)
    assert v1.superseded_at is not None


def test_load_deployed_raises_when_nothing_deployed(tmp_path: Path):
    registry = DynamicSchemaRegistry(root_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.load_deployed("cscas")


def test_load_version_raises_for_unknown_version(tmp_path: Path):
    registry = DynamicSchemaRegistry(root_dir=tmp_path)
    registry.deploy("cscas", _build_fn)
    with pytest.raises(KeyError):
        registry.load_version("cscas", 99)


def test_manifest_shape(tmp_path: Path):
    registry = DynamicSchemaRegistry(root_dir=tmp_path)
    registry.deploy("cscas", _build_fn)

    manifest_path = tmp_path / "cscas" / "dynamic" / "manifest.json"
    assert manifest_path.exists()

    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["scenario_name"] == "cscas"
    assert manifest["deployed"] == 1
    assert manifest["history"][0]["path"] == "schemas/1.json"


def test_scenarios_are_isolated(tmp_path: Path):
    registry = DynamicSchemaRegistry(root_dir=tmp_path)
    registry.deploy("cscas", _build_fn)
    registry.deploy("fox", _build_fn)

    assert registry.load_deployed("cscas").version == 1
    assert registry.load_deployed("fox").version == 1

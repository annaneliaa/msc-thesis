import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from thesis.inference.runtime_models import SklearnTabularModel


def _make_mock_model() -> SklearnTabularModel:
    inner = MagicMock()
    inner.predict_proba.return_value = np.array([[0.2, 0.8]])
    return SklearnTabularModel(
        model_name="test-model",
        model_version="0.0.1",
        schema_name="test-schema",
        features=["feat_a", "feat_b"],
        model=inner,
    )


@pytest.fixture(scope="session")
def app():
    # main.py calls load_model() and ensure_artifact_dirs() at module level.
    # patch("thesis.api.main.X") won't work here because entering that patch
    # context triggers an import of main.py (running the real functions first).
    # Instead we patch at the *source* modules so that when main.py is freshly
    # imported inside the context it picks up the mocks via its own imports.
    sys.modules.pop("thesis.api.main", None)
    with (
        patch(
            "thesis.inference.model_loader.load_model", return_value=_make_mock_model()
        ),
        patch("thesis.paths.ensure_artifact_dirs"),
    ):
        from thesis.api.main import app as _app

        yield _app


@pytest.fixture(scope="session")
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_model():
    return _make_mock_model()

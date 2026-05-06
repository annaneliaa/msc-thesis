"""Tests for thesis.api.main.

The app loads a model at import time; conftest.py patches load_model
so these tests run without real model artifacts on disk.
"""

# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_body(client):
    r = client.get("/health")
    assert r.json() == {"status": "ok"}


def test_health_is_idempotent(client):
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.json() == r2.json()


# ---------------------------------------------------------------------------
# Unknown routes
# ---------------------------------------------------------------------------


def test_unknown_route_returns_404(client):
    r = client.get("/nonexistent")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /predict  (uncomment when the endpoint is wired up)
# ---------------------------------------------------------------------------

# def test_predict_returns_200(client):
#     r = client.post("/predict", json={"text": "possible attack detected"})
#     assert r.status_code == 200
#
#
# def test_predict_response_schema(client):
#     r = client.post("/predict", json={"text": "possible attack detected"})
#     data = r.json()
#     assert set(data.keys()) >= {"label", "score", "model_name", "model_version"}
#
#
# def test_predict_label_is_binary(client):
#     r = client.post("/predict", json={"text": "possible attack detected"})
#     assert r.json()["label"] in (0, 1)
#
#
# def test_predict_score_in_range(client):
#     r = client.post("/predict", json={"text": "possible attack detected"})
#     score = r.json()["score"]
#     assert 0.0 <= score <= 1.0
#
#
# def test_predict_missing_text_returns_422(client):
#     r = client.post("/predict", json={})
#     assert r.status_code == 422
#
#
# def test_predict_wrong_type_returns_422(client):
#     r = client.post("/predict", json={"text": 42})
#     assert r.status_code == 422

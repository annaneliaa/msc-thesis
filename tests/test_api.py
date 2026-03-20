from fastapi.testclient import TestClient

from thesis.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    r = client.post("/predict", json={"text": "possible attack detected"})
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in [0, 1]
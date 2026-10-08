import os
import sys
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import main as app_main  # noqa: E402
from emotion_model import predict  # noqa: E402
from emotion_model.model_store import ModelStore  # noqa: E402


class FakeModel:
    """28-label model: 'happy' -> joy (17), 'sad' -> sadness (25), else nothing."""

    def predict_proba(self, texts):
        probs = np.full((len(texts), 28), 0.1)
        for i, t in enumerate(texts):
            if "happy" in t:
                probs[i, 17] = 0.9
            if "sad" in t:
                probs[i, 25] = 0.6
        return probs


class BrokenModel:
    def predict_proba(self, texts):
        raise RuntimeError("boom")


class FakeRegistry:
    """Stands in for the MLflow registry: alias -> version, version -> model."""

    def __init__(self, version="3", available=True):
        self.version = version
        self.available = available
        self.models = {}
        self.loads = []

    def resolve(self):
        if not self.available:
            raise ConnectionError("registry unreachable")
        return SimpleNamespace(version=self.version, run_id=f"run-{self.version}")

    def load(self, version):
        self.loads.append(version)
        return self.models.get(version, FakeModel())


@pytest.fixture
def registry():
    return FakeRegistry()


@pytest.fixture
def client(monkeypatch, registry):
    store = ModelStore(
        name="EmotionRecognitionModel",
        alias="staging",
        resolve=registry.resolve,
        load=registry.load,
        validate=predict.validate_model,
        refresh_seconds=0,  # tests drive refresh() explicitly
    )
    monkeypatch.setattr(app_main, "store", store)
    with TestClient(app_main.app) as c:
        yield c


# ---------- predictions ----------

def test_batch_returns_one_prediction_per_input(client):
    texts = ["so happy", "very sad", "happy but sad", "meh"]
    r = client.post("/predict", json={"text": texts})

    assert r.status_code == 200
    assert r.json() == {
        "emotions": [["joy"], ["sadness"], ["joy", "sadness"], ["neutral"]],
        "model_version": "3",
    }


def test_threshold_is_applied(client):
    r = client.post("/predict", json={"text": "happy but sad", "threshold": 0.7})
    assert r.json()["emotions"] == ["joy"]


def test_get_predict(client):
    r = client.get("/predict", params={"text": "so happy"})
    assert r.json() == {"emotions": ["joy"], "model_version": "3"}


# ---------- input limits ----------

def test_threshold_out_of_range_is_rejected(client):
    assert client.post("/predict", json={"text": "hi", "threshold": 1.5}).status_code == 422


def test_batch_larger_than_limit_is_rejected(client):
    texts = ["hi"] * (app_main.MAX_BATCH_SIZE + 1)
    assert client.post("/predict", json={"text": texts}).status_code == 422


def test_batch_at_limit_is_accepted(client):
    texts = ["hi"] * app_main.MAX_BATCH_SIZE
    assert client.post("/predict", json={"text": texts}).status_code == 200


def test_empty_batch_is_rejected(client):
    assert client.post("/predict", json={"text": []}).status_code == 422


def test_text_longer_than_limit_is_rejected(client):
    long_text = "a" * (app_main.MAX_TEXT_LENGTH + 1)
    assert client.post("/predict", json={"text": long_text}).status_code == 422
    assert client.get("/predict", params={"text": long_text}).status_code == 422


# ---------- health ----------

def test_ready_reports_served_registry_version(client):
    r = client.get("/health/ready")

    assert r.status_code == 200
    model = r.json()["model"]
    assert (model["name"], model["alias"], model["version"], model["run_id"]) == (
        "EmotionRecognitionModel",
        "staging",
        "3",
        "run-3",
    )
    assert client.get("/health").json()["model"]["version"] == "3"  # legacy alias


def test_live_is_ok(client):
    assert client.get("/health/live").json() == {"status": "alive"}


# ---------- registry failure ----------

def test_starts_unready_when_registry_is_down(monkeypatch):
    registry = FakeRegistry(available=False)
    store = ModelStore("M", "staging", registry.resolve, registry.load, predict.validate_model, 0)
    monkeypatch.setattr(app_main, "store", store)

    with TestClient(app_main.app) as client:  # startup must not crash
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert "registry unreachable" in ready.json()["last_error"]
        assert client.post("/predict", json={"text": "hi"}).status_code == 503

        registry.available = True  # registry recovers; next poll loads the model
        assert store.refresh() is True
        assert client.get("/health/ready").status_code == 200
        assert client.post("/predict", json={"text": "so happy"}).status_code == 200


# ---------- refresh without restart ----------

def test_serves_new_version_after_alias_moves(client, registry):
    registry.version = "4"

    assert app_main.store.refresh() is True
    assert client.post("/predict", json={"text": "hi"}).json()["model_version"] == "4"


def test_unchanged_alias_does_not_reload(client, registry):
    assert app_main.store.refresh() is False
    assert registry.loads == ["3"]


def test_failed_validation_keeps_serving_previous_version(client, registry):
    registry.version = "4"
    registry.models["4"] = BrokenModel()

    assert app_main.store.refresh() is False
    r = client.post("/predict", json={"text": "so happy"})
    assert r.status_code == 200 and r.json()["model_version"] == "3"
    assert "boom" in client.get("/health/ready").json()["last_error"]


def test_registry_outage_after_startup_keeps_serving(client, registry):
    registry.available = False

    assert app_main.store.refresh() is False
    assert client.get("/health/ready").status_code == 200  # still has a model
    assert client.post("/predict", json={"text": "hi"}).status_code == 200


# ---------- errors and request IDs ----------

def test_prediction_error_returns_500_without_internals(client, monkeypatch):
    broken = dataclasses.replace(app_main.store.current, model=BrokenModel())
    monkeypatch.setattr(app_main.store, "_current", broken)

    r = client.post("/predict", json={"text": "hi"}, headers={"X-Request-ID": "abc-123"})

    assert r.status_code == 500
    assert r.json()["detail"] == "Prediction failed (request_id=abc-123)"
    assert "boom" not in r.text


def test_request_id_is_echoed_or_generated(client):
    echoed = client.get("/health/live", headers={"X-Request-ID": "trace-42"})
    generated = client.get("/health/live")
    unsafe = client.get("/health/live", headers={"X-Request-ID": "bad id\nwith newline"})

    assert echoed.headers["X-Request-ID"] == "trace-42"
    assert len(generated.headers["X-Request-ID"]) == 32
    assert unsafe.headers["X-Request-ID"] != "bad id\nwith newline"


def test_validate_model_rejects_wrong_output_shape():
    class WrongShape:
        def predict_proba(self, texts):
            return np.zeros((len(texts), 5))

    with pytest.raises(ValueError, match="expected output shape"):
        predict.validate_model(WrongShape())


def test_load_version_uses_exact_registry_version(monkeypatch):
    seen = {}
    monkeypatch.setattr(predict.mlflow.sklearn, "load_model", lambda uri: seen.setdefault("uri", uri))
    predict.load_version("7")
    assert seen["uri"] == "models:/EmotionRecognitionModel/7"

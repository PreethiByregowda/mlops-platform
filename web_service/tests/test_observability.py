import os
import sys
import json
import time
import logging
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from emotion_model.log import JsonFormatter  # noqa: E402
from emotion_model.model_store import ModelStore  # noqa: E402


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("emotion_api", logging.INFO, __file__, 1, "request", None, None)
    record.request_id = "abc"
    record.status = 200

    entry = json.loads(JsonFormatter().format(record))

    assert entry["msg"] == "request" and entry["level"] == "INFO"
    assert entry["request_id"] == "abc" and entry["status"] == 200
    assert entry["ts"].endswith("+00:00")


def test_request_log_line_has_structured_fields(caplog, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main as app_main
    from tests.test_api import FakeRegistry

    registry = FakeRegistry()
    store = ModelStore("M", "staging", registry.resolve, registry.load, lambda m: None, 0)
    monkeypatch.setattr(app_main, "store", store)
    with TestClient(app_main.app) as client, caplog.at_level(logging.INFO, logger="emotion_api"):
        client.post("/predict", json={"text": ["so happy", "sad"]}, headers={"X-Request-ID": "r-1"})

    (line,) = [r for r in caplog.records if getattr(r, "event", None) == "request"]
    assert (line.request_id, line.method, line.path, line.status) == ("r-1", "POST", "/predict", 200)
    assert line.batch_size == 2 and line.model_version == "3"
    assert line.duration_ms >= 0


def test_background_poll_picks_up_new_version_without_restart():
    state = {"version": "1"}
    store = ModelStore(
        "M",
        "staging",
        resolve=lambda: SimpleNamespace(version=state["version"], run_id="r"),
        load=lambda version: f"model-{version}",
        validate=lambda model: None,
        refresh_seconds=0.05,
    )
    store.start()
    try:
        assert store.current.version == "1"
        state["version"] = "2"  # alias moves in the registry
        deadline = time.time() + 2
        while store.current.version != "2" and time.time() < deadline:
            time.sleep(0.02)
        assert store.current.version == "2" and store.current.model == "model-2"
    finally:
        store.stop()


def test_start_does_not_block_on_a_slow_registry():
    def slow_resolve():
        time.sleep(1.0)  # an unreachable registry retrying
        raise ConnectionError("registry unreachable")

    store = ModelStore("M", "staging", slow_resolve, lambda v: None, lambda m: None, 60)
    began = time.time()
    store.start()
    try:
        assert time.time() - began < 0.2  # returned before the first check finished
        assert not store.ready
    finally:
        store.stop()

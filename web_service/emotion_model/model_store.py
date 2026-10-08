"""Holds the served model and keeps it in step with the registry alias.

A background thread polls which version the alias points to. A new version is
loaded and validated before it replaces the current one, so a failed load or a
broken model never interrupts serving; the error is reported through readiness.
"""
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger("emotion_model.model_store")


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    name: str
    alias: str
    version: str
    run_id: str
    loaded_at: str


class ModelStore:
    def __init__(
        self,
        name: str,
        alias: str,
        resolve: Callable[[], Any],
        load: Callable[[str], Any],
        validate: Callable[[Any], None],
        refresh_seconds: float = 60,
    ):
        self.name = name
        self.alias = alias
        self._resolve = resolve  # () -> ModelVersion behind the alias
        self._load = load  # (version) -> model
        self._validate = validate  # (model) -> None, raises if unusable
        self.refresh_seconds = refresh_seconds
        self._current: Optional[LoadedModel] = None
        self.last_error: Optional[str] = None
        self.last_checked: Optional[str] = None
        self._lock = threading.Lock()  # one refresh at a time
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def current(self) -> Optional[LoadedModel]:
        """Snapshot of the served model; replaced atomically, never mutated."""
        return self._current

    @property
    def ready(self) -> bool:
        return self._current is not None

    def refresh(self) -> bool:
        """Load the alias's version if it differs from the served one.

        Returns True when a new version was swapped in. Never raises: on
        failure the previous model (if any) keeps serving.
        """
        with self._lock:
            self.last_checked = datetime.now(timezone.utc).isoformat(timespec="seconds")
            current = self._current
            try:
                target = self._resolve()
                version = str(target.version)
                if current is not None and current.version == version:
                    self.last_error = None
                    return False

                model = self._load(version)
                self._validate(model)
            except Exception as exc:  # registry down, missing alias, bad artifact...
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "model refresh failed",
                    extra={
                        "event": "model_refresh_failed",
                        "error": self.last_error,
                        "serving_version": current.version if current else None,
                    },
                )
                return False

            self._current = LoadedModel(
                model=model,
                name=self.name,
                alias=self.alias,
                version=version,
                run_id=target.run_id,
                loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            self.last_error = None
            logger.info(
                "model loaded",
                extra={
                    "event": "model_loaded",
                    "model_version": version,
                    "previous_version": current.version if current else None,
                    "run_id": target.run_id,
                },
            )
            return True

    def start(self):
        """Begin loading without blocking the caller.

        With polling enabled, the first load runs on the background thread, so
        a slow or unreachable registry never delays startup: the service is
        live immediately and becomes ready once a model loads. With polling
        disabled (refresh_seconds <= 0) a single synchronous load is attempted.
        """
        if self.refresh_seconds > 0:
            self._thread = threading.Thread(target=self._poll, name="model-refresh", daemon=True)
            self._thread.start()
        else:
            self.refresh()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _poll(self):
        self.refresh()
        while not self._stop.wait(self.refresh_seconds):
            self.refresh()

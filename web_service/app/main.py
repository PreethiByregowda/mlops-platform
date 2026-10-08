import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fail registry calls fast; the model store retries on its own schedule.
# MLflow's defaults (7 retries, exponential backoff, 120 s timeout) can block a
# single check for minutes. Explicit environment variables still win.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

import re
import time
import uuid
import logging
from typing import List, Union, Annotated
from contextlib import asynccontextmanager

import mlflow
from pydantic import Field, BaseModel, StringConstraints
from fastapi import Body, Query, Depends, FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from emotion_model import predict
from emotion_model.log import configure_logging
from emotion_model.model_store import ModelStore

# ---------------- CONFIG ----------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REFRESH_SECONDS = float(os.getenv("MODEL_REFRESH_SECONDS", "60"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

configure_logging(LOG_LEVEL)
logger = logging.getLogger("emotion_api")

store = ModelStore(
    name=predict.MODEL_NAME,
    alias=predict.MODEL_ALIAS,
    resolve=lambda: predict.resolve_alias(),
    load=lambda version: predict.load_version(version),
    validate=lambda model: predict.validate_model(model),
    refresh_seconds=MODEL_REFRESH_SECONDS,
)


# ---------------- LIFECYCLE ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Serve the registry version behind MODEL_NAME@MODEL_ALIAS. A registry
    # outage at startup is not fatal: the service reports not-ready and the
    # refresher keeps retrying.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    store.start()
    yield
    store.stop()


# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="Emotion Recognition API",
    description="Predict emotions from text using a multi-label ML model.",
    version="1.1.0",
    lifespan=lifespan,
)

Text = Annotated[str, StringConstraints(max_length=MAX_TEXT_LENGTH)]


class TextRequest(BaseModel):
    text: Union[Text, Annotated[List[Text], Field(min_length=1, max_length=MAX_BATCH_SIZE)]]
    threshold: float = Field(0.5, ge=0, le=1)  # Optional threshold override


_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request ID and emit one structured log line per request."""
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    if not request.url.path.startswith("/health"):
        logger.info(
            "request",
            extra={
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                "batch_size": getattr(request.state, "batch_size", None),
                "model_version": getattr(request.state, "model_version", None),
            },
        )
    return response


def serving_model(request: Request):
    """Dependency: the model snapshot for this request, or 503 if none is loaded."""
    current = store.current
    if current is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet; see /health/ready")
    request.state.model_version = current.version
    return current


def run_prediction(request: Request, current, texts, threshold):
    request.state.batch_size = len(texts)
    try:
        return predict.predict_emotion(current.model, texts, threshold=threshold)
    except Exception as e:
        logger.exception(
            "prediction failed",
            extra={"event": "prediction_failed", "request_id": request.state.request_id},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed (request_id={request.state.request_id})",
        ) from e


# ---------------- HEALTH ----------------
@app.get("/health/live")
def live():
    """Liveness: the process is up and serving HTTP."""
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    """Readiness: a validated model is loaded and can serve predictions."""
    current = store.current
    body = {
        "alias": store.alias,
        "last_checked": store.last_checked,
        "last_error": store.last_error,
    }
    if current is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", **body})
    return {
        "status": "ok",
        "model": {
            "name": current.name,
            "alias": current.alias,
            "version": current.version,
            "run_id": current.run_id,
            "loaded_at": current.loaded_at,
        },
        **body,
    }


@app.get("/health")
def health():
    """Backwards-compatible alias for /health/ready."""
    return ready()


# ---------------- PREDICTION ----------------
@app.post("/predict")
def predict_endpoint(request: Request, body: TextRequest = Body(...), current=Depends(serving_model)):
    texts = [body.text] if isinstance(body.text, str) else body.text
    emotions = run_prediction(request, current, texts, body.threshold)
    # Return format depends on input type
    result = emotions[0] if isinstance(body.text, str) else emotions
    return {"emotions": result, "model_version": current.version}


@app.get("/predict")
def predict_get(
    request: Request,
    text: str = Query(..., max_length=MAX_TEXT_LENGTH, description="Text to analyze"),
    threshold: float = Query(0.5, ge=0, le=1, description="Prediction threshold (0-1)"),
    current=Depends(serving_model),
):
    emotions = run_prediction(request, current, [text], threshold)
    return {"emotions": emotions[0], "model_version": current.version}

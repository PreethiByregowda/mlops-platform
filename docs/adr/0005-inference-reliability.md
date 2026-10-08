# 0005. Inference service reliability: health probes, registry failure handling and live model refresh

- Status: Accepted
- Date: 2026-10-08

## Context

The inference service loaded `EmotionRecognitionModel@<alias>` once, inside the FastAPI startup hook:

- If the MLflow registry was unreachable at startup, the hook raised and the process exited, so an orchestrator would crash-loop the container until MLflow came back.
- There was one `/health` endpoint, so "the process is up" and "the service can predict" could not be told apart.
- A newly promoted model was served only after a restart.
- Requests had no size limits, errors returned raw exception text, and logs were unstructured `print` statements.

End-to-end testing also showed that MLflow's HTTP client defaults (7 retries with exponential backoff, 120 s timeout) can block a single registry call for minutes, which would stall anything that calls the registry synchronously.

## Decision

**Model store** (`emotion_model/model_store.py`). A `ModelStore` owns the served model as an immutable snapshot (model, name, alias, version, run ID, load time). A background thread checks every `MODEL_REFRESH_SECONDS` (default 60) which version the alias points to. When it changes, the store:

1. loads that **exact version** (`models:/<name>/<version>`, not the alias, so the reported version is the one loaded even if the alias moves mid-load),
2. validates it with a smoke prediction (output shape `(1, 28)`),
3. replaces the snapshot in a single assignment.

If any step fails, the previous snapshot keeps serving and the error is kept for readiness. Requests read the snapshot once, so each request sees one consistent model and version.

**Non-blocking startup.** The first load runs on the background thread; the startup hook returns immediately. The service is live at once and becomes ready when a model loads.

**Bounded registry calls.** The app sets `MLFLOW_HTTP_REQUEST_MAX_RETRIES=2`, `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR=1` and `MLFLOW_HTTP_REQUEST_TIMEOUT=10` unless they are already set, so one failed check takes seconds; the store retries on its own schedule.

**Health probes.**

| Endpoint | Meaning | 200 when | Otherwise |
|---|---|---|---|
| `/health/live` | Process is serving HTTP | always | (no response) |
| `/health/ready` | Can predict | a validated model is loaded | 503 with `last_error` |
| `/health` | Backwards-compatible alias of `/health/ready` | | |

The Docker `HEALTHCHECK` uses readiness. `/predict` returns 503 while no model is loaded.

**Input limits.** `MAX_BATCH_SIZE` (64) texts per request, `MAX_TEXT_LENGTH` (2000) characters per text, and non-empty batches, enforced by request validation (422).

**Structured logging** (`emotion_model/log.py`). JSON lines on stdout using the standard library: one `request` event per non-health request (request ID, method, path, status, duration, batch size, model version), and `model_loaded` / `model_refresh_failed` events. `X-Request-ID` is accepted when it matches `[A-Za-z0-9._-]{1,64}`, otherwise generated, and is returned on every response. A prediction failure returns a generic 500 message with the request ID; the traceback goes to the log only.

**Response.** `/predict` responses add `model_version`; the `emotions` field is unchanged.

## Consequences

- A registry outage no longer takes the service down: before the first load it is live but not ready; after it, the last good model keeps serving.
- Promotions reach the running service within about `MODEL_REFRESH_SECONDS` without a restart, and every prediction states which version produced it.
- During a refresh two models are in memory; fine at this model size, a constraint for large models.
- Each gunicorn worker polls independently, so with several workers they can briefly serve different versions after a promotion (one worker today).
- There is no push notification from MLflow and no reload endpoint; an unauthenticated admin endpoint was judged a bigger risk than up to a minute of delay.
- The service now depends on its own background thread; a bug that kills the thread would stop refreshes silently. Readiness `last_checked` exposes this, but nothing alerts on it yet (Tier 2 observability).

## Alternatives considered

- **Load by alias (`models:/name@alias`).** Simpler, but a promotion between resolving and loading could serve one version while reporting another.
- **Fail fast at startup and let the orchestrator restart.** Common, but it turns a registry outage into an outage of every replica, and with MLflow's default retries a single start can take minutes.
- **Admin `/reload` endpoint or MLflow webhooks.** Faster propagation; needs authentication (endpoint) or MLflow features not in this deployment (webhooks).
- **Structured logging library (structlog, python-json-logger).** Nicer API; the standard library covers the need without another dependency in the inference image.

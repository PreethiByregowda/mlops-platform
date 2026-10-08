# 🚀 Emotion Recognition Web Service

FastAPI service that serves `EmotionRecognitionModel@staging` from the MLflow model registry. The model is not baked into the image; it is downloaded from the MLflow server at startup.

## Run in Docker
```bash
make run_webservice      # build image, start on :9696, wait for /health/ready, run client.py
make stop_webservice
```
The container reads `MLFLOW_TRACKING_URI` (default `http://host.docker.internal:5000`). Override with `MLFLOW_TRACKING_URI=http://<host>:5000 make run_webservice`. The MLflow server must proxy artifacts (`--artifacts-destination`) so the container can download the model.

## Run locally
```bash
pipenv install --dev
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 pipenv run \
  gunicorn --bind=127.0.0.1:9696 -k uvicorn.workers.UvicornWorker app.main:app
```

## Endpoints
- `GET /health/live`: the process is up (always 200 while serving HTTP)
- `GET /health/ready`: a validated model is loaded: name, alias, version, run ID, load time; 503 with the last error otherwise. `GET /health` is an alias, used by the Docker `HEALTHCHECK`
- `POST /predict`: `{"text": "..." | ["...", ...], "threshold": 0.5}`, up to `MAX_BATCH_SIZE` texts of at most `MAX_TEXT_LENGTH` characters; the response includes `model_version`
- `GET /predict?text=...&threshold=0.5`
- `GET /docs`: Swagger UI

The service polls the registry alias every `MODEL_REFRESH_SECONDS` and swaps in a new version without a restart; a registry outage never stops it from starting or from serving the last good model. Logs are JSON lines on stdout. See [ADR 0005](../docs/adr/0005-inference-reliability.md).

## Configuration
| Variable | Default |
|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` (`http://host.docker.internal:5000` in the image) |
| `MODEL_NAME` | `EmotionRecognitionModel` |
| `MODEL_ALIAS` | `staging` |
| `MODEL_REFRESH_SECONDS` | `60` (0 disables polling) |
| `MAX_BATCH_SIZE` | `64` |
| `MAX_TEXT_LENGTH` | `2000` |
| `LOG_LEVEL` | `INFO` |

## Tests
```bash
make test
```

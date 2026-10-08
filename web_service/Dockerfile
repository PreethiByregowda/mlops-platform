FROM python:3.12.7-slim-bookworm

# Install pip and pipenv
RUN pip install --upgrade pip
RUN pip install pipenv

# Set working directory
WORKDIR /app

# Copy Pipenv files first for better layer caching
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pipenv install --system --deploy

# Copy the application code
COPY app ./app
COPY emotion_model ./emotion_model

# Model is pulled from the MLflow registry at startup; point this at a
# tracking server reachable from inside the container
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# Expose the API port
EXPOSE 9696

# Healthy = a validated model is loaded (readiness). Liveness is /health/live.
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9696/health/ready', timeout=2)"

# Run FastAPI app with Gunicorn (ASGI needs the Uvicorn worker class)
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "-k", "uvicorn.workers.UvicornWorker", "app.main:app"]

import os

from main import mlflow_training
from stage import mlflow_staging
from approve import mlflow_production_approval

# MLflow server as reachable from inside the Docker work-pool containers
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "goemotions-classification")
JOB_ENV = {"MLFLOW_TRACKING_URI": TRACKING_URI, "MLFLOW_EXPERIMENT_NAME": EXPERIMENT_NAME}
JOB_VARIABLES = {
    "env": JOB_ENV,
    # The image is built locally and never pushed; without this, a ":latest"
    # tag makes the Docker worker pull from Docker Hub on every run
    "image_pull_policy": "Never",
    # On Linux the worker defaults to host networking for a localhost API URL;
    # under Docker Desktop that is the Docker VM, not this machine. Bridge mode
    # makes the worker rewrite localhost to host.docker.internal instead.
    "network_mode": "bridge",
}

# Deploy training flow (no schedule)
mlflow_training.deploy(
    name="mlflow-training-deployment",
    work_pool_name="docker-pool",
    image="preethibyregowda/mlops-platform:latest",
    push=False,  # 👈 Don't try to push or build a Docker image
    build=False,
    job_variables=JOB_VARIABLES,
    tags=["training", "ml"],
)

# Deploy staging flow (no schedule)
mlflow_staging.deploy(
    name="mlflow-staging-deployment",
    work_pool_name="docker-pool",
    image="preethibyregowda/mlops-platform:latest",
    push=False,  # 👈 Don't try to push or build a Docker image
    build=False,
    job_variables=JOB_VARIABLES,
    parameters={"tracking_uri": TRACKING_URI, "experiment_name": EXPERIMENT_NAME},
    tags=["staging", "ml"],
)

# Production approval (manual only, never scheduled). Run with, for example:
#   prefect deployment run 'mlflow_production_approval/mlflow-production-approval' \
#     --param version=2 --param approved_by="Jane Doe" --param reason="..."
mlflow_production_approval.deploy(
    name="mlflow-production-approval",
    work_pool_name="docker-pool",
    image="preethibyregowda/mlops-platform:latest",
    push=False,  # 👈 Don't try to push or build a Docker image
    build=False,
    job_variables=JOB_VARIABLES,
    parameters={"tracking_uri": TRACKING_URI},
    tags=["production", "ml"],
)

# Inference service: tests, image, and a local container run against MLflow.

IMAGE := emotion-model
TAG := $(shell date +%Y%m%d-%H%M%S)
CONTAINER := emotion-api
PORT := 9696
# MLflow as reachable from inside the container (Docker Desktop host alias)
MLFLOW_TRACKING_URI ?= http://host.docker.internal:5000

.PHONY: setup test format build run_webservice stop_webservice

setup:  ## install this environment exactly as locked
	pipenv sync --dev

test:
	pipenv run python -m pytest tests

format:
	pipenv run isort .
	pipenv run black .

build:
	docker build --tag $(IMAGE):$(TAG) --tag $(IMAGE):latest .

# Start the container, wait until a model is loaded, then send example requests
run_webservice: build
	-docker rm --force $(CONTAINER) >/dev/null 2>&1
	docker run --detach --rm --name $(CONTAINER) \
		--env MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		--publish $(PORT):$(PORT) $(IMAGE):$(TAG)
	@for attempt in $$(seq 60); do \
		curl --silent --fail http://127.0.0.1:$(PORT)/health/ready && echo && break; \
		sleep 1; \
	done
	pipenv run python client.py

stop_webservice:
	docker rm --force $(CONTAINER)

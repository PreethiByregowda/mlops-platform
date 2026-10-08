import ast
import os
import re
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from prefect import flow, task, get_run_logger
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "goemotions-classification")
NUM_LABELS = 28  # GoEmotions: 27 emotions + neutral


def build_candidates():
    """Candidate pipelines compared on every training run.

    Every candidate exposes predict_proba so the web service can apply a
    per-request decision threshold.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "SGDClassifier": SGDClassifier(
            loss="log_loss", class_weight="balanced", random_state=42
        ),
        "MultinomialNB": MultinomialNB(),
    }


def parse_labels(s):
    """Parse GoEmotions label strings such as "[27]" or "[ 8 20]" into lists."""
    if not isinstance(s, str):
        return []
    inner = re.sub(r"\s+", ",", s.strip("[] "))
    try:
        return list(ast.literal_eval(f"[{inner}]"))
    except (ValueError, SyntaxError):
        return []


def process_data(data_dir="data"):
    """Load the train, validation and held-out test splits.

    Validation scores select the best candidate; test scores are logged for
    the promotion gate only (see stage.py) and never used for selection.
    """
    mlb = MultiLabelBinarizer(classes=list(range(NUM_LABELS)))
    splits = {}
    for split in ("train", "valid", "test"):
        df = pd.read_csv(f"{data_dir}/{split}.csv")
        labels = df["labels"].apply(parse_labels)
        y = mlb.fit_transform(labels) if split == "train" else mlb.transform(labels)
        splits[split] = (df["text"], y)
    return splits


def score(y_true, y_pred, split):
    """Multi-label scores for one split, keyed like `micro_f1_valid`."""
    return {
        f"micro_f1_{split}": f1_score(y_true, y_pred, average="micro", zero_division=0),
        f"macro_f1_{split}": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"micro_precision_{split}": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        f"micro_recall_{split}": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
    }


RUN_TAGS = {"author/developer": "PreethiB", "task": "multilabel"}


def train_candidate(name, estimator, splits):
    """Fit one candidate inside its own MLflow run and log scores and pipeline.

    The run covers fitting and evaluation, so its duration reflects both.
    """
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = (
        splits["train"],
        splits["valid"],
        splits["test"],
    )
    with mlflow.start_run(tags={**RUN_TAGS, "Model": name}):
        pipeline = make_pipeline(TfidfVectorizer(), OneVsRestClassifier(estimator))
        pipeline.fit(X_train, y_train)

        began = time.perf_counter()
        valid_predictions = pipeline.predict(X_valid)
        seconds_per_sample = (time.perf_counter() - began) / len(X_valid)

        mlflow.log_metrics(
            {
                **score(y_valid, valid_predictions, "valid"),
                **score(y_test, pipeline.predict(X_test), "test"),
                "inference_time_per_sample": seconds_per_sample,
            }
        )
        mlflow.sklearn.log_model(pipeline, name="model")


@task(name="Run models")
def train_candidates(splits):
    for name, estimator in build_candidates().items():
        train_candidate(name, estimator, splits)


@flow(name="mlflow_training")
def mlflow_training():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    log = get_run_logger()

    log.info("Loading and processing GoEmotions dataset")
    splits = process_data()
    shapes = ", ".join(f"y_{split}={y.shape}" for split, (_, y) in splits.items())
    log.info(f"Split shapes: {shapes}")

    log.info("Training models")
    train_candidates(splits)


if __name__ == "__main__":
    mlflow_training()

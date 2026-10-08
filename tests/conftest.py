import os
import sys

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TEXTS = ["so happy", "very sad", "really angry", "thank you", "happy and thankful", "ok"]


@pytest.fixture
def tiny_model():
    """A 28-label TF-IDF + one-vs-rest pipeline shaped like the real model."""
    y = np.zeros((len(TEXTS), 28), dtype=int)
    for row, labels in enumerate([[17], [25], [2], [15], [17, 15], [27]]):
        y[row, labels] = 1
    y[:, 0] = [1, 0, 1, 0, 1, 0]  # every column needs both classes to fit
    for col in range(1, 28):
        if y[:, col].sum() == 0:
            y[col % len(TEXTS), col] = 1
    model = make_pipeline(TfidfVectorizer(), OneVsRestClassifier(LogisticRegression()))
    return model.fit(TEXTS, y)


@pytest.fixture
def tracking_uri(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    import mlflow

    mlflow.set_tracking_uri(uri)
    return uri

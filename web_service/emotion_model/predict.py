import os
from typing import List, Union

import mlflow
import mlflow.sklearn
import numpy as np

# -------- CONFIG --------
MODEL_NAME = os.getenv("MODEL_NAME", "EmotionRecognitionModel")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")

# -------- LABEL MAP --------
label_map = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}


# -------- MODEL LOADING --------
def resolve_alias(model_name: str = MODEL_NAME, alias: str = MODEL_ALIAS):
    """Return the ModelVersion currently behind `alias` in the MLflow registry."""
    return mlflow.MlflowClient().get_model_version_by_alias(model_name, alias)


def load_version(version: str, model_name: str = MODEL_NAME):
    """Load one exact registered version.

    Loading by version (not by alias) guarantees the version reported by the
    service is the one actually loaded, even if the alias moves meanwhile.
    Uses the sklearn flavor (not pyfunc) so predict_proba is available and raw
    text can be passed straight into the TF-IDF pipeline.
    """
    return mlflow.sklearn.load_model(f"models:/{model_name}/{version}")


def validate_model(model):
    """Smoke-test a freshly loaded model before it is allowed to serve."""
    probs = np.asarray(model.predict_proba(["model validation probe"]))
    if probs.shape != (1, len(label_map)):
        raise ValueError(f"expected output shape (1, {len(label_map)}), got {probs.shape}")


# -------- TEXT PREPROCESSING --------
def preprocess_text(text: str) -> str:
    return text.lower().strip()


# -------- PREDICTION --------
def predict_emotion(model, text: Union[str, List[str]], threshold: float = 0.5) -> List[List[str]]:
    if isinstance(text, str):
        text = [text]

    processed_text = [preprocess_text(t) for t in text]

    # Get prediction probabilities or fallback to binary predictions
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(processed_text)
        preds = (np.array(probs) >= threshold).astype(int)
    else:
        preds = model.predict(processed_text)

    # Ensure 2D shape for both single and batch inputs
    preds = np.atleast_2d(preds)

    # Map prediction vectors to emotion labels
    results = []
    for row in preds:
        emotion_ids = [i for i, val in enumerate(row) if val == 1]
        emotions = [label_map[i] for i in emotion_ids]
        results.append(emotions if emotions else ["neutral"])

    return results


# -------- MAIN TEST --------
if __name__ == "__main__":
    model = load_version(resolve_alias().version)

    sample_texts = [
        "I am so happy today!",
        "I feel very sad and lonely.",
        "This is surprising news!",
        "You made me so angry and disappointed.",
        "Thanks a lot, I appreciate your help!"
    ]
    predictions = predict_emotion(model, sample_texts, threshold=0.4)

    for text, pred in zip(sample_texts, predictions, strict=True):
        print(f"\nText: {text}\nPredicted emotions: {', '.join(pred)}")

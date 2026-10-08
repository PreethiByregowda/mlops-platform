"""Send example requests to a running inference service and print the replies.

Used by `make run_webservice` as a smoke check after the container starts.
"""
import requests

PREDICT_URL = "http://127.0.0.1:9696/predict"

EXAMPLES = {
    "Single text": "I feel fantastic and everything is going great!",
    "Batch": [
        "I feel miserable and alone",
        "I'm so angry with what happened",
        "This is terrifying news",
        "I love spending time with my family",
        "Wow! I wasn't expecting that at all!",
    ],
}


def predict(text):
    """POST one string or a list of strings; return the decoded reply or raise."""
    reply = requests.post(PREDICT_URL, json={"text": text}, timeout=30)
    reply.raise_for_status()
    return reply.json()


def main():
    for label, text in EXAMPLES.items():
        try:
            print(f"{label} prediction result: {predict(text)}")
        except requests.RequestException as error:
            print(f"{label} prediction failed: {error}")


if __name__ == "__main__":
    main()

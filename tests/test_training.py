import numpy as np
import pandas as pd

from main import NUM_LABELS, score, parse_labels, process_data


def test_parse_labels_handles_goemotions_formats():
    assert parse_labels("[27]") == [27]
    assert parse_labels("[ 8 20]") == [8, 20]
    assert parse_labels("[3 10 25]") == [3, 10, 25]
    assert parse_labels("[]") == []
    assert parse_labels(None) == []
    assert parse_labels("not a list") == []


def test_process_data_keeps_every_label(tmp_path):
    rows = pd.DataFrame({"text": ["a", "b"], "labels": ["[27]", "[ 8 20]"], "id": ["x", "y"]})
    for split in ("train", "valid", "test"):
        rows.to_csv(tmp_path / f"{split}.csv", index=False)

    splits = process_data(tmp_path)

    assert set(splits) == {"train", "valid", "test"}
    X_train, y_train = splits["train"]
    assert list(X_train) == ["a", "b"]
    assert y_train.shape == (2, NUM_LABELS)
    assert y_train[0].nonzero()[0].tolist() == [27]
    assert y_train[1].nonzero()[0].tolist() == [8, 20]  # multi-label rows are not truncated
    assert splits["test"][1].shape == (2, NUM_LABELS)


def test_score_keys_are_suffixed_by_split():
    y = np.array([[1, 0], [0, 1]])
    scores = score(y, y, "test")

    assert set(scores) == {
        "micro_f1_test",
        "macro_f1_test",
        "micro_precision_test",
        "micro_recall_test",
    }
    assert scores["micro_f1_test"] == 1.0

import numpy as np
import pytest
from sklearn.metrics import multilabel_confusion_matrix

from emotion import CLASSES
from emotion.metrics import summarize_confusion


def test_metrics_hand_calculation():
    matrix = np.diag([3, 4, 5, 6, 7, 8])
    matrix[0, 1], matrix[1, 0] = 2, 1
    result = summarize_confusion(matrix)
    anger = result["per_class"]["anger"]
    assert (anger["tp"], anger["fn"], anger["fp"], anger["tn"]) == (3, 2, 1, 30)
    assert anger["sensitivity"] == pytest.approx(3 / 5)
    assert anger["specificity"] == pytest.approx(30 / 31)
    assert anger["precision"] == pytest.approx(3 / 4)
    assert anger["f1"] == pytest.approx(6 / 9)
    assert result["accuracy"] == pytest.approx(33 / 36)


def test_confusion_agrees_with_sklearn():
    y = [0, 0, 1, 2, 3, 4, 5]
    predictions = [0, 1, 1, 2, 3, 4, 5]
    matrix = np.bincount(np.array(y) * 6 + predictions, minlength=36).reshape(6, 6)
    reference = multilabel_confusion_matrix(y, predictions, labels=range(6))
    result = summarize_confusion(matrix)
    for name, expected in zip(CLASSES, reference):
        row = result["per_class"][name]
        assert [[row["tn"], row["fp"]], [row["fn"], row["tp"]]] == expected.tolist()


def test_undefined_metrics_are_null():
    with pytest.warns(UserWarning, match="Undefined"):
        report = summarize_confusion(np.zeros((6, 6), dtype=int))
    assert report["accuracy"] is None
    assert report["per_class"]["anger"]["specificity"] is None
    assert report["macro_f1"] is None


def test_unpredicted_class_has_zero_recall_and_f1():
    matrix = np.eye(6, dtype=int)
    matrix[0, 0], matrix[0, 1] = 0, 1
    with pytest.warns(UserWarning):
        report = summarize_confusion(matrix)
    assert report["per_class"]["anger"]["precision"] is None
    assert report["per_class"]["anger"]["sensitivity"] == 0
    assert report["per_class"]["anger"]["f1"] == 0

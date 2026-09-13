"""Metrics from a fixed six-class confusion matrix (rows=true, columns=predicted)."""
import warnings

import numpy as np

from emotion import CLASSES


def summarize_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(CLASSES), len(CLASSES)) or (matrix < 0).any():
        raise ValueError("Expected a nonnegative 6 by 6 confusion matrix.")
    total = int(matrix.sum())
    undefined = []

    def ratio(numerator, denominator, label):
        if denominator == 0:
            undefined.append(label)
            return None
        return float(numerator / denominator)

    rows = {}
    for i, name in enumerate(CLASSES):
        tp = int(matrix[i, i])
        fn = int(matrix[i].sum()) - tp
        fp = int(matrix[:, i].sum()) - tp
        tn = total - tp - fn - fp
        rows[name] = {
            "support": tp + fn, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": ratio(tp, tp + fp, f"{name}/precision"),
            "sensitivity": ratio(tp, tp + fn, f"{name}/sensitivity"),
            "specificity": ratio(tn, tn + fp, f"{name}/specificity"),
            "f1": ratio(2 * tp, 2 * tp + fp + fn, f"{name}/f1"),
        }

    def macro(metric):
        values = [r[metric] for r in rows.values() if r[metric] is not None]
        return float(np.mean(values)) if values else None

    result = {
        "accuracy": ratio(int(matrix.trace()), total, "accuracy"),
        "macro_f1": macro("f1"), "balanced_accuracy": macro("sensitivity"),
        "macro_sensitivity": macro("sensitivity"),
        "macro_specificity": macro("specificity"),
        "macro_precision": macro("precision"), "per_class": rows,
        "confusion_matrix": matrix.tolist(), "undefined_metrics": undefined,
        "macro_policy": "mean of defined classes only",
    }
    if undefined:
        warnings.warn("Undefined metrics recorded as null: " + ", ".join(undefined), stacklevel=2)
    return result

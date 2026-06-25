"""Dependency-light classification metrics."""

from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float((y_true == y_pred).mean())


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """Compute an integer confusion matrix."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[truth, pred] += 1
    return matrix


def macro_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
    eps: float = 1e-12,
) -> float:
    """Compute unweighted mean F1 across classes."""

    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    tp = np.diag(cm).astype(np.float64)
    precision = tp / (cm.sum(axis=0) + eps)
    recall = tp / (cm.sum(axis=1) + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return float(np.nanmean(f1))


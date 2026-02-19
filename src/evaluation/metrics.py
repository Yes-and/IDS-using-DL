"""Per-class and aggregate evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Return a dict with per-class and macro/weighted aggregate metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Optional list of class name strings for display.

    Returns:
        dict with keys: per_class, macro_f1, weighted_f1, report_str, cm
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    per_class = {
        (class_names[i] if class_names else str(i)): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(f1))
    }
    return {
        "per_class": per_class,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "report_str": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred),
    }

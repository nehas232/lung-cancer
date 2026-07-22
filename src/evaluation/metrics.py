"""Classification metrics, weighted toward sensitivity (missing cancer is costly)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """Return a dict of standard classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }

    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)

    return metrics
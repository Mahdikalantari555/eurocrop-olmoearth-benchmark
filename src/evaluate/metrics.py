"""
Evaluation metrics.
"""

from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix,
    precision_score, recall_score, classification_report
)
import json
import os
import numpy as np


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """
    Compute classification metrics including per-class stats.
    """
    result = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "n_classes": int(len(set(y_true) | set(y_pred))),
    }

    if labels is not None:
        per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        result["per_class_f1"] = {str(l): float(f) for l, f in zip(labels, per_class_f1)}

    return result


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_confusion_matrix(y_true, y_pred, path, labels=None):
    """Save confusion matrix as CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(path, cm, delimiter=",", fmt="%d")


def load_metrics(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

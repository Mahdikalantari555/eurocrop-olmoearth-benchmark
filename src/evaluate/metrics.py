"""
Evaluation metrics.
"""

from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
)
import json
import os


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute classification metrics.
    """
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def save_metrics(metrics: dict, path: str):
    """
    Save metrics to JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> dict:
    """
    Load metrics from JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)

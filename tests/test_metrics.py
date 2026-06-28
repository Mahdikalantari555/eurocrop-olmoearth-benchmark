"""
Tests for src/evaluate/metrics.py
"""

import pytest
import numpy as np
import os
import json
import tempfile
from src.evaluate.metrics import compute_metrics, save_metrics, load_metrics, save_confusion_matrix


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["overall_accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["weighted_f1"] == 1.0
        assert metrics["kappa"] == 1.0

    def test_random_predictions(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 5, 100)
        y_pred = np.random.randint(0, 5, 100)
        metrics = compute_metrics(y_true, y_pred)
        assert 0 <= metrics["overall_accuracy"] <= 1
        assert 0 <= metrics["macro_f1"] <= 1
        assert 0 <= metrics["weighted_f1"] <= 1
        assert -1 <= metrics["kappa"] <= 1

    def test_all_wrong_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["overall_accuracy"] == 0.0
        assert metrics["kappa"] < 0

    def test_single_class(self):
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["overall_accuracy"] == 1.0

    def test_new_keys_present(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        metrics = compute_metrics(y_true, y_pred)
        assert "macro_precision" in metrics
        assert "macro_recall" in metrics
        assert "n_samples" in metrics
        assert "n_classes" in metrics

    def test_n_samples_correct(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["n_samples"] == 5

    def test_n_classes_correct(self):
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 2]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["n_classes"] == 4

    def test_per_class_f1(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 0]
        metrics = compute_metrics(y_true, y_pred, labels=[0, 1, 2])
        assert "per_class_f1" in metrics
        assert "0" in metrics["per_class_f1"]
        assert "1" in metrics["per_class_f1"]
        assert "2" in metrics["per_class_f1"]
        assert metrics["per_class_f1"]["1"] == 1.0

    def test_per_class_f1_not_included_without_labels(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        metrics = compute_metrics(y_true, y_pred)
        assert "per_class_f1" not in metrics


class TestSaveConfusionMatrix:
    def test_creates_csv(self, temp_dir):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]
        path = os.path.join(temp_dir, "cm.csv")
        save_confusion_matrix(y_true, y_pred, path)
        assert os.path.exists(path)

    def test_csv_content(self, temp_dir):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        path = os.path.join(temp_dir, "cm.csv")
        save_confusion_matrix(y_true, y_pred, path, labels=[0, 1])
        cm = np.loadtxt(path, delimiter=",", dtype=int)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # class 0 correct
        assert cm[1, 1] == 1  # class 1 correct

    def test_creates_directory(self, temp_dir):
        y_true = [0, 1]
        y_pred = [0, 1]
        path = os.path.join(temp_dir, "subdir", "cm.csv")
        save_confusion_matrix(y_true, y_pred, path)
        assert os.path.exists(path)


class TestSaveMetrics:
    def test_creates_file(self, temp_dir, sample_metrics):
        path = os.path.join(temp_dir, "metrics.json")
        save_metrics(sample_metrics, path)
        assert os.path.exists(path)

    def test_creates_directory(self, temp_dir, sample_metrics):
        path = os.path.join(temp_dir, "subdir", "metrics.json")
        save_metrics(sample_metrics, path)
        assert os.path.exists(path)

    def test_valid_json(self, temp_dir, sample_metrics):
        path = os.path.join(temp_dir, "metrics.json")
        save_metrics(sample_metrics, path)
        with open(path) as f:
            data = json.load(f)
        assert data == sample_metrics


class TestLoadMetrics:
    def test_loads_valid_json(self, temp_dir, sample_metrics):
        path = os.path.join(temp_dir, "metrics.json")
        save_metrics(sample_metrics, path)
        loaded = load_metrics(path)
        assert loaded == sample_metrics

    def test_loads_invalid_json_raises(self, temp_dir):
        path = os.path.join(temp_dir, "bad.json")
        with open(path, "w") as f:
            f.write("not json")
        with pytest.raises(json.JSONDecodeError):
            load_metrics(path)

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_metrics("nonexistent.json")

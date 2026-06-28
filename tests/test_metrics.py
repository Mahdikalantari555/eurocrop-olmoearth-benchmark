"""
Tests for src/evaluate/metrics.py
"""

import pytest
import numpy as np
import os
import json
import tempfile
from src.evaluate.metrics import compute_metrics, save_metrics, load_metrics


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

    def test_empty_predictions_raises(self):
        with pytest.raises(ValueError):
            compute_metrics([], [])

    def test_metrics_are_floats(self, sample_data):
        y = np.random.randint(0, 5, 100)
        metrics = compute_metrics(y, y)
        for value in metrics.values():
            assert isinstance(value, float)

    def test_returns_all_keys(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        metrics = compute_metrics(y_true, y_pred)
        expected_keys = {"overall_accuracy", "macro_f1", "weighted_f1", "kappa"}
        assert set(metrics.keys()) == expected_keys


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

    def test_overwrites_existing(self, temp_dir, sample_metrics):
        path = os.path.join(temp_dir, "metrics.json")
        save_metrics(sample_metrics, path)
        sample_metrics["overall_accuracy"] = 0.99
        save_metrics(sample_metrics, path)
        loaded = load_metrics(path)
        assert loaded["overall_accuracy"] == 0.99


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

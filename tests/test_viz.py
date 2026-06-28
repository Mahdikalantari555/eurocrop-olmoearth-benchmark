"""
Tests for src/viz/*.py
"""

import pytest
import numpy as np
import os
import tempfile
from src.viz.confusion import plot_confusion_matrix
from src.viz.fewshot_curve import plot_fewshot_curve
from src.viz.umap_viz import plot_umap


class TestPlotConfusionMatrix:
    def test_creates_figure(self, temp_dir):
        y_true = np.random.randint(0, 3, 100)
        y_pred = np.random.randint(0, 3, 100)
        save_path = os.path.join(temp_dir, "confusion.png")
        plot_confusion_matrix(y_true, y_pred, save_path=save_path)
        assert os.path.exists(save_path)

    def test_with_class_names(self, temp_dir):
        y_true = np.random.randint(0, 3, 100)
        y_pred = np.random.randint(0, 3, 100)
        class_names = ["Class A", "Class B", "Class C"]
        save_path = os.path.join(temp_dir, "confusion.png")
        plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=save_path)
        assert os.path.exists(save_path)

    def test_returns_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        cm = plot_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        cm = plot_confusion_matrix(y_true, y_pred)
        assert np.diag(cm).sum() == len(y_true)


class TestPlotFewshotCurve:
    def test_creates_figure(self, temp_dir):
        results = {
            10: {"rf_f1": 0.5, "lgbm_f1": 0.55, "olmo_lgbm_f1": 0.6},
            25: {"rf_f1": 0.6, "lgbm_f1": 0.65, "olmo_lgbm_f1": 0.7},
            50: {"rf_f1": 0.7, "lgbm_f1": 0.75, "olmo_lgbm_f1": 0.8},
            100: {"rf_f1": 0.8, "lgbm_f1": 0.85, "olmo_lgbm_f1": 0.9},
        }
        save_path = os.path.join(temp_dir, "fewshot.png")
        plot_fewshot_curve(results, save_path=save_path)
        assert os.path.exists(save_path)

    def test_handles_single_shot(self, temp_dir):
        results = {10: {"rf_f1": 0.5, "lgbm_f1": 0.55, "olmo_lgbm_f1": 0.6}}
        save_path = os.path.join(temp_dir, "fewshot.png")
        plot_fewshot_curve(results, save_path=save_path)
        assert os.path.exists(save_path)


class TestPlotUmap:
    def test_creates_figure(self, temp_dir):
        embeddings = np.random.rand(100, 64).astype(np.float32)
        labels = np.random.randint(0, 5, 100)
        save_path = os.path.join(temp_dir, "umap.png")
        plot_umap(embeddings, labels, save_path=save_path)
        assert os.path.exists(save_path)

    def test_with_title(self, temp_dir):
        embeddings = np.random.rand(50, 32).astype(np.float32)
        labels = np.random.randint(0, 3, 50)
        save_path = os.path.join(temp_dir, "umap.png")
        plot_umap(embeddings, labels, title="Test UMAP", save_path=save_path)
        assert os.path.exists(save_path)

    def test_handles_single_class(self, temp_dir):
        embeddings = np.random.rand(20, 16).astype(np.float32)
        labels = np.zeros(20, dtype=int)
        save_path = os.path.join(temp_dir, "umap.png")
        plot_umap(embeddings, labels, save_path=save_path)
        assert os.path.exists(save_path)

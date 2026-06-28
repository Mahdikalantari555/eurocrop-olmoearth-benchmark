"""
Tests for src/viz/*.py
"""

import pytest
import numpy as np
import os
from src.viz.fewshot_curve import plot_fewshot_curve
from src.viz.umap_viz import plot_umap, _get_cmap


class TestPlotFewshotCurve:
    def test_creates_figure(self, temp_dir):
        results = {
            "10": {"rf_f1": 0.5, "rf_f1_std": 0.05, "lgbm_f1": 0.55, "lgbm_f1_std": 0.04,
                   "olmo_lgbm_f1": 0.6, "olmo_lgbm_f1_std": 0.03},
            "25": {"rf_f1": 0.6, "rf_f1_std": 0.04, "lgbm_f1": 0.65, "lgbm_f1_std": 0.03,
                   "olmo_lgbm_f1": 0.7, "olmo_lgbm_f1_std": 0.02},
            "50": {"rf_f1": 0.7, "rf_f1_std": 0.03, "lgbm_f1": 0.75, "lgbm_f1_std": 0.02,
                   "olmo_lgbm_f1": 0.8, "olmo_lgbm_f1_std": 0.02},
        }
        save_path = os.path.join(temp_dir, "fewshot.png")
        plot_fewshot_curve(results, save_path=save_path)
        assert os.path.exists(save_path)

    def test_handles_single_shot(self, temp_dir):
        results = {
            "10": {"rf_f1": 0.5, "rf_f1_std": 0.05, "lgbm_f1": 0.55, "lgbm_f1_std": 0.04,
                   "olmo_lgbm_f1": 0.6, "olmo_lgbm_f1_std": 0.03}
        }
        save_path = os.path.join(temp_dir, "fewshot.png")
        plot_fewshot_curve(results, save_path=save_path)
        assert os.path.exists(save_path)

    def test_handles_missing_std(self, temp_dir):
        results = {
            "10": {"rf_f1": 0.5, "lgbm_f1": 0.55, "olmo_lgbm_f1": 0.6},
            "25": {"rf_f1": 0.6, "lgbm_f1": 0.65, "olmo_lgbm_f1": 0.7},
        }
        save_path = os.path.join(temp_dir, "fewshot.png")
        plot_fewshot_curve(results, save_path=save_path)
        assert os.path.exists(save_path)

    def test_shots_sorted_numerically(self, temp_dir):
        results = {
            "500": {"rf_f1": 0.9, "rf_f1_std": 0.01, "lgbm_f1": 0.91, "lgbm_f1_std": 0.01,
                    "olmo_lgbm_f1": 0.92, "olmo_lgbm_f1_std": 0.01},
            "5": {"rf_f1": 0.3, "rf_f1_std": 0.1, "lgbm_f1": 0.35, "lgbm_f1_std": 0.09,
                  "olmo_lgbm_f1": 0.4, "olmo_lgbm_f1_std": 0.08},
            "100": {"rf_f1": 0.8, "rf_f1_std": 0.02, "lgbm_f1": 0.82, "lgbm_f1_std": 0.02,
                    "olmo_lgbm_f1": 0.85, "olmo_lgbm_f1_std": 0.02},
        }
        save_path = os.path.join(temp_dir, "fewshot.png")
        plot_fewshot_curve(results, save_path=save_path)
        assert os.path.exists(save_path)


class TestGetCmap:
    def test_small_classes(self):
        colors = _get_cmap(5)
        assert len(colors) == 5

    def test_exact_20(self):
        colors = _get_cmap(20)
        assert len(colors) == 20

    def test_over_20(self):
        colors = _get_cmap(45)
        assert len(colors) == 45

    def test_large_classes(self):
        colors = _get_cmap(100)
        assert len(colors) == 100


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

    def test_many_classes(self, temp_dir):
        embeddings = np.random.rand(200, 32).astype(np.float32)
        labels = np.random.randint(0, 25, 200)
        save_path = os.path.join(temp_dir, "umap.png")
        plot_umap(embeddings, labels, save_path=save_path)
        assert os.path.exists(save_path)

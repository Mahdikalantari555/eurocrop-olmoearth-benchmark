"""
Integration tests for the full pipeline.
"""

import pytest
import numpy as np
import os
import tempfile
import yaml
from src.data.loader import load_dataset, train_test_split_stratified
from src.data.features import ndvi_features, band_stat_features
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics, load_metrics


class TestEndToEndPipeline:
    def test_feature_extraction_to_classification(self, temp_dir):
        np.random.seed(42)
        X = np.random.rand(100, 12, 13).astype(np.float32) * 1000
        y = np.random.randint(0, 3, 100)

        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2)

        X_train_feat = ndvi_features(X_train)
        X_test_feat = ndvi_features(X_test)

        clf = get_classifier("rf")
        clf.fit(X_train_feat, y_train)
        y_pred = clf.predict(X_test_feat)

        metrics = compute_metrics(y_test, y_pred)
        assert 0 <= metrics["overall_accuracy"] <= 1
        assert 0 <= metrics["macro_f1"] <= 1

    def test_bandstat_features_pipeline(self, temp_dir):
        np.random.seed(42)
        X = np.random.rand(80, 12, 13).astype(np.float32) * 1000
        y = np.random.randint(0, 4, 80)

        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2)

        X_train_feat = band_stat_features(X_train)
        X_test_feat = band_stat_features(X_test)

        clf = get_classifier("lgbm")
        clf.fit(X_train_feat, y_train)
        y_pred = clf.predict(X_test_feat)

        metrics = compute_metrics(y_test, y_pred)
        assert metrics["overall_accuracy"] >= 0

    def test_metrics_save_load_cycle(self, temp_dir):
        metrics = {
            "overall_accuracy": 0.85,
            "macro_f1": 0.78,
            "weighted_f1": 0.84,
            "kappa": 0.81,
        }
        path = os.path.join(temp_dir, "metrics.json")
        save_metrics(metrics, path)
        loaded = load_metrics(path)
        assert loaded == metrics

    def test_multiple_classifiers_comparison(self):
        np.random.seed(42)
        X = np.random.rand(100, 12, 13).astype(np.float32) * 1000
        y = np.random.randint(0, 3, 100)

        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2)
        X_train_feat = ndvi_features(X_train)
        X_test_feat = ndvi_features(X_test)

        results = {}
        for clf_name in ["rf", "lgbm"]:
            clf = get_classifier(clf_name)
            clf.fit(X_train_feat, y_train)
            y_pred = clf.predict(X_test_feat)
            results[clf_name] = compute_metrics(y_test, y_pred)

        assert "rf" in results
        assert "lgbm" in results
        assert results["rf"]["overall_accuracy"] >= 0
        assert results["lgbm"]["overall_accuracy"] >= 0


class TestConfigLoading:
    def test_load_config(self, temp_dir):
        config = {
            "data": {
                "country": "Estonia",
                "data_dir": "./data",
                "top_n_classes": 15,
                "test_split": 0.2,
                "random_seed": 42,
            },
            "model": {
                "olmoearth_id": "allenai/OlmoEarth-v1_1-Nano",
                "batch_size": 32,
                "device": "cpu",
            },
            "fewshot": {
                "shots": [10, 25, 50, 100],
                "n_repeats": 5,
            },
            "output": {
                "metrics_dir": "./results/metrics",
                "figures_dir": "./results/figures",
            },
        }
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with open(config_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["data"]["country"] == "Estonia"
        assert loaded["model"]["batch_size"] == 32
        assert loaded["fewshot"]["shots"] == [10, 25, 50, 100]

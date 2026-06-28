"""
Tests for src/data/loader.py
"""

import pytest
import numpy as np
import os
import tempfile
from src.data.loader import load_dataset, filter_top_classes, train_test_split_stratified


class TestFilterTopClasses:
    def test_filters_to_n_classes(self, sample_data):
        X, y = sample_data
        X_filtered, y_filtered = filter_top_classes(X, y, n=3)
        assert len(np.unique(y_filtered)) == 3

    def test_preserves_total_samples(self, sample_data):
        X, y = sample_data
        X_filtered, y_filtered = filter_top_classes(X, y, n=3)
        assert X_filtered.shape[0] == y_filtered.shape[0]

    def test_remaps_labels_to_contiguous(self, sample_data):
        X, y = sample_data
        X_filtered, y_filtered = filter_top_classes(X, y, n=3)
        unique_labels = sorted(np.unique(y_filtered))
        assert unique_labels == [0, 1, 2]

    def test_keeps_most_frequent_classes(self):
        y = np.array([0] * 50 + [1] * 30 + [2] * 15 + [3] * 5)
        X = np.random.rand(len(y), 6, 13)
        X_filtered, y_filtered = filter_top_classes(X, y, n=2)
        assert len(np.unique(y_filtered)) == 2
        assert np.sum(y_filtered == 0) == 50
        assert np.sum(y_filtered == 1) == 30

    def test_handles_single_class(self):
        y = np.array([0] * 10)
        X = np.random.rand(10, 6, 13)
        X_filtered, y_filtered = filter_top_classes(X, y, n=1)
        assert len(np.unique(y_filtered)) == 1


class TestTrainTestSplitStratified:
    def test_split_ratio(self, sample_data):
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2)
        assert len(X_test) == pytest.approx(len(X) * 0.2, abs=2)
        assert len(X_train) == pytest.approx(len(X) * 0.8, abs=2)

    def test_stratified_preserves_class_distribution(self):
        y = np.array([0] * 50 + [1] * 50)
        X = np.random.rand(100, 6, 13)
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.2)
        train_ratio = np.sum(y_train == 0) / len(y_train)
        test_ratio = np.sum(y_test == 0) / len(y_test)
        assert abs(train_ratio - test_ratio) < 0.1

    def test_no_data_leakage(self, sample_data):
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
        train_indices = set(range(len(X_train)))
        test_indices = set(range(len(X_train), len(X_train) + len(X_test)))
        assert train_indices.isdisjoint(test_indices)

    def test_random_seed_reproducibility(self, sample_data):
        X, y = sample_data
        split1 = train_test_split_stratified(X, y, seed=42)
        split2 = train_test_split_stratified(X, y, seed=42)
        np.testing.assert_array_equal(split1[0], split2[0])


class TestLoadDataset:
    def test_load_creates_valid_arrays(self, temp_dir):
        country = "Estonia"
        country_dir = os.path.join(temp_dir, country)
        os.makedirs(country_dir)

        np.random.seed(42)
        X = np.random.rand(50, 12, 13).astype(np.float32)
        y = np.random.randint(0, 5, 50)
        np.savez(os.path.join(country_dir, "test.npz"), X=X, y=y)

        X_loaded, y_loaded, label_names = load_dataset(temp_dir, country, top_n_classes=3)
        assert X_loaded.shape[1:] == (12, 13)
        assert len(np.unique(y_loaded)) <= 3
        assert isinstance(label_names, dict)

    def test_load_handles_multiple_files(self, temp_dir):
        country = "Latvia"
        country_dir = os.path.join(temp_dir, country)
        os.makedirs(country_dir)

        for i in range(3):
            X = np.random.rand(20, 12, 13).astype(np.float32)
            y = np.random.randint(0, 5, 20)
            np.savez(os.path.join(country_dir, f"file{i}.npz"), X=X, y=y)

        X_loaded, y_loaded, _ = load_dataset(temp_dir, country, top_n_classes=5)
        assert X_loaded.shape[0] == 60

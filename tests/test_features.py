"""
Tests for src/data/features.py
"""

import pytest
import numpy as np
from src.data.features import ndvi_features, band_stat_features, temporal_features


class TestNdviFeatures:
    def test_output_shape(self, sample_data):
        X, _ = sample_data
        features = ndvi_features(X)
        assert features.shape == (X.shape[0], 4)

    def test_ndvi_range(self, sample_data):
        X, _ = sample_data
        features = ndvi_features(X)
        assert np.all(features[:, 0] >= -1.1)
        assert np.all(features[:, 0] <= 1.1)

    def test_mean_equals_mean(self):
        np.random.seed(42)
        X = np.random.rand(10, 12, 13).astype(np.float32) * 1000
        features = ndvi_features(X)
        assert np.all(np.isfinite(features))

    def test_max_greater_than_min(self, sample_data):
        X, _ = sample_data
        features = ndvi_features(X)
        assert np.all(features[:, 1] >= features[:, 2])

    def test_handles_uniform_values(self):
        X = np.ones((5, 10, 13)) * 500
        features = ndvi_features(X)
        assert np.all(features[:, 3] == 0)


class TestBandStatFeatures:
    def test_output_shape(self, sample_data):
        X, _ = sample_data
        features = band_stat_features(X)
        assert features.shape == (X.shape[0], X.shape[2] * 3)

    def test_mean_within_range(self, sample_data):
        X, _ = sample_data
        features = band_stat_features(X)
        mean_part = features[:, :X.shape[2]]
        assert np.all(mean_part >= 0)
        assert np.all(mean_part <= 1000)

    def test_std_non_negative(self, sample_data):
        X, _ = sample_data
        features = band_stat_features(X)
        C = X.shape[2]
        std_part = features[:, C:2*C]
        assert np.all(std_part >= 0)

    def test_max_greater_than_mean(self, sample_data):
        X, _ = sample_data
        features = band_stat_features(X)
        C = X.shape[2]
        mean_part = features[:, :C]
        max_part = features[:, 2*C:3*C]
        assert np.all(max_part >= mean_part - 1e-6)


class TestTemporalFeatures:
    def test_output_shape(self, small_data):
        X, _ = small_data
        features = temporal_features(X)
        assert features.shape == (X.shape[0], 2)

    def test_finite_output(self, small_data):
        X, _ = small_data
        features = temporal_features(X)
        assert np.all(np.isfinite(features))

    def test_handles_constant_time_series(self):
        X = np.ones((5, 10, 13)) * 100
        features = temporal_features(X)
        assert np.allclose(features[:, 0], 0, atol=1e-10)

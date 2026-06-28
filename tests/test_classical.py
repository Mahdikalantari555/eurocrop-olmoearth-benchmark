"""
Tests for src/models/classical.py
"""

import pytest
import numpy as np
from src.models.classical import get_classifier


class TestGetClassifier:
    def test_returns_random_forest(self):
        clf = get_classifier("rf")
        assert clf is not None
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict")

    def test_returns_lightgbm(self):
        clf = get_classifier("lgbm")
        assert clf is not None
        assert hasattr(clf, "fit")

    def test_returns_xgboost(self):
        clf = get_classifier("xgb")
        assert clf is not None
        assert hasattr(clf, "fit")

    def test_returns_logistic_regression(self):
        clf = get_classifier("logreg")
        assert clf is not None
        assert hasattr(clf, "fit")

    def test_invalid_classifier_raises(self):
        with pytest.raises(KeyError):
            get_classifier("invalid")

    def test_random_state_reproducibility(self):
        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        clf1 = get_classifier("rf", seed=42)
        clf1.fit(X, y)
        pred1 = clf1.predict(X[:10])

        clf2 = get_classifier("rf", seed=42)
        clf2.fit(X, y)
        pred2 = clf2.predict(X[:10])

        np.testing.assert_array_equal(pred1, pred2)


class TestClassifierIntegration:
    def test_rf_fit_predict(self, small_data):
        X, y = small_data
        X_flat = X.reshape(X.shape[0], -1)
        clf = get_classifier("rf")
        clf.fit(X_flat, y)
        preds = clf.predict(X_flat)
        assert preds.shape == (len(y),)
        assert set(np.unique(preds)).issubset(set(np.unique(y)))

    def test_lgbm_fit_predict(self, small_data):
        X, y = small_data
        X_flat = X.reshape(X.shape[0], -1)
        clf = get_classifier("lgbm")
        clf.fit(X_flat, y)
        preds = clf.predict(X_flat)
        assert preds.shape == (len(y),)

    def test_xgb_fit_predict(self, small_data):
        X, y = small_data
        X_flat = X.reshape(X.shape[0], -1)
        clf = get_classifier("xgb")
        clf.fit(X_flat, y)
        preds = clf.predict(X_flat)
        assert preds.shape == (len(y),)

    def test_logreg_fit_predict(self, small_data):
        X, y = small_data
        X_flat = X.reshape(X.shape[0], -1)
        clf = get_classifier("logreg")
        clf.fit(X_flat, y)
        preds = clf.predict(X_flat)
        assert preds.shape == (len(y),)

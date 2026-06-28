"""
Classical feature extraction from time series.

Input: X shape (N, T, C) — 13 band Sentinel-2
"""

import numpy as np

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06",
            "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
B4_IDX = S2_BANDS.index("B04")
B8_IDX = S2_BANDS.index("B08")


def ndvi_features(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, 4) — [mean, max, min, std] of NDVI time series
    """
    red = X[:, :, B4_IDX]
    nir = X[:, :, B8_IDX]
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.stack([
        ndvi.mean(axis=1),
        ndvi.max(axis=1),
        ndvi.min(axis=1),
        ndvi.std(axis=1),
    ], axis=1)


def band_stat_features(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, C*3) — mean, std, max per band across time
    """
    return np.concatenate([
        X.mean(axis=1),
        X.std(axis=1),
        X.max(axis=1),
    ], axis=1)


def temporal_features(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, C*2) — slope and variance of first principal component per sample
    """
    N, T, C = X.shape
    features = []
    for i in range(N):
        sample = X[i]  # (T, C)
        _, S, Vt = np.linalg.svd(sample, full_matrices=False)
        pca = sample @ Vt[0]  # (T,) — projection onto first PC
        slope = np.polyfit(np.arange(T), pca, 1)[0]
        features.append([slope, np.var(pca)])
    return np.array(features)

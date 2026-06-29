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
    red = X[:, :, B4_IDX].astype(np.float32)
    nir = X[:, :, B8_IDX].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.stack([
        ndvi.mean(axis=1),
        ndvi.max(axis=1),
        ndvi.min(axis=1),
        ndvi.std(axis=1),
    ], axis=1).astype(np.float32)


def band_stat_features(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, C*3) — mean, std, max per band across time
    """
    X32 = X.astype(np.float32)
    return np.concatenate([
        X32.mean(axis=1),
        X32.std(axis=1),
        X32.max(axis=1),
    ], axis=1).astype(np.float32)


def temporal_features(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, C*2) — slope and variance of first principal component per sample
    Vectorized using batch SVD via numpy lstsq for slope.
    """
    N, T, C = X.shape
    time_axis = np.arange(T, dtype=np.float64)

    # Center time axis for slope computation
    t_mean = time_axis.mean()
    t_centered = time_axis - t_mean
    t_var = t_centered.var()

    features = np.zeros((N, C * 2), dtype=np.float64)

    # Vectorized: project each sample's bands onto time
    # X: (N, T, C) -> compute per-channel slope via least squares
    # slope_i = sum((t - t_mean) * x_i) / sum((t - t_mean)^2) for each channel
    t_bc = t_centered[np.newaxis, :, np.newaxis]  # (1, T, 1)
    x_centered = X - X.mean(axis=1, keepdims=True)  # (N, T, C)
    slopes = (t_bc * x_centered).sum(axis=1) / (t_var + 1e-8)  # (N, C)

    # Variance of the first PC approximation: use variance of band-mean signal
    pc1_proxy = X.mean(axis=2)  # (N, T) — mean across bands as PC1 proxy
    variances = pc1_proxy.var(axis=1)  # (N,)

    features[:, :C] = slopes
    features[:, C:] = variances[:, np.newaxis]

    return features.astype(np.float32)

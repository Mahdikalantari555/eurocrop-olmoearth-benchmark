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


def mean_ndvi(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, 1) — mean NDVI across time
    """
    red = X[:, :, B4_IDX].astype(np.float32)
    nir = X[:, :, B8_IDX].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi.mean(axis=1, keepdims=True).astype(np.float32)


def std_ndvi(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, 1) — standard deviation of NDVI across time
    """
    red = X[:, :, B4_IDX].astype(np.float32)
    nir = X[:, :, B8_IDX].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi.std(axis=1, keepdims=True).astype(np.float32)


def mean_red(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, 1) — mean Red band (B04) across time
    """
    red = X[:, :, B4_IDX].astype(np.float32)
    return red.mean(axis=1, keepdims=True).astype(np.float32)


def mean_nir(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, 1) — mean NIR band (B08) across time
    """
    nir = X[:, :, B8_IDX].astype(np.float32)
    return nir.mean(axis=1, keepdims=True).astype(np.float32)


def mean_green(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, 1) — mean Green band (B03) across time
    """
    B3_IDX = S2_BANDS.index("B03")
    green = X[:, :, B3_IDX].astype(np.float32)
    return green.mean(axis=1, keepdims=True).astype(np.float32)


def mean_blue(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, 1) — mean Blue band (B02) across time
    """
    B2_IDX = S2_BANDS.index("B02")
    blue = X[:, :, B2_IDX].astype(np.float32)
    return blue.mean(axis=1, keepdims=True).astype(np.float32)


def spectral_statistics(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, C*4) — mean, std, min, max per band across time
    """
    X32 = X.astype(np.float32)
    return np.concatenate([
        X32.mean(axis=1),
        X32.std(axis=1),
        X32.min(axis=1),
        X32.max(axis=1),
    ], axis=1).astype(np.float32)


def ndvi_percentiles(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, 4) — 25th, 50th, 75th, 90th percentiles of NDVI
    """
    red = X[:, :, B4_IDX].astype(np.float32)
    nir = X[:, :, B8_IDX].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.stack([
        np.percentile(ndvi, 25, axis=1),
        np.percentile(ndvi, 50, axis=1),
        np.percentile(ndvi, 75, axis=1),
        np.percentile(ndvi, 90, axis=1),
    ], axis=1).astype(np.float32)


def combined_baseline_features(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, C)
    Returns: (N, feature_dim) — combined baseline features
    """
    return np.concatenate([
        ndvi_features(X),
        mean_ndvi(X),
        std_ndvi(X),
        mean_red(X),
        mean_nir(X),
        mean_green(X),
        mean_blue(X),
        ndvi_percentiles(X),
    ], axis=1).astype(np.float32)


def stream_and_save_features(data_dir, split_dir, use_case, split_key,
                             output_path, top_n_classes=None,
                             batch_size=256, n_workers=8):
    """
    Stream data from disk, extract features per batch, save incrementally.
    Never loads full (N, T, C) into memory.

    Args:
        data_dir: Path to preprocess directory
        split_dir: Path to split directory
        use_case: Use case name
        split_key: "train", "val", or "test"
        output_path: Path to save .npz file
        top_n_classes: Optional filter to top-N classes
        batch_size: Samples per batch
        n_workers: Number of parallel workers for I/O
    """
    import os
    import gc
    import json
    import time
    import numpy as np
    from collections import Counter

    t_start = time.time()

    # Step 1: Get class filter
    print(f"  [1/4] Scanning classes...", end=" ", flush=True)
    class_counter = Counter()
    for f in os.listdir(data_dir):
        if f.endswith(".npz"):
            cls = f.split("_")[-1].replace(".npz", "")
            class_counter[cls] += 1
    print(f"{len(class_counter)} classes found ({time.time()-t_start:.1f}s)")

    class_filter = None
    label_map = None
    if top_n_classes:
        top_classes = [c for c, _ in class_counter.most_common(top_n_classes)]
        class_filter = set(top_classes)
        label_map = {c: i for i, c in enumerate(sorted(top_classes))}
        print(f"  Filtered to {len(class_filter)} classes: {top_classes[:5]}...")

    # Step 2: Read split file and filter filenames
    print(f"  [2/4] Reading split file...", end=" ", flush=True)
    split_file = os.path.join(split_dir, use_case, "finetune", "region_split_all.json")
    with open(split_file) as f:
        split_data = json.load(f)
    filenames = split_data[split_key]

    if class_filter:
        filenames = [fn for fn in filenames
                     if fn.split("_")[-1].replace(".npz", "") in class_filter]
    print(f"{len(filenames)} files to process ({time.time()-t_start:.1f}s)")

    # Step 3: Load and extract features in batches
    print(f"  [3/4] Extracting features (batch_size={batch_size})...")

    all_feats = []
    all_labels = []
    processed = 0
    batch_count = 0

    for i in range(0, len(filenames), batch_size):
        batch_files = filenames[i:i + batch_size]
        batch_X = []
        batch_y = []

        for fn in batch_files:
            filepath = os.path.join(data_dir, fn)
            try:
                data = np.load(filepath, allow_pickle=True)
                X = data["data"]
                cls_label = fn.split("_")[-1].replace(".npz", "")
                batch_X.append(X)
                batch_y.append(cls_label)
            except Exception:
                continue

        if batch_X:
            feat = _extract_batch_features(batch_X)
            all_feats.append(feat)

            if label_map is not None:
                y_encoded = np.array([label_map[c] for c in batch_y], dtype=np.int64)
            else:
                y_encoded = np.array(list(range(len(batch_y))), dtype=np.int64)
            all_labels.append(y_encoded)

            processed += len(batch_X)
            batch_count += 1

            if batch_count % 10 == 0:
                print(f"    Batch {batch_count}: {processed} files ({time.time()-t_start:.1f}s)")

        del batch_X, batch_y
        gc.collect()

    # Step 4: Save
    print(f"  [4/4] Saving...", end=" ", flush=True)
    if all_feats:
        X_feats = np.concatenate(all_feats, axis=0)
        y_labels = np.concatenate(all_labels, axis=0)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez(output_path, X=X_feats, y=y_labels)
        elapsed = time.time() - t_start
        print(f"{output_path} | shape={X_feats.shape} ({elapsed:.1f}s)")
    else:
        print("Warning: No data processed")

    return {"output_path": output_path, "processed": processed}


def _extract_batch_features(batch_X):
    """Extract features from a batch of variable-length sequences."""
    import numpy as np

    max_T = max(x.shape[0] for x in batch_X)
    C = batch_X[0].shape[1]
    X_padded = np.zeros((len(batch_X), max_T, C), dtype=np.float32)
    for i, x in enumerate(batch_X):
        T = min(x.shape[0], max_T)
        X_padded[i, :T, :] = x[:T, :]

    return combined_baseline_features(X_padded)

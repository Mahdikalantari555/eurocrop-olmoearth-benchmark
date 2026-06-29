"""
Generic multi-format data I/O for satellite imagery.

Supports: NPZ, TIF/GeoTIFF, NPY, PNG, HDF5
Auto-detects format from extension.

For EuroCropsML: NPZ files with (T, C) time series
For EuroSAT-MS: TIF files with (C, H, W) or (H, W, C) spatial images

Usage:
    from src.data.io import load_sample, detect_format, stream_samples

    data, label = load_sample("path/to/file.tif")
    data, label = load_sample("path/to/file.npz")
"""

import os
import gc
import json
import numpy as np
from typing import Generator, Tuple, Optional, Dict, List
from pathlib import Path


def detect_format(filepath: str) -> str:
    """Detect file format from extension."""
    ext = os.path.splitext(filepath)[1].lower()
    format_map = {
        ".npz": "npz",
        ".npy": "npy",
        ".tif": "tif",
        ".tiff": "tif",
        ".png": "png",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
    }
    return format_map.get(ext, "unknown")


def load_sample(filepath: str, label_from: str = "filename") -> Tuple[np.ndarray, str]:
    """
    Load a single sample from any supported format.

    Args:
        filepath: Path to data file
        label_from: How to extract label:
            - "filename": extract from filename (NPZ convention)
            - "dirname": use parent directory name
            - "metadata": read from file metadata
            - "none": return empty label

    Returns:
        Tuple of (data_array, class_label)
    """
    fmt = detect_format(filepath)

    if fmt == "npz":
        return _load_npz(filepath, label_from)
    elif fmt == "npy":
        return _load_npy(filepath, label_from)
    elif fmt == "tif":
        return _load_tif(filepath, label_from)
    elif fmt == "png":
        return _load_png(filepath, label_from)
    elif fmt == "hdf5":
        return _load_hdf5(filepath, label_from)
    else:
        raise ValueError(f"Unsupported format: {ext} for {filepath}")


def _load_npz(filepath: str, label_from: str) -> Tuple[np.ndarray, str]:
    """Load NPZ file. Returns (data, label)."""
    fname = os.path.basename(filepath)
    npz = np.load(filepath, allow_pickle=True)

    # Try common keys
    data = None
    for key in ["data", "X", "x", "array", "values"]:
        if key in npz:
            data = npz[key]
            break
    if data is None:
        # Use first array found
        for key in npz.files:
            if not key.startswith("_"):
                data = npz[key]
                break
    if data is None:
        raise ValueError(f"No data found in {filepath}")

    label = _extract_label(fname, label_from)
    return data.astype(np.float32), label


def _load_npy(filepath: str, label_from: str) -> Tuple[np.ndarray, str]:
    """Load NPY file. Returns (data, label)."""
    fname = os.path.basename(filepath)
    data = np.load(filepath)
    label = _extract_label(fname, label_from)
    return data.astype(np.float32), label


def _load_tif(filepath: str, label_from: str) -> Tuple[np.ndarray, str]:
    """
    Load TIF/GeoTIFF file. Returns (data, label).

    Supports:
    - Single-band: (H, W) -> (1, H, W)
    - Multi-band: (C, H, W) or (H, W, C) -> (C, H, W)
    - Time series: (T, C, H, W) -> (T, C, H, W)
    """
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            data = src.read()  # (C, H, W) or (T, C, H, W)
    except ImportError:
        try:
            import tifffile
            data = tifffile.imread(filepath)
        except ImportError:
            raise ImportError(
                "TIF support requires rasterio or tifffile. "
                "Install: pip install rasterio  OR  pip install tifffile"
            )

    # Ensure float32
    data = data.astype(np.float32)

    # Normalize shape to (C, H, W) or (T, C, H, W)
    if data.ndim == 2:
        # (H, W) -> (1, H, W)
        data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        # Check if (H, W, C) or (C, H, W)
        # Heuristic: if last dim is small (bands), it's likely (H, W, C)
        if data.shape[2] <= data.shape[0] and data.shape[2] <= data.shape[1]:
            # Likely (H, W, C) -> transpose to (C, H, W)
            data = data.transpose(2, 0, 1)
    # ndim == 4: assume (T, C, H, W) already correct

    label = _extract_label(os.path.basename(filepath), label_from)
    return data, label


def _load_png(filepath: str, label_from: str) -> Tuple[np.ndarray, str]:
    """Load PNG image. Returns (data, label) with shape (C, H, W)."""
    try:
        from PIL import Image
        img = Image.open(filepath)
        data = np.array(img).astype(np.float32)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 3:
            data = data.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    except ImportError:
        raise ImportError("PNG support requires Pillow. Install: pip install Pillow")

    label = _extract_label(os.path.basename(filepath), label_from)
    return data, label


def _load_hdf5(filepath: str, label_from: str) -> Tuple[np.ndarray, str]:
    """Load HDF5 file. Returns (data, label)."""
    try:
        import h5py
        with h5py.File(filepath, "r") as f:
            # Try common keys
            data = None
            for key in ["data", "X", "x", "image", "array"]:
                if key in f:
                    data = np.array(f[key])
                    break
            if data is None:
                data = np.array(f[list(f.keys())[0]])
    except ImportError:
        raise ImportError("HDF5 support requires h5py. Install: pip install h5py")

    label = _extract_label(os.path.basename(filepath), label_from)
    return data.astype(np.float32), label


def _extract_label(filename: str, label_from: str) -> str:
    """Extract class label from filename."""
    if label_from == "none":
        return ""

    name = os.path.splitext(filename)[0]

    if label_from == "filename":
        # NPZ convention: <prefix>_<label>.npz
        parts = name.split("_")
        if len(parts) > 1:
            return parts[-1]
        return name

    elif label_from == "dirname":
        return os.path.basename(os.path.dirname(filename))

    elif label_from == "custom":
        # EuroSAT convention: sample__<class>__<id>.tif
        if "__" in name:
            parts = name.split("__")
            if len(parts) >= 2:
                return parts[1]
        # Fallback: try underscore-separated
        parts = name.split("_")
        if len(parts) > 1:
            return parts[-1]
        return name

    return name


def stream_files(data_dir: str,
                 extensions: set = None,
                 class_filter: set = None,
                 label_from: str = "filename",
                 max_samples: int = None) -> Generator[Tuple[str, np.ndarray, str], None, None]:
    """
    Generator that yields files from a directory, one at a time.

    Args:
        data_dir: Path to directory with data files
        extensions: Set of extensions to include (e.g., {".tif", ".npz"})
        class_filter: Optional set of class labels to include
        label_from: How to extract label from filename
        max_samples: Maximum number of samples to yield

    Yields:
        Tuple of (filename, data_array, class_label)
    """
    if extensions is None:
        extensions = {".npz", ".npy", ".tif", ".tiff"}

    count = 0
    for f in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in extensions:
            continue

        filepath = os.path.join(data_dir, f)

        try:
            data, label = load_sample(filepath, label_from)

            if class_filter and label not in class_filter:
                continue

            yield f, data, label
            count += 1

            if max_samples and count >= max_samples:
                break

        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue

        finally:
            gc.collect()


def stream_from_split_generic(data_dir: str,
                              split_file: str,
                              split_key: str = "train",
                              extensions: set = None,
                              class_filter: set = None,
                              label_from: str = "filename",
                              max_samples: int = None) -> Generator[Tuple[str, np.ndarray, str], None, None]:
    """
    Generator that yields samples from a split JSON file.

    Compatible with EuroCropsML split format and generic split files.

    Args:
        data_dir: Path to data directory
        split_file: Path to split JSON file
        split_key: Split key (train, val, test)
        extensions: Set of extensions to include
        class_filter: Optional set of class labels to include
        label_from: How to extract label
        max_samples: Maximum number of samples

    Yields:
        Tuple of (filename, data_array, class_label)
    """
    if extensions is None:
        extensions = {".npz", ".npy", ".tif", ".tiff"}

    with open(split_file) as f:
        split_data = json.load(f)

    filenames = split_data[split_key]

    count = 0
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in extensions:
            # Try adding extension
            for ext_candidate in extensions:
                test_fn = fn + ext_candidate
                filepath = os.path.join(data_dir, test_fn)
                if os.path.exists(filepath):
                    fn = test_fn
                    ext = ext_candidate
                    break
            else:
                continue

        filepath = os.path.join(data_dir, fn)
        if not os.path.exists(filepath):
            continue

        try:
            data, label = load_sample(filepath, label_from)

            if class_filter and label not in class_filter:
                continue

            yield fn, data, label
            count += 1

            if max_samples and count >= max_samples:
                break

        except Exception as e:
            print(f"Warning: Failed to load {fn}: {e}")
            continue

        finally:
            gc.collect()


def get_class_counts(data_dir: str,
                     extensions: set = None,
                     label_from: str = "filename") -> Dict[str, int]:
    """
    Count samples per class without loading data.

    Args:
        data_dir: Path to directory with data files
        extensions: Set of extensions to count
        label_from: How to extract label

    Returns:
        Dictionary of class -> count
    """
    from collections import Counter

    if extensions is None:
        extensions = {".npz", ".npy", ".tif", ".tiff"}

    counter = Counter()
    for f in os.listdir(data_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext not in extensions:
            continue
        label = _extract_label(f, label_from)
        counter[label] += 1

    return dict(counter)


def get_top_classes(data_dir: str,
                    n: int = 20,
                    extensions: set = None,
                    label_from: str = "filename") -> List[str]:
    """Get top N most frequent classes."""
    counts = get_class_counts(data_dir, extensions, label_from)
    sorted_classes = sorted(counts.items(), key=lambda x: -x[1])
    return [c for c, _ in sorted_classes[:n]]

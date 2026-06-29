"""
EuroCropML data loader.

Supports two modes:
- local: loads pre-processed .npz files using split JSONs (parallel loading)
- cloud: loads from Zenodo downloaded data (preprocess + split directories)
"""

import os
import json
import numpy as np
from collections import Counter
from multiprocessing import Pool
from functools import partial
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _load_single_npz(filepath):
    """Load a single .npz file. Returns (data_array, class_label) or None on error."""
    try:
        fname = os.path.basename(filepath)
        npz = np.load(filepath, allow_pickle=True)
        data = npz["data"]
        class_label = fname.split("_")[-1].replace(".npz", "")
        return data, class_label
    except Exception:
        return None


def _load_single_npz_from_dict(item):
    """Load a single .npz file from a dict item. Returns (data_array, class_label) or None."""
    filepath, class_label = item
    try:
        npz = np.load(filepath, allow_pickle=True)
        data = npz["data"]
        return data, class_label
    except Exception:
        return None


def load_split(preprocess_dir: str, split_dir: str, use_case: str,
               split_name: str = "all", n_workers: int = 8,
               max_samples: int = None):
    """
    Load data using local split JSONs with parallel I/O.

    Args:
        max_samples: if set, limit each split to this many samples
    """
    if split_name == "all":
        split_file = os.path.join(split_dir, use_case, "finetune",
                                  "region_split_all.json")
    else:
        split_file = os.path.join(split_dir, use_case, "finetune",
                                  f"region_split_{split_name}.json")

    with open(split_file) as f:
        split_data = json.load(f)

    splits = {}
    for split_key in ["train", "val", "test"]:
        filenames = split_data[split_key]
        if max_samples is not None:
            filenames = filenames[:max_samples]
        filepaths = [os.path.join(preprocess_dir, fn) for fn in filenames]
        existing = [fp for fp in filepaths if os.path.exists(fp)]

        print(f"  Loading {split_key}: {len(existing)}/{len(filenames)} files")
        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap(_load_single_npz, existing),
                                total=len(existing), desc=f"  {split_key}"))

        results = [r for r in results if r is not None]
        X_list = [r[0] for r in results]
        y_list = [r[1] for r in results]

        unique_labels = sorted(set(y_list))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        y_mapped = np.array([label_map[yl] for yl in y_list], dtype=np.int64)

        splits[split_key] = (X_list, y_mapped, unique_labels)

    return splits


def load_split_zenodo(preprocess_dir: str, split_dir: str, use_case: str,
                      split_name: str = "all", n_workers: int = 8,
                      max_samples: int = None):
    """
    Load data from Zenodo structure: preprocess/*.npz + split/<use_case>/*.json
    Handles flat preprocess directory with files named: <NUTS3>_<parcelID>_<hcat>.npz
    """
    if split_name == "all":
        split_file = os.path.join(split_dir, use_case, "finetune",
                                  "region_split_all.json")
    else:
        split_file = os.path.join(split_dir, use_case, "finetune",
                                  f"region_split_{split_name}.json")

    with open(split_file) as f:
        split_data = json.load(f)

    splits = {}
    for split_key in ["train", "val", "test"]:
        filenames = split_data[split_key]
        if max_samples is not None:
            filenames = filenames[:max_samples]

        items = []
        for fn in filenames:
            fp = os.path.join(preprocess_dir, fn)
            if os.path.exists(fp):
                class_label = fn.split("_")[-1].replace(".npz", "")
                items.append((fp, class_label))

        print(f"  Loading {split_key}: {len(items)}/{len(filenames)} files")
        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap(_load_single_npz_from_dict, items),
                                total=len(items), desc=f"  {split_key}"))

        results = [r for r in results if r is not None]
        X_list = [r[0] for r in results]
        y_list = [r[1] for r in results]

        if not X_list:
            splits[split_key] = (np.array([]), np.array([]), [])
            continue

        unique_labels = sorted(set(y_list))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        y_mapped = np.array([label_map[yl] for yl in y_list], dtype=np.int64)

        splits[split_key] = (X_list, y_mapped, unique_labels)

    return splits


def load_split_padded(preprocess_dir: str, split_dir: str, use_case: str,
                      split_name: str = "all", max_timesteps: int = None,
                      n_workers: int = 8, max_samples: int = None,
                      use_zenodo: bool = False):
    """
    Load data with padding to fixed timestep dimension.

    Args:
        max_samples: if set, limit each split to this many samples
        use_zenodo: if True, use Zenodo flat directory structure
    """
    if use_zenodo:
        splits = load_split_zenodo(preprocess_dir, split_dir, use_case,
                                   split_name, n_workers, max_samples)
    else:
        splits = load_split(preprocess_dir, split_dir, use_case, split_name,
                            n_workers, max_samples)

    result = {}
    for split_key in ["train", "val", "test"]:
        X_list, y, label_names = splits[split_key]
        if len(X_list) == 0:
            result[split_key] = (np.array([]), np.array([]), label_names)
            continue

        if max_timesteps is None:
            max_timesteps = max(x.shape[0] for x in X_list)

        C = X_list[0].shape[1]
        X_padded = np.zeros((len(X_list), max_timesteps, C), dtype=np.float32)
        for i, x in enumerate(X_list):
            T = min(x.shape[0], max_timesteps)
            X_padded[i, :T, :] = x[:T, :]

        result[split_key] = (X_padded, y, label_names)

    return result


def load_split_padded_cached(preprocess_dir: str, split_dir: str, use_case: str,
                              split_name: str = "all", max_timesteps: int = None,
                              n_workers: int = 8, cache_dir: str = None,
                              use_zenodo: bool = False):
    """
    Load with caching to .npz to avoid re-reading 100K+ files.
    """
    if cache_dir is None:
        cache_dir = os.path.join(split_dir, use_case, "cache", split_name)
    cache_file = os.path.join(cache_dir, "padded_data.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        label_names = data["label_names"].tolist()
        return {
            "train": (X_train, y_train, label_names),
            "val": (X_val, y_val, label_names),
            "test": (X_test, y_test, label_names),
        }

    splits = load_split_padded(preprocess_dir, split_dir, use_case,
                                split_name, max_timesteps, n_workers,
                                use_zenodo=use_zenodo)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_file,
             X_train=splits["train"][0], y_train=splits["train"][1],
             X_val=splits["val"][0], y_val=splits["val"][1],
             X_test=splits["test"][0], y_test=splits["test"][1],
             label_names=splits["train"][2])

    return splits


def load_dataset(data_dir: str, country: str, top_n_classes: int = 15):
    """Cloud mode: load EuroCropML dataset."""
    country_dir = os.path.join(data_dir, country)
    npz_files = [f for f in os.listdir(country_dir) if f.endswith('.npz')]

    all_X = []
    all_y = []

    for npz_file in npz_files:
        filepath = os.path.join(country_dir, npz_file)
        data = np.load(filepath, allow_pickle=True)
        all_X.append(data['X'])
        all_y.append(data['y'])

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    unique_labels = np.unique(y)
    label_names = {int(label): f"class_{int(label)}" for label in unique_labels}

    X, y = filter_top_classes(X, y, n=top_n_classes)
    return X, y, label_names


def filter_top_classes(X: np.ndarray, y: np.ndarray, n: int = 15):
    """Keep only the n most frequent classes."""
    class_counts = Counter(y)
    top_classes = [cls for cls, _ in class_counts.most_common(n)]
    mask = np.isin(y, top_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    class_map = {old: new for new, old in enumerate(sorted(top_classes))}
    y_mapped = np.array([class_map[label] for label in y_filtered])
    return X_filtered, y_mapped


def train_test_split_stratified(X: np.ndarray, y: np.ndarray,
                                 test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y)

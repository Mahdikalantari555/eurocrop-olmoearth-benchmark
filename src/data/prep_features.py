"""
Chunked feature extraction for memory-constrained environments.

Processes .npz files in chunks, extracts features per-sample,
and saves compact .npy files. Never loads all time series at once.
"""

import os
import json
import gc
import numpy as np
from tqdm import tqdm


def extract_features_chunked(preprocess_dir, split_dir, use_case, feature_fn,
                              output_dir, chunk_size=5000, use_zenodo=False):
    """
    Extract features in chunks and save to .npy files.

    Args:
        preprocess_dir: path to preprocess/*.npz files
        split_dir: path to split/<use_case>/finetune/*.json
        use_case: e.g. "latvia_vs_estonia"
        feature_fn: callable, takes (T, C) array -> (F,) feature vector
        output_dir: directory to save .npy files
        chunk_size: files to process per batch
    """
    os.makedirs(output_dir, exist_ok=True)

    split_file = os.path.join(split_dir, use_case, "finetune", "region_split_all.json")
    with open(split_file) as f:
        split_data = json.load(f)

    label_names = None

    for split_key in ["train", "test"]:
        filenames = split_data[split_key]

        items = []
        for fn in filenames:
            fp = os.path.join(preprocess_dir, fn)
            if os.path.exists(fp):
                class_label = fn.split("_")[-1].replace(".npz", "")
                items.append((fp, class_label))

        n = len(items)
        print(f"\n{split_key}: {n} samples, chunk_size={chunk_size}")

        feat_list = []
        y_list = []

        for start in tqdm(range(0, n, chunk_size), desc=f"  {split_key}"):
            chunk = items[start:start + chunk_size]
            for fp, label in chunk:
                try:
                    npz = np.load(fp, allow_pickle=True)
                    data = npz["data"]
                    feat = feature_fn(data)
                    feat_list.append(feat)
                    y_list.append(label)
                except Exception:
                    pass

            gc.collect()

        unique_labels = sorted(set(y_list))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        y_mapped = np.array([label_map[yl] for yl in y_list], dtype=np.int64)
        X_feat = np.array(feat_list, dtype=np.float32)

        np.save(os.path.join(output_dir, f"X_{split_key}.npy"), X_feat)
        np.save(os.path.join(output_dir, f"y_{split_key}.npy"), y_mapped)

        print(f"  Saved X_{split_key}.npy {X_feat.shape}, y_{split_key}.npy {y_mapped.shape}")

        if label_names is None:
            label_names = unique_labels

        del feat_list, y_list, X_feat, y_mapped
        gc.collect()

    label_names_path = os.path.join(output_dir, "label_names.json")
    with open(label_names_path, "w") as f:
        json.dump(label_names, f)
    print(f"  Saved label_names.json")


def load_prepared_features(output_dir):
    """Load previously saved features from .npy files."""
    X_train = np.load(os.path.join(output_dir, "X_train.npy"))
    y_train = np.load(os.path.join(output_dir, "y_train.npy"))
    X_test = np.load(os.path.join(output_dir, "X_test.npy"))
    y_test = np.load(os.path.join(output_dir, "y_test.npy"))
    with open(os.path.join(output_dir, "label_names.json")) as f:
        label_names = json.load(f)
    return X_train, y_train, X_test, y_test, label_names

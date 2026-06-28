"""
Full pipeline test script.

Demonstrates:
1. Local data loading (parallel .npz reading)
2. PatchEmbed adaptation: (N, T, 13) -> drop B8A -> (N, T, 12) -> (N, T, 128)
3. Encoder forward pass: (N, T, 128) -> transformer blocks -> (N, 128)
4. Classical features on same data
5. Classifier training + evaluation

Run: python scripts/test_pipeline.py
"""

import sys
import os
import time
import json
import numpy as np
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import _load_single_npz, load_split_padded
from src.data.features import ndvi_features, band_stat_features
from src.encoder.olmoearth import OlmoEarthEncoder, OLMoEarthEncoder
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_small_fraction(preprocess_dir, split_dir, use_case, max_samples=200):
    """Load a small fraction without multiprocessing (avoids Windows issues)."""
    split_file = os.path.join(split_dir, use_case, "finetune",
                              "region_split_all.json")
    with open(split_file) as f:
        split_data = json.load(f)

    splits = {}
    for split_key in ["train", "val", "test"]:
        filenames = split_data[split_key][:max_samples]
        X_list = []
        y_list = []
        for fn in filenames:
            fp = os.path.join(preprocess_dir, fn)
            if not os.path.exists(fp):
                continue
            result = _load_single_npz(fp)
            if result is not None:
                X_list.append(result[0])
                y_list.append(result[1])

        unique_labels = sorted(set(y_list))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        y_mapped = np.array([label_map[yl] for yl in y_list], dtype=np.int64)

        splits[split_key] = (X_list, y_mapped, unique_labels)

    return splits


def pad_splits(splits, max_timesteps=None):
    """Pad variable-length arrays to fixed shape."""
    result = {}
    for split_key in ["train", "val", "test"]:
        X_list, y, labels = splits[split_key]
        if len(X_list) == 0:
            result[split_key] = (np.array([]), np.array([]), labels)
            continue
        if max_timesteps is None:
            max_timesteps = max(x.shape[0] for x in X_list)
        C = X_list[0].shape[1]
        X_padded = np.zeros((len(X_list), max_timesteps, C), dtype=np.float32)
        for i, x in enumerate(X_list):
            T = min(x.shape[0], max_timesteps)
            X_padded[i, :T, :] = x[:T, :]
        result[split_key] = (X_padded, y, labels)
    return result


def trace_patchembed_adaptation():
    """Demonstrate how PatchEmbed adapts non-spatial (T, 13) to (T, 128)."""
    print("=" * 60)
    print("TRACE: PatchEmbed Adaptation (N, T, 13) -> (N, T, 128)")
    print("=" * 60)

    model = OlmoEarthEncoder(dim=128, n_blocks=4)

    B, T, C = 2, 10, 13
    x = torch.randn(B, T, C)
    print(f"\nInput:  {list(x.shape)}  (B={B}, T={T}, C={C} bands)")

    # Step 1: Drop B8A (index 8)
    x_12 = torch.cat([x[:, :, :8], x[:, :, 9:]], dim=-1)
    print(f"After dropping B8A: {list(x_12.shape)}  (B={B}, T={T}, C=12)")

    # Step 2: pixel_proj = Linear(12, 12)
    pe = model.patch_embeddings["sentinel2_l2a"]
    h = pe.pixel_proj(x_12)
    print(f"After pixel_proj:   {list(h.shape)}  Linear(12->12)")

    # Step 3: Pad to 768 for proj
    pad = torch.zeros(B, T, 768 - 12)
    h_padded = torch.cat([h, pad], dim=-1)
    print(f"After pad to 768:   {list(h_padded.shape)}")

    # Step 4: proj = Linear(768, 128)
    h_proj = pe.proj(h_padded)
    print(f"After proj:         {list(h_proj.shape)}  Linear(768->128)")

    # Step 5: composite_encodings (broadcast add)
    h_proj = h_proj + model.composite_encodings
    print(f"After pos encoding: {list(h_proj.shape)}  + composite_encodings")

    # Step 6: 4 transformer blocks
    for i, block in enumerate(model.blocks):
        h_proj = block(h_proj)
    print(f"After 4 blocks:     {list(h_proj.shape)}  TransformerBlock x 4")

    # Step 7: LayerNorm
    h_proj = model.norm(h_proj)
    print(f"After LayerNorm:    {list(h_proj.shape)}")

    # Step 8: mean pool over T
    h_pool = h_proj.mean(dim=1)
    print(f"After mean pool:    {list(h_pool.shape)}  (B, D) -- T collapsed")

    # Step 9: project
    h_out = model.project_and_aggregate(h_pool)
    print(f"After project:      {list(h_out.shape)}  Linear(128->128)")

    print(f"\nFinal embedding:    {list(h_out.shape)}  -- (N, 128)")
    print("=" * 60)


def test_data_loading(cfg):
    """Load a small fraction of data (200 samples per split)."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading local data (200 samples per split)")
    print("=" * 60)

    t0 = time.time()
    splits = load_small_fraction(
        cfg["data"]["local_preprocess_dir"],
        cfg["data"]["local_split_dir"],
        cfg["data"]["use_case"],
        max_samples=200
    )
    splits = pad_splits(splits)
    elapsed = time.time() - t0

    X_train, y_train, labels_train = splits["train"]
    X_val, y_val, labels_val = splits["val"]
    X_test, y_test, labels_test = splits["test"]

    print(f"Loaded in {elapsed:.1f}s")
    print(f"  train: X={list(X_train.shape)}, y={list(y_train.shape)}, {len(labels_train)} classes")
    print(f"  val:   X={list(X_val.shape)}, y={list(y_val.shape)}, {len(labels_val)} classes")
    print(f"  test:  X={list(X_test.shape)}, y={list(y_test.shape)}, {len(labels_test)} classes")
    print(f"  sample class names: {labels_train[:10]}")

    return X_train, y_train, X_val, y_val, X_test, y_test, labels_train


def test_encoder(X, cfg):
    """Encode a small batch through the encoder."""
    print("\n" + "=" * 60)
    print("STEP 2: Encoding samples through OLMoEarth encoder")
    print("=" * 60)

    encoder = OLMoEarthEncoder(
        mode="local",
        local_weights_path=cfg["model"]["local_weights_path"],
        device="cpu"
    )

    t0 = time.time()
    embeddings = encoder.encode(X, batch_size=20)
    elapsed = time.time() - t0

    print(f"Input:  {list(X.shape)}  (N={len(X)}, T={X.shape[1]}, C={X.shape[2]})")
    print(f"Output: {list(embeddings.shape)}  (N={len(X)}, D=128)")
    print(f"Time:   {elapsed:.2f}s ({elapsed/len(X)*1000:.1f}ms/sample)")
    print(f"Sample: {embeddings[0, :5]}")

    return embeddings


def test_classical_features(X):
    """Extract classical features."""
    print("\n" + "=" * 60)
    print("STEP 3: Classical feature extraction")
    print("=" * 60)

    t0 = time.time()
    ndvi = ndvi_features(X)
    band_stats = band_stat_features(X)
    elapsed = time.time() - t0

    print(f"Input:  {list(X.shape)}")
    print(f"NDVI:   {list(ndvi.shape)}  -- [mean, max, min, std]")
    print(f"Band:   {list(band_stats.shape)}  -- [mean, std, max] x 13 bands")
    print(f"Time:   {elapsed:.3f}s")

    return ndvi, band_stats


def test_classifier(X_features, y, X_test_feat, y_test):
    """Train and evaluate a classifier."""
    print("\n" + "=" * 60)
    print("STEP 4: Classifier training + evaluation")
    print("=" * 60)

    clf = get_classifier("rf", seed=42)
    t0 = time.time()
    clf.fit(X_features, y)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test_feat)
    t_pred = time.time() - t0

    metrics = compute_metrics(y_test, y_pred)
    print(f"Train time:  {t_train:.2f}s")
    print(f"Predict time:{t_pred:.2f}s")
    print(f"OA:  {metrics['overall_accuracy']:.4f}")
    print(f"F1:  {metrics['macro_f1']:.4f}")
    print(f"Kappa: {metrics['kappa']:.4f}")

    return metrics


def main():
    cfg = load_config()

    print("EuroCropML x OLMoEarth -- Full Pipeline Test")
    print(f"Config: mode={cfg['data']['mode']}, use_case={cfg['data']['use_case']}")
    print(f"Model:  {cfg['model']['local_weights_path']}")
    print()

    # Trace the PatchEmbed adaptation
    trace_patchembed_adaptation()

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, labels = test_data_loading(cfg)

    # Encode with OLMoEarth
    embeddings = test_encoder(X_train, cfg)

    # Classical features
    ndvi, band_stats = test_classical_features(X_train)

    # Train classifier on band stats
    metrics = test_classifier(band_stats, y_train, band_stat_features(X_test), y_test)

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Phase 0b: Embedding End-to-End Pipeline Validation

Answers: "Do OLMoEarth embeddings provide useful representations for EuroCropsML?"

Flow:
  Top-3 classes -> 20 samples/class -> StreamingDataset -> OLMoEarth embeddings -> LogReg -> metrics

Usage:
  python test_embedding_e2e.py
"""

import sys
import os
import time
import gc
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.streaming import StreamingDataset
from src.data.memory_profiler import get_memory_usage
from src.encoder.olmoearth import OLMoEarthEncoder
from src.evaluate.metrics import compute_metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR = "E:/RSGIS/OlmoEarth/Project/Data/preprocess/preprocess"
WEIGHTS_PATH = "E:/RSGIS/OlmoEarth/Project/Models/V1.1_Nano/weights.pth"
TOP_N_CLASSES = 3
SAMPLES_PER_CLASS = 20
BATCH_SIZE = 16
MEMORY_THRESHOLD_MB = 500


def main():
    print("=" * 60)
    print("Phase 0b: Embedding End-to-End Pipeline Validation")
    print("=" * 60)

    # --- 1. Find top-3 classes ---
    print("\n[1] Finding top-3 classes...")
    class_counter = Counter()
    for f in os.listdir(DATA_DIR):
        if f.endswith(".npz"):
            cls = f.split("_")[-1].replace(".npz", "")
            class_counter[cls] += 1
    top3 = [c for c, _ in class_counter.most_common(TOP_N_CLASSES)]
    print(f"    Top-3: {top3}")

    # --- 2. Load balanced data via StreamingDataset ---
    print(f"\n[2] Loading {SAMPLES_PER_CLASS} samples per class via streaming...")
    loaded_per_class = {c: 0 for c in top3}
    X_list, y_list = [], []
    target_total = TOP_N_CLASSES * SAMPLES_PER_CLASS

    dataset = StreamingDataset(DATA_DIR, batch_size=BATCH_SIZE, class_filter=set(top3))
    mem_before = get_memory_usage()

    for X_batch, y_batch in dataset:
        for i, label_int in enumerate(y_batch):
            cls = top3[label_int]
            if loaded_per_class[cls] < SAMPLES_PER_CLASS:
                X_list.append(X_batch[i])
                y_list.append(label_int)
                loaded_per_class[cls] += 1
        if sum(loaded_per_class.values()) >= target_total:
            break
        del X_batch, y_batch
        gc.collect()

    del dataset
    gc.collect()

    # Pad variable-length time series
    max_T = max(x.shape[0] for x in X_list)
    C = X_list[0].shape[1]
    X = np.zeros((len(X_list), max_T, C), dtype=np.float32)
    for i, x in enumerate(X_list):
        T = min(x.shape[0], max_T)
        X[i, :T, :] = x[:T, :]
    y = np.array(y_list, dtype=np.int64)
    print(f"    Loaded: {X.shape[0]} samples, shape={X.shape}")

    # --- 3. Stratified train/test split ---
    print("\n[3] Stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"    Train: {len(X_train)} | Test: {len(y_test)}")

    # --- 4. Load OLMoEarth encoder ---
    print("\n[4] Loading OLMoEarth encoder...")
    t0 = time.time()
    encoder = OLMoEarthEncoder(
        mode="local",
        local_weights_path=WEIGHTS_PATH,
        device="cpu"
    )
    print(f"    Encoder loaded in {time.time() - t0:.2f}s")

    # --- 5. Extract embeddings ---
    print("\n[5] Extracting embeddings...")
    t0 = time.time()
    emb_train = encoder.encode(X_train, batch_size=BATCH_SIZE)
    emb_test = encoder.encode(X_test, batch_size=BATCH_SIZE)
    encode_time = time.time() - t0
    print(f"    Train embeddings: {emb_train.shape}")
    print(f"    Test embeddings:  {emb_test.shape}")
    print(f"    Encoded in {encode_time:.2f}s")

    del encoder
    gc.collect()

    # --- 6. Embedding sanity checks ---
    print("\n[6] Embedding sanity checks...")
    assert emb_train.shape[0] > 0, "Empty training embeddings"
    assert emb_train.shape[1] > 0, "Zero-dim embeddings"
    assert np.isfinite(emb_train).all(), "Non-finite values in train embeddings"
    assert np.isfinite(emb_test).all(), "Non-finite values in test embeddings"
    assert emb_train.std() > 0, "Constant embeddings (no variance)"
    print(f"    Mean:   {emb_train.mean():.4f}")
    print(f"    Std:    {emb_train.std():.4f}")
    print(f"    Min:    {emb_train.min():.4f}")
    print(f"    Max:    {emb_train.max():.4f}")
    n_unique = np.unique(emb_train[:10], axis=0).shape[0]
    print(f"    Unique (first 10): {n_unique}/10")
    print("    All checks passed -- PASS")

    # --- 7. Train LogisticRegression ---
    print("\n[7] Training LogisticRegression (with StandardScaler)...")
    t0 = time.time()
    scaler = StandardScaler()
    emb_train_scaled = scaler.fit_transform(emb_train)
    emb_test_scaled = scaler.transform(emb_test)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(emb_train_scaled, y_train)
    print(f"    Trained in {time.time() - t0:.2f}s")

    # --- 8. Evaluate ---
    print("\n[8] Evaluating...")
    y_pred = clf.predict(emb_test_scaled)
    labels = sorted(set(y_test))
    m = compute_metrics(y_test, y_pred, labels=labels)
    acc = m["overall_accuracy"]
    f1 = m["macro_f1"]
    kappa = m["kappa"]
    print(f"    Accuracy:  {acc:.3f}")
    print(f"    Macro F1:  {f1:.3f}")
    print(f"    Kappa:     {kappa:.3f}")

    # --- 9. Assert finite metrics ---
    assert not np.isnan(acc), "Accuracy is NaN"
    assert not np.isnan(f1), "Macro F1 is NaN"
    assert not np.isnan(kappa), "Kappa is NaN"
    print("    Metrics are finite -- PASS")

    # --- 10. Memory check ---
    mem_after = get_memory_usage()
    delta_mb = mem_after["rss_mb"] - mem_before["rss_mb"]
    print(f"\n[9] Memory: {mem_before['rss_mb']:.1f} -> {mem_after['rss_mb']:.1f} MB (delta={delta_mb:+.1f} MB)")
    if delta_mb < MEMORY_THRESHOLD_MB:
        print(f"    Memory increase < {MEMORY_THRESHOLD_MB} MB -- PASS")
    else:
        print(f"    WARNING: Memory increase {delta_mb:.1f} MB exceeds {MEMORY_THRESHOLD_MB} MB threshold")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"RESULT: PASS  |  acc={acc:.3f}  f1={f1:.3f}  kappa={kappa:.3f}  mem_delta={delta_mb:+.1f}MB")
    print("=" * 60)


if __name__ == "__main__":
    main()

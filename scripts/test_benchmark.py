"""Test benchmark notebook loading and data functions."""
import sys, os, time, gc, json, warnings
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.inspect import get_top_classes, generate_class_distribution
from src.data.streaming import stream_from_split
from src.data.features import combined_baseline_features

DATA_DIR = "E:/RSGIS/OlmoEarth/Project/Data/preprocess/preprocess"
SPLIT_DIR = "E:/RSGIS/OlmoEarth/Project/Data/split/split"
USE_CASE = "overlap_latvia_vs_estonia"

def load_split_data(top_n_classes):
    top_classes = get_top_classes(DATA_DIR, top_n_classes)
    class_filter = set(top_classes)
    label_encoder = LabelEncoder()
    label_encoder.fit(top_classes)
    splits = {}
    for split_key in ["train", "val", "test"]:
        X_list, y_list = [], []
        for _, data, cls_label in stream_from_split(
            DATA_DIR, SPLIT_DIR, USE_CASE,
            split_name="all", split_key=split_key,
            class_filter=class_filter
        ):
            X_list.append(data)
            y_list.append(cls_label)
        if X_list:
            max_T = max(x.shape[0] for x in X_list)
            C = X_list[0].shape[1]
            X = np.zeros((len(X_list), max_T, C), dtype=np.float32)
            for i, x in enumerate(X_list):
                T = min(x.shape[0], max_T)
                X[i, :T, :] = x[:T, :]
            y = label_encoder.transform(y_list)
        else:
            X, y = np.array([]), np.array([])
        splits[split_key] = (X, y)
        del X_list, y_list
        gc.collect()
    return splits, top_classes, label_encoder

# Test with top-10
t0 = time.time()
splits, classes, le = load_split_data(10)
print("Loaded top-10 in %.1fs" % (time.time() - t0))
print("Train: %d | Val: %d | Test: %d" % (len(splits["train"][0]), len(splits["val"][0]), len(splits["test"][0])))
print("Classes: %d" % len(classes))

# Test features
X_tr, y_tr = splits["train"]
feat = combined_baseline_features(X_tr[:10])
print("Features shape: %s" % str(feat.shape))

# Test classifier
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics
from sklearn.preprocessing import StandardScaler

clf = get_classifier("logreg", seed=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feat)
clf.fit(X_scaled, y_tr[:10])
y_pred = clf.predict(X_scaled)
m = compute_metrics(y_tr[:10], y_pred)
print("Accuracy: %.3f | F1: %.3f" % (m["overall_accuracy"], m["macro_f1"]))

print("ALL OK")

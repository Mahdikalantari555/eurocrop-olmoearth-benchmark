"""Test script to verify refactored modules."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Testing refactored modules")
print("=" * 60)

# 1. Test imports
print("\n1. Testing imports...")
from src.data.loader import load_split_padded, load_split_padded_cached
from src.data.streaming import StreamingDataset, StreamingSplitDataset, create_batch_generator
from src.data.features import (ndvi_features, band_stat_features, spectral_statistics,
                                combined_baseline_features, mean_ndvi, std_ndvi)
from src.data.inspect import (inspect_directory_structure, count_samples, count_classes,
                               generate_class_distribution, get_top_classes)
from src.data.memory_profiler import (get_memory_usage, measure_single_npz,
                                       measure_batch_loading, identify_memory_bottlenecks)
from src.data.scaling import (get_system_resources, scale_sample_count,
                               evaluate_scalability)
from src.encoder.olmoearth import OLMoEarthEncoder
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics
from experiments.run_classification import ClassificationExperiments
print("   All imports OK")

# 2. Test memory profiler
print("\n2. Testing memory profiler...")
mem = get_memory_usage()
print(f"   Current RAM: {mem['rss_mb']} MB (of {mem['total_mb']} MB, {mem['percent_used']}% used)")

# 3. Test dataset inspection
print("\n3. Testing dataset inspection...")
data_dir = "E:/RSGIS/OlmoEarth/Project/Data/preprocess/preprocess"
if os.path.exists(data_dir):
    class_info = count_classes(data_dir)
    print(f"   Total classes: {class_info['total_classes']}")
    total = sum(class_info['class_distribution'].values())
    print(f"   Total samples: {total}")
    
    dist = generate_class_distribution(data_dir, top_n=5)
    print(f"   Top 5 classes:")
    for i, cls in enumerate(dist['top_n_classes'], 1):
        print(f"     {i}. {cls['class']}: {cls['count']} ({cls['percentage']}%)")
else:
    print(f"   Data dir not found: {data_dir}")

# 4. Test streaming loader
print("\n4. Testing streaming loader...")
if os.path.exists(data_dir):
    dataset = StreamingDataset(data_dir, batch_size=10, max_samples=20)
    print(f"   Dataset size: {len(dataset)} batches")
    for X_batch, y_batch in dataset:
        print(f"   Batch shape: X={X_batch.shape}, y={y_batch.shape}")
        break
else:
    print(f"   Data dir not found, skipping")

# 5. Test features
print("\n5. Testing features...")
import numpy as np
X_dummy = np.random.rand(5, 20, 13).astype(np.float32)
ndvi = ndvi_features(X_dummy)
print(f"   NDVI features shape: {ndvi.shape}")
band_stats = band_stat_features(X_dummy)
print(f"   Band stats shape: {band_stats.shape}")
combined = combined_baseline_features(X_dummy)
print(f"   Combined features shape: {combined.shape}")

# 6. Test classifiers
print("\n6. Testing classifiers...")
for name in ["logreg", "lgbm", "rf", "xgb"]:
    clf = get_classifier(name)
    print(f"   {name}: {type(clf).__name__}")

# 7. Test metrics
print("\n7. Testing metrics...")
y_true = np.array([0, 0, 1, 1, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0])
m = compute_metrics(y_true, y_pred)
print(f"   Accuracy: {m['overall_accuracy']:.3f}")
print(f"   Macro F1: {m['macro_f1']:.3f}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)

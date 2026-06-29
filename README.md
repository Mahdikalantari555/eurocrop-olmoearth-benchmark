# EuroCropML × OLMoEarth Benchmark v2.0

A lightweight and reproducible benchmark pipeline for evaluating OLMoEarth embeddings on EuroCropsML without loading the entire dataset into memory.

## Pipeline Phases

| Phase | Description | Module |
|-------|-------------|--------|
| 1 | Dataset Inspection | `src/data/inspect.py` |
| 2 | Memory Profiling | `src/data/memory_profiler.py` |
| 3 | Streaming Data Loader | `src/data/streaming.py` |
| 4 | Embedding Extraction | `src/encoder/olmoearth.py` |
| 5 | Baseline Features | `src/data/features.py` |
| 6 | Classification Experiments | `experiments/run_classification.py` |
| 7 | Scaling Evaluation | `src/data/scaling.py` |

## Quick Start

```bash
# End-to-end pipeline test (60 samples, ~5 seconds)
python test_e2e.py

# Embedding pipeline test (60 samples, ~10 seconds)
python test_embedding_e2e.py

# Full benchmark notebook (local)
jupyter notebook benchmark.ipynb

# Colab benchmark (4 models: Nano, Tiny, Base, Large)
# Upload benchmark_colab.ipynb to Colab and run all cells
```

## Architecture

The pipeline supports two independent modes controlled by `config.yaml`:

| Aspect | Local Mode | Cloud Mode |
|--------|-----------|------------|
| Data | Individual `.npz` parcel files + split JSONs | `eurocropsml` library (Zenodo) |
| Model | Local `weights.pth` state_dict | Auto-downloads from HuggingFace |
| Dependencies | No rslearn needed | No rslearn needed |
| Use case | Offline development, full dataset | Colab, quick prototyping |

Switch by setting `data.mode` and `model.mode` in `config.yaml`.

## Data Format

### Local mode

Each parcel is a single `.npz` file:
```
<NUTS3region>_<parcelID>_<EC_hcat_c>.npz
```

Keys inside each `.npz`:
| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `data` | `(T, 13)` | int64 | Sentinel-2 L1C time series — 13 bands, T timesteps |
| `dates` | `(T,)` | datetime64 | Observation dates |
| `center` | `(2,)` | float64 | Parcel centroid `[lon, lat]` |

Split JSONs define train/val/test parcel filenames:
```json
{
  "train": ["EE008_21485419_3301010102.npz", ...],
  "val":   ["EE004_22209357_3301060401.npz", ...],
  "test":  ["EE008_21369841_3302000000.npz", ...]
}
```

Use cases (cross-region splits):
- `latvia_vs_estonia` — pretrain on Latvia, finetune on Estonia
- `latvia_portugal_vs_estonia` — pretrain on Latvia+Portugal, finetune on Estonia
- `overlap_latvia_vs_estonia` — pretrain on overlapping classes, finetune on Estonia
- `overlap_latvia_portugal_vs_estonia` — same with Portugal added

Few-shot splits: `region_split_5.json`, `_10`, `_20`, `_100`, `_200`, `_500`, `_all`.

### Cloud mode

Standard EuroCropML format with keys `X` and `y` per `.npz` file, loaded via `eurocropsml` library.

## Model

### Local weights

`weights.pth` is an `OrderedDict` state_dict with three top-level components:

| Prefix | Purpose |
|--------|---------|
| `encoder.*` | **Used for embeddings** — 4 transformer blocks, 128-dim |
| `decoder.*` | Not used in this benchmark |
| `target_encoder.*` | Not used (EMA copy) |

Encoder architecture (128-dim, 4 blocks):
```
patch_embeddings → composite_encodings → TransformerBlock×4 → LayerNorm → mean_pool → project
```

Per-modality patch embeddings:
| Modality | Input channels | Description |
|----------|---------------|-------------|
| `sentinel2_l2a` | 12 (13→12, drops B8A) | Sentinel-2 Level-2A |
| `sentinel1` | 2 | Sentinel-1 SAR |
| `landsat` | 11 | Landsat |
| `worldcover` | 1 | ESA WorldCover |
| `srtm` | 1 | SRTM elevation |
| `openstreetmap_raster` | 30 | OSM raster |
| `wri_canopy_height_map` | 1 | WRI canopy height |
| `cdl` | 1 | USDA CDL |
| `worldcereal` | 8 | WorldCereal |

### Band handling

EuroCropML data has 13 bands (B01-B12 + B8A). The model's `sentinel2_l2a` pixel_proj expects 12 channels. The encoder auto-drops B8A (index 8) when input has 13 bands.

## Source Modules

### `src/data/loader.py`

Data loading with parallel I/O for large datasets.

```python
from src.data.loader import load_split_padded, load_split_padded_cached

# Load all splits (parallel, ~140K files)
splits = load_split_padded(
    preprocess_dir="Data/preprocess/preprocess",
    split_dir="Data/split/split",
    use_case="overlap_latvia_vs_estonia",
    split_name="all",
    n_workers=16
)
# splits = {"train": (X, y, labels), "val": (...), "test": (...)}

# Load with caching (first run slow, subsequent runs fast)
splits = load_split_padded_cached(...)

# Cloud mode
from src.data.loader import load_dataset, train_test_split_stratified
X, y, label_names = load_dataset("./data", "Estonia", top_n_classes=15)
X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
```

Functions:
| Function | Description |
|----------|-------------|
| `load_split()` | Load split JSONs + .npz files in parallel, returns variable-length lists |
| `load_split_padded()` | Same but pads to fixed `(N, T, C)` array |
| `load_split_padded_cached()` | Caches padded result to disk, skips reload on subsequent runs |
| `load_dataset()` | Cloud mode — loads from eurocropsml download |
| `filter_top_classes()` | Keep N most frequent classes |
| `train_test_split_stratified()` | Stratified train/test split |

### `src/data/features.py`

Classical feature extraction from Sentinel-2 time series.

```python
from src.data.features import ndvi_features, band_stat_features, temporal_features

ndvi = ndvi_features(X)           # (N, 4) — mean, max, min, std of NDVI
band_stats = band_stat_features(X) # (N, 39) — mean, std, max per band (13×3)
temporal = temporal_features(X)    # (N, 2) — PC1 slope, PC1 variance
```

Band indices: B04=Red (index 3), B08=NIR (index 7).

NDVI formula: `(NIR - Red) / (NIR + Red + 1e-8)`

### `src/encoder/olmoearth.py`

OLMoEarth encoder — no rslearn dependency.

```python
from src.encoder.olmoearth import OLMoEarthEncoder

# Local mode
encoder = OLMoEarthEncoder(
    mode="local",
    local_weights_path="Models/V1.1_Nano/weights.pth",
    device="cuda"
)

# Cloud mode
encoder = OLMoEarthEncoder(
    mode="cloud",
    cloud_model_id="allenai/OlmoEarth-v1_1-Nano",
    device="cuda"
)

embeddings = encoder.encode(X, batch_size=32)  # (N, 128)
```

Internal classes (not used directly):
| Class | Description |
|-------|-------------|
| `OlmoEarthEncoder(nn.Module)` | Minimal encoder matching state_dict keys |
| `PatchEmbed` | Per-modality Linear projection (pixel_proj → 768 → 128) |
| `TransformerBlock` | Pre-norm transformer block (LayerNorm → MHSA → MLP) |

`from_state_dict(path)` loads encoder-only weights, strips `encoder.` prefix.

### `src/models/classical.py`

```python
from src.models.classical import get_classifier

clf = get_classifier("rf")    # RandomForestClassifier(n_estimators=200)
clf = get_classifier("lgbm")  # LGBMClassifier(n_estimators=300)
clf = get_classifier("xgb")   # XGBClassifier(n_estimators=300)
clf = get_classifier("logreg") # LogisticRegression(max_iter=1000)
```

### `src/evaluate/metrics.py`

```python
from src.evaluate.metrics import compute_metrics, save_metrics, load_metrics

metrics = compute_metrics(y_true, y_pred)
# {"overall_accuracy": 0.85, "macro_f1": 0.78, "weighted_f1": 0.84, "kappa": 0.81}

save_metrics(metrics, "results/metrics/phase1_rf.json")
loaded = load_metrics("results/metrics/phase1_rf.json")
```

### `src/viz/`

```python
from src.viz.confusion import plot_confusion_matrix
from src.viz.fewshot_curve import plot_fewshot_curve
from src.viz.umap_viz import plot_umap

plot_confusion_matrix(y_true, y_pred, save_path="results/figures/cm.png")
plot_fewshot_curve(results_dict, save_path="results/figures/fewshot.png")
plot_umap(embeddings, labels, save_path="results/figures/umap.png")
```

### `src/data/inspect.py`

Dataset inspection utilities (Phase 1).

```python
from src.data.inspect import (
    inspect_directory_structure, count_samples, count_classes,
    generate_class_distribution, get_top_classes, create_reduced_subset,
    print_inspection_report
)

# Print full inspection report
print_inspection_report("Data/preprocess/preprocess", "Data/split/split", "overlap_latvia_vs_estonia")

# Get top 20 classes
top_classes = get_top_classes("Data/preprocess/preprocess", n=20)

# Create reduced subset
create_reduced_subset("Data/preprocess/preprocess", "Data/preprocess_subset", top_n=20)
```

### `src/data/memory_profiler.py`

Memory profiling utilities (Phase 2).

```python
from src.data.memory_profiler import (
    get_memory_usage, measure_single_npz, measure_batch_loading,
    identify_memory_bottlenecks, print_memory_report
)

# Print full memory report
print_memory_report("Data/preprocess/preprocess", n_samples=100)

# Measure single file
result = measure_single_npz("path/to/file.npz")
print(f"Memory delta: {result['memory_delta_mb']} MB")
```

### `src/data/streaming.py`

Memory-efficient streaming data loader (Phase 3).

```python
from src.data.streaming import (
    StreamingDataset, StreamingSplitDataset,
    create_batch_generator, create_split_batch_generator
)

# Stream from directory
dataset = StreamingDataset("Data/preprocess/preprocess", batch_size=32)
for X_batch, y_batch in dataset:
    # Process batch - never loads all files at once
    pass

# Stream from split
dataset = StreamingSplitDataset(
    "Data/preprocess/preprocess", "Data/split/split",
    "overlap_latvia_vs_estonia", batch_size=32, split_key="train"
)
for X_batch, y_batch in dataset:
    pass
```

### `src/data/features.py`

Expanded baseline feature extraction (Phase 5).

```python
from src.data.features import (
    ndvi_features, band_stat_features, spectral_statistics,
    combined_baseline_features, mean_ndvi, std_ndvi,
    mean_red, mean_nir, mean_green, mean_blue, ndvi_percentiles
)

# Combined features (recommended)
features = combined_baseline_features(X)  # (N, 14)

# Individual features
ndvi = ndvi_features(X)           # (N, 4)
band_stats = band_stat_features(X) # (N, 39)
spectral = spectral_statistics(X)  # (N, 52)
```

### `src/data/scaling.py`

Scaling evaluation utilities (Phase 7).

```python
from src.data.scaling import evaluate_scalability, print_scaling_report

# Run full scalability evaluation
results = evaluate_scalability("Data/preprocess/preprocess")
print_scaling_report(results)
```

### `experiments/run_classification.py`

Classification experiments runner (Phase 6).

```bash
# Run all experiments (A, B, C, D)
python experiments/run_classification.py --config config.yaml

# Run specific experiment
python experiments/run_classification.py --config config.yaml --experiment A
python experiments/run_classification.py --config config.yaml --experiment B
```

Experiments:
- **A**: Handcrafted features (NDVI, band stats, combined) + Logistic Regression/LightGBM
- **B**: OLMoEarth embeddings + Logistic Regression
- **C**: OLMoEarth embeddings + LightGBM
- **D**: Concatenated embeddings + features + LightGBM

## Experiments

### Current (v2.0)

| Script | Description |
|--------|-------------|
| `experiments/run_classification.py` | Classification experiments (A/B/C/D matrix) |

### Archive (v1.0)

Previous experiments archived in `archive/experiments/`:

| Script | Description |
|--------|-------------|
| `phase1_baselines.py` | RF/LightGBM/XGBoost on NDVI and band statistics features |
| `phase2_embeddings.py` | OLMoEarth embeddings + classifiers |
| `phase3_fewshot.py` | Few-shot comparison using pre-defined splits |
| `phase4_linear_probe.py` | Linear probe on pre-computed embeddings |

## Config

```yaml
data:
  mode: "local"                          # "local" | "cloud"
  local_preprocess_dir: "E:/.../preprocess"
  local_split_dir: "E:/.../split"
  use_case: "overlap_latvia_vs_estonia"  # split use case
  cloud_data_dir: "./data"               # cloud mode only
  cloud_country: "Estonia"               # cloud mode only
  top_n_classes: 15                      # cloud mode only
  random_seed: 42

model:
  mode: "local"                          # "local" | "cloud"
  local_weights_path: "E:/.../weights.pth"
  cloud_model_id: "allenai/OlmoEarth-v1_1-Nano"
  batch_size: 32
  device: "cuda"                         # "cuda" | "cpu"

fewshot:
  shots: [5, 10, 20, 100, 200, 500]
  n_repeats: 5
```

## Tests

```bash
# Using geospatial conda env
E:\Programs\conda\envs\geospatial\python.exe -m pytest tests/ -v
```

71 tests covering: loader, features, encoder (mocked), classifiers, metrics, visualization, integration.

## Google Colab

Upload `benchmark_colab.ipynb` to Google Colab and run all cells.

The notebook will:
1. Clone this repo and install dependencies
2. Download EuroCropsML data from HuggingFace (`mahdi555/eurocrop-data`)
3. Download OLMoEarth models from HuggingFace (`allenai/OlmoEarth-v1_1-{Nano,Tiny,Base,Large}`)
4. Extract embeddings and run classification experiments
5. Generate comparison table and visualizations

**Models compared:**

| Model | HF Repo | Parameters |
|-------|---------|------------|
| Nano | `allenai/OlmoEarth-v1_1-Nano` | ~12M |
| Tiny | `allenai/OlmoEarth-v1_1-Tiny` | ~30M |
| Base | `allenai/OlmoEarth-v1_1-Base` | ~100M |
| Large | `allenai/OlmoEarth-v1_1-Large` | ~300M |

**Experiments per model:**
- Embeddings + LogisticRegression
- Embeddings + LightGBM
- Hybrid (features + embeddings) + LightGBM

Plus baseline (features + LightGBM) for comparison.

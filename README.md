# EuroCropML × OLMoEarth Benchmark

Benchmark comparing classical ML with OLMoEarth foundation model embeddings for crop type classification.

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

## Experiments

| Script | Description |
|--------|-------------|
| `phase1_baselines.py` | RF/LightGBM/XGBoost on NDVI and band statistics features |
| `phase2_embeddings.py` | OLMoEarth embeddings + classifiers (caches embeddings to .npy) |
| `phase3_fewshot.py` | Few-shot comparison using pre-defined splits (5-500 shots) |
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

```python
# 1. Clone repo
!git clone https://github.com/YOUR_USERNAME/eurocrop-olmoearth-benchmark.git
%cd eurocrop-olmoearth-benchmark

# 2. Install deps
!pip install -r requirements.txt huggingface_hub -q

# 3. Download data
!eurocropsml download --country Estonia --output-dir ./data

# 4. Run experiments (weights auto-download from HuggingFace)
!python experiments/phase1_baselines.py
!python experiments/phase2_embeddings.py
!python experiments/phase3_fewshot.py
```

Or use the pre-built notebook: `notebooks/99_colab_runner.ipynb`

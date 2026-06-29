# Multi-Notebook Restructure for 12GB Colab

> **For agentic workers:** REQUIRED SUB-SKILL: Use compose:subagent (recommended) or compose:execute to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the benchmark into separate Colab notebooks with Google Drive persistence, so each step fits within 12GB RAM.

**Architecture:** Follow the UT project pattern: separate data preparation from training. Notebook 1 downloads data. Notebook 2 extracts features in chunks and saves to Google Drive as .npy files. Notebooks 3-5 load only the compact .npy features they need. Each notebook restarts with a fresh kernel, releasing all prior memory.

**Tech Stack:** Python, NumPy, scikit-learn, LightGBM, XGBoost, PyTorch, Google Colab, Google Drive via `google.colab.drive`

## Global Constraints

- Target platform: Google Colab free tier (12GB RAM, T4 GPU optional)
- Data source: Zenodo record 15095445 (EuroCropML preprocess + split)
- 105,543 train / 1,000 val / 35,182 test samples
- Each sample: variable-length Sentinel-2 time series, shape (T, 13)
- All intermediate results persist to Google Drive under `MyDrive/eurocrop_benchmark/`
- No single notebook may hold all raw time series in memory simultaneously

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `notebooks/01_setup_download.ipynb` | Clone repo, install deps, download Zenodo data |
| Create | `notebooks/02_data_prep.ipynb` | Chunked feature extraction, save .npy to Drive |
| Create | `notebooks/03_phase1_baselines.ipynb` | Load features from Drive, train RF/LGBM/XGB |
| Create | `notebooks/04_phase2_embeddings.ipynb` | Load data, encode with OLMoEarth, train classifiers |
| Create | `notebooks/05_phase3_fewshot.ipynb` | Few-shot comparison |
| Create | `notebooks/06_visualization.ipynb` | Plots and summary |
| Create | `src/data/prep_features.py` | Chunked feature extraction utility |
| Keep | `notebooks/99_colab_runner.ipynb` | Original single-notebook version (for reference) |

---

## Task 1: Create data prep utility

**Covers:** Chunked feature extraction that never loads all samples at once

**Files:**
- Create: `src/data/prep_features.py`

**Interfaces:**
- Consumes: preprocess directory path, split JSON path, feature function
- Produces: Saves `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy` to output directory

- [ ] **Step 1: Create `src/data/prep_features.py`**

```python
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
        feature_fn: callable, takes (T, C) array → (F,) feature vector
        output_dir: directory to save .npy files
        chunk_size: files to process per batch
        use_zenodo: if True, use Zenodo flat directory structure
    """
    os.makedirs(output_dir, exist_ok=True)

    split_file = os.path.join(split_dir, use_case, "finetune", "region_split_all.json")
    with open(split_file) as f:
        split_data = json.load(f)

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

        del feat_list, y_list, X_feat, y_mapped
        gc.collect()

    label_names_path = os.path.join(output_dir, "label_names.json")
    with open(label_names_path, "w") as f:
        json.dump(unique_labels, f)
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
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.data.prep_features import extract_features_chunked, load_prepared_features; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/data/prep_features.py
git commit -m "feat: add chunked feature extraction utility"
```

---

## Task 2: Create Notebook 01 — Setup + Download

**Covers:** Clone repo, install dependencies, download Zenodo data

**Files:**
- Create: `notebooks/01_setup_download.ipynb`

- [ ] **Step 1: Create notebook with these cells**

Cell 1 (markdown):
```
# EuroCropML Benchmark - Step 1: Setup & Download
Clone repo, install dependencies, download data from Zenodo.
```

Cell 2 (code):
```python
# Clone repo and install dependencies
!git clone https://github.com/mahdikalantari555/eurocrop-olmoearth-benchmark.git
%cd eurocrop-olmoearth-benchmark
!pip install -r requirements.txt huggingface_hub -q
!apt-get install -y aria2 -q
```

Cell 3 (code):
```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/eurocrop_benchmark
```

Cell 4 (code):
```python
# Download EuroCropML data from Zenodo
import requests, os

record_id = 15095445
record = requests.get(f"https://zenodo.org/api/records/{record_id}").json()

for file in record["files"]:
    if file["key"] not in ["preprocess.zip", "split.zip"]:
        continue
    filename = file["key"]
    filesize = file["size"]
    url = file["links"]["self"]

    if os.path.exists(filename) and os.path.getsize(filename) == filesize:
        print(f"Already downloaded: {filename}")
    else:
        print(f"Downloading: {filename} ({filesize / 1e6:.1f} MB)")
        !aria2c -x 16 -s 16 -o "{filename}" "{url}"

    if os.path.exists(filename):
        print(f"Extracting: {filename}")
        !unzip -q -o "{filename}"
        os.remove(filename)

print("Done!")
!ls -d preprocess/ split/
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/01_setup_download.ipynb
git commit -m "feat: add setup and download notebook"
```

---

## Task 3: Create Notebook 02 — Data Preparation

**Covers:** Chunked feature extraction, save to Google Drive

**Files:**
- Create: `notebooks/02_data_prep.ipynb`

- [ ] **Step 1: Create notebook with these cells**

Cell 1 (markdown):
```
# EuroCropML Benchmark - Step 2: Data Preparation
Extract classical features in chunks and save to Google Drive.
This notebook processes 140K samples without exceeding 12GB RAM.
```

Cell 2 (code):
```python
# Setup
%cd eurocrop-olmoearth-benchmark
import numpy as np
import os
```

Cell 3 (code):
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/eurocrop_benchmark'
```

Cell 4 (code):
```python
# NDVI features: (T, C) → (4,)
from src.data.features import ndvi_features

def ndvi_single(x):
    B4, B8 = 3, 7
    red = x[:, B4].astype(np.float32)
    nir = x[:, B8].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.array([ndvi.mean(), ndvi.max(), ndvi.min(), ndvi.std()], dtype=np.float32)

# Band stat features: (T, C) → (39,)
def band_stat_single(x):
    x32 = x.astype(np.float32)
    return np.concatenate([x32.mean(0), x32.std(0), x32.max(0)]).astype(np.float32)
```

Cell 5 (code):
```python
# Extract NDVI features (chunked, ~2 min)
from src.data.prep_features import extract_features_chunked

ndvi_dir = os.path.join(DRIVE_DIR, 'features_ndvi')
extract_features_chunked(
    preprocess_dir='./preprocess',
    split_dir='./split',
    use_case='latvia_vs_estonia',
    feature_fn=ndvi_single,
    output_dir=ndvi_dir,
    chunk_size=5000
)
```

Cell 6 (code):
```python
# Extract band stat features (chunked, ~2 min)
bandstat_dir = os.path.join(DRIVE_DIR, 'features_bandstat')
extract_features_chunked(
    preprocess_dir='./preprocess',
    split_dir='./split',
    use_case='latvia_vs_estonia',
    feature_fn=band_stat_single,
    output_dir=bandstat_dir,
    chunk_size=5000
)
```

Cell 7 (code):
```python
# Verify saved features
from src.data.prep_features import load_prepared_features

for name, d in [("NDVI", ndvi_dir), ("BandStat", bandstat_dir)]:
    X_tr, y_tr, X_te, y_te, labels = load_prepared_features(d)
    print(f"{name}: train={X_tr.shape}, test={X_te.shape}, classes={len(labels)}")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/02_data_prep.ipynb
git commit -m "feat: add data preparation notebook"
```

---

## Task 4: Create Notebook 03 — Phase 1 Baselines

**Covers:** Load features from Drive, train RF/LGBM/XGB classifiers

**Files:**
- Create: `notebooks/03_phase1_baselines.ipynb`

- [ ] **Step 1: Create notebook with these cells**

Cell 1 (markdown):
```
# EuroCropML Benchmark - Step 3: Classical Baselines
Load pre-extracted features from Drive and train classifiers.
```

Cell 2 (code):
```python
%cd eurocrop-olmoearth-benchmark
import json, os, time
import numpy as np
from src.data.prep_features import load_prepared_features
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics, save_confusion_matrix
```

Cell 3 (code):
```python
# Mount Drive and load features
from google.colab import drive
drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/eurocrop_benchmark'
```

Cell 4 (code):
```python
# NDVI + Random Forest
X_tr, y_tr, X_te, y_te, labels = load_prepared_features(os.path.join(DRIVE_DIR, 'features_ndvi'))
print(f"NDVI features: train={X_tr.shape}, test={X_te.shape}")

t = time.time()
clf = get_classifier("rf", 42)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)

m = compute_metrics(y_te, y_pred, labels=labels)
save_metrics(m, 'results/metrics/phase1_ndvi_rf.json')
save_confusion_matrix(y_te, y_pred, 'results/metrics/phase1_ndvi_rf_cm.csv', labels=labels)
print(f"ndvi_rf: OA={m['overall_accuracy']:.3f} F1={m['macro_f1']:.3f} ({time.time()-t:.1f}s)")

del X_tr, y_tr, clf
import gc; gc.collect()
```

Cell 5 (code):
```python
# BandStat + RF
X_tr, y_tr, X_te, y_te, labels = load_prepared_features(os.path.join(DRIVE_DIR, 'features_bandstat'))
print(f"BandStat features: train={X_tr.shape}, test={X_te.shape}")

for clf_name in ["rf", "lgbm", "xgb"]:
    t = time.time()
    clf = get_classifier(clf_name, 42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    m = compute_metrics(y_te, y_pred, labels=labels)
    save_metrics(m, f'results/metrics/phase1_bandstat_{clf_name}.json')
    save_confusion_matrix(y_te, y_pred, f'results/metrics/phase1_bandstat_{clf_name}_cm.csv', labels=labels)
    print(f"bandstat_{clf_name}: OA={m['overall_accuracy']:.3f} F1={m['macro_f1']:.3f} ({time.time()-t:.1f}s)")

    del clf
    gc.collect()
```

Cell 6 (code):
```python
# Copy results to Drive
!cp -r results/metrics/*.json /content/drive/MyDrive/eurocrop_benchmark/
print("Results saved to Drive")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/03_phase1_baselines.ipynb
git commit -m "feat: add phase 1 baselines notebook"
```

---

## Task 5: Create Notebook 04 — Phase 2 Embeddings

**Covers:** OLMoEarth encoding, classifier training

**Files:**
- Create: `notebooks/04_phase2_embeddings.ipynb`

- [ ] **Step 1: Create notebook with these cells**

Cell 1 (markdown):
```
# EuroCropML Benchmark - Step 4: OLMoEarth Embeddings
Load raw data, encode with OLMoEarth, train classifiers.
Uses GPU for encoding. Saves embeddings to Drive for reuse.
```

Cell 2 (code):
```python
%cd eurocrop-olmoearth-benchmark
import gc, json, os, time
import numpy as np
import torch
from google.colab import drive
drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/eurocrop_benchmark'

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Cell 3 (code):
```python
# Load raw data (this loads all time series - ~8GB)
from src.utils.runner import load_config, setup_logging, log
from src.data.loader import load_split_padded

cfg = {
    'data': {
        'mode': 'local',
        'local_preprocess_dir': './preprocess',
        'local_split_dir': './split',
        'use_case': 'latvia_vs_estonia',
        'use_zenodo': True
    }
}

log_file = setup_logging("phase2", "./results/metrics")
splits = load_split_padded(
    cfg['data']['local_preprocess_dir'],
    cfg['data']['local_split_dir'],
    cfg['data']['use_case'],
    split_name="all",
    use_zenodo=True
)
X_train, y_train, _ = splits["train"]
X_test, y_test, _ = splits["test"]
del splits; gc.collect()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

Cell 4 (code):
```python
# Initialize OLMoEarth encoder (downloads weights from HuggingFace)
from src.encoder.olmoearth import OLMoEarthEncoder

encoder = OLMoEarthEncoder(
    mode="cloud",
    cloud_model_id="allenai/OlmoEarth-v1_1-Nano",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

Cell 5 (code):
```python
# Encode training set
t = time.time()
emb_train = encoder.encode(X_train, batch_size=32)
print(f"Train embeddings: {emb_train.shape} ({time.time()-t:.1f}s)")

# Encode test set
t = time.time()
emb_test = encoder.encode(X_test, batch_size=32)
print(f"Test embeddings: {emb_test.shape} ({time.time()-t:.1f}s)")

# Free raw data
del X_train, X_test
gc.collect()

# Save embeddings to Drive
np.save(os.path.join(DRIVE_DIR, 'emb_train.npy'), emb_train)
np.save(os.path.join(DRIVE_DIR, 'emb_test.npy'), emb_test)
np.save(os.path.join(DRIVE_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(DRIVE_DIR, 'y_test.npy'), y_test)
print("Embeddings saved to Drive")
```

Cell 6 (code):
```python
# Train classifiers on embeddings
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics, save_confusion_matrix

for clf_name in ["logreg", "rf", "lgbm", "xgb"]:
    t = time.time()
    clf = get_classifier(clf_name)
    clf.fit(emb_train, y_train)
    y_pred = clf.predict(emb_test)

    labels = sorted(set(y_test))
    m = compute_metrics(y_test, y_pred, labels=labels)
    save_metrics(m, f'results/metrics/phase2_olmo_{clf_name}.json')
    save_confusion_matrix(y_test, y_pred, f'results/metrics/phase2_olmo_{clf_name}_cm.csv', labels=labels)
    print(f"olmo_{clf_name}: OA={m['overall_accuracy']:.3f} F1={m['macro_f1']:.3f} ({time.time()-t:.1f}s)")

    del clf; gc.collect()
```

Cell 7 (code):
```python
!cp results/metrics/phase2_*.json /content/drive/MyDrive/eurocrop_benchmark/
print("Results saved to Drive")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/04_phase2_embeddings.ipynb
git commit -m "feat: add phase 2 embeddings notebook"
```

---

## Task 6: Create Notebook 05 — Phase 3 Few-Shot

**Covers:** Few-shot comparison using pre-defined splits

**Files:**
- Create: `notebooks/05_phase3_fewshot.ipynb`

- [ ] **Step 1: Create notebook with these cells**

Cell 1 (markdown):
```
# EuroCropML Benchmark - Step 5: Few-Shot Comparison
Compare RF/LGBM/OLMoEarth at different shot counts.
Loads embeddings from Drive (saved in Step 4).
```

Cell 2 (code):
```python
%cd eurocrop-olmoearth-benchmark
import gc, json, os, time
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/eurocrop_benchmark'
```

Cell 3 (code):
```python
# Load embeddings and labels from Drive
emb_train = np.load(os.path.join(DRIVE_DIR, 'emb_train.npy'))
emb_test = np.load(os.path.join(DRIVE_DIR, 'emb_test.npy'))
y_train = np.load(os.path.join(DRIVE_DIR, 'y_train.npy'))
y_test = np.load(os.path.join(DRIVE_DIR, 'y_test.npy'))
print(f"emb_train: {emb_train.shape}, emb_test: {emb_test.shape}")
```

Cell 4 (code):
```python
# Few-shot experiment
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics

shots = [5, 10, 20, 100, 200, 500]
repeats = 5
results = {}

for n in shots:
    print(f"\n--- {n}-shot ---")
    scores_rf, scores_lgbm, scores_olmo = [], [], []

    for r in range(repeats):
        rng = np.random.RandomState(r)
        indices = []
        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            sampled = rng.choice(cls_idx, min(n, len(cls_idx)),
                                 replace=len(cls_idx) < n)
            indices.extend(sampled)
        idx = np.array(indices)

        emb_fs = emb_train[idx]
        y_fs = y_train[idx]

        clf_rf = get_classifier("rf", seed=r)
        clf_rf.fit(emb_fs, y_fs)
        scores_rf.append(compute_metrics(y_test, clf_rf.predict(emb_test))["macro_f1"])

        clf_lgbm = get_classifier("lgbm", seed=r)
        clf_lgbm.fit(emb_fs, y_fs)
        scores_lgbm.append(compute_metrics(y_test, clf_lgbm.predict(emb_test))["macro_f1"])

        del clf_rf, clf_lgbm, emb_fs, y_fs
        gc.collect()

    results[str(n)] = {
        "rf_f1": float(np.mean(scores_rf)),
        "rf_f1_std": float(np.std(scores_rf)),
        "lgbm_f1": float(np.mean(scores_lgbm)),
        "lgbm_f1_std": float(np.std(scores_lgbm)),
        "olmo_lgbm_f1": float(np.mean(scores_olmo)) if scores_olmo else 0.0,
        "olmo_lgbm_f1_std": float(np.std(scores_olmo)) if scores_olmo else 0.0,
    }
    print(f"  RF={results[str(n)]['rf_f1']:.3f} | LGBM={results[str(n)]['lgbm_f1']:.3f}")

# Save results
os.makedirs('results/metrics', exist_ok=True)
with open('results/metrics/phase3_fewshot.json', 'w') as f:
    json.dump(results, f, indent=2)
!cp results/metrics/phase3_fewshot.json /content/drive/MyDrive/eurocrop_benchmark/
print("\nDone!")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/05_phase3_fewshot.ipynb
git commit -m "feat: add phase 3 few-shot notebook"
```

---

## Task 7: Create Notebook 06 — Visualization

**Covers:** Plots and results summary

**Files:**
- Create: `notebooks/06_visualization.ipynb`

- [ ] **Step 1: Create notebook with these cells**

Cell 1 (markdown):
```
# EuroCropML Benchmark - Step 6: Visualization
Generate plots and print results summary.
```

Cell 2 (code):
```python
%cd eurocrop-olmoearth-benchmark
import json, glob, os
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/eurocrop_benchmark'

# Copy results from Drive
!cp /content/drive/MyDrive/eurocrop_benchmark/*.json results/metrics/ 2>/dev/null || true
```

Cell 3 (code):
```python
# Few-shot curve
if os.path.exists('results/metrics/phase3_fewshot.json'):
    from src.viz.fewshot_curve import plot_fewshot_curve
    with open('results/metrics/phase3_fewshot.json') as f:
        results = json.load(f)
    plot_fewshot_curve(results, 'results/figures/fewshot_curve.png')
    plt.show()
```

Cell 4 (code):
```python
# Results summary
print("=" * 70)
print("BENCHMARK RESULTS SUMMARY")
print("=" * 70)

for phase_name in ["phase1", "phase2", "phase3"]:
    files = sorted(glob.glob(f'results/metrics/{phase_name}*.json'))
    if not files:
        continue
    print(f"\n{'─' * 50}")
    print(f"  {phase_name.upper()}")
    print(f"{'─' * 50}")
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        name = f.split('/')[-1].replace('.json', '')
        if isinstance(data, dict) and 'overall_accuracy' in data:
            print(f"  {name}: OA={data['overall_accuracy']:.3f} F1={data['macro_f1']:.3f}")
        elif isinstance(data, dict) and 'rf_f1' in data:
            for shot, m in sorted(data.items(), key=lambda x: int(x[0])):
                print(f"    {shot}-shot: RF={m['rf_f1']:.3f} LGBM={m['lgbm_f1']:.3f}")
```

Cell 5 (code):
```python
# Download all results
from google.colab import files
for f in sorted(glob.glob('results/metrics/*.json') + glob.glob('results/figures/*.png')):
    print(f"  {f}")
    files.download(f)
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/06_visualization.ipynb
git commit -m "feat: add visualization notebook"
```

---

## Task 8: Update Colab runner notebook

**Covers:** Update existing 99_colab_runner.ipynb to reference new notebooks

**Files:**
- Modify: `notebooks/99_colab_runner.ipynb`

- [ ] **Step 1: Add a markdown cell at the top pointing to new notebooks**

Add cell at position 0:
```
# ⚠️ For 12GB Colab: Use the individual notebooks (01-06) instead
# This single-file runner may exceed memory on free Colab.
# See: 01_setup_download.ipynb → 02_data_prep.ipynb → 03-05 experiment notebooks
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/99_colab_runner.ipynb
git commit -m "docs: add pointer to new multi-notebook structure"
```

---

## Verification

After implementing all tasks, verify by:

1. Open `01_setup_download.ipynb` in Colab — run all cells, confirm data downloads
2. Open `02_data_prep.ipynb` — run all cells, confirm .npy files appear in Drive
3. Open `03_phase1_baselines.ipynb` — run all cells, confirm phase1 results appear
4. Check `!free -h` in any notebook to confirm RAM stays under 10GB
5. Verify all result JSONs in Drive match expected metrics

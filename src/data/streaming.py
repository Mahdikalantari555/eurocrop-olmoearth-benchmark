"""
Streaming data loader for EuroCropsML.

Provides:
- Generator-based loading that never loads all files at once
- Iterable dataset for PyTorch
- Batch processing with constant RAM usage
- Memory-efficient preprocessing
"""

import os
import json
import numpy as np
from typing import Generator, Tuple, List, Dict, Optional, Callable
from pathlib import Path
import gc


def stream_npz_files(data_dir: str, 
                     class_filter: set = None) -> Generator[Tuple[str, np.ndarray, str], None, None]:
    """
    Generator that yields NPZ files one at a time.
    
    Args:
        data_dir: Path to directory with NPZ files
        class_filter: Optional set of class labels to include
        
    Yields:
        Tuple of (filename, data_array, class_label)
    """
    for f in os.listdir(data_dir):
        if not f.endswith('.npz'):
            continue
            
        class_label = f.split("_")[-1].replace(".npz", "")
        
        if class_filter and class_label not in class_filter:
            continue
        
        filepath = os.path.join(data_dir, f)
        try:
            data = np.load(filepath, allow_pickle=True)
            yield f, data['data'], class_label
            del data
            gc.collect()
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue


def stream_from_split(data_dir: str, split_dir: str, use_case: str,
                      split_name: str = "all",
                      split_key: str = "train",
                      class_filter: set = None) -> Generator[Tuple[str, np.ndarray, str], None, None]:
    """
    Generator that yields samples from a specific split.
    
    Args:
        data_dir: Path to preprocess directory
        split_dir: Path to split directory
        use_case: Use case name
        split_name: Split name (all, 5, 10, etc.)
        split_key: Split key (train, val, test)
        class_filter: Optional set of class labels to include
        
    Yields:
        Tuple of (filename, data_array, class_label)
    """
    if split_name == "all":
        split_file = os.path.join(split_dir, use_case, "finetune",
                                  "region_split_all.json")
    else:
        split_file = os.path.join(split_dir, use_case, "finetune",
                                  f"region_split_{split_name}.json")
    
    with open(split_file) as f:
        split_data = json.load(f)
    
    filenames = split_data[split_key]
    
    for fn in filenames:
        class_label = fn.split("_")[-1].replace(".npz", "")
        
        if class_filter and class_label not in class_filter:
            continue
        
        filepath = os.path.join(data_dir, fn)
        if not os.path.exists(filepath):
            continue
        
        try:
            data = np.load(filepath, allow_pickle=True)
            yield fn, data['data'], class_label
            del data
            gc.collect()
        except Exception as e:
            print(f"Warning: Failed to load {fn}: {e}")
            continue


def create_batch_generator(data_dir: str, batch_size: int = 32,
                           class_filter: set = None,
                           max_samples: int = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields batches of data.
    
    Args:
        data_dir: Path to directory with NPZ files
        batch_size: Number of samples per batch
        class_filter: Optional set of class labels to include
        max_samples: Maximum number of samples to yield
        
    Yields:
        Tuple of (X_batch, y_batch) where X_batch is (B, T, C) and y_batch is (B,)
    """
    batch_X = []
    batch_y = []
    sample_count = 0
    
    for _, data, class_label in stream_npz_files(data_dir, class_filter):
        if max_samples and sample_count >= max_samples:
            break
        
        batch_X.append(data)
        batch_y.append(class_label)
        sample_count += 1
        
        if len(batch_X) >= batch_size:
            yield _collate_batch(batch_X, batch_y)
            batch_X = []
            batch_y = []
            gc.collect()
    
    if batch_X:
        yield _collate_batch(batch_X, batch_y)


def create_split_batch_generator(data_dir: str, split_dir: str,
                                 use_case: str, batch_size: int = 32,
                                 split_name: str = "all",
                                 split_key: str = "train",
                                 class_filter: set = None,
                                 max_samples: int = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields batches from a specific split.
    
    Args:
        data_dir: Path to preprocess directory
        split_dir: Path to split directory
        use_case: Use case name
        batch_size: Number of samples per batch
        split_name: Split name
        split_key: Split key (train, val, test)
        class_filter: Optional set of class labels to include
        max_samples: Maximum number of samples to yield
        
    Yields:
        Tuple of (X_batch, y_batch)
    """
    batch_X = []
    batch_y = []
    sample_count = 0
    
    for _, data, class_label in stream_from_split(
        data_dir, split_dir, use_case, split_name, split_key, class_filter
    ):
        if max_samples and sample_count >= max_samples:
            break
        
        batch_X.append(data)
        batch_y.append(class_label)
        sample_count += 1
        
        if len(batch_X) >= batch_size:
            yield _collate_batch(batch_X, batch_y)
            batch_X = []
            batch_y = []
            gc.collect()
    
    if batch_X:
        yield _collate_batch(batch_X, batch_y)


def _collate_batch(X_list: List[np.ndarray], y_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collate variable-length sequences into padded batch.
    
    Args:
        X_list: List of arrays with shape (T_i, C)
        y_list: List of class labels
        
    Returns:
        Tuple of (X_padded, y_encoded) where X_padded is (B, T_max, C)
    """
    if not X_list:
        return np.array([]), np.array([])
    
    max_timesteps = max(x.shape[0] for x in X_list)
    C = X_list[0].shape[1]
    
    X_padded = np.zeros((len(X_list), max_timesteps, C), dtype=np.float32)
    for i, x in enumerate(X_list):
        T = min(x.shape[0], max_timesteps)
        X_padded[i, :T, :] = x[:T, :]
    
    unique_labels = sorted(set(y_list))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_encoded = np.array([label_map[yl] for yl in y_list], dtype=np.int64)
    
    return X_padded, y_encoded


class StreamingDataset:
    """
    Iterable dataset for streaming data from disk.
    
    Usage:
        dataset = StreamingDataset(data_dir, batch_size=32)
        for X_batch, y_batch in dataset:
            # Process batch
            pass
    """
    
    def __init__(self, data_dir: str, batch_size: int = 32,
                 class_filter: set = None, max_samples: int = None,
                 transform: Callable = None):
        """
        Initialize streaming dataset.
        
        Args:
            data_dir: Path to directory with NPZ files
            batch_size: Number of samples per batch
            class_filter: Optional set of class labels to include
            max_samples: Maximum number of samples
            transform: Optional transform function applied to each sample
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.class_filter = class_filter
        self.max_samples = max_samples
        self.transform = transform
        
        self._npz_files = [
            f for f in os.listdir(data_dir) 
            if f.endswith('.npz')
        ]
        
        if class_filter:
            self._npz_files = [
                f for f in self._npz_files
                if f.split("_")[-1].replace(".npz", "") in class_filter
            ]
        
        if max_samples:
            self._npz_files = self._npz_files[:max_samples]
        
        self._total_samples = len(self._npz_files)
    
    def __len__(self) -> int:
        """Return number of batches."""
        return (self._total_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Iterate over batches."""
        batch_X = []
        batch_y = []
        
        for f in self._npz_files:
            filepath = os.path.join(self.data_dir, f)
            try:
                data = np.load(filepath, allow_pickle=True)
                X = data['data']
                y = f.split("_")[-1].replace(".npz", "")
                
                if self.transform:
                    X = self.transform(X)
                
                batch_X.append(X)
                batch_y.append(y)
                
                if len(batch_X) >= self.batch_size:
                    yield _collate_batch(batch_X, batch_y)
                    batch_X = []
                    batch_y = []
                    gc.collect()
                    
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
                continue
        
        if batch_X:
            yield _collate_batch(batch_X, batch_y)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution without loading data."""
        from collections import Counter
        class_counter = Counter()
        for f in self._npz_files:
            class_label = f.split("_")[-1].replace(".npz", "")
            class_counter[class_label] += 1
        return dict(class_counter)


class StreamingSplitDataset:
    """
    Iterable dataset for streaming data from a specific split.
    """
    
    def __init__(self, data_dir: str, split_dir: str, use_case: str,
                 batch_size: int = 32, split_name: str = "all",
                 split_key: str = "train", class_filter: set = None,
                 max_samples: int = None, transform: Callable = None):
        """
        Initialize streaming split dataset.
        """
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.use_case = use_case
        self.batch_size = batch_size
        self.split_name = split_name
        self.split_key = split_key
        self.class_filter = class_filter
        self.max_samples = max_samples
        self.transform = transform
        
        if split_name == "all":
            split_file = os.path.join(split_dir, use_case, "finetune",
                                      "region_split_all.json")
        else:
            split_file = os.path.join(split_dir, use_case, "finetune",
                                      f"region_split_{split_name}.json")
        
        with open(split_file) as f:
            split_data = json.load(f)
        
        self._filenames = split_data[split_key]
        
        if class_filter:
            self._filenames = [
                fn for fn in self._filenames
                if fn.split("_")[-1].replace(".npz", "") in class_filter
            ]
        
        if max_samples:
            self._filenames = self._filenames[:max_samples]
        
        self._total_samples = len(self._filenames)
    
    def __len__(self) -> int:
        """Return number of batches."""
        return (self._total_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Iterate over batches."""
        batch_X = []
        batch_y = []
        
        for fn in self._filenames:
            filepath = os.path.join(self.data_dir, fn)
            if not os.path.exists(filepath):
                continue
            
            try:
                data = np.load(filepath, allow_pickle=True)
                X = data['data']
                y = fn.split("_")[-1].replace(".npz", "")
                
                if self.transform:
                    X = self.transform(X)
                
                batch_X.append(X)
                batch_y.append(y)
                
                if len(batch_X) >= self.batch_size:
                    yield _collate_batch(batch_X, batch_y)
                    batch_X = []
                    batch_y = []
                    gc.collect()
                    
            except Exception as e:
                print(f"Warning: Failed to load {fn}: {e}")
                continue
        
        if batch_X:
            yield _collate_batch(batch_X, batch_y)

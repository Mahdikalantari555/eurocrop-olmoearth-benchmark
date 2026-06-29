"""
Memory profiling utilities for EuroCropsML.

Provides:
- RAM usage measurement for single NPZ files
- RAM usage measurement for batch loading
- Full dataset memory profiling
- Memory bottleneck identification
"""

import os
import sys
import numpy as np
import psutil
from typing import Dict, List, Optional, Callable
from functools import wraps
import time


def get_memory_usage() -> Dict:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory statistics in MB
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
        "vms_mb": round(mem_info.vms / (1024 * 1024), 2),
        "available_mb": round(psutil.virtual_memory().available / (1024 * 1024), 2),
        "total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2),
        "percent_used": psutil.virtual_memory().percent
    }


def measure_single_npz(filepath: str) -> Dict:
    """
    Measure RAM usage when loading a single NPZ file.
    
    Args:
        filepath: Path to NPZ file
        
    Returns:
        Dictionary with memory measurements
    """
    mem_before = get_memory_usage()
    start_time = time.time()
    
    data = np.load(filepath, allow_pickle=True)
    
    load_time = time.time() - start_time
    mem_after = get_memory_usage()
    
    keys = list(data.keys())
    shapes = {k: data[k].shape for k in keys}
    dtypes = {k: str(data[k].dtype) for k in keys}
    sizes_mb = {k: round(data[k].nbytes / (1024 * 1024), 4) for k in keys}
    
    return {
        "filepath": os.path.basename(filepath),
        "load_time_ms": round(load_time * 1000, 2),
        "memory_before_mb": mem_before["rss_mb"],
        "memory_after_mb": mem_after["rss_mb"],
        "memory_delta_mb": round(mem_after["rss_mb"] - mem_before["rss_mb"], 2),
        "keys": keys,
        "shapes": shapes,
        "dtypes": dtypes,
        "sizes_mb": sizes_mb,
        "total_size_mb": round(sum(sizes_mb.values()), 4)
    }


def measure_batch_loading(data_dir: str, n_samples: int = 100,
                          sample_rate: float = None) -> Dict:
    """
    Measure RAM usage for loading N samples.
    
    Args:
        data_dir: Path to directory with NPZ files
        n_samples: Number of samples to load
        sample_rate: Optional sampling rate (0-1) for random sampling
        
    Returns:
        Dictionary with batch memory measurements
    """
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    if sample_rate and sample_rate < 1:
        import random
        n_sample = max(1, int(len(npz_files) * sample_rate))
        npz_files = random.sample(npz_files, min(n_sample, len(npz_files)))
    
    npz_files = npz_files[:n_samples]
    
    mem_before = get_memory_usage()
    start_time = time.time()
    
    loaded_data = []
    for f in npz_files:
        filepath = os.path.join(data_dir, f)
        data = np.load(filepath, allow_pickle=True)
        loaded_data.append(data['data'])
    
    load_time = time.time() - start_time
    mem_after = get_memory_usage()
    
    if loaded_data:
        total_elements = sum(d.size for d in loaded_data)
        total_bytes = sum(d.nbytes for d in loaded_data)
    else:
        total_elements = 0
        total_bytes = 0
    
    return {
        "n_samples_loaded": len(npz_files),
        "load_time_ms": round(load_time * 1000, 2),
        "memory_before_mb": mem_before["rss_mb"],
        "memory_after_mb": mem_after["rss_mb"],
        "memory_delta_mb": round(mem_after["rss_mb"] - mem_before["rss_mb"], 2),
        "total_elements": total_elements,
        "total_bytes_mb": round(total_bytes / (1024 * 1024), 2),
        "bytes_per_sample_kb": round(total_bytes / len(npz_files) / 1024, 2) if npz_files else 0
    }


def profile_full_dataset(data_dir: str, batch_size: int = 100,
                         max_samples: int = None) -> Dict:
    """
    Profile memory usage for loading the full dataset.
    
    Args:
        data_dir: Path to directory with NPZ files
        batch_size: Number of samples per batch
        max_samples: Maximum number of samples to profile
        
    Returns:
        Dictionary with full dataset profiling results
    """
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    if max_samples:
        npz_files = npz_files[:max_samples]
    
    total_files = len(npz_files)
    mem_initial = get_memory_usage()
    
    memory_history = []
    peak_memory = mem_initial["rss_mb"]
    
    for i in range(0, total_files, batch_size):
        batch_files = npz_files[i:i + batch_size]
        
        batch_data = []
        for f in batch_files:
            filepath = os.path.join(data_dir, f)
            data = np.load(filepath, allow_pickle=True)
            batch_data.append(data['data'])
        
        mem_current = get_memory_usage()
        memory_history.append({
            "batch_start": i,
            "batch_end": min(i + batch_size, total_files),
            "memory_mb": mem_current["rss_mb"]
        })
        
        if mem_current["rss_mb"] > peak_memory:
            peak_memory = mem_current["rss_mb"]
        
        del batch_data
    
    return {
        "total_files": total_files,
        "batch_size": batch_size,
        "initial_memory_mb": mem_initial["rss_mb"],
        "peak_memory_mb": peak_memory,
        "memory_growth_mb": round(peak_memory - mem_initial["rss_mb"], 2),
        "memory_history": memory_history,
        "is_constant": all(
            abs(h["memory_mb"] - memory_history[0]["memory_mb"]) < 10
            for h in memory_history
        )
    }


def identify_memory_bottlenecks(data_dir: str, 
                                 load_fn: Callable = None) -> Dict:
    """
    Identify memory bottlenecks in data loading.
    
    Args:
        data_dir: Path to directory with NPZ files
        load_fn: Optional custom loading function
        
    Returns:
        Dictionary with bottleneck analysis
    """
    mem_before = get_memory_usage()
    
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')][:10]
    
    object_sizes = {}
    
    for f in npz_files:
        filepath = os.path.join(data_dir, f)
        data = np.load(filepath, allow_pickle=True)
        
        for key in data.keys():
            obj = data[key]
            if key not in object_sizes:
                object_sizes[key] = []
            object_sizes[key].append({
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "size_mb": round(obj.nbytes / (1024 * 1024), 4)
            })
    
    avg_sizes = {}
    for key, sizes in object_sizes.items():
        avg_sizes[key] = {
            "avg_shape": sizes[0]["shape"],
            "avg_dtype": sizes[0]["dtype"],
            "avg_size_mb": round(
                sum(s["size_mb"] for s in sizes) / len(sizes), 4
            )
        }
    
    mem_after = get_memory_usage()
    
    return {
        "object_sizes": avg_sizes,
        "memory_delta_mb": round(mem_after["rss_mb"] - mem_before["rss_mb"], 2),
        "recommendation": _generate_memory_recommendation(avg_sizes)
    }


def _generate_memory_recommendation(object_sizes: Dict) -> str:
    """Generate memory optimization recommendation."""
    total_avg_mb = sum(s["avg_size_mb"] for s in object_sizes.values())
    
    if total_avg_mb > 10:
        return (
            f"Large data objects detected (avg {total_avg_mb:.2f} MB per file). "
            "Consider: 1) Streaming/generator-based loading, "
            "2) Processing in smaller batches, "
            "3) Using memory-mapped files."
        )
    elif total_avg_mb > 1:
        return (
            f"Moderate data objects (avg {total_avg_mb:.2f} MB per file). "
            "Streaming loading recommended for full dataset."
        )
    else:
        return (
            f"Small data objects (avg {total_avg_mb:.4f} MB per file). "
            "Standard loading should be fine for most use cases."
        )


def print_memory_report(data_dir: str, n_samples: int = 100):
    """
    Print a comprehensive memory profiling report.
    
    Args:
        data_dir: Path to directory with NPZ files
        n_samples: Number of samples to profile
    """
    print("=" * 60)
    print("Memory Profiling Report")
    print("=" * 60)
    
    print("\n1. Single NPZ File Profile:")
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    if npz_files:
        single_profile = measure_single_npz(os.path.join(data_dir, npz_files[0]))
        print(f"   File: {single_profile['filepath']}")
        print(f"   Load time: {single_profile['load_time_ms']} ms")
        print(f"   Memory delta: {single_profile['memory_delta_mb']} MB")
        print(f"   Keys: {single_profile['keys']}")
        print(f"   Shapes: {single_profile['shapes']}")
    
    print(f"\n2. Batch Loading Profile ({n_samples} samples):")
    batch_profile = measure_batch_loading(data_dir, n_samples)
    print(f"   Samples loaded: {batch_profile['n_samples_loaded']}")
    print(f"   Load time: {batch_profile['load_time_ms']} ms")
    print(f"   Memory delta: {batch_profile['memory_delta_mb']} MB")
    print(f"   Bytes per sample: {batch_profile['bytes_per_sample_kb']} KB")
    
    print("\n3. Bottleneck Analysis:")
    bottlenecks = identify_memory_bottlenecks(data_dir)
    for key, info in bottlenecks["object_sizes"].items():
        print(f"   {key}: {info['avg_shape']} ({info['avg_dtype']}) - {info['avg_size_mb']} MB")
    print(f"   Recommendation: {bottlenecks['recommendation']}")
    
    print("=" * 60)

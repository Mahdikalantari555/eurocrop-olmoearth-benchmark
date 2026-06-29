"""
Scaling utilities for EuroCropsML benchmark.

Provides:
- Sample count scaling
- Class count scaling
- Memory and runtime monitoring
- Scalability evaluation
"""

import os
import json
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import gc


def get_system_resources() -> Dict:
    """Get current system resource usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "ram_used_mb": round(mem_info.rss / (1024 * 1024), 2),
        "ram_available_mb": round(psutil.virtual_memory().available / (1024 * 1024), 2),
        "ram_total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2),
        "ram_percent": psutil.virtual_memory().percent,
        "cpu_percent": psutil.cpu_percent(interval=0.1)
    }


def scale_sample_count(data_dir: str, 
                       sample_counts: List[int],
                       process_fn: Callable = None,
                       **kwargs) -> Dict:
    """
    Evaluate scalability with increasing sample counts.
    
    Args:
        data_dir: Path to directory with NPZ files
        sample_counts: List of sample counts to test
        process_fn: Function to process each sample count
        **kwargs: Additional arguments for process_fn
        
    Returns:
        Dictionary with scaling results
    """
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    total_available = len(npz_files)
    
    results = {
        "total_available": total_available,
        "sample_counts_tested": [],
        "timings": [],
        "memory_usage": []
    }
    
    for n_samples in sample_counts:
        if n_samples > total_available:
            print(f"Warning: Requested {n_samples} samples but only {total_available} available")
            continue
        
        print(f"\nTesting with {n_samples} samples...")
        
        mem_before = get_system_resources()
        start_time = time.time()
        
        if process_fn:
            process_fn(data_dir, n_samples, **kwargs)
        
        elapsed = time.time() - start_time
        mem_after = get_system_resources()
        
        results["sample_counts_tested"].append(n_samples)
        results["timings"].append(round(elapsed, 2))
        results["memory_usage"].append({
            "before_mb": mem_before["ram_used_mb"],
            "after_mb": mem_after["ram_used_mb"],
            "delta_mb": round(mem_after["ram_used_mb"] - mem_before["ram_used_mb"], 2)
        })
        
        gc.collect()
    
    results["scaling_factor"] = _compute_scaling_factor(
        results["sample_counts_tested"],
        results["timings"]
    )
    
    return results


def scale_class_count(data_dir: str,
                      class_counts: List[int],
                      process_fn: Callable = None,
                      **kwargs) -> Dict:
    """
    Evaluate scalability with increasing number of classes.
    
    Args:
        data_dir: Path to directory with NPZ files
        class_counts: List of class counts to test
        process_fn: Function to process each class count
        **kwargs: Additional arguments for process_fn
        
    Returns:
        Dictionary with scaling results
    """
    from collections import Counter
    
    class_counter = Counter()
    for f in os.listdir(data_dir):
        if f.endswith('.npz'):
            class_label = f.split("_")[-1].replace(".npz", "")
            class_counter[class_label] += 1
    
    total_classes = len(class_counter)
    top_classes = [cls for cls, _ in class_counter.most_common()]
    
    results = {
        "total_classes": total_classes,
        "class_counts_tested": [],
        "timings": [],
        "memory_usage": [],
        "samples_per_class": []
    }
    
    for n_classes in class_counts:
        if n_classes > total_classes:
            print(f"Warning: Requested {n_classes} classes but only {total_classes} available")
            continue
        
        selected_classes = set(top_classes[:n_classes])
        n_samples = sum(class_counter[c] for c in selected_classes)
        
        print(f"\nTesting with {n_classes} classes ({n_samples} samples)...")
        
        mem_before = get_system_resources()
        start_time = time.time()
        
        if process_fn:
            process_fn(data_dir, n_classes, selected_classes, **kwargs)
        
        elapsed = time.time() - start_time
        mem_after = get_system_resources()
        
        results["class_counts_tested"].append(n_classes)
        results["samples_per_class"].append(n_samples)
        results["timings"].append(round(elapsed, 2))
        results["memory_usage"].append({
            "before_mb": mem_before["ram_used_mb"],
            "after_mb": mem_after["ram_used_mb"],
            "delta_mb": round(mem_after["ram_used_mb"] - mem_before["ram_used_mb"], 2)
        })
        
        gc.collect()
    
    results["scaling_factor"] = _compute_scaling_factor(
        results["class_counts_tested"],
        results["timings"]
    )
    
    return results


def _compute_scaling_factor(x_values: List[int], y_values: List[float]) -> Dict:
    """Compute scaling factor from x and y values."""
    if len(x_values) < 2:
        return {"linear": None, "polynomial": None}
    
    x = np.array(x_values, dtype=np.float64)
    y = np.array(y_values, dtype=np.float64)
    
    log_x = np.log(x)
    log_y = np.log(y + 1e-8)
    
    coeffs = np.polyfit(log_x, log_y, 1)
    exponent = coeffs[0]
    
    if 0.8 <= exponent <= 1.2:
        scaling_type = "linear"
    elif 1.8 <= exponent <= 2.2:
        scaling_type = "quadratic"
    elif exponent < 0.8:
        scaling_type = "sub-linear"
    else:
        scaling_type = "super-linear"
    
    return {
        "type": scaling_type,
        "exponent": round(exponent, 3),
        "description": f"O(n^{exponent:.2f})"
    }


def monitor_resources_during(func: Callable, *args, **kwargs) -> Tuple:
    """
    Monitor resource usage during function execution.
    
    Returns:
        Tuple of (result, resource_profile)
    """
    profile = {
        "timestamps": [],
        "memory_mb": [],
        "cpu_percent": []
    }
    
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    elapsed = time.time() - start_time
    
    return result, {
        "elapsed_seconds": round(elapsed, 2),
        "profile": profile
    }


def evaluate_scalability(data_dir: str, 
                         config: Dict = None) -> Dict:
    """
    Run comprehensive scalability evaluation.
    
    Args:
        data_dir: Path to directory with NPZ files
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with scalability results
    """
    from collections import Counter
    
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    total_files = len(npz_files)
    
    class_counter = Counter()
    for f in npz_files:
        class_label = f.split("_")[-1].replace(".npz", "")
        class_counter[class_label] += 1
    total_classes = len(class_counter)
    
    sample_counts = [100, 500, 1000, 5000, min(10000, total_files)]
    sample_counts = [n for n in sample_counts if n <= total_files]
    
    class_counts = [5, 10, 20, min(50, total_classes)]
    class_counts = [n for n in class_counts if n <= total_classes]
    
    results = {
        "dataset_info": {
            "total_samples": total_files,
            "total_classes": total_classes,
            "class_distribution": dict(class_counter.most_common(20))
        },
        "sample_scaling": None,
        "class_scaling": None,
        "recommendations": []
    }
    
    print("\n" + "=" * 60)
    print("Scalability Evaluation")
    print("=" * 60)
    
    print(f"\nDataset: {total_files} samples, {total_classes} classes")
    
    print("\n1. Sample Count Scaling:")
    sample_scaling = scale_sample_count(data_dir, sample_counts)
    results["sample_scaling"] = sample_scaling
    print(f"   Scaling type: {sample_scaling['scaling_factor']['type']}")
    print(f"   Exponent: {sample_scaling['scaling_factor']['exponent']}")
    
    print("\n2. Class Count Scaling:")
    class_scaling = scale_class_count(data_dir, class_counts)
    results["class_scaling"] = class_scaling
    print(f"   Scaling type: {class_scaling['scaling_factor']['type']}")
    print(f"   Exponent: {class_scaling['scaling_factor']['exponent']}")
    
    recommendations = _generate_recommendations(results)
    results["recommendations"] = recommendations
    
    print("\n3. Recommendations:")
    for rec in recommendations:
        print(f"   - {rec}")
    
    print("=" * 60)
    
    return results


def _generate_recommendations(results: Dict) -> List[str]:
    """Generate scaling recommendations."""
    recommendations = []
    
    total_samples = results["dataset_info"]["total_samples"]
    total_classes = results["dataset_info"]["total_classes"]
    
    if total_samples > 50000:
        recommendations.append(
            "Large dataset detected. Use streaming/generator-based loading "
            "to avoid memory issues."
        )
    
    if total_classes > 20:
        recommendations.append(
            f"Many classes ({total_classes}). Consider starting with top-20 "
            "classes before scaling to full dataset."
        )
    
    sample_scaling = results.get("sample_scaling", {})
    if sample_scaling.get("scaling_factor", {}).get("type") == "super-linear":
        recommendations.append(
            "Sample scaling is super-linear. Optimize data loading "
            "and consider batch processing."
        )
    
    class_scaling = results.get("class_scaling", {})
    if class_scaling.get("scaling_factor", {}).get("type") == "super-linear":
        recommendations.append(
            "Class scaling is super-linear. Consider reducing feature "
            "dimensionality or using feature selection."
        )
    
    if not recommendations:
        recommendations.append("Dataset appears to scale well. Proceed with full evaluation.")
    
    return recommendations


def print_scaling_report(results: Dict):
    """Print scaling evaluation report."""
    print("\n" + "=" * 80)
    print("SCALABILITY EVALUATION REPORT")
    print("=" * 80)
    
    info = results["dataset_info"]
    print(f"\nDataset: {info['total_samples']} samples, {info['total_classes']} classes")
    
    if results.get("sample_scaling"):
        ss = results["sample_scaling"]
        print(f"\nSample Scaling: {ss['scaling_factor']['type']} ({ss['scaling_factor']['description']})")
        print("  Sample Count | Time (s) | Memory Delta (MB)")
        print("  " + "-" * 45)
        for i, n in enumerate(ss["sample_counts_tested"]):
            time_val = ss["timings"][i]
            mem_delta = ss["memory_usage"][i]["delta_mb"]
            print(f"  {n:>11} | {time_val:>8.2f} | {mem_delta:>15.2f}")
    
    if results.get("class_scaling"):
        cs = results["class_scaling"]
        print(f"\nClass Scaling: {cs['scaling_factor']['type']} ({cs['scaling_factor']['description']})")
        print("  Class Count | Samples | Time (s) | Memory Delta (MB)")
        print("  " + "-" * 55)
        for i, n in enumerate(cs["class_counts_tested"]):
            samples = cs["samples_per_class"][i]
            time_val = cs["timings"][i]
            mem_delta = cs["memory_usage"][i]["delta_mb"]
            print(f"  {n:>10} | {samples:>7} | {time_val:>8.2f} | {mem_delta:>15.2f}")
    
    if results.get("recommendations"):
        print("\nRecommendations:")
        for rec in results["recommendations"]:
            print(f"  - {rec}")
    
    print("=" * 80)

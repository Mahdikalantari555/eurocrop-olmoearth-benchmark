"""
Dataset inspection utilities for EuroCropsML.

Provides:
- Directory structure analysis
- Sample counting
- Class distribution statistics
- Top-N class identification
- Reduced subset creation
"""

import os
import json
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def inspect_directory_structure(data_dir: str) -> Dict:
    """
    Inspect EuroCropsML directory structure.
    
    Args:
        data_dir: Path to preprocess directory containing .npz files
        
    Returns:
        Dictionary with structure information
    """
    info = {
        "total_files": 0,
        "total_size_mb": 0,
        "file_extensions": Counter(),
        "sample_files": [],
        "directory_tree": {}
    }
    
    for root, dirs, files in os.walk(data_dir):
        rel_path = os.path.relpath(root, data_dir)
        info["directory_tree"][rel_path] = len(files)
        
        for f in files:
            info["total_files"] += 1
            ext = os.path.splitext(f)[1]
            info["file_extensions"][ext] += 1
            
            filepath = os.path.join(root, f)
            info["total_size_mb"] += os.path.getsize(filepath) / (1024 * 1024)
            
            if len(info["sample_files"]) < 5:
                info["sample_files"].append(os.path.join(rel_path, f))
    
    info["file_extensions"] = dict(info["file_extensions"])
    info["total_size_mb"] = round(info["total_size_mb"], 2)
    
    return info


def count_samples(data_dir: str, split_dir: str = None, 
                  use_case: str = None) -> Dict:
    """
    Count total samples in the dataset.
    
    Args:
        data_dir: Path to preprocess directory
        split_dir: Optional path to split directory
        use_case: Optional use case name for split files
        
    Returns:
        Dictionary with sample counts
    """
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    counts = {
        "total_npz_files": len(npz_files),
        "split_counts": {}
    }
    
    if split_dir and use_case:
        split_path = os.path.join(split_dir, use_case, "finetune")
        if os.path.exists(split_path):
            for split_file in os.listdir(split_path):
                if split_file.endswith('.json'):
                    with open(os.path.join(split_path, split_file)) as f:
                        split_data = json.load(f)
                    split_name = split_file.replace("region_split_", "").replace(".json", "")
                    counts["split_counts"][split_name] = {
                        k: len(v) for k, v in split_data.items()
                    }
    
    return counts


def count_classes(data_dir: str) -> Dict:
    """
    Count total classes in the dataset.
    
    Args:
        data_dir: Path to preprocess directory
        
    Returns:
        Dictionary with class information
    """
    class_counter = Counter()
    
    for f in os.listdir(data_dir):
        if f.endswith('.npz'):
            class_label = f.split("_")[-1].replace(".npz", "")
            class_counter[class_label] += 1
    
    return {
        "total_classes": len(class_counter),
        "class_distribution": dict(class_counter)
    }


def generate_class_distribution(data_dir: str, 
                                 top_n: int = 20) -> Dict:
    """
    Generate class distribution statistics.
    
    Args:
        data_dir: Path to preprocess directory
        top_n: Number of top classes to return
        
    Returns:
        Dictionary with distribution statistics
    """
    class_counter = Counter()
    
    for f in os.listdir(data_dir):
        if f.endswith('.npz'):
            class_label = f.split("_")[-1].replace(".npz", "")
            class_counter[class_label] += 1
    
    total = sum(class_counter.values())
    top_classes = class_counter.most_common(top_n)
    
    stats = {
        "total_samples": total,
        "total_classes": len(class_counter),
        "top_n_classes": [
            {
                "class": cls,
                "count": count,
                "percentage": round(count / total * 100, 2)
            }
            for cls, count in top_classes
        ],
        "class_imbalance": {
            "most_common": top_classes[0][1] if top_classes else 0,
            "least_common": class_counter.most_common()[-1][1] if class_counter else 0,
            "imbalance_ratio": round(
                top_classes[0][1] / class_counter.most_common()[-1][1], 2
            ) if class_counter and class_counter.most_common()[-1][1] > 0 else 0
        }
    }
    
    return stats


def get_top_classes(data_dir: str, n: int = 20) -> List[str]:
    """
    Get top N most frequent classes.
    
    Args:
        data_dir: Path to preprocess directory
        n: Number of top classes to return
        
    Returns:
        List of class labels
    """
    class_counter = Counter()
    
    for f in os.listdir(data_dir):
        if f.endswith('.npz'):
            class_label = f.split("_")[-1].replace(".npz", "")
            class_counter[class_label] += 1
    
    return [cls for cls, _ in class_counter.most_common(n)]


def create_reduced_subset(data_dir: str, output_dir: str,
                          top_n: int = 20,
                          split_dir: str = None,
                          use_case: str = None) -> Dict:
    """
    Create a reduced subset using only top N classes.
    
    Args:
        data_dir: Path to preprocess directory
        output_dir: Path to output directory
        top_n: Number of top classes to keep
        split_dir: Optional path to split directory
        use_case: Optional use case name for split files
        
    Returns:
        Dictionary with subset statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    top_classes = get_top_classes(data_dir, top_n)
    top_class_set = set(top_classes)
    
    copied = 0
    skipped = 0
    
    for f in os.listdir(data_dir):
        if f.endswith('.npz'):
            class_label = f.split("_")[-1].replace(".npz", "")
            if class_label in top_class_set:
                src = os.path.join(data_dir, f)
                dst = os.path.join(output_dir, f)
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)
                copied += 1
            else:
                skipped += 1
    
    result = {
        "top_classes": top_classes,
        "copied": copied,
        "skipped": skipped,
        "output_dir": output_dir
    }
    
    if split_dir and use_case:
        new_split_dir = os.path.join(os.path.dirname(output_dir), "split_subset")
        os.makedirs(new_split_dir, exist_ok=True)
        
        split_path = os.path.join(split_dir, use_case, "finetune")
        if os.path.exists(split_path):
            for split_file in os.listdir(split_path):
                if split_file.endswith('.json'):
                    with open(os.path.join(split_path, split_file)) as f:
                        split_data = json.load(f)
                    
                    filtered_split = {}
                    for k, v in split_data.items():
                        filtered_split[k] = [
                            fn for fn in v
                            if fn.split("_")[-1].replace(".npz", "") in top_class_set
                        ]
                    
                    out_split_path = os.path.join(new_split_dir, split_file)
                    with open(out_split_path, 'w') as f:
                        json.dump(filtered_split, f, indent=2)
        
        result["split_dir"] = new_split_dir
    
    return result


def print_inspection_report(data_dir: str, split_dir: str = None,
                           use_case: str = None, top_n: int = 20):
    """
    Print a comprehensive inspection report.
    
    Args:
        data_dir: Path to preprocess directory
        split_dir: Optional path to split directory
        use_case: Optional use case name for split files
        top_n: Number of top classes to show
    """
    print("=" * 60)
    print("EuroCropsML Dataset Inspection Report")
    print("=" * 60)
    
    print("\n1. Directory Structure:")
    structure = inspect_directory_structure(data_dir)
    print(f"   Total files: {structure['total_files']}")
    print(f"   Total size: {structure['total_size_mb']} MB")
    print(f"   File extensions: {structure['file_extensions']}")
    
    print("\n2. Sample Counts:")
    counts = count_samples(data_dir, split_dir, use_case)
    print(f"   Total NPZ files: {counts['total_npz_files']}")
    if counts['split_counts']:
        for split_name, split_counts in counts['split_counts'].items():
            print(f"   {split_name}: {split_counts}")
    
    print("\n3. Class Distribution:")
    dist = generate_class_distribution(data_dir, top_n)
    print(f"   Total classes: {dist['total_classes']}")
    print(f"   Total samples: {dist['total_samples']}")
    print(f"   Imbalance ratio: {dist['class_imbalance']['imbalance_ratio']}")
    
    print(f"\n4. Top {top_n} Classes:")
    for i, cls_info in enumerate(dist['top_n_classes'], 1):
        print(f"   {i:2d}. {cls_info['class']}: {cls_info['count']} ({cls_info['percentage']}%)")
    
    print("=" * 60)

"""
EuroCropsML / EuroSAT data loading and processing modules.

Modules:
    io: Multi-format data I/O (NPZ, TIF, NPY, PNG, HDF5)
    loader: Traditional data loading with parallel I/O (NPZ only)
    streaming: Memory-efficient streaming data loader (NPZ only)
    features: Classical feature extraction
    inspect: Dataset inspection utilities
    memory_profiler: Memory profiling utilities
    scaling: Scaling evaluation utilities
"""

from .io import (
    load_sample,
    detect_format,
    stream_files,
    stream_from_split_generic,
    get_class_counts,
    get_top_classes as get_top_classes_generic,
)

from .loader import (
    load_split,
    load_split_zenodo,
    load_split_padded,
    load_split_padded_cached,
    load_dataset,
    filter_top_classes,
    train_test_split_stratified
)

from .streaming import (
    stream_npz_files,
    stream_from_split,
    create_batch_generator,
    create_split_batch_generator,
    StreamingDataset,
    StreamingSplitDataset
)

from .features import (
    ndvi_features,
    band_stat_features,
    temporal_features,
    mean_ndvi,
    std_ndvi,
    mean_red,
    mean_nir,
    mean_green,
    mean_blue,
    spectral_statistics,
    ndvi_percentiles,
    combined_baseline_features,
    stream_and_save_features,
)

from .inspect import (
    inspect_directory_structure,
    count_samples,
    count_classes,
    generate_class_distribution,
    get_top_classes,
    create_reduced_subset,
    print_inspection_report
)

from .memory_profiler import (
    get_memory_usage,
    measure_single_npz,
    measure_batch_loading,
    profile_full_dataset,
    identify_memory_bottlenecks,
    print_memory_report
)

from .scaling import (
    get_system_resources,
    scale_sample_count,
    scale_class_count,
    evaluate_scalability,
    print_scaling_report
)

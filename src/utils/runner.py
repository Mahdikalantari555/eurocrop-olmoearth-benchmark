"""
Shared runner utilities for all experiment phases.
"""

import yaml
import os
import sys
import time
from datetime import datetime


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(phase_name, metrics_dir):
    os.makedirs(metrics_dir, exist_ok=True)
    log_file = os.path.join(metrics_dir, f"{phase_name}_log.txt")
    with open(log_file, "w") as f:
        f.write("")
    return log_file


def log(msg, log_file=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def log_header(title, log_file):
    log("=" * 60, log_file)
    log(title, log_file)
    log("=" * 60, log_file)


def log_footer(phase_name, start_time, log_file):
    total = time.time() - start_time
    log("", log_file)
    log("=" * 60, log_file)
    log(f"{phase_name} COMPLETE - Total time: {total:.1f}s", log_file)
    log("=" * 60, log_file)


def load_data(cfg, log_file):
    import gc
    from src.data.loader import load_split_padded, load_dataset, train_test_split_stratified

    start = time.time()

    if cfg["data"]["mode"] == "local":
        log(f"Loading data in LOCAL mode...", log_file)
        log(f"  Preprocess dir: {cfg['data']['local_preprocess_dir']}", log_file)
        log(f"  Split dir: {cfg['data']['local_split_dir']}", log_file)
        log(f"  Use case: {cfg['data']['use_case']}", log_file)
        use_zenodo = cfg["data"].get("use_zenodo", False)
        splits = load_split_padded(
            cfg["data"]["local_preprocess_dir"],
            cfg["data"]["local_split_dir"],
            cfg["data"]["use_case"],
            split_name="all",
            use_zenodo=use_zenodo
        )
        X_train, y_train, _ = splits["train"]
        X_test, y_test, _ = splits["test"]
        del splits
        gc.collect()
    else:
        log(f"Loading data in CLOUD mode...", log_file)
        X, y, _ = load_dataset(cfg["data"]["cloud_data_dir"],
                                cfg["data"]["cloud_country"],
                                cfg["data"]["top_n_classes"])
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, test_size=cfg["data"]["test_split"],
            seed=cfg["data"]["random_seed"]
        )
        del X, y
        gc.collect()

    log(f"Data loaded in {time.time() - start:.1f}s", log_file)
    log(f"  Train: {len(X_train)} | Test: {len(X_test)}", log_file)

    return X_train, y_train, X_test, y_test

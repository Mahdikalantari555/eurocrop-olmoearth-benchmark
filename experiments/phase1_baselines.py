"""
Phase 1: RF / LightGBM / XGBoost on classical features.
Run: python experiments/phase1_baselines.py
"""

import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import (load_split_padded, load_dataset,
                              train_test_split_stratified)
from src.data.features import ndvi_features, band_stat_features
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics


def run(cfg):
    if cfg["data"]["mode"] == "local":
        splits = load_split_padded(
            cfg["data"]["local_preprocess_dir"],
            cfg["data"]["local_split_dir"],
            cfg["data"]["use_case"],
            split_name="all"
        )
        X_train, y_train, _ = splits["train"]
        X_test, y_test, _ = splits["test"]
    else:
        X, y, _ = load_dataset(cfg["data"]["cloud_data_dir"],
                                cfg["data"]["cloud_country"],
                                cfg["data"]["top_n_classes"])
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, test_size=cfg["data"]["test_split"],
            seed=cfg["data"]["random_seed"]
        )

    experiments = {
        "ndvi_rf": (ndvi_features, "rf"),
        "bandstat_rf": (band_stat_features, "rf"),
        "bandstat_lgbm": (band_stat_features, "lgbm"),
        "bandstat_xgb": (band_stat_features, "xgb"),
    }

    results = {}
    for name, (feat_fn, clf_name) in experiments.items():
        X_tr = feat_fn(X_train)
        X_te = feat_fn(X_test)
        clf = get_classifier(clf_name, cfg["data"]["random_seed"])
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        m = compute_metrics(y_test, y_pred)
        save_metrics(m, f"results/metrics/phase1_{name}.json")
        results[name] = m
        print(f"{name}: OA={m['overall_accuracy']:.3f} | F1={m['macro_f1']:.3f}")

    return results


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

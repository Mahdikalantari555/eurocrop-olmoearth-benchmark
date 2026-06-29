"""
Phase 1: RF / LightGBM / XGBoost on classical features.
Run: python experiments/phase1_baselines.py
"""

import os
import sys
import gc
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.runner import load_config, setup_logging, log, log_header, log_footer
from src.data.loader import load_features_chunked
from src.data.features import ndvi_single, band_stat_single
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics, save_confusion_matrix


def run(cfg):
    log_file = setup_logging("phase1", cfg["output"]["metrics_dir"])

    log_header("PHASE 1: Classical Baselines (RF / LightGBM / XGBoost)", log_file)
    start_time = time.time()

    preprocess_dir = cfg["data"]["local_preprocess_dir"]
    split_dir = cfg["data"]["local_split_dir"]
    use_case = cfg["data"]["use_case"]
    use_zenodo = cfg["data"].get("use_zenodo", False)

    experiments = [
        ("ndvi_rf", ndvi_single, "rf"),
        ("bandstat_rf", band_stat_single, "rf"),
        ("bandstat_lgbm", band_stat_single, "lgbm"),
        ("bandstat_xgb", band_stat_single, "xgb"),
    ]

    results = {}
    for i, (name, feat_fn, clf_name) in enumerate(experiments, 1):
        log(f"\n[{i}/{len(experiments)}] {name}", log_file)
        exp_start = time.time()

        X_tr, y_train, X_te, y_test, labels = load_features_chunked(
            preprocess_dir, split_dir, use_case, feat_fn,
            use_zenodo=use_zenodo, chunk_size=5000
        )
        log(f"  Features: train={X_tr.shape}, test={X_te.shape}", log_file)

        clf = get_classifier(clf_name, cfg["data"]["random_seed"])
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        del X_tr, X_te, clf
        gc.collect()

        m = compute_metrics(y_test, y_pred, labels=labels)
        save_metrics(m, f"results/metrics/phase1_{name}.json")
        save_confusion_matrix(y_test, y_pred, f"results/metrics/phase1_{name}_cm.csv", labels=labels)
        results[name] = m

        log(f"  OA={m['overall_accuracy']:.3f} | F1={m['macro_f1']:.3f} | "
            f"Kappa={m['kappa']:.3f} ({time.time() - exp_start:.1f}s)", log_file)

    log_footer("PHASE 1", start_time, log_file)
    return results


if __name__ == "__main__":
    run(load_config())

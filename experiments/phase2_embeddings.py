"""
Phase 2: OLMoEarth embeddings + classifiers.
Run: python experiments/phase2_embeddings.py
"""

import os
import sys
import gc
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.runner import load_config, setup_logging, log, log_header, log_footer, load_data
from src.encoder.olmoearth import OLMoEarthEncoder
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics, save_confusion_matrix


def run(cfg):
    log_file = setup_logging("phase2", cfg["output"]["metrics_dir"])

    log_header("PHASE 2: OLMoEarth Embeddings + Classifiers", log_file)
    start_time = time.time()

    X_train, y_train, X_test, y_test = load_data(cfg, log_file)

    log(f"\nInitializing OLMoEarth encoder...", log_file)
    log(f"  Mode: {cfg['model']['mode']} | Device: {cfg['model']['device']}", log_file)

    encoder = OLMoEarthEncoder(
        mode=cfg["model"]["mode"],
        local_weights_path=cfg["model"].get("local_weights_path"),
        cloud_model_id=cfg["model"].get("cloud_model_id"),
        device=cfg["model"]["device"]
    )

    log(f"\nEncoding training set ({len(X_train)} samples)...", log_file)
    t = time.time()
    emb_train = encoder.encode(X_train, cfg["model"]["batch_size"])
    log(f"  Shape: {emb_train.shape} ({time.time() - t:.1f}s)", log_file)

    log(f"\nEncoding test set ({len(X_test)} samples)...", log_file)
    t = time.time()
    emb_test = encoder.encode(X_test, cfg["model"]["batch_size"])
    log(f"  Shape: {emb_test.shape} ({time.time() - t:.1f}s)", log_file)

    del X_train, X_test
    gc.collect()

    os.makedirs("results/metrics", exist_ok=True)
    np.save("results/metrics/emb_train.npy", emb_train)
    np.save("results/metrics/emb_test.npy", emb_test)
    log(f"Embeddings saved to results/metrics/", log_file)

    results = {}
    classifiers = ["logreg", "rf", "lgbm", "xgb"]
    for i, clf_name in enumerate(classifiers, 1):
        log(f"\n[{i}/{len(classifiers)}] {clf_name}", log_file)
        t = time.time()

        clf = get_classifier(clf_name)
        clf.fit(emb_train, y_train)
        y_pred = clf.predict(emb_test)

        labels = sorted(set(y_test))
        m = compute_metrics(y_test, y_pred, labels=labels)
        save_metrics(m, f"results/metrics/phase2_olmo_{clf_name}.json")
        save_confusion_matrix(y_test, y_pred, f"results/metrics/phase2_olmo_{clf_name}_cm.csv", labels=labels)
        results[clf_name] = m

        log(f"  OA={m['overall_accuracy']:.3f} | F1={m['macro_f1']:.3f} | "
            f"Kappa={m['kappa']:.3f} ({time.time() - t:.1f}s)", log_file)

    del encoder
    gc.collect()

    log_footer("PHASE 2", start_time, log_file)
    return results


if __name__ == "__main__":
    run(load_config())

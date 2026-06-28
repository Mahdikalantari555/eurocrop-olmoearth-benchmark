"""
Phase 4: Linear probe evaluation on pre-computed embeddings.
Run: python experiments/phase4_linear_probe.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.runner import load_config, setup_logging, log, log_header, log_footer
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics, save_confusion_matrix


def run(cfg):
    log_file = setup_logging("phase4", cfg["output"]["metrics_dir"])

    log_header("PHASE 4: Linear Probe Evaluation", log_file)
    start_time = time.time()

    log(f"\nLoading pre-computed embeddings...", log_file)
    emb_train = np.load("results/metrics/emb_train.npy")
    emb_test = np.load("results/metrics/emb_test.npy")
    log(f"  Train: {emb_train.shape} | Test: {emb_test.shape}", log_file)

    if cfg["data"]["mode"] == "local":
        log(f"\nLoading labels from local splits...", log_file)
        log(f"  Use case: {cfg['data']['use_case']}", log_file)
        use_zenodo = cfg["data"].get("use_zenodo", False)
        from src.data.loader import load_split_padded
        splits = load_split_padded(
            cfg["data"]["local_preprocess_dir"],
            cfg["data"]["local_split_dir"],
            cfg["data"]["use_case"],
            split_name="all",
            use_zenodo=use_zenodo
        )
        _, y_train, _ = splits["train"]
        _, y_test, _ = splits["test"]
    else:
        log(f"\nLoading labels from cloud mode...", log_file)
        from src.data.loader import load_dataset, train_test_split_stratified
        X, y, _ = load_dataset(cfg["data"]["cloud_data_dir"],
                                cfg["data"]["cloud_country"],
                                cfg["data"]["top_n_classes"])
        _, _, y_train, y_test = train_test_split_stratified(
            X, y, test_size=cfg["data"]["test_split"],
            seed=cfg["data"]["random_seed"]
        )

    log(f"  Train: {len(y_train)} | Test: {len(y_test)}", log_file)

    results = {}
    classifiers = ["logreg", "rf", "lgbm"]
    for i, clf_name in enumerate(classifiers, 1):
        log(f"\n[{i}/{len(classifiers)}] {clf_name}", log_file)
        t = time.time()

        clf = get_classifier(clf_name)
        clf.fit(emb_train, y_train)
        y_pred = clf.predict(emb_test)

        labels = sorted(set(y_test))
        m = compute_metrics(y_test, y_pred, labels=labels)
        save_metrics(m, f"results/metrics/phase4_probe_{clf_name}.json")
        save_confusion_matrix(y_test, y_pred, f"results/metrics/phase4_probe_{clf_name}_cm.csv", labels=labels)
        results[clf_name] = m

        log(f"  OA={m['overall_accuracy']:.3f} | F1={m['macro_f1']:.3f} | "
            f"Kappa={m['kappa']:.3f} ({time.time() - t:.1f}s)", log_file)

    log_footer("PHASE 4", start_time, log_file)
    return results


if __name__ == "__main__":
    run(load_config())

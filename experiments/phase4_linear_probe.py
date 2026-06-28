"""
Phase 4: Linear probe evaluation on pre-computed embeddings.
Run: python experiments/phase4_linear_probe.py
"""

import yaml
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics


def run(cfg):
    emb_train = np.load("results/metrics/emb_train.npy")
    emb_test = np.load("results/metrics/emb_test.npy")

    if cfg["data"]["mode"] == "local":
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
        from src.data.loader import load_dataset, train_test_split_stratified
        X, y, _ = load_dataset(cfg["data"]["cloud_data_dir"],
                                cfg["data"]["cloud_country"],
                                cfg["data"]["top_n_classes"])
        _, _, y_train, y_test = train_test_split_stratified(
            X, y, test_size=cfg["data"]["test_split"],
            seed=cfg["data"]["random_seed"]
        )

    results = {}
    for clf_name in ["logreg", "rf", "lgbm"]:
        clf = get_classifier(clf_name)
        clf.fit(emb_train, y_train)
        m = compute_metrics(y_test, clf.predict(emb_test))
        save_metrics(m, f"results/metrics/phase4_probe_{clf_name}.json")
        results[clf_name] = m
        print(f"probe+{clf_name}: OA={m['overall_accuracy']:.3f} | F1={m['macro_f1']:.3f}")

    return results


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

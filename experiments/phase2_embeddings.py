"""
Phase 2: OLMoEarth embeddings + classifiers.
Run: python experiments/phase2_embeddings.py
"""

import yaml
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import (load_split_padded, load_dataset,
                              train_test_split_stratified)
from src.encoder.olmoearth import OLMoEarthEncoder
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics


def run(cfg):
    if cfg["data"]["mode"] == "local":
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
    else:
        X, y, _ = load_dataset(cfg["data"]["cloud_data_dir"],
                                cfg["data"]["cloud_country"],
                                cfg["data"]["top_n_classes"])
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, test_size=cfg["data"]["test_split"],
            seed=cfg["data"]["random_seed"]
        )

    encoder = OLMoEarthEncoder(
        mode=cfg["model"]["mode"],
        local_weights_path=cfg["model"].get("local_weights_path"),
        cloud_model_id=cfg["model"].get("cloud_model_id"),
        device=cfg["model"]["device"]
    )

    emb_train = encoder.encode(X_train, cfg["model"]["batch_size"])
    emb_test = encoder.encode(X_test, cfg["model"]["batch_size"])

    os.makedirs("results/metrics", exist_ok=True)
    np.save("results/metrics/emb_train.npy", emb_train)
    np.save("results/metrics/emb_test.npy", emb_test)

    results = {}
    for clf_name in ["logreg", "rf", "lgbm", "xgb"]:
        clf = get_classifier(clf_name)
        clf.fit(emb_train, y_train)
        m = compute_metrics(y_test, clf.predict(emb_test))
        save_metrics(m, f"results/metrics/phase2_olmo_{clf_name}.json")
        results[clf_name] = m
        print(f"olmo+{clf_name}: OA={m['overall_accuracy']:.3f} | F1={m['macro_f1']:.3f}")

    return results


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

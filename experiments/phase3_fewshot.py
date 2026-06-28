"""
Phase 3: Few-shot comparison using pre-defined splits.
Run: python experiments/phase3_fewshot.py
"""

import yaml
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_split_padded, load_dataset, train_test_split_stratified
from src.data.features import band_stat_features
from src.encoder.olmoearth import OLMoEarthEncoder
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics


def run(cfg):
    shots = cfg["fewshot"]["shots"]
    repeats = cfg["fewshot"]["n_repeats"]

    if cfg["data"]["mode"] == "local":
        preprocess_dir = cfg["data"]["local_preprocess_dir"]
        split_dir = cfg["data"]["local_split_dir"]
        use_case = cfg["data"]["use_case"]

        base_splits = load_split_padded(preprocess_dir, split_dir, use_case, "all")
        X_test, y_test, _ = base_splits["test"]

        X_test_feat = band_stat_features(np.array(X_test, dtype=np.float32))

        encoder = OLMoEarthEncoder(
            mode="local",
            local_weights_path=cfg["model"]["local_weights_path"],
            device=cfg["model"]["device"]
        )
        emb_test = encoder.encode(np.array(X_test, dtype=np.float32),
                                   cfg["model"]["batch_size"])

        results = {}
        for n in shots:
            shot_splits = load_split_padded(
                preprocess_dir, split_dir, use_case, str(n)
            )
            X_train_n, y_train_n, _ = shot_splits["train"]
            X_train_n = np.array(X_train_n, dtype=np.float32)

            X_train_feat = band_stat_features(X_train_n)
            emb_train_n = encoder.encode(X_train_n, cfg["model"]["batch_size"])

            scores_rf, scores_lgbm, scores_olmo = [], [], []
            for r in range(repeats):
                clf_rf = get_classifier("rf", seed=r)
                clf_rf.fit(X_train_feat, y_train_n)
                scores_rf.append(compute_metrics(y_test, clf_rf.predict(X_test_feat))["macro_f1"])

                clf_lgbm = get_classifier("lgbm", seed=r)
                clf_lgbm.fit(X_train_feat, y_train_n)
                scores_lgbm.append(compute_metrics(y_test, clf_lgbm.predict(X_test_feat))["macro_f1"])

                clf_olmo = get_classifier("lgbm", seed=r)
                clf_olmo.fit(emb_train_n, y_train_n)
                scores_olmo.append(compute_metrics(y_test, clf_olmo.predict(emb_test))["macro_f1"])

            results[str(n)] = {
                "rf_f1": float(np.mean(scores_rf)),
                "rf_f1_std": float(np.std(scores_rf)),
                "lgbm_f1": float(np.mean(scores_lgbm)),
                "lgbm_f1_std": float(np.std(scores_lgbm)),
                "olmo_lgbm_f1": float(np.mean(scores_olmo)),
                "olmo_lgbm_f1_std": float(np.std(scores_olmo)),
            }
            print(f"Shots={n}: RF={results[str(n)]['rf_f1']:.3f} | "
                  f"LGBM={results[str(n)]['lgbm_f1']:.3f} | "
                  f"OLMO+LGBM={results[str(n)]['olmo_lgbm_f1']:.3f}")
    else:
        X, y, _ = load_dataset(cfg["data"]["cloud_data_dir"],
                                cfg["data"]["cloud_country"],
                                cfg["data"]["top_n_classes"])
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X, y, test_size=cfg["data"]["test_split"],
            seed=cfg["data"]["random_seed"]
        )
        X_train_feat = band_stat_features(X_train)
        X_test_feat = band_stat_features(X_test)

        encoder = OLMoEarthEncoder(
            mode="cloud",
            cloud_model_id=cfg["model"]["cloud_model_id"],
            device=cfg["model"]["device"]
        )
        emb_test = encoder.encode(X_test, cfg["model"]["batch_size"])

        results = {}
        for n in shots:
            scores_rf, scores_lgbm, scores_olmo = [], [], []
            for r in range(repeats):
                rng = np.random.RandomState(r)
                indices = []
                for cls in np.unique(y_train):
                    cls_idx = np.where(y_train == cls)[0]
                    sampled = rng.choice(cls_idx, min(n, len(cls_idx)),
                                         replace=len(cls_idx) < n)
                    indices.extend(sampled)
                idx = np.array(indices)

                X_fs_feat = X_train_feat[idx]
                y_fs = y_train[idx]
                emb_fs = encoder.encode(X_train[idx], cfg["model"]["batch_size"])

                clf_rf = get_classifier("rf", seed=r)
                clf_rf.fit(X_fs_feat, y_fs)
                scores_rf.append(compute_metrics(y_test, clf_rf.predict(X_test_feat))["macro_f1"])

                clf_lgbm = get_classifier("lgbm", seed=r)
                clf_lgbm.fit(X_fs_feat, y_fs)
                scores_lgbm.append(compute_metrics(y_test, clf_lgbm.predict(X_test_feat))["macro_f1"])

                clf_olmo = get_classifier("lgbm", seed=r)
                clf_olmo.fit(emb_fs, y_fs)
                scores_olmo.append(compute_metrics(y_test, clf_olmo.predict(emb_test))["macro_f1"])

            results[str(n)] = {
                "rf_f1": float(np.mean(scores_rf)),
                "rf_f1_std": float(np.std(scores_rf)),
                "lgbm_f1": float(np.mean(scores_lgbm)),
                "lgbm_f1_std": float(np.std(scores_lgbm)),
                "olmo_lgbm_f1": float(np.mean(scores_olmo)),
                "olmo_lgbm_f1_std": float(np.std(scores_olmo)),
            }
            print(f"Shots={n}: RF={results[str(n)]['rf_f1']:.3f} | "
                  f"LGBM={results[str(n)]['lgbm_f1']:.3f} | "
                  f"OLMO+LGBM={results[str(n)]['olmo_lgbm_f1']:.3f}")

    save_metrics(results, "results/metrics/phase3_fewshot.json")
    return results


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

"""
Phase 3: Few-shot comparison using pre-defined splits.
Run: python experiments/phase3_fewshot.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.runner import load_config, setup_logging, log, log_header, log_footer
from src.data.loader import load_split_padded, load_dataset, train_test_split_stratified
from src.data.features import band_stat_features
from src.encoder.olmoearth import OLMoEarthEncoder
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics


def run(cfg):
    log_file = setup_logging("phase3", cfg["output"]["metrics_dir"])

    shots = cfg["fewshot"]["shots"]
    repeats = cfg["fewshot"]["n_repeats"]

    log_header("PHASE 3: Few-Shot Comparison", log_file)
    log(f"Shots: {shots}", log_file)
    log(f"Repeats: {repeats}", log_file)
    start_time = time.time()

    if cfg["data"]["mode"] == "local":
        log(f"\nLoading data in LOCAL mode...", log_file)
        log(f"  Use case: {cfg['data']['use_case']}", log_file)
        use_zenodo = cfg["data"].get("use_zenodo", False)
        preprocess_dir = cfg["data"]["local_preprocess_dir"]
        split_dir = cfg["data"]["local_split_dir"]
        use_case = cfg["data"]["use_case"]

        base_splits = load_split_padded(preprocess_dir, split_dir, use_case, "all",
                                         use_zenodo=use_zenodo)
        X_test, y_test, _ = base_splits["test"]
        log(f"  Test samples: {len(X_test)}", log_file)

        X_test_feat = band_stat_features(np.array(X_test, dtype=np.float32))

        log(f"\nInitializing OLMoEarth encoder...", log_file)
        encoder = OLMoEarthEncoder(
            mode="local",
            local_weights_path=cfg["model"].get("local_weights_path"),
            device=cfg["model"]["device"]
        )

        log(f"\nEncoding test set...", log_file)
        t = time.time()
        emb_test = encoder.encode(np.array(X_test, dtype=np.float32),
                                   cfg["model"]["batch_size"])
        log(f"  Shape: {emb_test.shape} ({time.time() - t:.1f}s)", log_file)

        results = {}
        for i, n in enumerate(shots, 1):
            log(f"\n{'='*40}", log_file)
            log(f"[{i}/{len(shots)}] SHOT={n}", log_file)
            log(f"{'='*40}", log_file)

            t = time.time()
            shot_splits = load_split_padded(
                preprocess_dir, split_dir, use_case, str(n),
                use_zenodo=use_zenodo
            )
            X_train_n, y_train_n, _ = shot_splits["train"]
            X_train_n = np.array(X_train_n, dtype=np.float32)
            log(f"  Train samples: {len(X_train_n)} ({time.time() - t:.1f}s load)", log_file)

            X_train_feat = band_stat_features(X_train_n)

            t = time.time()
            emb_train_n = encoder.encode(X_train_n, cfg["model"]["batch_size"])
            log(f"  Encoding: {time.time() - t:.1f}s", log_file)

            scores_rf, scores_lgbm, scores_olmo = [], [], []
            for r in range(repeats):
                log(f"  Repeat {r+1}/{repeats}...", log_file)

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

            log(f"  RF={results[str(n)]['rf_f1']:.3f} | LGBM={results[str(n)]['lgbm_f1']:.3f} | "
                f"OLMO={results[str(n)]['olmo_lgbm_f1']:.3f}", log_file)

    else:
        log(f"\nLoading data in CLOUD mode...", log_file)
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
            cloud_model_id=cfg["model"].get("cloud_model_id"),
            device=cfg["model"]["device"]
        )
        emb_test = encoder.encode(X_test, cfg["model"]["batch_size"])

        results = {}
        for i, n in enumerate(shots, 1):
            log(f"\n[{i}/{len(shots)}] SHOT={n}", log_file)
            t = time.time()
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
            log(f"  RF={results[str(n)]['rf_f1']:.3f} | LGBM={results[str(n)]['lgbm_f1']:.3f} | "
                f"OLMO={results[str(n)]['olmo_lgbm_f1']:.3f} ({time.time() - t:.1f}s)", log_file)

    save_metrics(results, "results/metrics/phase3_fewshot.json")
    log(f"\nResults saved", log_file)

    log_footer("PHASE 3", start_time, log_file)
    return results


if __name__ == "__main__":
    run(load_config())

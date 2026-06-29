"""
Classification Experiments Runner.

Implements the 4-experiment matrix from the checklist:
- Experiment A: Handcrafted features only (Logistic Regression + LightGBM)
- Experiment B: OLMoEarth embeddings only (Logistic Regression)
- Experiment C: OLMoEarth embeddings only (LightGBM)
- Experiment D: Concatenated embeddings + features (LightGBM)

Usage:
    python experiments/run_classification.py --config config.yaml
    python experiments/run_classification.py --config config.yaml --experiment A
    python experiments/run_classification.py --config config.yaml --experiment all
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.runner import load_config, setup_logging, log, log_header, log_footer
from src.data.loader import load_split_padded, load_split_padded_cached
from src.data.features import (
    ndvi_features, band_stat_features, spectral_statistics,
    combined_baseline_features
)
from src.encoder.olmoearth import OLMoEarthEncoder
from src.models.classical import get_classifier
from src.evaluate.metrics import compute_metrics, save_metrics, save_confusion_matrix


class ClassificationExperiments:
    """Run classification experiments per the checklist matrix."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.log_file = None
        self.results = {}
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
        """Load data and return train/test splits."""
        log_file = self.log_file
        
        log("Loading data...", log_file)
        start = time.time()
        
        if self.cfg["data"]["mode"] == "local":
            use_zenodo = self.cfg["data"].get("use_zenodo", False)
            
            cache_dir = os.path.join(
                self.cfg["data"]["local_split_dir"],
                self.cfg["data"]["use_case"],
                "cache"
            )
            
            splits = load_split_padded_cached(
                self.cfg["data"]["local_preprocess_dir"],
                self.cfg["data"]["local_split_dir"],
                self.cfg["data"]["use_case"],
                split_name="all",
                cache_dir=cache_dir,
                use_zenodo=use_zenodo
            )
            
            X_train, y_train, label_names = splits["train"]
            X_test, y_test, _ = splits["test"]
        else:
            from src.data.loader import load_dataset, train_test_split_stratified
            X, y, label_names = load_dataset(
                self.cfg["data"]["cloud_data_dir"],
                self.cfg["data"]["cloud_country"],
                self.cfg["data"]["top_n_classes"]
            )
            X_train, X_test, y_train, y_test = train_test_split_stratified(
                X, y, 
                test_size=self.cfg["data"]["test_split"],
                seed=self.cfg["data"]["random_seed"]
            )
        
        log(f"Data loaded in {time.time() - start:.1f}s", log_file)
        log(f"  Train: {len(X_train)} | Test: {len(X_test)}", log_file)
        log(f"  Classes: {len(label_names)}", log_file)
        
        return X_train, y_train, X_test, y_test, label_names
    
    def extract_features(self, X: np.ndarray, feature_type: str) -> np.ndarray:
        """Extract features based on type."""
        if feature_type == "ndvi":
            return ndvi_features(X)
        elif feature_type == "bandstat":
            return band_stat_features(X)
        elif feature_type == "spectral":
            return spectral_statistics(X)
        elif feature_type == "combined":
            return combined_baseline_features(X)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract OLMoEarth embeddings."""
        encoder = OLMoEarthEncoder(
            mode=self.cfg["model"]["mode"],
            local_weights_path=self.cfg["model"].get("local_weights_path"),
            cloud_model_id=self.cfg["model"].get("cloud_model_id"),
            device=self.cfg["model"]["device"]
        )
        
        embeddings = encoder.encode(X, self.cfg["model"]["batch_size"])
        
        del encoder
        import gc
        gc.collect()
        
        return embeddings
    
    def run_experiment_a(self, X_train, y_train, X_test, y_test, 
                         labels: List) -> Dict:
        """
        Experiment A: Handcrafted features only.
        - Logistic Regression
        - LightGBM
        """
        log("\n" + "=" * 60, self.log_file)
        log("EXPERIMENT A: Handcrafted Features Only", self.log_file)
        log("=" * 60, self.log_file)
        
        results = {}
        
        feature_sets = {
            "ndvi": ndvi_features,
            "bandstat": band_stat_features,
            "combined": combined_baseline_features
        }
        
        classifiers = ["logreg", "lgbm"]
        
        for feat_name, feat_fn in feature_sets.items():
            log(f"\nFeature set: {feat_name}", self.log_file)
            
            X_train_feat = feat_fn(X_train)
            X_test_feat = feat_fn(X_test)
            
            log(f"  Shape: train={X_train_feat.shape}, test={X_test_feat.shape}", self.log_file)
            
            for clf_name in classifiers:
                exp_name = f"{feat_name}_{clf_name}"
                log(f"\n  Classifier: {clf_name}", self.log_file)
                
                start = time.time()
                
                clf = get_classifier(clf_name, self.cfg["data"]["random_seed"])
                clf.fit(X_train_feat, y_train)
                y_pred = clf.predict(X_test_feat)
                
                elapsed = time.time() - start
                
                m = compute_metrics(y_test, y_pred, labels=labels)
                m["feature_type"] = feat_name
                m["classifier"] = clf_name
                m["experiment"] = "A"
                m["elapsed_seconds"] = round(elapsed, 2)
                
                results[exp_name] = m
                
                save_metrics(m, f"results/metrics/expA_{exp_name}.json")
                save_confusion_matrix(y_test, y_pred, 
                                     f"results/metrics/expA_{exp_name}_cm.csv",
                                     labels=labels)
                
                log(f"    OA={m['overall_accuracy']:.3f} | F1={m['macro_f1']:.3f} | "
                    f"Kappa={m['kappa']:.3f} ({elapsed:.1f}s)", self.log_file)
        
        return results
    
    def run_experiment_b(self, X_train, y_train, X_test, y_test,
                         labels: List) -> Dict:
        """
        Experiment B: OLMoEarth embeddings only + Logistic Regression.
        """
        log("\n" + "=" * 60, self.log_file)
        log("EXPERIMENT B: OLMoEarth Embeddings + Logistic Regression", self.log_file)
        log("=" * 60, self.log_file)
        
        start = time.time()
        emb_train = self.extract_embeddings(X_train)
        emb_test = self.extract_embeddings(X_test)
        encoding_time = time.time() - start
        
        log(f"  Embeddings extracted in {encoding_time:.1f}s", self.log_file)
        log(f"  Shape: train={emb_train.shape}, test={emb_test.shape}", self.log_file)
        
        clf_start = time.time()
        clf = get_classifier("logreg", self.cfg["data"]["random_seed"])
        clf.fit(emb_train, y_train)
        y_pred = clf.predict(emb_test)
        clf_time = time.time() - clf_start
        
        m = compute_metrics(y_test, y_pred, labels=labels)
        m["experiment"] = "B"
        m["classifier"] = "logreg"
        m["encoding_seconds"] = round(encoding_time, 2)
        m["clf_seconds"] = round(clf_time, 2)
        m["embedding_dim"] = emb_train.shape[1]
        
        save_metrics(m, "results/metrics/expB_logreg.json")
        save_confusion_matrix(y_test, y_pred, 
                             "results/metrics/expB_logreg_cm.csv",
                             labels=labels)
        
        log(f"  OA={m['overall_accuracy']:.3f} | F1={m['macro_f1']:.3f} | "
            f"Kappa={m['kappa']:.3f}", self.log_file)
        
        return {"logreg": m, "embeddings": (emb_train, emb_test)}
    
    def run_experiment_c(self, X_train, y_train, X_test, y_test,
                         labels: List, embeddings: Tuple = None) -> Dict:
        """
        Experiment C: OLMoEarth embeddings only + LightGBM.
        """
        log("\n" + "=" * 60, self.log_file)
        log("EXPERIMENT C: OLMoEarth Embeddings + LightGBM", self.log_file)
        log("=" * 60, self.log_file)
        
        if embeddings:
            emb_train, emb_test = embeddings
        else:
            start = time.time()
            emb_train = self.extract_embeddings(X_train)
            emb_test = self.extract_embeddings(X_test)
            log(f"  Embeddings extracted in {time.time() - start:.1f}s", self.log_file)
        
        log(f"  Shape: train={emb_train.shape}, test={emb_test.shape}", self.log_file)
        
        clf_start = time.time()
        clf = get_classifier("lgbm", self.cfg["data"]["random_seed"])
        clf.fit(emb_train, y_train)
        y_pred = clf.predict(emb_test)
        clf_time = time.time() - clf_start
        
        m = compute_metrics(y_test, y_pred, labels=labels)
        m["experiment"] = "C"
        m["classifier"] = "lgbm"
        m["clf_seconds"] = round(clf_time, 2)
        m["embedding_dim"] = emb_train.shape[1]
        
        save_metrics(m, "results/metrics/expC_lgbm.json")
        save_confusion_matrix(y_test, y_pred,
                             "results/metrics/expC_lgbm_cm.csv",
                             labels=labels)
        
        log(f"  OA={m['overall_accuracy']:.3f} | F1={m['macro_f1']:.3f} | "
            f"Kappa={m['kappa']:.3f}", self.log_file)
        
        return {"lgbm": m}
    
    def run_experiment_d(self, X_train, y_train, X_test, y_test,
                         labels: List, embeddings: Tuple = None) -> Dict:
        """
        Experiment D: Concatenated embeddings + features + LightGBM.
        """
        log("\n" + "=" * 60, self.log_file)
        log("EXPERIMENT D: Embeddings + Features + LightGBM", self.log_file)
        log("=" * 60, self.log_file)
        
        if embeddings:
            emb_train, emb_test = embeddings
        else:
            start = time.time()
            emb_train = self.extract_embeddings(X_train)
            emb_test = self.extract_embeddings(X_test)
            log(f"  Embeddings extracted in {time.time() - start:.1f}s", self.log_file)
        
        feat_train = combined_baseline_features(X_train)
        feat_test = combined_baseline_features(X_test)
        
        X_train_concat = np.concatenate([emb_train, feat_train], axis=1)
        X_test_concat = np.concatenate([emb_test, feat_test], axis=1)
        
        log(f"  Concatenated shape: train={X_train_concat.shape}, test={X_test_concat.shape}", self.log_file)
        
        clf_start = time.time()
        clf = get_classifier("lgbm", self.cfg["data"]["random_seed"])
        clf.fit(X_train_concat, y_train)
        y_pred = clf.predict(X_test_concat)
        clf_time = time.time() - clf_start
        
        m = compute_metrics(y_test, y_pred, labels=labels)
        m["experiment"] = "D"
        m["classifier"] = "lgbm"
        m["clf_seconds"] = round(clf_time, 2)
        m["total_dim"] = X_train_concat.shape[1]
        m["embedding_dim"] = emb_train.shape[1]
        m["feature_dim"] = feat_train.shape[1]
        
        save_metrics(m, "results/metrics/expD_concat_lgbm.json")
        save_confusion_matrix(y_test, y_pred,
                             "results/metrics/expD_concat_lgbm_cm.csv",
                             labels=labels)
        
        log(f"  OA={m['overall_accuracy']:.3f} | F1={m['macro_f1']:.3f} | "
            f"Kappa={m['kappa']:.3f}", self.log_file)
        
        return {"concat_lgbm": m}
    
    def run_all_experiments(self, experiment: str = "all") -> Dict:
        """Run specified experiments."""
        self.log_file = setup_logging("classification", self.cfg["output"]["metrics_dir"])
        
        log_header("Classification Experiments", self.log_file)
        start_time = time.time()
        
        X_train, y_train, X_test, y_test, labels = self.load_data()
        
        all_results = {}
        embeddings_cache = None
        
        if experiment in ["A", "all"]:
            all_results["A"] = self.run_experiment_a(
                X_train, y_train, X_test, y_test, labels
            )
        
        if experiment in ["B", "all"]:
            result_b = self.run_experiment_b(
                X_train, y_train, X_test, y_test, labels
            )
            all_results["B"] = {k: v for k, v in result_b.items() if k != "embeddings"}
            embeddings_cache = result_b.get("embeddings")
        
        if experiment in ["C", "all"]:
            all_results["C"] = self.run_experiment_c(
                X_train, y_train, X_test, y_test, labels, embeddings_cache
            )
        
        if experiment in ["D", "all"]:
            all_results["D"] = self.run_experiment_d(
                X_train, y_train, X_test, y_test, labels, embeddings_cache
            )
        
        self._print_summary(all_results)
        
        log_footer("Classification Experiments", start_time, self.log_file)
        
        return all_results
    
    def _print_summary(self, results: Dict):
        """Print comparison table."""
        log("\n" + "=" * 80, self.log_file)
        log("COMPARISON TABLE", self.log_file)
        log("=" * 80, self.log_file)
        log(f"{'Method':<35} {'Accuracy':>10} {'Macro F1':>10} {'Kappa':>10}", self.log_file)
        log("-" * 80, self.log_file)
        
        summary_rows = []
        
        for exp_name, exp_results in results.items():
            for method_name, metrics in exp_results.items():
                if isinstance(metrics, dict) and "overall_accuracy" in metrics:
                    row = {
                        "experiment": exp_name,
                        "method": method_name,
                        "accuracy": metrics["overall_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "kappa": metrics["kappa"]
                    }
                    summary_rows.append(row)
        
        for row in summary_rows:
            method_label = f"Exp {row['experiment']}: {row['method']}"
            log(f"{method_label:<35} {row['accuracy']:>10.3f} {row['macro_f1']:>10.3f} {row['kappa']:>10.3f}", self.log_file)
        
        log("-" * 80, self.log_file)
        
        if summary_rows:
            best = max(summary_rows, key=lambda x: x["macro_f1"])
            log(f"\nBest method: Exp {best['experiment']}: {best['method']} (F1={best['macro_f1']:.3f})", self.log_file)
        
        summary_path = os.path.join(self.cfg["output"]["metrics_dir"], "comparison_table.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_rows, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run classification experiments")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--experiment", choices=["A", "B", "C", "D", "all"], 
                       default="all", help="Experiment to run")
    args = parser.parse_args()
    
    runner = ClassificationExperiments(args.config)
    runner.run_all_experiments(args.experiment)


if __name__ == "__main__":
    main()

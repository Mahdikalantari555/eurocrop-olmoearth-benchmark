"""
Tests for src/utils/runner.py
"""

import pytest
import os
import yaml
import time
from src.utils.runner import load_config, setup_logging, log, log_header, log_footer, load_data


class TestLoadConfig:
    def test_loads_yaml(self, temp_dir):
        config = {
            "data": {"mode": "local"},
            "model": {"device": "cpu"},
            "output": {"metrics_dir": "./results"},
        }
        path = os.path.join(temp_dir, "config.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f)

        loaded = load_config(path)
        assert loaded["data"]["mode"] == "local"
        assert loaded["model"]["device"] == "cpu"

    def test_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestSetupLogging:
    def test_creates_log_file(self, temp_dir):
        log_file = setup_logging("phase1", temp_dir)
        assert os.path.exists(log_file)
        assert "phase1_log.txt" in log_file

    def test_creates_metrics_dir(self, temp_dir):
        metrics_dir = os.path.join(temp_dir, "results", "metrics")
        log_file = setup_logging("phase2", metrics_dir)
        assert os.path.exists(metrics_dir)
        assert os.path.exists(log_file)


class TestLog:
    def test_prints_message(self, capsys):
        log("Hello World")
        captured = capsys.readouterr()
        assert "Hello World" in captured.out

    def test_writes_to_file(self, temp_dir):
        log_file = os.path.join(temp_dir, "test.log")
        log("Test message", log_file)
        with open(log_file) as f:
            content = f.read()
        assert "Test message" in content
        assert "[" in content  # timestamp

    def test_appends_to_file(self, temp_dir):
        log_file = os.path.join(temp_dir, "test.log")
        log("First", log_file)
        log("Second", log_file)
        with open(log_file) as f:
            content = f.read()
        assert "First" in content
        assert "Second" in content


class TestLogHeader:
    def test_creates_header(self, temp_dir):
        log_file = os.path.join(temp_dir, "test.log")
        log_header("TEST PHASE", log_file)
        with open(log_file) as f:
            content = f.read()
        assert "TEST PHASE" in content
        assert "=" * 60 in content


class TestLogFooter:
    def test_creates_footer(self, temp_dir):
        log_file = os.path.join(temp_dir, "test.log")
        start = time.time() - 10
        log_footer("PHASE 1", start, log_file)
        with open(log_file) as f:
            content = f.read()
        assert "PHASE 1 COMPLETE" in content
        assert "Total time:" in content


class TestLoadData:
    def test_local_mode(self, temp_dir):
        import numpy as np

        # Create mock data structure
        preprocess_dir = os.path.join(temp_dir, "preprocess")
        split_dir = os.path.join(temp_dir, "split")
        os.makedirs(preprocess_dir)
        os.makedirs(split_dir)

        # Create split JSON
        use_case = "test_case"
        finetune_dir = os.path.join(split_dir, use_case, "finetune")
        os.makedirs(finetune_dir)

        npz_files = []
        for i in range(10):
            fname = f"region_{i}_class{i % 3}.npz"
            data = np.random.rand(6, 13).astype(np.float32)
            np.savez(os.path.join(preprocess_dir, fname), data=data)
            npz_files.append(fname)

        import json
        split_data = {
            "train": npz_files[:7],
            "val": npz_files[7:9],
            "test": npz_files[9:],
        }
        with open(os.path.join(finetune_dir, "region_split_all.json"), "w") as f:
            json.dump(split_data, f)

        cfg = {
            "data": {
                "mode": "local",
                "local_preprocess_dir": preprocess_dir,
                "local_split_dir": split_dir,
                "use_case": use_case,
                "use_zenodo": True,
            }
        }

        log_file = os.path.join(temp_dir, "test.log")
        X_train, y_train, X_test, y_test = load_data(cfg, log_file)
        assert len(X_train) > 0
        assert len(X_test) > 0

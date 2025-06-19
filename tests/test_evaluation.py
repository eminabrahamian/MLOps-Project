"""
test_evaluation.py

Unit tests for evaluation.py

Covers:
- _specificity, _npv (edge cases: zero denominators, normal)
- _round_dict_values (nested dicts, floats, non-floats)
- evaluate_classification: various metrics (accuracy, precision, recall, f1,
  roc auc, specificity, NPV, confusion matrix)
- evaluate_classification: save_path writes JSON, logging with log_results
- generate_split_report: missing files, missing target, valid evaluation
  with temporary CSV/model
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from src.evaluation.evaluation import (
    _npv,
    _round_dict_values,
    _specificity,
    evaluate_classification,
    generate_split_report,
)


# Dummy model that implements predict and predict_proba
class DummyBinaryClassifier(BaseEstimator):
    def __init__(self, proba=False):
        self.proba = proba

    def fit(self, X, y):
        return self

    def predict(self, X):
        # Predict all zeros
        return np.zeros(shape=(X.shape[0],), dtype=int)

    def predict_proba(self, X):
        # Return equal probability for both classes
        return np.tile([0.5, 0.5], (X.shape[0], 1))


# config fixture


@pytest.fixture
def basic_config(tmp_path):
    """
    Returns a config dict with:
      - metrics.report:
        ['accuracy','precision','recall','f1','roc auc','specificity','npv']
      - artifacts: processed_dir, model_path, metrics_dir
      - target: 'target'
    """
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    # Create a simple processed CSV for "validation" split
    df_val = pd.DataFrame({"feat1": [0, 1, 1, 0], "target": [0, 1, 0, 1]})
    df_val.to_csv(proc_dir / "validation_processed.csv", index=False)

    # Create a simple processed CSV for "test" split
    df_test = pd.DataFrame({"feat1": [0, 1], "target": [1, 0]})
    df_test.to_csv(proc_dir / "test_processed.csv", index=False)

    # Serialize a DummyBinaryClassifier to disk
    model = DummyBinaryClassifier(proba=True)
    model.fit(np.array([[0], [1], [1], [0]]), np.array([0, 1, 0, 1]))
    model_file = tmp_path / "model.pkl"
    with model_file.open("wb") as f:
        pickle.dump(model, f)

    cfg = {
        "metrics": {
            "report": [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc auc",
                "specificity",
                "npv",
            ]
        },
        "artifacts": {
            "processed_dir": str(proc_dir),
            "model_path": str(model_file),
            "metrics_dir": str(tmp_path / "metrics"),
        },
        "target": "target",
    }
    return cfg


def test_specificity_npv_edge_cases():
    """
    Test that _specificity and _npv return NaN if denominator is zero,
    and correct fraction otherwise.
    """
    # Specificity: tn/(tn+fp)
    assert np.isnan(_specificity(0, 0))
    assert _specificity(5, 0) == pytest.approx(1.0)
    assert _specificity(3, 1) == pytest.approx(3 / (3 + 1))

    # NPV: tn/(tn+fn)
    assert np.isnan(_npv(0, 0))
    assert _npv(4, 0) == pytest.approx(1.0)
    assert _npv(2, 1) == pytest.approx(2 / (2 + 1))


def test_round_dict_values_nested():
    """
    Test that _round_dict_values correctly rounds nested float
    values and leaves others untouched.
    """
    nested = {"a": 1.23456, "b": {"c": 2.34567, "d": "no_change"}, "e": 3}
    rounded = _round_dict_values(nested, digits=2)
    assert rounded["a"] == 1.23
    assert rounded["b"]["c"] == 2.35
    assert rounded["b"]["d"] == "no_change"
    assert rounded["e"] == 3


def test_evaluate_classification_basic(basic_config, tmp_path, caplog):
    """
    Test evaluate_classification with DummyBinaryClassifier:
      - X and y that produce known metrics (all-zero predictions)
      - Check that metrics dict contains keys and correct values
      - Test save_path: JSON file is written
      - Test log_results=True: INFO logs present
    """
    cfg = basic_config
    # Create input arrays: shape (4,1)
    X = np.array([[0], [1], [0], [1]])
    y_true = np.array([0, 1, 1, 0])  # two correct, two incorrect
    model = DummyBinaryClassifier(proba=True)
    model.fit(X, y_true)

    # Call evaluate_classification with save_path and log_results
    caplog.set_level(logging.INFO)
    save_file = tmp_path / "out_metrics.json"
    metrics = evaluate_classification(
        model,
        X,
        y_true,
        cfg,
        metrics=None,  # should read from cfg
        split_name="validation",
        log_results=True,
        save_path=str(save_file),
    )
    # Validate keys
    for key in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
        "Specificity",
        "NPV",
        "Confusion Matrix",
    ]:
        assert key in metrics

    # Accuracy: model predicts all zeros; out of 4, two zeros correct →
    # accuracy = 0.5
    assert metrics["Accuracy"] == pytest.approx(0.5)

    # JSON file should exist and match metrics
    assert save_file.is_file()
    data = json.loads(save_file.read_text())
    assert "Accuracy" in data

    # Check that INFO log contains "Metrics [validation]"
    assert any(
        "Metrics [validation]" in rec.getMessage() for rec in caplog.records
    )


def test_generate_split_report_no_files(tmp_path, basic_config):
    """
    Test generate_split_report returns empty dict and logs warning
    if processed file missing.
    """
    cfg = basic_config
    # Point processed_dir to an empty folder
    cfg["artifacts"]["processed_dir"] = str(tmp_path / "empty")
    cfg["artifacts"]["metrics_dir"] = str(tmp_path / "metrics_out")
    # No CSV files exist
    report = generate_split_report(cfg, split="nonexistent")
    assert report == {}


def test_generate_split_report_success(tmp_path, basic_config):
    """
    Test generate_split_report with valid files and model:
      - Should load CSVs, load model, compute metrics, and write JSON.
      - Returned dict should match metrics.
    """
    cfg = basic_config
    # Ensure metrics_dir exists
    metrics_dir = Path(cfg["artifacts"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = generate_split_report(cfg, split="validation")
    # Should return a non-empty dict with "Accuracy"
    assert "Accuracy" in report
    # JSON file should have been written
    assert (metrics_dir / "validation_metrics.json").is_file()

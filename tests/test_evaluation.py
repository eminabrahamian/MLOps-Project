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
import io

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


# Dummy model with predict and predict_proba
class DummyBinaryClassifier(BaseEstimator):
    def __init__(self, proba=False):
        self.proba = proba

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(shape=(X.shape[0],), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (X.shape[0], 1))


class NoProbaClassifier(BaseEstimator):
    def predict(self, X):
        return np.ones(X.shape[0])


@pytest.fixture
def basic_config(tmp_path):
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    df_val = pd.DataFrame({"feat1": [0, 1, 1, 0], "target": [0, 1, 0, 1]})
    df_val.to_excel(proc_dir / "validation_processed.xlsx", index=False)

    df_test = pd.DataFrame({"feat1": [0, 1], "target": [1, 0]})
    df_test.to_excel(proc_dir / "test_processed.xlsx", index=False)

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
    assert np.isnan(_specificity(0, 0))
    assert _specificity(5, 0) == pytest.approx(1.0)
    assert _specificity(3, 1) == pytest.approx(3 / (3 + 1))

    assert np.isnan(_npv(0, 0))
    assert _npv(4, 0) == pytest.approx(1.0)
    assert _npv(2, 1) == pytest.approx(2 / (2 + 1))


def test_round_dict_values_nested():
    nested = {"a": 1.23456, "b": {"c": 2.34567, "d": "no_change"}, "e": 3}
    rounded = _round_dict_values(nested, digits=2)
    assert rounded["a"] == 1.23
    assert rounded["b"]["c"] == 2.35
    assert rounded["b"]["d"] == "no_change"
    assert rounded["e"] == 3


def test_evaluate_classification_basic(basic_config, tmp_path, caplog):
    cfg = basic_config
    X = np.array([[0], [1], [0], [1]])
    y_true = np.array([0, 1, 1, 0])
    model = DummyBinaryClassifier(proba=True)
    model.fit(X, y_true)

    caplog.set_level(logging.INFO)
    save_file = tmp_path / "out_metrics.json"
    metrics = evaluate_classification(
        model,
        X,
        y_true,
        cfg,
        metrics=None,
        split_name="validation",
        log_results=True,
        save_path=str(save_file),
    )

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

    assert metrics["Accuracy"] == pytest.approx(0.5)
    assert save_file.is_file()
    data = json.loads(save_file.read_text())
    assert "Accuracy" in data
    assert any("Metrics [validation]" in rec.getMessage()
               for rec in caplog.records)


def test_generate_split_report_no_files(tmp_path, basic_config):
    cfg = basic_config
    cfg["artifacts"]["processed_dir"] = str(tmp_path / "empty")
    cfg["artifacts"]["metrics_dir"] = str(tmp_path / "metrics_out")
    report = generate_split_report(cfg, split="nonexistent")
    assert report == {}


def test_generate_split_report_success(tmp_path, basic_config):
    cfg = basic_config
    metrics_dir = Path(cfg["artifacts"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = generate_split_report(cfg, split="validation")
    assert "Accuracy" in report
    assert (metrics_dir / "validation_metrics.json").is_file()


# ---------- NEW TESTS TO INCREASE COVERAGE ----------

def test_roc_auc_missing_predict_proba(basic_config):
    model = NoProbaClassifier()
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 1])
    cfg = basic_config
    result = evaluate_classification(model, X, y, cfg, metrics=["roc auc"])
    assert np.isnan(result["ROC AUC"])


def test_confusion_matrix_one_class_only(basic_config):
    model = DummyBinaryClassifier()
    X = np.array([[0], [0], [0]])
    y = np.array([0, 0, 0])  # only negatives
    result = evaluate_classification(model, X, y, basic_config,
                                     metrics=["confusion matrix"])
    cm = result["Confusion Matrix"]
    assert cm["fp"] == 0 and cm["fn"] == 0 and cm["tp"] == 0


def test_save_path_exception(monkeypatch, basic_config):
    model = DummyBinaryClassifier()
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    bad_path = Path("/invalid/dir/metrics.json")

    def bad_mkdir(*args, **kwargs):
        raise OSError("fail mkdir")

    monkeypatch.setattr("pathlib.Path.mkdir", bad_mkdir)

    with pytest.raises(OSError):
        evaluate_classification(model, X, y, basic_config,
                                save_path=str(bad_path))


def test_generate_split_report_missing_target(tmp_path, basic_config):
    cfg = basic_config
    file = (Path(cfg["artifacts"]["processed_dir"]) /
            "validation_processed.xlsx")
    pd.DataFrame({"feat1": [0, 1, 0]}).to_excel(file, index=False)
    result = generate_split_report(cfg, split="validation")
    assert result == {}


def test_generate_split_report_bad_model_file(tmp_path, basic_config):
    cfg = basic_config
    model_path = Path(cfg["artifacts"]["model_path"])
    model_path.write_text("not a pickle")
    result = generate_split_report(cfg, split="validation")
    assert result == {}


def test_confusion_matrix_single_class_skips_conf_matrix(basic_config):
    model = DummyBinaryClassifier()
    X = np.array([[0], [1], [2]])
    y = np.array([1, 1, 1])  # only one class present
    result = evaluate_classification(model, X, y, basic_config,
                                     metrics=["confusion matrix"])
    assert result["Confusion Matrix"] == {"tn": 0, "fp": 0, "fn": 3, "tp": 0}


def test_round_dict_values_complex_structure():
    nested = {
        "a": 1.23456789,
        "b": {
            "c": 3.141592,
            "d": [1.98765, "no_round"],  # stays unchanged
            "e": {
                "f": 2.71828,
                "g": {"h": 0.333333},
            },
        },
        "x": "string",
    }
    rounded = _round_dict_values(nested, digits=3)
    assert rounded["a"] == 1.235
    assert rounded["b"]["c"] == 3.142
    assert rounded["b"]["d"] == [1.98765, "no_round"]  # no change
    assert rounded["b"]["e"]["f"] == 2.718
    assert rounded["b"]["e"]["g"]["h"] == 0.333


def test_evaluate_classification_save_path_failure(
        tmp_path, basic_config, monkeypatch):
    model = DummyBinaryClassifier()
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    save_file = tmp_path / "metrics.json"

    def fail_open(*args, **kwargs):
        raise IOError("fail write")

    monkeypatch.setattr("builtins.open", lambda *a, **kw: fail_open())

    result = evaluate_classification(model, X, y, basic_config,
                                     save_path=str(save_file))
    assert "Accuracy" in result  # still returns metrics


def test_round_dict_values_deep_dict():
    input_dict = {
        "outer": {
            "inner": {
                "number": 1.23456,
                "text": "skip",
                "list": [1.11111, "text"]
            },
            "nested": {
                "float": 9.87654,
                "none": None
            }
        }
    }
    rounded = _round_dict_values(input_dict, digits=2)
    assert rounded["outer"]["inner"]["number"] == 1.23
    assert rounded["outer"]["inner"]["text"] == "skip"
    assert rounded["outer"]["nested"]["float"] == 9.88
    assert rounded["outer"]["nested"]["none"] is None


def test_evaluate_classification_json_write_failure(
        monkeypatch, tmp_path, basic_config):
    model = DummyBinaryClassifier()
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    target_file = tmp_path / "fail.json"

    # Simulate failure when trying to open file for writing
    def fail_open(*args, **kwargs):
        raise IOError("Write failure")

    monkeypatch.setattr("builtins.open", lambda *a, **kw: fail_open())

    result = evaluate_classification(model, X, y, basic_config,
                                     save_path=str(target_file))
    assert "Accuracy" in result  # still returns result despite write failure


def test_json_dump_write_failure(monkeypatch, tmp_path, basic_config):
    model = DummyBinaryClassifier()
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    save_path = tmp_path / "failing_metrics.json"

    class FailingFile(io.StringIO):
        def write(self, *args, **kwargs):
            raise IOError("Write failed")

    def failing_open(*args, **kwargs):
        return FailingFile()

    monkeypatch.setattr("builtins.open", failing_open)

    result = evaluate_classification(model, X, y, basic_config,
                                     save_path=str(save_path))
    assert "Accuracy" in result


def test_rounding_and_logging_triggered(basic_config, caplog):
    model = DummyBinaryClassifier()
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    caplog.set_level("INFO")

    result = evaluate_classification(
        model,
        X,
        y,
        basic_config,
        log_results=True,
        split_name="validation"
    )

    assert "Metrics [validation]" in caplog.text
    assert "Accuracy" in result

"""
test_model.py

Unit tests for model.py

Covers:
- train_model (valid KNN, invalid type)
- save_artifact (successful write, permission error)
- format_metrics (rounding floats, leaving ints/strings)
- run_model_pipeline integration on a minimal DataFrame:
  - Splitting, pipeline building, training, saving artifacts,
    evaluating metrics
  - Uses tmp_path for all file outputs to avoid polluting project
"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier

from src.model.model import (
    format_metrics,
    run_model_pipeline,
    save_artifact,
    train_model,
)


@pytest.fixture
def minimal_train_config(tmp_path):
    # Balanced dataset (4 per class) to support stratified splitting
    data = pd.DataFrame(
        {
            "x1": list(range(1, 13)),
            "x2": list(range(12, 0, -1)),
            "target": [0, 1] * 6,  # 6 zeros, 6 ones
        }
    )

    cfg = {
        "raw_features": ["x1", "x2"],
        "target": "target",
        "data_split": {
            "test_size": 0.25,
            "valid_size": 0.25,
            "random_state": 0,
            "stratify": True,
        },
        "preprocessing": {
            "rename_columns": {},
            "weight_col": None,
            "height_col": None,
            "icd10_chapter_flags": [],
            "interaction_columns": [],
            "outlier_columns": [],
            "z_threshold": 3.0,
            "datetime_column": None,
        },
        "features": {
            "continuous": ["x1", "x2"],
            "categorical": [],
            "feature_columns": ["x1", "x2"],
        },
        "model": {
            "type": "knn",
            "knn": {
                "params": {
                    "n_neighbors": 1,
                    "weights": "uniform",
                    "metric": "minkowski",
                },
                "save_path": str(tmp_path / "extra_model.pkl"),
            },
            "save_path": str(tmp_path / "model.pkl"),
        },
        "artifacts": {
            "splits_dir": str(tmp_path / "splits"),
            "processed_dir": str(tmp_path / "processed"),
            "preprocessing_pipeline": str(tmp_path / "pipe.pkl"),
            "model_path": str(tmp_path / "model.pkl"),
            "metrics_dir": str(tmp_path / "metrics"),
        },
    }

    return data, cfg


def test_train_model_valid_and_invalid():
    X = pd.DataFrame({"a": [1, 2], "b": [2, 1]})
    y = pd.Series([0, 1])
    model = train_model(
        X,
        y,
        "knn",
        {"n_neighbors": 1, "weights": "uniform", "metric": "euclidean"},
    )
    assert isinstance(model, KNeighborsClassifier)

    with pytest.raises(ValueError):
        train_model(X, y, "not_supported", {})


def test_format_metrics_rounding():
    metrics = {
        "accuracy": 0.98765,
        "precision": 1.0,
        "label": "ok",
        "raw_count": 42,
    }
    rounded = format_metrics(metrics, ndigits=2)
    assert rounded["accuracy"] == 0.99
    assert rounded["precision"] == 1.0
    assert rounded["label"] == "ok"
    assert rounded["raw_count"] == 42


def test_save_artifact_success_and_fail(tmp_path):
    obj = {"a": 1}
    path = tmp_path / "model.pkl"
    save_artifact(obj, str(path))
    assert path.exists()

    # Fail case (e.g., invalid dir)
    with pytest.raises(Exception):
        save_artifact(obj, "/invalid_dir/save.pkl")


def test_run_model_pipeline_end_to_end(minimal_train_config):
    df, cfg = minimal_train_config
    run_model_pipeline(df, cfg)

    # Assert model & pipeline saved
    assert Path(cfg["model"]["save_path"]).exists()
    assert Path(cfg["artifacts"]["model_path"]).exists()
    assert Path(cfg["artifacts"]["preprocessing_pipeline"]).exists()

    # Assert processed data exists
    for split in ["train", "valid", "test"]:
        assert (
            Path(cfg["artifacts"]["processed_dir"]) / f"{split}_processed.xlsx"
        ).exists()
        assert (Path(cfg["artifacts"]["splits_dir"]) / f"{split}_raw.xlsx").exists()


def test_run_model_pipeline_invalid_model(minimal_train_config):
    df, cfg = minimal_train_config
    cfg["model"]["type"] = "svm"  # not in MODEL_REGISTRY
    with pytest.raises(ValueError):
        run_model_pipeline(df, cfg)


def test_run_model_pipeline_missing_target(minimal_train_config):
    df, cfg = minimal_train_config
    df = df.drop(columns=["target"])
    with pytest.raises(ValueError):
        run_model_pipeline(df, cfg)


def test_run_model_pipeline_with_model_override(monkeypatch, minimal_train_config):
    df, cfg = minimal_train_config

    # Provide model-specific params at incorrect level
    cfg["model"]["model"] = {
        "params": {"n_neighbors": 1, "weights": "uniform", "metric": "minkowski"}
    }

    run_model_pipeline(df, cfg)

    # Ensure model still saved
    assert Path(cfg["model"]["save_path"]).exists()

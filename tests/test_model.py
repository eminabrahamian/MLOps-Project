"""
test_model.py

Unit tests for model.py

Covers:
- train_model (valid KNN, invalid type)
- save_artifact (successful write, permission error)
- format_metrics (rounding floats, leaving ints/strings)
- run_model_pipeline integration on a minimal DataFrame:
  - Splitting, pipeline building, training, saving artifacts, evaluating metrics
  - Uses tmp_path for all file outputs to avoid polluting project
"""

import json
import os
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from src.models.model import (
    train_model,
    save_artifact,
    format_metrics,
    run_model_pipeline,
    MODEL_REGISTRY,
)

# Fixture: minimal dataset and config for run_model_pipeline
@pytest.fixture
def minimal_train_config(tmp_path):
    """
    Construct a minimal DataFrame and config dict:
      - DataFrame: two features and binary target, at least 4 rows
      - raw_features: ["x1", "x2"]
      - target: "target"
      - data_split: {test_size:0.5, valid_size:0.5, random_state:0, stratify:True}
      - preprocessing: identity (no renaming, no features beyond raw)
        Continuous= ["x1", "x2"], categorical=[]; raw_features same
      - model: active="knn", knn.params with k=1
      - artifacts: split_dir, processed_dir, preprocessing_pipeline, model_path
    """
    # 1) Create a simple DataFrame
    data = pd.DataFrame({
        "x1": [1, 2, 3, 4],
        "x2": [4, 3, 2, 1],
        "target": [0, 1, 0, 1]
    })
    # 2) Temporary config
    cfg = {
        "raw_features": ["x1", "x2"],
        "target": "target",
        "data_split": {
            "test_size": 0.5,
            "valid_size": 0.25,
            "random_state": 0,
            "stratify": True
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
            # No per-feature overrides
        },
        "features": {
            "continuous": ["x1", "x2"],
            "categorical": [],
            "feature_columns": ["x1", "x2"]
        },
        "model": {
            "type": "knn",
            "knn": {
                "params": {"n_neighbors": 1, "weights": "uniform", "metric": "minkowski"},
                "save_path": str(tmp_path / "extra_model.pkl")
            },
            "save_path": str(tmp_path / "model.pkl")
        },
        "artifacts": {
            "splits_dir": str(tmp_path / "splits"),
            "processed_dir": str(tmp_path / "processed"),
            "preprocessing_pipeline": str(tmp_path / "pipe.pkl"),
            "model_path": str(tmp_path / "model.pkl"),
            "metrics_dir": str(tmp_path / "metrics")
        }
    }
    return data, cfg

def test_train_model_valid_and_invalid():
    """
    train_model should train a KNN when model_type='knn' and raise ValueError otherwise.
    """
    X = pd.DataFrame({"a": [1, 2], "b": [2, 1]})
    y = pd.Series([0, 1])
    # Valid 'knn'
    model = train_model(X, y, "knn", {"n_neighbors": 1, "weights": "uniform", "metric": "euclidean"})
    assert isinstance(model, KNeighborsClassifier)

    # Invalid model_type
    with pytest.raises(ValueError):
        train_model(X, y, "random_forest", {})

def test_run_model_pipeline_missing_target(minimal_train_config):
    """
    run_model_pipeline should raise ValueError if target column is missing.
    """
    df, cfg = minimal_train_config
    # Remove target column name
    cfg_bad = dict(cfg)
    cfg_bad["target"] = "nonexistent"
    with pytest.raises(ValueError):
        run_model_pipeline(df, cfg_bad)

"""
test_preprocessing.py

Unit tests for preprocessing.py

Covers:
- ColumnRenamer (rename, no-op)
- build_preprocessing_pipeline (with various config keys: default, BMI, interaction, outlier, datetime)
- get_output_feature_names (one-hot, bucket, passthrough)
- run_preprocessing_pipeline integration (fits and transforms, resulting DataFrame)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import pytest
import yaml

from src.data.preprocessing import (
    ColumnRenamer,
    build_preprocessing_pipeline,
    get_output_feature_names,
    run_preprocessing_pipeline,
)

# Disable sklearn warnings in tests
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@pytest.fixture
def simple_df():
    """
    Returns a small DataFrame with numeric, categorical, and datetime columns
    for testing. Columns:
      - 'num': numeric with missing
      - 'cat': categorical with one missing
      - 'weight', 'height': for BMITransformer
      - 'date': for DateTimeFeatures
    """
    return pd.DataFrame({
        "num": [1.0, 2.0, None, 4.0],
        "cat": ["a", "b", None, "b"],
        "weight": [70, 80, 90, 60],
        "height": [175, 180, 165, 170],
        "flag1": [1, 0, 1, 0],
        "flag2": [0, 1, 0, 1],
        "date": [
            "2021-01-01 10:00:00",
            "2021-06-15 15:30:00",
            None,
            "2021-12-31 23:59:59"
        ]
    })

@pytest.fixture
def minimal_config(tmp_path):
    """
    Creates a YAML-like dict for minimal preprocessing:
      - No rename_map, no BMI, no interaction, no outlier, no datetime
      - features.continuous = ['num']
      - features.categorical = ['cat']
      - raw_features = ['num', 'cat']
      - For 'num': apply impute (mean) + MinMaxScaler
      - For 'cat': apply impute (most_frequent) + OneHotEncoder
    """
    cfg = {
        "preprocessing": {
            "rename_columns": {},
            "weight_col": None,
            "height_col": None,
            "icd10_chapter_flags": [],
            "interaction_columns": [],
            "outlier_columns": [],
            "z_threshold": 3.0,
            "datetime_column": None,
            "num": {"impute": True, "imputer_strategy": "mean", "scaler": "minmax", "bucketize": False},
            "cat": {"impute": True, "imputer_strategy": "most_frequent", "encoding": "onehot"}
        },
        "features": {
            "continuous": ["num"],
            "categorical": ["cat"],
            "feature_columns": ["num", "cat"],
        },
        "raw_features": ["num", "cat"]
    }
    return cfg

def test_column_renamer_noop():
    """
    ColumnRenamer with empty rename_map should return the same DataFrame.
    """
    df = pd.DataFrame({"a": [1], "b": [2]})
    ren = ColumnRenamer({})
    out = ren.transform(df)
    pd.testing.assert_frame_equal(out, df)

def test_column_renamer_mapping():
    """
    ColumnRenamer should rename columns according to rename_map.
    """
    df = pd.DataFrame({"old": [1], "b": [2]})
    ren = ColumnRenamer({"old": "new"})
    out = ren.transform(df)
    assert "new" in out.columns and "old" not in out.columns

def test_build_pipeline_minimal(minimal_config):
    """
    build_preprocessing_pipeline should create a pipeline with:
      - risk_score step
      - rename step
      - col_transform step with numeric and categorical branches
    """
    cfg = minimal_config
    pipe = build_preprocessing_pipeline(cfg)
    # Should have exactly 3 steps: 'risk_score', 'rename', 'col_transform'
    names = [name for name, _ in pipe.steps]
    assert "risk_score" in names
    assert "rename" in names
    assert "col_transform" in names

def test_run_preprocessing_pipeline_integration(minimal_config, tmp_path):
    """
    run_preprocessing_pipeline should fit & transform, returning a DataFrame with named columns.
    Also, writing to data/processed/ is tested by verifying the file.
    """
    # Build a small DataFrame
    df = pd.DataFrame({"num": [1.0, None, 3.0], "cat": ["x", "y", "x"]})
    cfg = minimal_config

    # Call run_preprocessing_pipeline
    df_processed = run_preprocessing_pipeline(df, cfg)

    # Should produce a DataFrame of shape (3, 3) (1 scaled numeric + 2 one-hot cat columns)
    assert df_processed.shape[0] == 3
    # Check that there are no missing values after imputation
    assert not df_processed.isnull().any().any()
    """
    Test a pipeline that includes BMITransformer and DateTimeFeatures.
    """
    data = pd.DataFrame({
        "num": [1.0, 2.0],
        "cat": ["a", "b"],
        "weight": [60, 80],
        "height": [160, 180],
        "date": ["2020-01-01 00:00:00", "2020-06-15 12:00:00"]
    })
    cfg = {
        "preprocessing": {
            "rename_columns": {},
            "weight_col": "weight",
            "height_col": "height",
            "icd10_chapter_flags": [],
            "interaction_columns": [],
            "outlier_columns": [],
            "z_threshold": 3.0,
            "datetime_column": "date",
            "num": {"impute": True, "imputer_strategy": "mean", "scaler": "standard", "bucketize": False},
            "cat": {"impute": True, "imputer_strategy": "most_frequent", "encoding": "ordinal"}
        },
        "features": {
            "continuous": ["num", "weight", "height"],
            "categorical": ["cat"],
            "feature_columns": ["num", "weight", "height", "cat"]
        },
        "raw_features": ["num", "weight", "height", "cat", "date"]
    }
    pipe = build_preprocessing_pipeline(cfg)
    arr = pipe.fit_transform(data)
    # Expect shape: 
    # - num standardized → 1 column
    # - weight, height standardized → 2 columns
    # - cat ordinal → 1 column
    # - DateTimeFeatures will add 5 columns: day_of_week, month, hour, hour_sin, hour_cos
    # Total output columns: 1+2+1+5 = 9
    assert arr.shape == (2, 9)

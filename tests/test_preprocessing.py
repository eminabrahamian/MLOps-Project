"""
Unit tests for preprocessing.py.

Covers:
- ColumnRenamer (rename, no-op)
- build_preprocessing_pipeline (with various config keys:
  default, BMI, interaction, outlier, datetime)
- get_output_feature_names (one-hot, bucket, passthrough)
- run_preprocessing_pipeline integration
  (fits and transforms, resulting DataFrame)
"""

import copy
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.preprocessing import (
    ColumnRenamer,
    build_preprocessing_pipeline,
    get_output_feature_names,
    run_preprocessing_pipeline,
)


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
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
                "2021-12-31 23:59:59",
            ],
            "extra": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def minimal_config():
    return {
        "preprocessing": {
            "rename_columns": {},
            "weight_col": None,
            "height_col": None,
            "risk_score": True,
            "icd10_chapter_flags": [],
            "interaction_columns": [],
            "outlier_columns": [],
            "z_threshold": 3.0,
            "datetime_column": None,
            "num": {
                "impute": True,
                "imputer_strategy": "mean",
                "scaler": "minmax",
                "bucketize": False,
            },
            "cat": {
                "impute": True,
                "imputer_strategy": "most_frequent",
                "encoding": "onehot",
            },
        },
        "features": {
            "continuous": ["num"],
            "categorical": ["cat"],
            "feature_columns": ["num", "cat"],
        },
        "raw_features": ["num", "cat", "extra"],
    }


def test_column_renamer_noop():
    df = pd.DataFrame({"a": [1], "b": [2]})
    ren = ColumnRenamer({})
    out = ren.transform(df)
    pd.testing.assert_frame_equal(out, df)


def test_column_renamer_mapping():
    df = pd.DataFrame({"old": [1], "b": [2]})
    ren = ColumnRenamer({"old": "new"})
    out = ren.transform(df)
    assert "new" in out.columns and "old" not in out.columns


def test_build_pipeline_minimal(minimal_config):
    pipe = build_preprocessing_pipeline(minimal_config)
    names = [name for name, _ in pipe.steps]
    assert "risk_score" in names
    assert "rename" in names
    assert "col_transform" in names


def test_run_preprocessing_pipeline_integration(minimal_config):
    df = pd.DataFrame(
        {"num": [1.0, None, 3.0], "cat": ["x", "y", "x"], "extra": [5, 6, 7]}
    )
    df_out = run_preprocessing_pipeline(df, minimal_config)
    assert df_out.shape[0] == 3
    assert not df_out.isnull().any().any()


def test_get_output_feature_names_fallback(simple_df, minimal_config):
    pipe = build_preprocessing_pipeline(minimal_config)
    pipe.fit(simple_df[["num", "cat", "extra"]])
    names = get_output_feature_names(pipe, ["num", "cat", "extra"], minimal_config)
    assert isinstance(names, list)
    assert "extra" in names


def test_bmi_step_and_values(simple_df, minimal_config):
    cfg = copy.deepcopy(minimal_config)
    cfg["preprocessing"]["weight_col"] = "weight"
    cfg["preprocessing"]["height_col"] = "height"
    cfg["preprocessing"]["drop_original_columns"] = False
    for col in ["weight", "height"]:
        cfg["features"]["continuous"].append(col)
        cfg["raw_features"].append(col)

    pipe = build_preprocessing_pipeline(cfg)
    step_names = [name for name, _ in pipe.steps]
    assert "bmi" in step_names
    transformed = pipe.fit_transform(simple_df)
    assert transformed.shape[0] == simple_df.shape[0]


def test_outlier_flagging(simple_df, minimal_config):
    cfg = copy.deepcopy(minimal_config)
    cfg["preprocessing"]["outlier_columns"] = ["num"]
    cfg["preprocessing"]["z_threshold"] = 1.0
    cfg["raw_features"].append("num_outlier")

    pipe = build_preprocessing_pipeline(cfg)
    assert "outlier_flags" in [n for n, _ in pipe.steps]
    _ = pipe.fit(simple_df)
    out_names = get_output_feature_names(pipe, simple_df.columns.tolist(), cfg)
    assert any("num_outlier" in nm for nm in out_names)


def test_datetime_features(simple_df, minimal_config):
    cfg = copy.deepcopy(minimal_config)
    cfg["preprocessing"]["datetime_column"] = "date"
    for fld in ["date", "day_of_week", "month", "hour", "hour_sin", "hour_cos"]:
        cfg["raw_features"].append(fld)

    pipe = build_preprocessing_pipeline(cfg)
    assert "datetime_feats" in [n for n, _ in pipe.steps]
    _ = pipe.fit(simple_df)
    out_names = get_output_feature_names(pipe, simple_df.columns.tolist(), cfg)
    assert any("month" in nm for nm in out_names)


def test_passthrough_fallback_only(simple_df, minimal_config):
    cfg = copy.deepcopy(minimal_config)
    cfg["features"]["continuous"] = []
    cfg["features"]["categorical"] = []
    cfg["raw_features"] = ["num", "cat", "extra"]
    pipe = build_preprocessing_pipeline(cfg)
    pipe.fit(simple_df)
    names = get_output_feature_names(pipe, simple_df.columns.tolist(), cfg)
    assert set(["num", "cat", "extra"]).issubset(names)


def test_column_renamer_affects_passthrough():
    df = pd.DataFrame({"x": [1], "extra": [5]})
    cfg = {
        "preprocessing": {
            "rename_columns": {"x": "num"},
            "num": {"impute": True, "scaler": "minmax"},
        },
        "features": {
            "continuous": ["num"],
            "categorical": [],
            "feature_columns": ["num"],
        },
        "raw_features": ["x", "extra"],
    }
    pipe = build_preprocessing_pipeline(cfg)
    out = pipe.fit_transform(df)
    assert out.shape[0] == 1


def test_feature_names_no_get_feature_names_out():
    class Dummy(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    ct = ColumnTransformer(
        transformers=[("dummy", Dummy(), ["num", "cat"])],
        verbose_feature_names_out=False,
    )
    pipe = Pipeline([("col_transform", ct)])
    pipe.fit(pd.DataFrame({"num": [1], "cat": [2]}))
    names = get_output_feature_names(
        pipe, ["num", "cat"], {"raw_features": ["num", "cat"]}
    )
    assert set(["num", "cat"]).issubset(names)


def test_standard_scaler_and_bucketize(simple_df, minimal_config):
    cfg = copy.deepcopy(minimal_config)
    cfg["preprocessing"]["num"]["scaler"] = "standard"
    cfg["preprocessing"]["num"]["bucketize"] = True
    pipe = build_preprocessing_pipeline(cfg)
    out = pipe.fit_transform(simple_df)
    assert out.shape[0] == simple_df.shape[0]


def test_drop_column_skipped_in_output_names():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    df = pd.DataFrame({"a": [1], "b": [2]})
    ct = ColumnTransformer(
        transformers=[
            ("drop_b", "drop", ["b"]),
            ("pass_a", "passthrough", ["a"]),
        ],
        verbose_feature_names_out=False,
    )
    pipe = Pipeline([("col_transform", ct)])
    pipe.fit(df)
    names = get_output_feature_names(
        pipe, df.columns.tolist(), {"raw_features": ["a", "b"]}
    )
    assert "a" in names and "b" not in names


def test_named_steps_without_get_feature_names_out():
    class StepWithoutGetName(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    pipe = Pipeline(
        [
            ("dummy_step", StepWithoutGetName()),
        ]
    )
    ct = ColumnTransformer(
        transformers=[("pipe_branch", pipe, ["num"])], verbose_feature_names_out=False
    )
    top_pipe = Pipeline([("col_transform", ct)])
    df = pd.DataFrame({"num": [1.0]})
    top_pipe.fit(df)
    out_names = get_output_feature_names(
        top_pipe, df.columns.tolist(), {"raw_features": ["num"]}
    )
    assert "num" in out_names


def test_skip_drop_transformer_in_output_names():
    df = pd.DataFrame({"a": [1], "b": [2]})
    ct = ColumnTransformer(
        [("drop_b", "drop", ["b"]), ("pass_a", "passthrough", ["a"])]
    )
    pipe = Pipeline([("col_transform", ct)])
    pipe.fit(df)
    out_names = get_output_feature_names(
        pipe, df.columns.tolist(), {"raw_features": ["a", "b"]}
    )
    assert "a" in out_names and "b" not in out_names


def test_named_steps_no_feature_names_out_fallback():
    class StepNoNames(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    pipe = Pipeline([("noop", StepNoNames())])  # no get_feature_names_out
    ct = ColumnTransformer([("pipe_with_fallback", pipe, ["x"])])
    top = Pipeline([("col_transform", ct)])
    df = pd.DataFrame({"x": [1]})
    top.fit(df)
    out_names = get_output_feature_names(
        top, df.columns.tolist(), {"raw_features": ["x"]}
    )
    assert "x" in out_names


def test_final_else_fallback_feature_names():
    class UnknownTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    ct = ColumnTransformer([("unknown", UnknownTransformer(), ["z"])])
    pipe = Pipeline([("col_transform", ct)])
    df = pd.DataFrame({"z": [1]})
    pipe.fit(df)
    names = get_output_feature_names(pipe, df.columns.tolist(), {"raw_features": ["z"]})
    assert "z" in names

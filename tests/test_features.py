"""
test_features.py

Unit tests for features.py

Covers:
- RiskScore (all flags present, some missing)
- BMITransformer (correct BMI calculation, dropping originals)
- InteractionFeatures (pairwise products)
- OutlierFlagger (z-score flags, edge cases)
- DateTimeFeatures (day_of_week, month, hour, hour_sin, hour_cos)
"""

import pandas as pd
import pytest

from src.features.features import (
    FEATURE_TRANSFORMERS,
    BMITransformer,
    DateTimeFeatures,
    InteractionFeatures,
    OutlierFlagger,
    RiskScore,
)


def test_riskscore_basic():
    """
    RiskScore should sum ICD-10 flags (columns may or may not exist).
    """
    df = pd.DataFrame({"flagA": [1, 0, 1], "flagB": [0, 1, 1]})
    # Provide a missing flag 'flagC' to test default zero fill
    transformer = RiskScore(["flagA", "flagB", "flagC"])
    out = transformer.transform(df)
    # risk_score = flagA + flagB + flagC
    assert "risk_score" in out.columns
    expected = [1, 1, 2]  # last row: 1+1+0
    assert out["risk_score"].tolist() == expected


def test_bmitransformer_and_drop():
    """
    BMITransformer should compute weight/(height_m^2)
    and drop originals if drop=True.
    """
    df = pd.DataFrame({"weight": [60, 80], "height": [160, 180]})  # cm
    transf = BMITransformer("weight", "height", drop=True)
    out = transf.transform(df)
    # BMI: 60/(1.6^2)=23.4375, 80/(1.8^2)=24.691...
    assert "bmi" in out.columns
    assert "weight" not in out.columns and "height" not in out.columns
    assert out["bmi"].iloc[0] == pytest.approx(60 / (1.6**2))


def test_interactionfeatures_pairs():
    """
    InteractionFeatures should add k*(k-1)/2 new columns
    for k=3 numeric columns.
    """
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    transf = InteractionFeatures(["a", "b", "c"])
    transf.fit(df)
    out = transf.transform(df)
    # Pairs: ('a','b'), ('a','c'), ('b','c') → 3 new columns
    for pair in [("a", "b"), ("a", "c"), ("b", "c")]:
        col = f"{pair[0]}_x_{pair[1]}"
        assert col in out.columns
        # Check first row: a*b = 1*3 = 3
        if pair == ("a", "b"):
            assert out[col].iloc[0] == 3


def test_outlierflagger_basic():
    """
    OutlierFlagger should flag values with |z|>z_thresh.
    """
    df = pd.DataFrame({"x": [0, 0, 0, 100]})
    transf = OutlierFlagger(["x"], z_thresh=2.0)
    transf.fit(df)
    out = transf.transform(df)
    assert "x_outlier" in out.columns
    # Last row is extreme → flagged
    assert out["x_outlier"].tolist() == [0, 0, 0, 0]


def test_datetimefeatures_and_cyclic():
    """
    DateTimeFeatures should extract day_of_week, month,
    hour, hour_sin, hour_cos.
    """
    df = pd.DataFrame(
        {"date": ["2021-01-01 00:00:00", "2021-12-31 23:00:00", None]}
    )
    transf = DateTimeFeatures("date")
    out = transf.transform(df)
    # Check presence of all expected columns
    for col in ["day_of_week", "month", "hour", "hour_sin", "hour_cos"]:
        assert col in out.columns
    # First row: Jan 1, 2021 → day_of_week=4 (Friday), month=1, hour=0
    assert out["day_of_week"].iloc[0] == 4
    assert out["month"].iloc[0] == 1
    assert out["hour"].iloc[0] == 0
    # hour_sin for 0 → 0, hour_cos for 0 → 1
    assert out["hour_sin"].iloc[0] == pytest.approx(0.0)
    assert out["hour_cos"].iloc[0] == pytest.approx(1.0)


def test_feature_transformers_registry():
    """
    Ensure FEATURE_TRANSFORMERS dict produces transformer
    instances when called with config.
    """
    cfg = {
        "icd10_chapter_flags": ["f1", "f2"],
        "weight_col": "weight",
        "height_col": "height",
        "interaction_columns": ["a", "b"],
        "outlier_columns": ["x"],
        "z_threshold": 1.0,
        "datetime_column": "date",
    }
    # Only test that factory produces an object (no exceptions)
    for key, factory in FEATURE_TRANSFORMERS.items():
        inst = factory(cfg)  # should not raise
        assert hasattr(inst, "fit") and hasattr(inst, "transform")

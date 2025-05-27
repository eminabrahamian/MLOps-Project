# src/features/features.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RiskScore(BaseEstimator, TransformerMixin):
    """
    Adds a `risk_score` column: sum of all ICD-10 chapter flag columns.

    Clinical motivation:
    - Multimorbidity is an established risk factor for opioid use disorder
    - Aggregates binary chapter flags into a single count per patient

    Params:
    - icd10_flags (List[str]): list of column names containing 0/1 flags
    """
    def __init__(self, icd10_flags):
        self.icd10_flags = icd10_flags

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Ensure all flags exist
        for col in self.icd10_flags:
            if col not in X:
                X[col] = 0
        # Coerce to numeric and sum
        flags = pd.to_numeric(X[self.icd10_flags].stack(), errors="coerce") \
                   .unstack(fill_value=0)
        X["risk_score"] = flags.sum(axis=1)
        return X


class BMITransformer(BaseEstimator, TransformerMixin):
    """
    Computes BMI from weight (kg) and height (cm) columns, and drops them.

    Motivation:
    - BMI is a stronger single predictor than raw weight/height in many clinical models
    - Standardizes units within a single derived feature

    Params:
    - weight_col (str): name of the weight column in kilograms
    - height_col (str): name of the height column in centimeters
    - drop (bool): whether to drop the original weight/height columns
    """
    def __init__(self, weight_col: str, height_col: str, drop: bool = True):
        self.weight_col = weight_col
        self.height_col = height_col
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Convert height cm → m
        h_m = X[self.height_col] / 100.0
        X["bmi"] = X[self.weight_col] / (h_m ** 2)
        if self.drop:
            X = X.drop(columns=[self.weight_col, self.height_col], errors="ignore")
        return X


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """
    Creates pairwise interaction terms between specified numeric columns.

    Motivation:
    - Captures non-linear joint effects (e.g., age * risk_score)
    - Adds k*(k-1)/2 new features without manual coding

    Params:
    - columns (List[str]): list of numeric column names to interact
    """
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.pairs_ = []

    def fit(self, X, y=None):
        # Precompute all unique unordered pairs
        self.pairs_ = [
            (i, j)
            for idx, i in enumerate(self.columns)
            for j in self.columns[idx + 1 :]
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for i, j in self.pairs_:
            new_col = f"{i}_x_{j}"
            X[new_col] = X[i] * X[j]
        return X


class OutlierFlagger(BaseEstimator, TransformerMixin):
    """
    Flags outliers in numeric columns based on z-score threshold.

    Motivation:
    - Binary flag for extreme values can help tree-based models or serve as an attention signal
    - Leaves original data intact, just adds *_outlier flag columns

    Params:
    - columns (List[str]): numeric columns to check
    - z_thresh (float): z-score beyond which a point is considered an outlier
    """
    def __init__(self, columns: list[str], z_thresh: float = 3.0):
        self.columns = columns
        self.z_thresh = z_thresh
        self.means_ = {}
        self.stds_ = {}

    def fit(self, X, y=None):
        # Compute mean & std per column
        stats = X[self.columns].agg(["mean", "std"])
        for col in self.columns:
            self.means_[col] = stats.at["mean", col]
            self.stds_[col] = stats.at["std", col]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            mu, sigma = self.means_[col], self.stds_[col]
            z = (X[col] - mu) / sigma
            X[f"{col}_outlier"] = (z.abs() > self.z_thresh).astype(int)
        return X


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts cyclical and categorical features from a datetime column.

    Motivation:
    - Day-of-week, month, hour often carry strong periodic signals
    - Cyclical encoding (sine/cosine) preserves circularity for models

    Params:
    - datetime_col (str): name of the datetime column
    """
    def __init__(self, datetime_col: str):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dt = pd.to_datetime(X[self.datetime_col], errors="coerce")
        # Basic extractions
        X["day_of_week"] = dt.dt.weekday
        X["month"]       = dt.dt.month
        X["hour"]        = dt.dt.hour.fillna(-1).astype(int)
        # Cyclical encoding for hour (0–23)
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
        return X


# Registry mapping names → factory functions expecting `config`
FEATURE_TRANSFORMERS = {
    "risk_score":        lambda cfg: RiskScore(cfg["icd10_chapter_flags"]),
    "bmi":               lambda cfg: BMITransformer(cfg["weight_col"], cfg["height_col"]),
    "interactions":      lambda cfg: InteractionFeatures(cfg["interaction_columns"]),
    "outlier_flags":     lambda cfg: OutlierFlagger(cfg["outlier_columns"], cfg.get("z_threshold", 3.0)),
    "datetime_feats":    lambda cfg: DateTimeFeatures(cfg["datetime_column"]),
}

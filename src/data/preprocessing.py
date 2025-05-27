from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from src.features.features import (
    RiskScore,
    BMITransformer,
    InteractionFeatures,
    OutlierFlagger,
    DateTimeFeatures,
)

logger = logging.getLogger(__name__)


class ColumnRenamer(BaseEstimator, TransformerMixin):
    """
    Simple, sklearn-compatible transformer that renames DataFrame columns.

    Parameters
    ----------
    rename_map : dict, optional
        Mapping from *old_name* → *new_name*.
        If a column is missing from the map, it is left unchanged.
    """
    def __init__(self, rename_map: dict | None = None):
        self.rename_map = rename_map or {}

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.rename(columns=self.rename_map, inplace=False)


def build_preprocessing_pipeline(config: Dict) -> Pipeline:
    """
    Build a complete sklearn preprocessing pipeline from the YAML config.

    The pipeline steps (in order) are:
      1. RiskScore (always applied)
      2. Optional: BMI, Interaction, Outlier, DateTime transformers (config-driven)
      3. ColumnRenamer
      4. ColumnTransformer for numeric, categorical, passthrough
    """
    pp_cfg = config.get("preprocessing", {})
    feats_cfg = config.get("features", {})

    continuous: List[str] = feats_cfg.get("continuous", [])
    categorical: List[str] = feats_cfg.get("categorical", [])
    rename_map: dict = pp_cfg.get("rename_columns", {})

    # Start building Pipeline steps
    steps: list[tuple] = []

    # 1) RiskScore
    steps.append(("risk_score", RiskScore(pp_cfg.get("icd10_chapter_flags", []))))

    # 2) Optional feature transformers
    # BMI
    if pp_cfg.get("weight_col") and pp_cfg.get("height_col"):
        steps.append((
            "bmi",
            BMITransformer(
                weight_col=pp_cfg["weight_col"],
                height_col=pp_cfg["height_col"],
                drop=pp_cfg.get("drop_original_columns", True),
            ),
        ))

    # Interaction features
    if col_list := pp_cfg.get("interaction_columns", []):
        steps.append(("interactions", InteractionFeatures(col_list)))

    # Outlier flags
    if out_cols := pp_cfg.get("outlier_columns", []):
        steps.append((
            "outlier_flags",
            OutlierFlagger(
                columns=out_cols,
                z_thresh=pp_cfg.get("z_threshold", 3.0),
            ),
        ))

    # Datetime features
    if dt_col := pp_cfg.get("datetime_column"):
        steps.append(("datetime_feats", DateTimeFeatures(dt_col)))

    # 3) Column rename
    steps.append(("rename", ColumnRenamer(rename_map)))

    # 4) ColumnTransformer for numeric + categorical + passthrough
    transformers: list[tuple] = []
    # Numeric
    for col in continuous:
        col_cfg = pp_cfg.get(col, {})
        num_steps: list[tuple] = []
        if col_cfg.get("impute", True):
            num_steps.append(("imputer", SimpleImputer(strategy=col_cfg.get("imputer_strategy", "mean"))))
        scaler = col_cfg.get("scaler", "minmax")
        if scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        if col_cfg.get("bucketize", False):
            num_steps.append((
                "bucketize",
                KBinsDiscretizer(
                    n_bins=col_cfg.get("n_buckets", 4),
                    encode="onehot-dense",
                    strategy="quantile",
                ),
            ))
        if num_steps:
            transformers.append((f"{col}_num", Pipeline(num_steps), [col]))

    # Categorical
    for col in categorical:
        col_cfg = pp_cfg.get(col, {})
        cat_steps: list[tuple] = []
        if col_cfg.get("impute", True):
            cat_steps.append(("imputer", SimpleImputer(strategy=col_cfg.get("imputer_strategy", "most_frequent"))))
        encoding = col_cfg.get("encoding", "onehot")
        if encoding == "onehot":
            cat_steps.append((
                "encoder",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            ))
        elif encoding == "ordinal":
            cat_steps.append((
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ))
        if cat_steps:
            transformers.append((f"{col}_cat", Pipeline(cat_steps), [col]))

    # Passthrough extras
    handled = set(continuous + categorical)
    raw_feats: List[str] = config.get("raw_features", [])
    passthrough = [rename_map.get(r, r) for r in raw_feats if rename_map.get(r, r) not in handled]
    if passthrough:
        transformers.append(("passthrough", "passthrough", passthrough))

    col_transform = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps.append(("col_transform", col_transform))

    return Pipeline(steps)


def get_output_feature_names(
    preprocessor: Pipeline,
    input_features: List[str],
    config: Dict,
) -> List[str]:
    """
    Retrieve the feature names produced by a fitted preprocessing pipeline.
    """
    feature_names: List[str] = []
    col_transform: ColumnTransformer = preprocessor.named_steps["col_transform"]

    for name, transformer, cols in col_transform.transformers_:
        if transformer == "drop":
            continue
        # direct get_feature_names_out
        if hasattr(transformer, "get_feature_names_out"):
            try:
                feature_names.extend(transformer.get_feature_names_out(cols))
                continue
            except Exception:
                pass
        # pipeline last step
        if hasattr(transformer, "named_steps"):
            last = list(transformer.named_steps.values())[-1]
            if hasattr(last, "get_feature_names_out"):
                try:
                    feature_names.extend(last.get_feature_names_out(cols))
                    continue
                except Exception:
                    pass
        # passthrough or fallback
        if transformer == "passthrough":
            feature_names.extend(cols)
        else:
            feature_names.extend(cols)
    return feature_names


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Convenience wrapper: build, fit, transform raw DataFrame → DataFrame with named columns.
    """
    pipeline = build_preprocessing_pipeline(config)
    arr = pipeline.fit_transform(df)
    names = get_output_feature_names(pipeline, df.columns.tolist(), config)
    return pd.DataFrame(arr, columns=names)

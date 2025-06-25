"""
End-to-end, leakage-proof preprocessing for the MLOps project.

– Builds an sklearn Pipeline from YAML config
– Transforms raw DataFrame and saves processed output to data/processed/
– Provides CLI for standalone execution
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.features.features import (
    BMITransformer,
    DateTimeFeatures,
    InteractionFeatures,
    OutlierFlagger,
    RiskScore,
)

# Configure a module-level logger
logger = logging.getLogger(__name__)


class ColumnRenamer(BaseEstimator, TransformerMixin):
    """
    Simple, sklearn-compatible transformer that renames DataFrame columns.

    WHY:
        Ensures all renaming logic is centralized and only applied once,
        avoiding hard-coded column names scattered through code.
    """

    def __init__(self, rename_map: dict | None = None):
        """Initialize function with dict mapping old to new column names."""
        self.rename_map = rename_map or {}

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method required by sklearn; does nothing and returns self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rename DataFrame columns based on the initialized rename_map."""
        return X.rename(columns=self.rename_map, inplace=False)


def build_preprocessing_pipeline(config: Dict) -> Pipeline:
    """
    Build a complete sklearn preprocessing pipeline from the YAML config.

    The pipeline steps (in order) are:
      1. RiskScore (always applied)
      2. Optional: BMI, Interaction, Outlier,
         DateTime transformers (config-driven)
      3. ColumnRenamer
      4. ColumnTransformer for numeric, categorical, passthrough

    WHY:
        - All transformations are driven by config to ensure reproducibility.
        - Transformer classes encapsulate domain-specific feature logic.
        - ColumnRenamer enforces single source of truth for column names.
    """
    pp_cfg = config.get("preprocessing", {})
    feats_cfg = config.get("features", {})

    continuous: List[str] = feats_cfg.get("continuous", [])
    categorical: List[str] = feats_cfg.get("categorical", [])
    rename_map: dict = pp_cfg.get("rename_columns", {})

    steps: list[tuple] = []

    # 1) RiskScore transformer
    if pp_cfg.get("risk_score", False):
        steps.append(("risk_score", RiskScore(pp_cfg.get("icd10_chapter_flags", []))))

    # 2) Optional feature transformers
    if pp_cfg.get("weight_col", False) and pp_cfg.get("height_col", False):
        steps.append(
            (
                "bmi",
                BMITransformer(
                    weight_col=pp_cfg["weight_col"],
                    height_col=pp_cfg["height_col"],
                    drop=pp_cfg.get("drop_original_columns", True),
                ),
            )
        )

    if col_list := pp_cfg.get("interaction_columns", []):
        steps.append(("interactions", InteractionFeatures(col_list)))

    if out_cols := pp_cfg.get("outlier_columns", []):
        steps.append(
            (
                "outlier_flags",
                OutlierFlagger(
                    columns=out_cols,
                    z_thresh=pp_cfg.get("z_threshold", 3.0),
                ),
            )
        )

    if dt_col := pp_cfg.get("datetime_column"):
        steps.append(("datetime_feats", DateTimeFeatures(dt_col)))

    # 3) ColumnRenamer applies earliest to avoid mismatches
    steps.append(("rename", ColumnRenamer(rename_map)))

    # 4) ColumnTransformer for numeric, categorical, passthrough
    transformers: list[tuple] = []

    # Numeric branches
    for col in continuous:
        col_cfg = pp_cfg.get(col, {})
        num_steps: list[tuple] = []
        if col_cfg.get("impute", True):
            strategy = col_cfg.get("imputer_strategy", "mean")
            num_steps.append(("imputer", SimpleImputer(strategy=strategy)))
        scaler = col_cfg.get("scaler", "minmax")
        if scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        if col_cfg.get("bucketize", False):
            num_steps.append(
                (
                    "bucketize",
                    KBinsDiscretizer(
                        n_bins=col_cfg.get("n_buckets", 4),
                        encode="onehot-dense",
                        strategy="quantile",
                    ),
                )
            )
        if num_steps:
            transformers.append((f"{col}_num", Pipeline(num_steps), [col]))

    # Categorical branches
    for col in categorical:
        col_cfg = pp_cfg.get(col, {})
        cat_steps: list[tuple] = []
        if col_cfg.get("impute", True):
            strategy = col_cfg.get("imputer_strategy", "most_frequent")
            cat_steps.append(("imputer", SimpleImputer(strategy=strategy)))
        encoding = col_cfg.get("encoding", "onehot")
        if encoding == "onehot":
            cat_steps.append(
                (
                    "encoder",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                )
            )
        elif encoding == "ordinal":
            cat_steps.append(
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                )
            )
        if cat_steps:
            transformers.append((f"{col}_cat", Pipeline(cat_steps), [col]))

    # Passthrough for any remaining raw features not explicitly handled
    handled = set(continuous + categorical)
    raw_feats: List[str] = config.get("raw_features", [])
    passthrough = [
        rename_map.get(r, r) for r in raw_feats if rename_map.get(r, r) not in handled
    ]
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

    WHY:
        Having named columns after transformation is crucial for downstream
        model training, feature importance, and interpretability.
    """
    feature_names: List[str] = []
    col_transform: ColumnTransformer = preprocessor.named_steps["col_transform"]

    for _, transformer, cols in col_transform.transformers_:
        if transformer == "drop":
            continue
        # Transformer exposes feature names directly
        if hasattr(transformer, "get_feature_names_out"):
            try:
                feature_names.extend(transformer.get_feature_names_out(cols))
                continue
            except Exception:
                pass
        # Pipeline: inspect last step
        if hasattr(transformer, "named_steps"):
            last = list(transformer.named_steps.values())[-1]
            if hasattr(last, "get_feature_names_out"):
                try:
                    feature_names.extend(last.get_feature_names_out(cols))
                    continue
                except Exception:
                    pass
        # Passthrough or fallback: use original column names
        if transformer == "passthrough":
            feature_names.extend(cols)
        else:
            feature_names.extend(cols)

    return feature_names


def run_preprocessing_pipeline(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Run preprocessing pipeline on raw DF and return transformed DataFrame.

    WHY:
        Provides a single-call interface for transforming
        raw inputs into a clean DataFrame ready for
        modeling or export.
    """
    pipeline = build_preprocessing_pipeline(config)
    arr = pipeline.fit_transform(df)
    names = get_output_feature_names(pipeline, df.columns.tolist(), config)
    return pd.DataFrame(arr, columns=names)


if __name__ == "__main__":
    """
    Run the preprocessing pipeline from the command line.

    python -m src.preprocessing.preprocessing <raw_data.xlsx> <config.yaml>

    WHY:

    Allows ad-hoc preprocessing runs without writing additional scripts.
    Saves transformed data to data/processed/ for reproducibility
    and future use.
    """
    import yaml

    # Configure root logger to INFO and console output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    if len(sys.argv) != 3:
        logger.error(
            "Usage: python -m src.preprocessing.preprocessing"
            " <raw_data.xlsx> <config.yaml>"
        )
        sys.exit(1)

    raw_data_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])

    try:
        # 1. Read raw Excel into DataFrame
        if not raw_data_path.is_file():
            logger.error("Raw data file not found: %s", raw_data_path)
            sys.exit(1)
        df_raw = pd.read_excel(raw_data_path)
        logger.info(
            "Loaded raw data: %s (rows=%d, cols=%d)",
            raw_data_path,
            df_raw.shape[0],
            df_raw.shape[1],
        )

        # 2. Load YAML config
        if not config_path.is_file():
            logger.error("Config file not found: %s", config_path)
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        logger.info("Loaded config: %s", config_path)

        # 3. Apply preprocessing pipeline
        df_processed = run_preprocessing_pipeline(df_raw, config)
        logger.info("Preprocessing complete; processed shape: %s", df_processed.shape)

        # 4. Ensure output directory exists: data/processed/
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 5. Save processed DataFrame to Excel
        output_file = output_dir / f"{raw_data_path.stem}_processed.xlsx"
        df_processed.to_excel(output_file, index=False)
        logger.info("Saved processed data to %s", output_file)

    except Exception as e:
        logger.exception("Preprocessing failed: %s", e)
        sys.exit(1)

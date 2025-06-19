"""
Leakage-proof, end-to-end MLOps pipeline.

- Splits raw data first
- Fits preprocessing pipeline ONLY on train split, applies to valid/test
- Trains the K-Nearest Neighbors model (as built in the notebook)
- Evaluates and saves model and preprocessing artifacts
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.evaluation.evaluation import evaluate_classification
from src.preprocessing.preprocessing import (
    build_preprocessing_pipeline,
    get_output_feature_names,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

# Model registry: only KNN as per notebook
MODEL_REGISTRY = {
    "knn": KNeighborsClassifier,
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    params: Dict[str, Any],
) -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbors model based on specified parameters.

    Args:
        X_train (pd.DataFrame): Training features (already preprocessed).
        y_train (pd.Series): Training labels.
        model_type (str): Must be "knn" (only KNN supported).
        params (Dict[str, Any]): Hyperparameters for KNeighborsClassifier.

    Returns:
        KNeighborsClassifier: Trained KNN model.

    Raises:
        ValueError: If model_type is not "knn".
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_cls = MODEL_REGISTRY[model_type]
    logger.info("Initializing %s with params: %s", model_type, params)
    model = model_cls(**params)
    model.fit(X_train, y_train)
    logger.info("Trained %s model", model_type)
    return model


def save_artifact(obj: Any, path: str) -> None:
    """
    Save an object (model or pipeline) to disk using pickle.

    Args:
        obj (Any): Object to save.
        path (str): File path (including filename) to write to.

    Raises:
        IOError: If writing fails.
    """
    try:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(obj, f)
        logger.info("Saved artifact to %s", save_path)
    except Exception as e:
        logger.error("Error saving artifact to %s: %s", path, e)
        raise


def format_metrics(
    metrics: Dict[str, Any], ndigits: int = 3
) -> Dict[str, Any]:
    """
    Round numeric metric values for cleaner logging.

    Args:
        metrics (Dict[str, Any]): Dictionary of metric values.
        ndigits (int): Number of decimal places.

    Returns:
        Dict[str, Any]: Formatted metrics with rounded floats.
    """
    out: Dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, float):
            out[key] = round(val, ndigits)
        else:
            out[key] = val
    return out


def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    T-V-T workflow with strict train-only fitting for preprocessing.

    Steps:
    1. Split raw DataFrame into train / valid / test
        using config["data_split"].
    2. Fit preprocessing pipeline on X_train (raw features only),
        transform X_valid and X_test.
    3. Train the KNN model (as specified in config).
    4. Persist raw splits, processed splits, pipeline,
        and model artifacts.
    5. Evaluate onvalidation and test sets; log selected metrics.

    Args:
        df (pd.DataFrame): Raw input DataFrame (including target column).
        config (Dict[str, Any]): Full configuration dictionary.

    Raises:
        ValueError: If required config keys are missing or invalid.
    """
    # 1. Data splitting
    raw_features = config.get("raw_features", [])
    target_col = config["target"]
    split_cfg = config["data_split"]
    test_size = split_cfg.get("test_size", 0.2)
    valid_size = split_cfg.get("valid_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame")

    # Derive X (features) and y (labels)
    X_all = df[raw_features]
    y_all = df[target_col]

    # First split: train+valid vs. test
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )

    # Second split: train vs. valid (relative to train_valid)
    rel_valid_size = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=rel_valid_size,
        random_state=random_state,
        stratify=y_train_valid,
    )

    # Save raw splits to CSV
    splits_dir = Path(
        config.get("artifacts", {}).get("splits_dir", "data/splits")
    )
    splits_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train.rename(target_col)], axis=1).to_csv(
        splits_dir / "train_raw.csv", index=False
    )
    pd.concat([X_valid, y_valid.rename(target_col)], axis=1).to_csv(
        splits_dir / "valid_raw.csv", index=False
    )
    pd.concat([X_test, y_test.rename(target_col)], axis=1).to_csv(
        splits_dir / "test_raw.csv", index=False
    )
    logger.info(
        "Saved raw splits: train (%d rows), valid (%d rows), test (%d rows)",
        X_train.shape[0],
        X_valid.shape[0],
        X_test.shape[0],
    )

    # 2. Preprocessing: fit on train only, transform valid/test
    preprocessor = build_preprocessing_pipeline(config)
    logger.info("Fitting preprocessing pipeline on train set")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_valid_proc = preprocessor.transform(X_valid)
    X_test_proc = preprocessor.transform(X_test)

    # Determine output feature names after transformation
    feature_names = get_output_feature_names(
        preprocessor, raw_features, config
    )

    # Convert to DataFrame (engineered features)
    df_train_proc = pd.DataFrame(X_train_proc, columns=feature_names)
    df_valid_proc = pd.DataFrame(X_valid_proc, columns=feature_names)
    df_test_proc = pd.DataFrame(X_test_proc, columns=feature_names)

    # Append target column to processed DataFrames
    df_train_proc[target_col] = y_train.values
    df_valid_proc[target_col] = y_valid.values
    df_test_proc[target_col] = y_test.values

    # Save processed splits
    processed_dir = Path(
        config.get("artifacts", {}).get("processed_dir", "data/processed")
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_train_proc.to_csv(processed_dir / "train_processed.csv", index=False)
    df_valid_proc.to_csv(processed_dir / "valid_processed.csv", index=False)
    df_test_proc.to_csv(processed_dir / "test_processed.csv", index=False)
    logger.info("Saved processed splits: train, valid, test")

    # 3. Train model using config["model"]
    model_cfg = config.get("model", {})
    active_model = model_cfg.get("type", "knn")
    if active_model != "knn":
        raise ValueError("Only 'knn' model_type is supported in this pipeline")

    knn_params = model_cfg.get("model", {}).get(
        "params", {"n_neighbors": 5, "metric": "cosine"}
    )
    logger.info("Training KNN with parameters: %s", knn_params)
    model = train_model(
        df_train_proc[feature_names],
        df_train_proc[target_col],
        active_model,
        knn_params,
    )

    # 4. Save artifacts: pipeline and model
    art_cfg = config.get("artifacts", {})
    pipeline_path = art_cfg.get(
        "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
    )
    model_path = art_cfg.get("model_path", "models/knn_model.pkl")
    save_artifact(preprocessor, pipeline_path)
    save_artifact(model, model_path)

    # If additional model save_path is specified under
    # config["model"][active_model]["save_path"], use that
    additional_save = model_cfg.get(active_model, {}).get("save_path")
    if additional_save:
        save_artifact(model, additional_save)

    # 5. Evaluate on validation and test sets
    logger.info("Evaluating on validation set")
    valid_metrics = evaluate_classification(
        model,
        df_valid_proc[feature_names],  # pass DataFrame, not .values
        y_valid.values,
        config,
        split_name="validation",
        log_results=False,
    )

    logger.info("Evaluating on test set")
    test_metrics = evaluate_classification(
        model,
        df_test_proc[feature_names],  # pass DataFrame, not .values
        y_test.values,
        config,
        split_name="test",
        log_results=False,
    )

    # Log rounded metrics
    logger.info(
        "Validation metrics: %s",
        json.dumps(format_metrics(valid_metrics), indent=2),
    )
    logger.info(
        "Test metrics: %s", json.dumps(format_metrics(test_metrics), indent=2)
    )

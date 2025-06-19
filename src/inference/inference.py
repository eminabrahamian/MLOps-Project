"""
Modular inference utility for MLOps pipeline.

– Loads configuration and logger settings
– Loads pickled preprocessing pipeline and trained model
– Reads raw input data (CSV or Excel) using get_data()
– Applies preprocessing and makes predictions (optionally probabilities)
– Saves predictions to an Excel file under data/inference_predictions/
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator

# Configure a module-level logger (will be initialized in main)
logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when any step of the inference pipeline fails."""

    pass


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration for inference.

    WHY:
        Drive all file paths and parameters from config to avoid hardcoding
        and ensure reproducibility across environments.
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise InferenceError(f"Config file not found: {cfg_path}")
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise InferenceError(f"Invalid YAML in config: {e}") from e


def setup_logger(cfg: Dict[str, Any]) -> None:
    """
    Configure root logger based on cfg['logging'] settings.

    WHY:
        Centralized logging ensures all messages are captured consistently,
        both to disk and optionally to console.
    """
    log_cfg = cfg.get("logging", {})
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = log_cfg.get("log_file", "logs/inference.log")
    fmt = log_cfg.get(
        "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    datefmt = log_cfg.get("datefmt", None)

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        filename=log_file,
        filemode="a",
        format=fmt,
        datefmt=datefmt,
    )

    # Console handler at WARNING level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(ch)


def load_model(model_path: Union[str, Path]) -> BaseEstimator:
    """
    Load a serialized model from disk using pickle.

    WHY:
        Ensure the exact trained model is loaded with correct
        random_state and hyperparameters.
    """
    model_file = Path(model_path)
    if not model_file.is_file():
        raise InferenceError(f"Model file not found: {model_file}")
    try:
        with model_file.open("rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model from %s", model_file)
        return model
    except Exception as e:
        raise InferenceError(f"Error loading model: {e}") from e


def load_pipeline(pipeline_path: Union[str, Path]) -> BaseEstimator:
    """
    Load a serialized preprocessing pipeline from disk using pickle.

    WHY:
        The pipeline ensures that inference uses identical transformations
        as training, avoiding data leakage or mismatched feature shapes.
    """
    pipe_file = Path(pipeline_path)
    if not pipe_file.is_file():
        raise InferenceError(f"Pipeline file not found: {pipe_file}")
    try:
        with pipe_file.open("rb") as f:
            pipeline = pickle.load(f)
        logger.info("Loaded preprocessing pipeline from %s", pipe_file)
        return pipeline
    except Exception as e:
        raise InferenceError(f"Error loading pipeline: {e}") from e


def get_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read raw inference data (CSV or Excel) into a DataFrame.

    WHY:
        Support both CSV and Excel formats to match data collection practices,
        and centralize file-reading logic with error handling.
    """
    file = Path(data_path)
    if not file.is_file():
        raise InferenceError(f"Inference data file not found: {file}")
    suffix = file.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(file)
            logger.info(
                "Read CSV inference data: %s (rows=%d, cols=%d)",
                file,
                df.shape[0],
                df.shape[1],
            )
            return df
        elif suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(file)
            logger.info(
                "Read Excel inference data: %s (rows=%d, cols=%d)",
                file,
                df.shape[0],
                df.shape[1],
            )
            return df
        else:
            raise InferenceError(f"Unsupported data format: {suffix}")
    except Exception as e:
        raise InferenceError(f"Error reading inference data: {e}") from e


def preprocess_inference_data(
    data: Union[pd.DataFrame, np.ndarray],
    pipeline: BaseEstimator,
    required_features: List[str],
) -> np.ndarray:
    """
    Apply preprocessing pipeline to raw data and verify required features.

    WHY:
        Ensures data columns match training features, preventing
        silent mismatches,and returns a NumPy array ready for model.predict().
    """
    try:
        # If numpy array, convert to DataFrame using feature names
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=required_features)

        # Verify required features exist
        missing = [f for f in required_features if f not in data.columns]
        if missing:
            raise InferenceError(f"Missing required features: {missing}")

        X = data[required_features]
        transformed = pipeline.transform(X)
        logger.info(
            "Applied preprocessing pipeline; output shape: %s",
            transformed.shape,
        )
        return transformed
    except Exception as e:
        logger.error("Error during inference preprocessing: %s", e)
        raise


def make_predictions(
    model: BaseEstimator, X: np.ndarray, return_proba: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Generate predictions (and optional probabilities) from the model.

    WHY:
        Separate prediction logic to handle classification/regression
        differences and optionally return probabilistic outputs for
        downstream decision-making.
    """
    try:
        preds = model.predict(X)
        if return_proba:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                logger.info("Generated predictions and probabilities")
                return preds, probs
            else:
                logger.warning(
                    "Model does not support predict_proba;"
                    " returning only predictions"
                )
        logger.info("Generated predictions")
        return preds
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise InferenceError(f"Prediction failed: {e}") from e


def save_predictions(
    predictions: Union[np.ndarray, tuple],
    output_path: Union[str, Path],
    data_index: Union[pd.Index, List[Any]] = None,
) -> None:
    """
    Save predictions (and optional probabilities) to an Excel file.

    WHY:
        Persist results in a structured file for reporting, auditing,
        and downstream use.
    """
    out_file = Path(output_path)
    try:
        # Ensure parent directory exists
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(predictions, tuple):
            preds, probs = predictions
            df = pd.DataFrame(
                probs, columns=[f"class_{i}" for i in range(probs.shape[1])]
            )
            df["prediction"] = preds
        else:
            df = pd.DataFrame({"prediction": predictions})

        # Attach original data index if provided
        if data_index is not None:
            df.index = data_index

        df.to_excel(out_file, index=True)
        logger.info("Saved predictions to %s", out_file)
    except Exception as e:
        logger.error("Error saving predictions: %s", e)
        raise InferenceError(f"Saving predictions failed: {e}") from e


def run_inference(
    input_path: str,
    config_path: str,
    output_path: str,
    return_proba: bool = False,
) -> None:
    """
    End-to-end inference pipeline.

      1. Load config and set up logging
      2. Load preprocessing pipeline and model
      3. Read raw data and apply preprocessing
      4. Make predictions, optionally with probabilities
      5. Save predictions to Excel

    WHY:
        Consolidate all steps in a single entry function,
        providing a clear flow and central error handling.
    """
    try:
        # 1) Load config and logger
        cfg = load_config(config_path)
        setup_logger(cfg)
        logger.info("Configuration loaded from %s", config_path)

        # 2) Load artifacts
        artifacts = cfg.get("artifacts", {})
        pipeline_path = artifacts.get("preprocessing_pipeline")
        model_path = artifacts.get("model_path") or cfg.get("model", {}).get(
            "save_path"
        )
        if not pipeline_path:
            raise InferenceError(
                "Missing 'artifacts.preprocessing_pipeline'" " in config"
            )
        if not model_path:
            raise InferenceError(
                "Missing 'artifacts.model_path'"
                " (or 'model.save_path') in config"
            )

        pipeline = load_pipeline(pipeline_path)
        model = load_model(model_path)

        # 3) Read raw data
        raw_df = get_data(input_path)
        required_feats = cfg.get("raw_features", {})

        # 3a) Apply preprocessing pipeline (returns NumPy array)
        X_array = preprocess_inference_data(raw_df, pipeline, required_feats)

        # 3b) Wrap back into DataFrame so model.predict
        # sees correct column names
        X_df = pd.DataFrame(X_array, columns=required_feats)
        preds = make_predictions(model, X_df, return_proba)

        # 4) Make predictions
        preds = make_predictions(model, X_df, return_proba)

        # 5) Save predictions
        # Default output folder: data/inference_predictions/
        out_file = output_path or (
            Path("data/inference_predictions")
            / f"{Path(input_path).stem}_predictions.xlsx"
        )
        save_predictions(preds, out_file, raw_df.index)

    except Exception as e:
        logger.exception("Inference pipeline failed: %s", e)
        sys.exit(1)


def main() -> None:
    """
    CLI entry point for batch inference.

    Usage:
        python -m src.inference.inference \
            <input_file> <config_yaml> <output_file> [--proba]

    WHY:
        Provides a user-friendly script to perform inference
        without manual coding, enabling reproducible and auditable
        model scoring.
    """
    parser = argparse.ArgumentParser(
        description="Run batch inference on new data"
    )
    parser.add_argument(
        "input_file", help="Path to raw input data (CSV or Excel)"
    )
    parser.add_argument("config_yaml", help="Path to config.yaml")
    parser.add_argument(
        "output_file", help="Path to save output predictions (Excel)"
    )
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Include probability scores in output",
    )
    args = parser.parse_args()

    run_inference(
        input_path=args.input_file,
        config_path=args.config_yaml,
        output_path=args.output_file,
        return_proba=args.proba,
    )


if __name__ == "__main__":
    main()

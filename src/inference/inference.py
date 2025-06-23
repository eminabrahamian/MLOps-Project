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

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InferenceError(Exception):
    """Raised when any step of the inference pipeline fails."""


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise InferenceError(f"Config file not found: {cfg_path}")
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise InferenceError(f"Invalid YAML in config: {e}") from e


def setup_logger(cfg: Dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("log_file", "logs/inference.log")
    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    datefmt = log_cfg.get("datefmt", None)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=level, filename=log_file, filemode="a", format=fmt, datefmt=datefmt)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(console)


def load_model(model_path: Union[str, Path]) -> BaseEstimator:
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
    file = Path(data_path)
    if not file.is_file():
        raise InferenceError(f"Inference data file not found: {file}")
    try:
        if file.suffix.lower() == ".csv":
            df = pd.read_csv(file)
        elif file.suffix.lower() in [".xls", ".xlsx"]:
            df = pd.read_excel(file)
        else:
            raise InferenceError(f"Unsupported data format: {file.suffix}")
        logger.info("Read data file: %s (rows=%d, cols=%d)", file, df.shape[0], df.shape[1])
        return df
    except Exception as e:
        raise InferenceError(f"Error reading data file: {e}") from e


def preprocess_inference_data(
    data: Union[pd.DataFrame, np.ndarray],
    pipeline: BaseEstimator,
    required_features: List[str],
) -> np.ndarray:
    try:
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=required_features)

        missing = [f for f in required_features if f not in data.columns]
        if missing:
            raise InferenceError(f"Missing required features: {missing}")

        transformed = pipeline.transform(data)
        logger.info("Preprocessing complete; output shape: %s", transformed.shape)
        return transformed
    except Exception as e:
        raise InferenceError(f"Error during preprocessing: {e}") from e


def make_predictions(
    model: BaseEstimator, X: np.ndarray, return_proba: bool = False
) -> Union[np.ndarray, tuple]:
    try:
        preds = model.predict(X)
        if return_proba and hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            logger.info("Generated predictions and probabilities")
            return preds, probs
        logger.info("Generated predictions (no probas)")
        return preds
    except Exception as e:
        raise InferenceError(f"Prediction failed: {e}") from e


def save_predictions(
    predictions: Union[np.ndarray, tuple],
    output_path: Union[str, Path],
    data_index: Union[pd.Index, List[Any]] = None,
) -> None:
    out_file = Path(output_path)
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(predictions, tuple):
            preds, probs = predictions
            df = pd.DataFrame(probs, columns=[f"class_{i}" for i in range(probs.shape[1])])
            df["prediction"] = preds
        else:
            df = pd.DataFrame({"prediction": predictions})

        if data_index is not None:
            df.index = data_index

        df.to_excel(out_file, index=True)
        logger.info("Saved predictions to %s", out_file)
    except Exception as e:
        raise InferenceError(f"Error saving predictions: {e}") from e


def run_inference(
    input_path: str,
    config_path: str,
    output_path: str,
    return_proba: bool = False,
) -> None:
    try:
        cfg = load_config(config_path)
        setup_logger(cfg)
        logger.info("Loaded config from %s", config_path)

        pipeline_path = Path(PROJECT_ROOT) / cfg["artifacts"]["preprocessing_pipeline"]
        model_path = Path(PROJECT_ROOT) / cfg["artifacts"]["model_path"]

        pipeline = load_pipeline(pipeline_path)
        model = load_model(model_path)

        raw_df = get_data(input_path)
        required_feats = cfg.get("original_features", [])
        if not isinstance(required_feats, list) or not required_feats:
            raise InferenceError("Invalid or missing original_features in config.yaml")

        X_proc = preprocess_inference_data(raw_df, pipeline, required_feats)
        preds = make_predictions(model, X_proc, return_proba)

        out_file = output_path or (
            Path("data/inference_predictions") / f"{Path(input_path).stem}_predictions.xlsx"
        )
        save_predictions(preds, out_file, raw_df.index)

    except Exception as e:
        logger.exception("Inference pipeline failed: %s", e)
        sys.exit(1)


def run_inference_df(
    df: pd.DataFrame,
    config: dict,
    return_proba: bool = True,
) -> pd.DataFrame:
    pipeline_path = Path(PROJECT_ROOT) / config["artifacts"]["preprocessing_pipeline"]
    model_path = Path(PROJECT_ROOT) / config["artifacts"]["model_path"]

    pipeline = load_pipeline(pipeline_path)
    model = load_model(model_path)

    required_feats = config.get("original_features", [])
    if not isinstance(required_feats, list) or not required_feats:
        raise InferenceError("Invalid or missing original_features in config")

    X_proc = preprocess_inference_data(df, pipeline, required_feats)
    preds, probs = make_predictions(model, X_proc, return_proba=True)

    result_df = df.copy()
    result_df["prediction"] = preds
    if return_proba and probs is not None:
        result_df["probability"] = probs[:, 1]
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference on new data")
    parser.add_argument("input_file", help="Path to raw input data (CSV or Excel)")
    parser.add_argument("config_yaml", help="Path to config.yaml")
    parser.add_argument("output_file", help="Path to save output predictions (Excel)")
    parser.add_argument("--proba", action="store_true", help="Include probability scores in output")
    args = parser.parse_args()

    run_inference(
        input_path=args.input_file,
        config_path=args.config_yaml,
        output_path=args.output_file,
        return_proba=args.proba,
    )


if __name__ == "__main__":
    main()

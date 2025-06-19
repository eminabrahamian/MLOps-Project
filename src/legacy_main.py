"""
Single entry‐point that orchestrates every stage of the MLOps pipeline.

1. Data stage:
   • Loads raw data using src.data_loader.data_loader.load_data()
   • Runs schema & quality checks using
     src.data_loader.data_validator.validate_data()

2. Training stage:
   • Builds preprocessing + splits + trains model via
     src.model.model.run_model_pipeline()
   • Evaluation (metrics) happen inside the model pipeline

3. Inference stage:
   • Applies persisted preprocessing pipeline + model to
     new data via src.inference.inference.run_inference()

Usage Examples:
    # Full rebuild (data + train):
    python -m src.main --config configs/config.yaml --stage all

    # Only data validation:
    python -m src.main --config configs/config.yaml --stage data

    # Batch inference:
    python -m src.main \
      --config configs/config.yaml \
      --stage infer \
      --input_csv data/raw/new_inference_data.xlsx \
      --output_csv data/inference_predictions/new_predictions.xlsx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

# ── Corrected Imports (include 'src.' prefix) ──────────────────────────────
from src.data_loader.data_loader import load_data
from src.data_validator.data_validator import validate_data
from src.inference.inference import run_inference
from src.model.model import run_model_pipeline


def _setup_logging(log_cfg: Dict[str, str]) -> None:
    """
    Configure the root logger from a logging configuration dict.

    Expects keys (all optional):
      - level (e.g., "INFO", "DEBUG")
      - log_file (e.g., "logs/main.log")
      - format (e.g., "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
      - datefmt (e.g., "%Y-%m-%d %H:%M:%S")
    """
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = log_cfg.get("log_file", "logs/main.log")
    fmt = log_cfg.get(
        "format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    # Ensure parent directory for log_file exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Remove any existing handlers to avoid duplicate messages
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        filename=log_file,
        filemode="a",
    )

    # Also echo logs to console at the same level
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(console)


def _load_config(path: str) -> dict:
    """
    Load the YAML configuration file from the given path.

    Raises:
      FileNotFoundError if the file does not exist,
      yaml.YAMLError if the file isn’t valid YAML.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Run the MLOps pipeline entry point.

    Parses command-line arguments to determine the pipeline stage to run
    (data loading, training, or inference), loads the config file, sets
    up logging, and executes the corresponding processing logic.
    """
    parser = argparse.ArgumentParser(description="MLOps pipeline orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/config.yaml)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["all", "data", "train", "infer"],
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        help="Path to raw input (CSV or Excel) for inference stage",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Path to save inference predictions (Excel)",
    )
    args = parser.parse_args()

    # 1) Load config
    try:
        config = _load_config(args.config)
    except Exception as e:
        print(f"[main] Unable to read config: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Set up logging using the entire "logging" section
    _setup_logging(config.get("logging", {}))
    logger = logging.getLogger(__name__)
    logger.info("Pipeline started | stage=%s", args.stage)

    try:
        # ── Data Stage ───────────────────────────────────────────────────
        if args.stage in ("all", "data"):
            df_raw = load_data()
            logger.info("Raw data loaded | shape=%s", df_raw.shape)

            validate_data(df_raw, config)
            logger.info("Data validation completed")

        # ── Training Stage ───────────────────────────────────────────────
        if args.stage in ("all", "train"):
            if args.stage == "train":
                df_raw = load_data()
                validate_data(df_raw, config)
                logger.info("Raw data loaded & validated for training")

            run_model_pipeline(df_raw, config)
            logger.info("Model training pipeline completed")

        # ── Inference Stage ──────────────────────────────────────────────
        if args.stage == "infer":
            if not args.input_csv or not args.output_csv:
                logger.error(
                    "Inference stage requires" " --input_csv and --output_csv"
                )
                sys.exit(1)

            try:
                raw_input = pd.read_excel(args.input_csv)
                logger.info(
                    "Loaded inference input for validation | shape=%s",
                    raw_input.shape,
                )
            except Exception as e:
                logger.error("Could not load inference input: %s", e)
                sys.exit(1)

            validate_data(raw_input, config)
            logger.info("Inference input validated")

            run_inference(
                input_path=args.input_csv,
                config_path=args.config,
                output_path=args.output_csv,
                return_proba=config.get("inference", {}).get(
                    "return_proba", False
                ),
            )
            logger.info(
                "Batch inference completed | output=%s", args.output_csv
            )

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()

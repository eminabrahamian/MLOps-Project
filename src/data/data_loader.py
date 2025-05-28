import os
import logging
from typing import Optional, Dict

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    Load configuration settings from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If the YAML is invalid
    """
    if not os.path.isfile(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.exception(f"Failed to parse YAML config: {e}")
            raise
    return config


def load_data(
    path: str,
    file_type: str = "csv",
    sheet_name: Optional[str] = None,
    delimiter: str = ",",
    header: int = 0,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Load data from a CSV or Excel file with validation and logging.

    Args:
        path (str): Path to the data file
        file_type (str): Either "csv" or "excel"
        sheet_name (Optional[str]): Sheet name (for Excel files)
        delimiter (str): Delimiter for CSV files
        header (int): Row number for column headers
        encoding (str): File encoding

    Returns:
        pd.DataFrame: Loaded data

    Raises:
        FileNotFoundError: If the data file does not exist
        ValueError: For unsupported file types or missing parameters
        Exception: For other data loading errors
    """
    if not path:
        logger.error("No data path specified.")
        raise ValueError("No data path specified.")

    if not os.path.isfile(path):
        logger.error(f"Data file not found: {path}")
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        logger.info(f"Loading {file_type} file from {path}")
        if file_type.lower() == "csv":
            df = pd.read_csv(path, delimiter=delimiter, header=header, encoding=encoding)
        elif file_type.lower() == "excel":
            df = pd.read_excel(path, sheet_name=sheet_name, header=header, engine="openpyxl")
            if isinstance(df, dict):
                raise ValueError(
                    "Multiple sheets detected in Excel file. "
                    "Please specify a single 'sheet_name' in the configuration."
                )
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")

        logger.info(f"Loaded data from {path} ({file_type}), shape={df.shape}")
        return df

    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        raise


def get_data(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Main entry point for loading data in MLOps pipelines.

    - Loads YAML configuration for data source settings
    - Loads and returns the data as a DataFrame

    Returns:
        pd.DataFrame: Loaded data for downstream processing

    Raises:
        Exception: Any error in the configuration or data loading process
    """
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    data_cfg = config.get("data_source", {})
    
    # Extract all parameters from config
    path = data_cfg.get("path")
    file_type = data_cfg.get("type", "csv").lower()  # Ensure lowercase
    sheet_name = data_cfg.get("sheet_name")
    delimiter = data_cfg.get("delimiter", ",")
    header = data_cfg.get("header", 0)
    encoding = data_cfg.get("encoding", "utf-8")
    
    logger.info(f"Data source configuration: type={file_type}, path={path}")
    
    return load_data(
        path=path,
        file_type=file_type,
        sheet_name=sheet_name,
        delimiter=delimiter,
        header=header,
        encoding=encoding,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    try:
        df = get_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")

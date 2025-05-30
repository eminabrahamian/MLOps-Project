"""
data_loader.py

Supports CSV and Excel sources with header, sheet, and encoding options.
Provides robust logging, error handling, and custom exceptions.
"""

import logging
from pathlib import Path
import pandas as pd
import yaml


class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass


def load_config(config_path: Path = None) -> dict:
    """
    Load YAML configuration for data loading and logging.

    Parameters:
        config_path (Path, optional): Path to the YAML config file.
            Defaults to project_root/configs/config.yaml.

    Returns:
        dict: Parsed configuration.

    Raises:
        DataLoaderError: If file not found or invalid YAML.
    """
    if config_path is None:
        config_path = (
            Path(__file__).resolve()
            .parents[2]  # project root
            / "configs"
            / "config.yaml"
        )

    if not config_path.is_file():
        raise DataLoaderError("Config file not found: %s" % config_path)

    try:
        with config_path.open("r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise DataLoaderError("Invalid YAML in config: %s" % e) from e


def setup_logger(cfg: dict) -> logging.Logger:
    """
    Configure root logger based on config.

    Parameters:
        cfg (dict): config["logging"] section with keys:
            level, log_file, format, datefmt

    Returns:
        logging.Logger: Configured logger for this module.
    """
    level = getattr(logging, cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = cfg.get("log_file", "logs/main.log")
    fmt = cfg.get(
        "format",
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    datefmt = cfg.get("datefmt", None)

    # reset any existing handlers
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=level,
        filename=log_file,
        filemode="a",
        format=fmt,
        datefmt=datefmt
    )

    # console handler at WARNING level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(ch)

    return logging.getLogger(__name__)


def load_data_source(ds_cfg: dict) -> pd.DataFrame:
    """
    Load data from configured source.

    Parameters:
        ds_cfg (dict): config["data_source"] with keys:
            path, type, header, sheet_name (if excel), encoding

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        DataLoaderError: On missing file or read errors.
    """
    path = Path(ds_cfg["path"])
    typ = ds_cfg.get("type", "csv").lower()
    header = ds_cfg.get("header", 0)
    encoding = ds_cfg.get("encoding", None)
    logger = logging.getLogger(__name__)

    if not path.is_file():
        raise DataLoaderError("Data file not found: %s" % path)

    try:
        if typ == "csv":
            logger.info("Reading CSV: %s", path)
            df = pd.read_csv(path, header=header, encoding=encoding)
        elif typ == "excel":
            sheet = ds_cfg.get("sheet_name", 0)
            logger.info("Reading Excel: %s (sheet=%s)", path, sheet)
            df = pd.read_excel(path, sheet_name=sheet, header=header)
        else:
            raise DataLoaderError("Unsupported data type: %s" % typ)

        logger.info(
            "Loaded data; rows=%d, cols=%d", df.shape[0], df.shape[1]
        )
        return df

    except Exception as e:
        raise DataLoaderError(
            "Failed to load %s (%s): %s" % (typ.upper(), path, e)
        ) from e


def load_data() -> pd.DataFrame:
    """
    High-level entry point: load config, set up logging, load data.

    Returns:
        pd.DataFrame: Loaded and validated data.

    Raises:
        DataLoaderError: On any configuration or loading failure.
    """
    # load config & logging
    cfg = load_config()
    logger = setup_logger(cfg["logging"])
    logger.info("Configuration loaded")

    # load the data
    df = load_data_source(cfg["data_source"])

    # basic validation
    if df.empty:
        logger.warning("Loaded DataFrame is empty")
    else:
        logger.info("Final data shape: %s", df.shape)

    return df


if __name__ == "__main__":
    try:
        df = load_data()
        print("Data loaded successfully; shape=%s" % (df.shape,))
    except DataLoaderError as e:
        logging.error("DataLoaderError: %s", e)
    except Exception as e:
        logging.exception("Unexpected error: %s", e)

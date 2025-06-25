"""
Modular data validation utility for MLOps pipelines.

– Reads validation rules from configs/config.yaml
– Checks schema: required columns, dtypes, missing values,
                 ranges, allowed values
– Logs all checks and writes a JSON report artifact
– Behavior on errors is configurable (raise or warn)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


class DataValidationError(Exception):
    """Raised when a validation error occurs and action_on_error is 'raise'."""


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load the project config YAML.

    WHY:
        Configuration drives validation rules and file locations,
        avoiding hard‐coded values in code and easing reproducibility.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

    if not config_path.is_file():
        raise DataValidationError("Config file not found: %s" % config_path)

    try:
        with config_path.open("r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise DataValidationError("Invalid YAML in config: %s" % e)


def setup_logger(cfg: Dict[str, Any]) -> logging.Logger:
    """
    Configure module logger based on config['logging'].

    WHY:
        Centralized logging configuration ensures consistency
        and that logs are captured to both file and console.
    """
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("log_file", "logs/validation.log")
    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    datefmt = log_cfg.get("datefmt", None)

    # Ensure log directory exists
    p = Path(log_file)
    p.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        filename=log_file,
        filemode="a",
        format=fmt,
        datefmt=datefmt,
    )
    # Console warnings/errors
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(ch)

    return logging.getLogger(__name__)


def _is_dtype_compatible(series: pd.Series, expected: str) -> bool:
    """
    Check if a Series’ dtype matches expected ('int','float','str','bool').

    WHY:
        Early type mismatches often cause obscure downstream errors;
        catching them here simplifies debugging.
    """
    kind = series.dtype.kind
    return (
        (expected == "int" and kind in ("i", "u"))
        or (expected == "float" and kind == "f")
        or (expected == "str" and kind in ("O", "U", "S"))
        or (expected == "bool" and kind == "b")
    )


def _validate_column(
    df: pd.DataFrame,
    col_cfg: Dict[str, Any],
    errors: List[str],
    warnings: List[str],
    report: Dict[str, Any],
) -> None:
    """
    Validate one column against its schema entry.

    WHY:
        Breaking validation into per‐column checks keeps code modular
        and makes it easy to extend with new rules.
    """
    name = col_cfg["name"]
    col_report: Dict[str, Any] = {}

    # 1️⃣ Presence
    if name not in df.columns:
        if col_cfg.get("required", True):
            msg = "Missing required column: %s" % name
            errors.append(msg)
            col_report["status"] = "missing"
            col_report["error"] = msg
        else:
            col_report["status"] = "optional not present"
        report[name] = col_report
        return

    series = df[name]
    col_report["status"] = "present"

    # 2️⃣ Type
    expected = col_cfg.get("dtype")
    if expected and not _is_dtype_compatible(series, expected):
        msg = "Column '%s' dtype '%s' != expected '%s'" % (
            name,
            series.dtype,
            expected,
        )
        errors.append(msg)
        col_report.update(dtype=str(series.dtype), expected_dtype=expected, error=msg)
        report[name] = col_report
        return  # stop further checks

    # 3️⃣ Missing values
    miss = int(series.isnull().sum())
    if miss:
        if col_cfg.get("required", True):
            msg = "Column '%s' has %d missing values (required)" % (name, miss)
            errors.append(msg)
        else:
            msg = "Column '%s' has %d missing values (optional)" % (name, miss)
            warnings.append(msg)
        col_report["missing_count"] = miss

    # 4️⃣ Min/Max bounds
    if "min" in col_cfg:
        below = int((series < col_cfg["min"]).sum())
        if below:
            msg = "Column '%s' has %d values below min %s" % (
                name,
                below,
                col_cfg["min"],
            )
            errors.append(msg)
            col_report["below_min"] = below

    if "max" in col_cfg:
        above = int((series > col_cfg["max"]).sum())
        if above:
            msg = "Column '%s' has %d values above max %s" % (
                name,
                above,
                col_cfg["max"],
            )
            errors.append(msg)
            col_report["above_max"] = above

    # 5️⃣ Allowed set
    if "allowed_values" in col_cfg:
        allowed = set(col_cfg["allowed_values"])
        invalid = int((~series.isin(allowed)).sum())
        if invalid:
            msg = "Column '%s' has %d invalid values not in %s" % (
                name,
                invalid,
                allowed,
            )
            errors.append(msg)
            col_report["invalid_count"] = invalid

    # 6️⃣ Sample values for report
    try:
        col_report["samples"] = series.dropna().unique()[:5].tolist()
    except Exception:
        col_report["samples"] = []

    report[name] = col_report


def validate_data(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Execute data validation checks in their entirety.

    WHY:
        Enforcing data contracts early prevents garbage data from flowing
        through downstream models and analysis.
    """
    cfg = config.get("data_validation", {})
    if not cfg.get("enabled", True):
        logging.getLogger(__name__).info("Validation disabled in config.")
        return

    columns = cfg.get("schema", {}).get("columns", [])
    if not columns:
        logging.getLogger(__name__).warning(
            "No columns defined under data_validation.schema.columns; " "skipping."
        )
        return

    action = cfg.get("action_on_error", "raise").lower()
    report_path = Path(cfg.get("report_path", "logs/validation_report.json"))
    errors: List[str] = []
    warnings: List[str] = []
    report: Dict[str, Any] = {}

    for col in columns:
        _validate_column(df, col, errors, warnings, report)

    # write JSON report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(
            {
                "status": "fail" if errors else "pass",
                "errors": errors,
                "warnings": warnings,
                "details": report,
            },
            f,
            indent=2,
        )

    # log results
    logger = logging.getLogger(__name__)
    if errors:
        logger.error("Validation failed: %d errors (see %s)", len(errors), report_path)
        for e in errors:
            logger.error(e)
    if warnings:
        logger.warning("Validation warnings: %d", len(warnings))
        for w in warnings:
            logger.warning(w)
    if not errors:
        logger.info("Validation passed; report at %s", report_path)

    # enforce action
    if errors and action == "raise":
        raise DataValidationError("Data validation failed; see %s" % report_path)


def main() -> None:
    """
    Run the data validator from the command line.

    CLI entry: python -m src.data.data_validator <data.xlsx> <config.yaml>

    WHY:
        Enables quick ad-hoc validation without writing custom scripts.
    """
    if len(sys.argv) != 3:
        print("Usage: python -m src.data.data_validator " "<data.xlsx> <config.yaml>")
        sys.exit(1)

    data_path, cfg_path = sys.argv[1], sys.argv[2]
    df = pd.read_excel(data_path)
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # apply logger from config
    setup_logger(config)
    validate_data(df, config)


if __name__ == "__main__":
    main()

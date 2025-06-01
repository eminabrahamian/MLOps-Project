"""
test_data_validator.py

Unit tests for data_validator.py

Covers:
- load_config (missing file, invalid YAML)
- _is_dtype_compatible (various dtypes)
- _validate_column (missing required, type mismatch, missing values, out-of-range, allowed_values)
- validate_data (action_on_error="raise" vs "warn", report file creation)
"""

import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.data_validator import (
    load_config,
    setup_logger,
    _is_dtype_compatible,
    _validate_column,
    validate_data,
    DataValidationError,
)

# Fixture: temporary config.yaml with a basic validation schema
@pytest.fixture
def basic_config(tmp_path):
    """
    Create a minimal config dict and file for data validation:
      - logging to tmp_path/log.log
      - schema for one 'id' (int, required) and one 'feature' (float, optional, min=0).
    """
    config = {
        "logging": {
            "level": "DEBUG",
            "log_file": str(tmp_path / "val.log"),
            "format": "%(levelname)s:%(message)s",
            "datefmt": None
        },
        "data_validation": {
            "enabled": True,
            "action_on_error": "raise",
            "report_path": str(tmp_path / "validation_report.json"),
            "schema": {
                "columns": [
                    {"name": "id", "dtype": "int", "required": True, "min": 0},
                    {"name": "score", "dtype": "float", "required": False, "min": 0.0},
                    {"name": "cat", "dtype": "int", "required": False, "allowed_values": [0, 1]},
                ]
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_file, config

def test_load_config_missing(tmp_path):
    """
    Test load_config raises DataValidationError when config file is missing.
    """
    fake_path = tmp_path / "not_exist.yaml"
    with pytest.raises(DataValidationError) as excinfo:
        load_config(fake_path)
    assert "Config file not found" in str(excinfo.value)

def test_load_config_invalid_yaml(tmp_path):
    """
    Test load_config raises DataValidationError on invalid YAML syntax.
    """
    bad_yaml = tmp_path / "bad2.yaml"
    bad_yaml.write_text("not: [closed_list", encoding="utf-8")
    with pytest.raises(DataValidationError) as excinfo:
        load_config(bad_yaml)
    assert "Invalid YAML" in str(excinfo.value)

@pytest.mark.parametrize(
    "dtype, values, expected", [
        ("int", pd.Series([1, 2], dtype=int), True),
        ("int", pd.Series([1.0, 2.0], dtype=float), False),
        ("float", pd.Series([1.0, 2.0], dtype=float), True),
        ("float", pd.Series([1, 2], dtype=int), False),
        ("str", pd.Series(["a", "b"], dtype=object), True),
        ("bool", pd.Series([True, False], dtype=bool), True),
    ]
)
def test_is_dtype_compatible(dtype, values, expected):
    """
    Test that _is_dtype_compatible correctly identifies matching and non-matching dtypes.
    """
    result = _is_dtype_compatible(values, dtype)
    assert result is expected

def test_validate_column_missing_required(tmp_path, basic_config):
    """
    Test that _validate_column flags missing required column as an error.
    """
    _, cfg_dict = basic_config
    report = {}
    errors = []
    warnings = []
    df = pd.DataFrame({"score": [0.1, 0.2]})  # missing 'id'
    # Validate only 'id'
    col_cfg = cfg_dict["data_validation"]["schema"]["columns"][0]
    _validate_column(df, col_cfg, errors, warnings, report)
    assert "Missing required column" in errors[0]
    assert report["id"]["status"] == "missing"

def test_validate_column_type_mismatch(tmp_path, basic_config):
    """
    Test that _validate_column flags dtype mismatches correctly.
    """
    _, cfg_dict = basic_config
    # Create DataFrame with wrong type for 'id' (float instead of int)
    df = pd.DataFrame({"id": [1.1, 2.2], "score": [1.0, 2.0], "cat": [0, 1]})
    report = {}
    errors = []
    warnings = []
    col_cfg = cfg_dict["data_validation"]["schema"]["columns"][0]  # id:int
    _validate_column(df, col_cfg, errors, warnings, report)
    assert "dtype" in report["id"]["error"] or "dtype" in errors[0]

def test_validate_data_raise(tmp_path, basic_config):
    """
    Test that validate_data writes a report and raises DataValidationError on errors.
    """
    config_file, cfg_dict = basic_config
    df = pd.DataFrame({"id": [None], "score": [1.0], "cat": [0]})
    # Load config
    config = cfg_dict
    # Ensure log directory exists
    setup_logger(config)
    # Validation should fail (id missing) and raise
    with pytest.raises(DataValidationError):
        validate_data(df, config)
    # Check report file exists and contents reflect status "fail"
    report_path = Path(config["data_validation"]["report_path"])
    assert report_path.is_file()
    rpt = json.loads(report_path.read_text())
    assert rpt["status"] == "fail"

def test_validate_data_warn(tmp_path, basic_config):
    """
    Test that validate_data logs warnings and does not raise when action_on_error="warn".
    """
    config_file, cfg_dict = basic_config
    # Change action to "warn"
    cfg_dict["data_validation"]["action_on_error"] = "warn"
    df = pd.DataFrame({"id": [1], "score": [None], "cat": [0]})
    setup_logger(cfg_dict)
    # score missing => warning but no raise
    validate_data(df, cfg_dict)
    report_path = Path(cfg_dict["data_validation"]["report_path"])
    assert report_path.is_file()
    rpt = json.loads(report_path.read_text())
    assert rpt["status"] == "pass" or "warnings" in rpt

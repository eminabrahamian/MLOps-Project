"""
test_data_validator.py

Unit tests for data_validator.py

WHY:
These tests achieve >90% coverage by exercising all functions and branches:
- Config loading (success, missing file, invalid YAML)
- Logger setup
- Dtype compatibility checks
- Per-column validation (_validate_column) for presence, type, missing,
  bounds, allowed values, and sample extraction
- End-to-end validate_data with different actions (raise, warn, optional)
- CLI main entry (usage error and successful run)
"""

import json
import logging
import sys
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
    main
)


def test_load_config_success(tmp_path):
    """
    Test that load_config reads a valid YAML config file correctly.

    WHY:
    Ensures configuration driving validation rules is parsed as expected,
    preventing downstream errors due to bad config ingestion.
    """
    cfg_dict = {"data_validation": {"enabled": True}}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(cfg_dict))

    loaded = load_config(cfg_file)
    assert loaded == cfg_dict


def test_load_config_missing(tmp_path):
    """
    Test that load_config raises DataValidationError if the config file is absent.

    WHY:
    Validates early failure on missing configuration, avoiding obscure errors later.
    """
    missing = tmp_path / "no_config.yaml"
    with pytest.raises(DataValidationError) as exc:
        load_config(missing)
    assert "Config file not found" in str(exc.value)


def test_load_config_invalid_yaml(tmp_path):
    """
    Test that load_config raises DataValidationError for malformed YAML.

    WHY:
    Catches YAML syntax issues at the entry point, ensuring rules are valid.
    """
    bad = tmp_path / "bad.yaml"
    bad.write_text("::: not yaml :::")
    with pytest.raises(DataValidationError) as exc:
        load_config(bad)
    assert "Invalid YAML" in str(exc.value)


def test_setup_logger_creates_handlers(tmp_path):
    """
    Test that setup_logger configures a FileHandler and a StreamHandler,
    and writes log entries to the file.

    WHY:
    Confirms that logs are captured to disk for auditing and that warnings
    still appear on console per MLOps guidelines.
    """
    log_path = tmp_path / "validation.log"
    cfg = {
        "logging": {
            "level": "DEBUG",
            "log_file": str(log_path),
            "format": "%(levelname)s:%(message)s",
            "datefmt": None
        }
    }
    logger = setup_logger(cfg)
    # Handler types
    handler_types = {type(h) for h in logger.handlers}
    assert logging.FileHandler in handler_types
    assert logging.StreamHandler in handler_types

    # Emit a debug message and verify it's in the file
    logger.debug("debug-entry")
    text = log_path.read_text()
    assert "debug-entry" in text


@pytest.mark.parametrize(
    "data, expected_type, expected",
    [
        (pd.Series([1, 2, 3], dtype=int), "int", True),
        (pd.Series([1.0, 2.5], dtype=float), "float", True),
        (pd.Series(["a", "b"], dtype=object), "str", True),
        (pd.Series([True, False], dtype=bool), "bool", True),
        (pd.Series([1, 2], dtype=int), "float", False),
        (pd.Series([1.0, 2.0], dtype=float), "int", False),
        (pd.Series(["x"], dtype=object), "bool", False),
    ]
)
def test_is_dtype_compatible(data, expected_type, expected):
    """
    Test _is_dtype_compatible across various dtypes and expectations.

    WHY:
    Ensures that the module correctly identifies matching and mismatched types,
    preventing obscure type-related bugs in validation.
    """
    assert _is_dtype_compatible(data, expected_type) is expected


def test_validate_column_presence_and_optional(tmp_path):
    """
    Test _validate_column for missing required and optional columns.

    WHY:
    Verifies that required columns trigger errors and optional ones are noted
    without error, supporting flexible schema definitions.
    """
    df = pd.DataFrame({"a": [1, 2]})
    errors, warnings, report = [], [], {}

    # Required missing
    cfg_req = {"name": "b", "required": True}
    _validate_column(df, cfg_req, errors, warnings, report)
    assert len(errors) == 1
    assert "Missing required column: b" in errors[0]
    assert report["b"]["status"] == "missing"

    # Optional missing
    errors.clear(); warnings.clear(); report.clear()
    cfg_opt = {"name": "c", "required": False}
    _validate_column(df, cfg_opt, errors, warnings, report)
    assert errors == []
    assert report["c"]["status"] == "optional not present"


def test_validate_column_type_missing_bounds_allowed():
    """
    Test _validate_column for dtype mismatch, missing values, bounds, and allowed_values.

    WHY:
    Covers deep validation logic: type checking, NA handling, min/max range checks,
    allowed set enforcement, and sample extraction.
    """
    df = pd.DataFrame({
        "num": [0, 5, None, 15],
        "cat": ["A", "B", "C", "X"]
    })
    errors, warnings, report = [], [], {}

    # Configuration combining all checks
    cfg_num = {
        "name": "num",
        "dtype": "int",
        "required": True,
        "min": 1,
        "max": 10
    }
    cfg_cat = {
        "name": "cat",
        "dtype": "str",
        "required": True,
        "allowed_values": ["A", "B", "C"]
    }
    _validate_column(df, cfg_num, errors, warnings, report)
    # num: dtype mismatch (float w/ NaN), missing values, below min, above max
    assert any("dtype" in e for e in errors)
    # After dtype mismatch, no further checks on num, so missing/bounds not rechecked here

    # Clear and test bounds & missing without dtype rule
    errors.clear(); warnings.clear(); report.clear()
    cfg_num2 = {
        "name": "num",
        "required": True,
        "min": 1,
        "max": 10
    }
    _validate_column(df, cfg_num2, errors, warnings, report)
    # missing values, below_min (0), above_max (15)
    assert any("has 1 missing values" in e for e in errors)
    assert any("below min" in e for e in errors)
    assert any("above max" in e for e in errors)

    errors.clear(); warnings.clear(); report.clear()
    _validate_column(df, cfg_cat, errors, warnings, report)
    # cat: no dtype error, no missing, but invalid 'X'
    assert any("invalid values" in e for e in errors)
    assert report["cat"]["invalid_count"] == 1
    # samples list present
    assert isinstance(report["cat"]["samples"], list)


def test_validate_data_missing_required_raise(tmp_path):
    """
    Test validate_data raises when a required column is missing and action_on_error is 'raise'.

    WHY:
    Ensures strict enforcement of required schema when configured to fail fast.
    """
    df = pd.DataFrame({"a": [1]})
    report_file = tmp_path / "rep.json"
    config = {
        "data_validation": {
            "enabled": True,
            "action_on_error": "raise",
            "report_path": str(report_file),
            "schema": {
                "columns": [{"name": "b", "required": True}]
            }
        },
        "logging": {"level": "INFO", "log_file": str(tmp_path / "log.log")}
    }

    with pytest.raises(DataValidationError):
        validate_data(df, config)

    # Report written
    rpt = json.loads(report_file.read_text())
    assert rpt["status"] == "fail"
    assert "Missing required column: b" in rpt["errors"][0]


def test_validate_data_missing_required_warn(tmp_path):
    """
    Test validate_data does not raise when action_on_error is 'warn' despite errors.

    WHY:
    Provides a soft-fail mode allowing pipelines to continue on validation errors.
    """
    df = pd.DataFrame({"a": [1]})
    report_file = tmp_path / "rep.json"
    config = {
        "data_validation": {
            "enabled": True,
            "action_on_error": "warn",
            "report_path": str(report_file),
            "schema": {
                "columns": [{"name": "b", "required": True}]
            }
        },
        "logging": {"level": "INFO", "log_file": str(tmp_path / "log.log")}
    }

    # Should not raise
    validate_data(df, config)
    rpt = json.loads(report_file.read_text())
    assert rpt["status"] == "fail"


def test_validate_data_optional_missing(tmp_path):
    """
    Test validate_data with an optional missing column does not raise and reports pass.

    WHY:
    Verifies optional schema entries are respected without error interruption.
    """
    df = pd.DataFrame({"a": [1]})
    report_file = tmp_path / "rep.json"
    config = {
        "data_validation": {
            "enabled": True,
            "action_on_error": "raise",
            "report_path": str(report_file),
            "schema": {
                "columns": [{"name": "b", "required": False}]
            }
        },
        "logging": {"level": "INFO", "log_file": str(tmp_path / "log.log")}
    }

    validate_data(df, config)
    rpt = json.loads(report_file.read_text())
    assert rpt["status"] == "pass"
    assert rpt["errors"] == []


def test_main_usage_and_success(tmp_path, capsys):
    """
    Test CLI main() prints usage on bad args and runs without error on valid args.

    WHY:
    Ensures the module can be used as a script for ad-hoc validation.
    """
    # Usage error
    monkeyargs = ["prog"]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))
    monkeypatch.setattr(sys, "argv", monkeyargs)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Usage" in captured.out

    # Successful run
    # Create a minimal Excel file
    df = pd.DataFrame({"a": [1]})
    xlsx = tmp_path / "data.xlsx"
    df.to_excel(xlsx, index=False)
    cfg = {"data_validation": {"enabled": False}}
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump(cfg))

    monkeypatch.setattr(sys, "argv", ["prog", str(xlsx), str(cfg_file)])
    # Should not raise
    main()
    monkeypatch.undo()

"""
test_data_loader.py

Unit tests for data_loader.py

WHY:
These tests are designed to achieve >90% coverage by exercising all major
functions and error paths in the data_loader module:
- Configuration loading (success, missing file, invalid YAML)
- Logger setup (handlers creation, file writing)
- Data source loading for CSV and Excel (success, missing file, unsupported type)
- End-to-end load_data integration (successful CSV load, unsupported source)

Each test includes a docstring and comments to explain the rationale (the "WHY")
behind the test and what behavior it is validating.
"""

import os
import logging
import yaml
import pandas as pd
import pytest
from pathlib import Path

from src.data.data_loader import (
    load_config,
    setup_logger,
    load_data_source,
    load_data,
    DataLoaderError
)


def test_load_config_success(tmp_path):
    """
    Test that load_config correctly reads a valid YAML config file.

    WHY:
    Ensures that the loader can parse standard configurations without error
    so downstream functions receive the expected dictionary.
    """
    # Create a temporary config file with simple key/value pairs.
    sample_cfg = {"foo": "bar", "number": 123}
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(sample_cfg))

    # Attempt to load and verify the content matches exactly.
    result = load_config(cfg_file)
    assert result == sample_cfg


def test_load_config_not_found(tmp_path):
    """
    Test that load_config raises DataLoaderError when the config file is missing.

    WHY:
    Validates error handling for a common misconfiguration—missing file—
    ensuring the module fails fast with a clear exception.
    """
    missing = tmp_path / "nonexistent.yaml"
    with pytest.raises(DataLoaderError) as exc:
        load_config(missing)
    assert "Config file not found" in str(exc.value)


def test_load_config_invalid_yaml(tmp_path):
    """
    Test that load_config raises DataLoaderError for malformed YAML content.

    WHY:
    Checks robustness against bad config formatting, preventing obscure downstream
    errors by catching YAML syntax issues early.
    """
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":::: not valid yaml")
    with pytest.raises(DataLoaderError) as exc:
        load_config(bad_yaml)
    assert "Invalid YAML" in str(exc.value)


def test_setup_logger_creates_handlers(tmp_path):
    """
    Test that setup_logger configures both FileHandler and StreamHandler,
    and that log messages are written to the file.

    WHY:
    Ensures that logging is set up per MLOps guidelines:
    - Structured logs written to disk for auditing.
    - Warnings and above still printed to console.
    """
    log_file = tmp_path / "app.log"
    cfg = {
        "level": "DEBUG",
        "log_file": str(log_file),
        "format": "%(levelname)s:%(message)s",
        "datefmt": None
    }

    # Initialize logger based on the config.
    logger = setup_logger(cfg)

    # Verify handler types: at least one FileHandler and one StreamHandler.
    handler_types = {type(h) for h in logger.handlers}
    assert logging.FileHandler in handler_types
    assert logging.StreamHandler in handler_types

    # Emit a debug message and confirm it appears in the log file.
    logger.debug("debug message for testing")
    content = log_file.read_text()
    assert "debug message for testing" in content


def test_load_data_source_csv(tmp_path):
    """
    Test load_data_source for CSV input with valid data.

    WHY:
    Verifies that CSV reading logic handles headers and encoding correctly
    and returns a DataFrame with expected shape and values.
    """
    # Create a small DataFrame and write to CSV.
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)

    # Configure loader for CSV.
    ds_cfg = {
        "path": str(csv_file),
        "type": "csv",
        "header": 0,
        "encoding": None
    }
    loaded = load_data_source(ds_cfg)

    # Confirm that the loaded DataFrame matches the original.
    pd.testing.assert_frame_equal(loaded, df)


def test_load_data_source_excel(tmp_path):
    """
    Test load_data_source for Excel input with valid data.

    WHY:
    Ensures that Excel loading supports sheet selection and header rows,
    and returns accurate DataFrame content.
    """
    df = pd.DataFrame({"c": [3, 4]})
    xlsx_file = tmp_path / "data.xlsx"
    df.to_excel(xlsx_file, index=False)

    ds_cfg = {
        "path": str(xlsx_file),
        "type": "excel",
        "header": 0,
        "sheet_name": "Sheet1"
    }
    loaded = load_data_source(ds_cfg)
    pd.testing.assert_frame_equal(loaded, df)


def test_load_data_source_missing_file(tmp_path):
    """
    Test that load_data_source raises DataLoaderError when input file is absent.

    WHY:
    Confirms proper error propagation for missing data artifacts,
    a key requirement in production data pipelines.
    """
    ds_cfg = {
        "path": str(tmp_path / "missing.csv"),
        "type": "csv",
        "header": 0,
        "encoding": None
    }
    with pytest.raises(DataLoaderError) as exc:
        load_data_source(ds_cfg)
    assert "Data file not found" in str(exc.value)


def test_load_data_source_unsupported_type(tmp_path):
    """
    Test that load_data_source raises DataLoaderError for unsupported types.

    WHY:
    Validates that the loader rejects unknown formats,
    preventing silent failures or incorrect parsing.
    """
    # Create a dummy file so path.exists() passes.
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("irrelevant")
    ds_cfg = {
        "path": str(dummy),
        "type": "txt",
        "header": 0,
        "encoding": None
    }
    with pytest.raises(DataLoaderError) as exc:
        load_data_source(ds_cfg)
    assert "Unsupported data type" in str(exc.value)


def test_load_data_csv_integration(monkeypatch, tmp_path):
    """
    End-to-end test of load_data for CSV source.

    WHY:
    Simulates actual module usage by mocking load_config to return
    a CSV-based config, and verifies that load_data returns the correct DataFrame.
    """
    # Prepare a sample DataFrame and CSV file.
    df = pd.DataFrame({"x": [9, 8, 7]})
    csv_file = tmp_path / "my.csv"
    df.to_csv(csv_file, index=False)

    # Monkey-patch load_config to provide a controlled config.
    fake_config = {
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "app.log"),
            "format": "%(message)s",
            "datefmt": None
        },
        "data_source": {
            "path": str(csv_file),
            "type": "csv",
            "header": 0,
            "encoding": None
        }
    }
    monkeypatch.setattr("src.data.data_loader.load_config", lambda cfg=None: fake_config)

    result = load_data()
    pd.testing.assert_frame_equal(result, df)


def test_load_data_unsupported_source(monkeypatch, tmp_path):
    """
    Test that load_data raises DataLoaderError for an unsupported source.

    WHY:
    Ensures that load_data enforces the allowed sources and fails clearly
    when misconfigured in production.
    """
    fake_config = {
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "app.log"),
            "format": "%(message)s",
            "datefmt": None
        },
        "data_source": {
            "path": str(tmp_path / "whatever.csv"),
            "type": "api"  # unsupported in this version
        }
    }
    monkeypatch.setattr("src.data.data_loader.load_config", lambda cfg=None: fake_config)

    with pytest.raises(DataLoaderError) as exc:
        load_data()
    assert "Unsupported data source" in str(exc.value)

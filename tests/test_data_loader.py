"""
test_data_loader.py

Unit tests for data_loader.py

Covers:
- load_config (missing file, invalid YAML, default path)
- setup_logger (returns a logger, writes to specified file)
- load_data_source (CSV vs. Excel, missing file, unsupported type)
- load_data (integration: reads YAML, loads data, empty vs. non-empty)
"""

import logging
import os
import tempfile
import yaml
from pathlib import Path

import pandas as pd
import pytest

from data_loader.data_loader import (
    load_config,
    setup_logger,
    load_data_source,
    load_data,
    DataLoaderError,
)

# Fixture: create a temporary config.yaml for testing
@pytest.fixture
def temp_config(tmp_path):
    """
    Create a temporary config.yaml with:
      - minimal logging section
      - data_source pointing to a CSV and then an Excel.
    """
    cfg = {
        "logging": {
            "level": "DEBUG",
            "log_file": str(tmp_path / "test.log"),
            "format": "%(levelname)s:%(message)s",
            "datefmt": None
        },
        "data_source": {
            "raw_path": str(tmp_path / "data.csv"),
            "type": "csv",
            "header": 0,
            "encoding": "utf-8"
        }
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return config_file, cfg

def test_load_config_missing(tmp_path):
    """
    Test load_config raises DataLoaderError when the config file does not exist.
    """
    fake_path = tmp_path / "nonexistent.yaml"
    with pytest.raises(DataLoaderError) as excinfo:
        load_config(fake_path)
    assert "Config file not found" in str(excinfo.value)

def test_load_config_invalid_yaml(tmp_path):
    """
    Test load_config raises DataLoaderError when YAML is invalid.
    """
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("this: [unclosed_list", encoding="utf-8")
    with pytest.raises(DataLoaderError) as excinfo:
        load_config(bad_yaml)
    assert "Invalid YAML" in str(excinfo.value)

def test_setup_logger_creates_file(tmp_path, temp_config):
    """
    Test that setup_logger returns a logger and creates the log file when logging messages.
    """
    config_file, cfg_dict = temp_config
    log_cfg = cfg_dict["logging"]
    logger = setup_logger(log_cfg)
    # Emit a log; should write to the specified file
    logger.debug("Test debug log")
    # File may not exist immediately until flush; explicitly flush handlers
    for h in logging.root.handlers:
        h.flush()
    log_path = Path(log_cfg["log_file"])
    assert log_path.is_file(), "Log file should be created"
    content = log_path.read_text()
    assert "Test debug log" in content

def test_load_data_source_csv(tmp_path, temp_config):
    """
    Test load_data_source can read a CSV with header and encoding correctly.
    """
    config_file, cfg_dict = temp_config
    csv_path = tmp_path / "data.csv"
    # Write a small CSV
    df_orig = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df_orig.to_csv(csv_path, index=False)
    ds_cfg = {
        "raw_path": str(csv_path),
        "type": "csv",
        "header": 0,
        "encoding": "utf-8"
    }
    df_loaded = load_data_source(ds_cfg)
    # DataFrame equality check
    pd.testing.assert_frame_equal(df_loaded.reset_index(drop=True), df_orig)

def test_load_data_source_excel(tmp_path):
    """
    Test load_data_source can read an Excel sheet with default sheet name.
    """
    excel_path = tmp_path / "data.xlsx"
    df_orig = pd.DataFrame({"x": [3, 4], "y": [True, False]})
    df_orig.to_excel(excel_path, sheet_name="Sheet1", index=False)
    ds_cfg = {
        "raw_path": str(excel_path),
        "type": "excel",
        "header": 0,
        "sheet_name": "Sheet1",
        "encoding": None
    }
    df_loaded = load_data_source(ds_cfg)
    pd.testing.assert_frame_equal(df_loaded, df_orig)

def test_load_data_source_missing_file(tmp_path):
    """
    Test load_data_source raises DataLoaderError when the file is missing.
    """
    ds_cfg = {"raw_path": str(tmp_path / "no.csv"), "type": "csv", "header": 0, "encoding": "utf-8"}
    with pytest.raises(DataLoaderError) as excinfo:
        load_data_source(ds_cfg)
    assert "Data file not found" in str(excinfo.value)

def test_load_data_source_unsupported_type(tmp_path):
    """
    Test load_data_source raises DataLoaderError on unsupported 'type' values.
    """
    # Create a dummy CSV so path exists
    csv_path = tmp_path / "data2.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)
    ds_cfg = {"raw_path": str(csv_path), "type": "txt", "header": 0, "encoding": "utf-8"}
    with pytest.raises(DataLoaderError) as excinfo:
        load_data_source(ds_cfg)
    assert "Unsupported data type" in str(excinfo.value)

def test_load_data_integration(tmp_path, monkeypatch, temp_config):
    """
    Integration test for load_data:
      - Monkeypatch load_config to return our temp config
      - Monkeypatch load_data_source to return a non-empty DataFrame
      - Verify load_data returns the DataFrame and logs correctly
    """
    config_file, cfg_dict = temp_config

    # Write a small CSV & update ds_cfg path
    csv_path = tmp_path / "dataraw.csv"
    df_test = pd.DataFrame({"foo": [10, 20]})
    df_test.to_csv(csv_path, index=False)
    cfg_dict["data_source"]["raw_path"] = str(csv_path)

    # Monkeypatch load_config to ignore default path
    def fake_load_config(arg=None):
        return cfg_dict
    monkeypatch.setattr("src.data.data_loader.load_config", fake_load_config)

    # Now call load_data; it should return df_test
    df_loaded = load_data()
    pd.testing.assert_frame_equal(df_loaded.reset_index(drop=True), df_test)

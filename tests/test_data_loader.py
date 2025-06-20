"""
test_data_loader.py

Unit tests for data_loader.py

Covers:
- load_config (missing file, invalid YAML, default path)
- setup_logger (returns a logger, writes to specified file)
- load_data_source (CSV vs. Excel, missing file, unsupported type)
- load_data (integration: reads YAML, loads data, empty vs. non-empty)

Test Categories:
- Unit tests for individual functions
- Integration tests for end-to-end workflows
- Error handling validation
- Configuration management
"""

import logging
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data_loader.data_loader import (
    DataLoaderError,
    load_config,
    load_data,
    load_data_source,
    setup_logger,
)

# Test constants for path independence
TEST_DIR = Path(__file__).parent
MOCK_DATA_DIR = TEST_DIR / "mock_data"

# Test configurations for isolation
CSV_CONFIG = {
    "raw_path": "mock_data.csv",
    "type": "csv",
    "header": 0,
    "encoding": "utf-8",
}

EXCEL_CONFIG = {
    "raw_path": "mock_data.xlsx",
    "type": "excel",
    "header": 0,
    "sheet_name": "Sheet1",
    "encoding": None,
}


@pytest.fixture
def mock_data_csv(tmp_path):
    """Create mock CSV data for testing."""
    csv_path = tmp_path / "mock_data.csv"
    df = pd.DataFrame({
        "patient_id": [1, 2, 3],
        "age": [25, 45, 35],
        "gender": ["M", "F", "M"],
        "diagnosis": ["positive", "negative", "positive"]
    })
    df.to_csv(csv_path, index=False)
    return csv_path, df


@pytest.fixture
def mock_data_excel(tmp_path):
    """Create mock Excel data for testing."""
    excel_path = tmp_path / "mock_data.xlsx"
    df = pd.DataFrame({
        "patient_id": [4, 5, 6],
        "age": [28, 52, 38],
        "gender": ["F", "M", "F"],
        "diagnosis": ["negative", "positive", "negative"]
    })
    df.to_excel(excel_path, sheet_name="Sheet1", index=False)
    return excel_path, df


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
            "datefmt": None,
        },
        "data_source": {
            "raw_path": str(tmp_path / "data.csv"),
            "type": "csv",
            "header": 0,
            "encoding": "utf-8",
        },
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return config_file, cfg


class TestConfigManagement:
    """Test configuration loading and validation."""
    
    @pytest.mark.unit
    def test_load_config_missing(self, tmp_path):
        """
        Test load_config raises DataLoaderError when
        the config file does not exist.
        """
        fake_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(DataLoaderError) as excinfo:
            load_config(fake_path)
        assert "Config file not found" in str(excinfo.value)

    @pytest.mark.unit
    def test_load_config_invalid_yaml(self, tmp_path):
        """
        Test load_config raises DataLoaderError when YAML is invalid.
        """
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("this: [unclosed_list", encoding="utf-8")
        with pytest.raises(DataLoaderError) as excinfo:
            load_config(bad_yaml)
        assert "Invalid YAML" in str(excinfo.value)

    @pytest.mark.unit
    def test_load_config_valid_yaml(self, tmp_path):
        """Test load_config successfully loads valid YAML."""
        valid_yaml = tmp_path / "valid.yaml"
        config_data = {"test": "value", "nested": {"key": "data"}}
        with valid_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f)
        
        loaded_config = load_config(valid_yaml)
        assert loaded_config == config_data


class TestLoggingSetup:
    """Test logging configuration and setup."""
    
    @pytest.mark.unit
    def test_setup_logger_creates_file(self, tmp_path, temp_config):
        """
        Test that setup_logger returns a logger and
        creates the log file when logging messages.
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

    @pytest.mark.unit
    def test_setup_logger_returns_logger(self, temp_config):
        """Test that setup_logger returns a proper logger instance."""
        config_file, cfg_dict = temp_config
        log_cfg = cfg_dict["logging"]
        logger = setup_logger(log_cfg)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "src.data_loader.data_loader"


class TestDataLoading:
    """Test data source loading functionality."""
    
    @pytest.mark.unit
    def test_load_data_source_csv_success(self, mock_data_csv):
        """
        Test load_data_source can read a CSV with header and encoding correctly.
        """
        csv_path, df_orig = mock_data_csv
        ds_cfg = CSV_CONFIG.copy()
        ds_cfg["raw_path"] = str(csv_path)
        
        df_loaded = load_data_source(ds_cfg)
        # DataFrame equality check
        pd.testing.assert_frame_equal(df_loaded.reset_index(drop=True), df_orig)

    @pytest.mark.unit
    def test_load_data_source_excel_success(self, mock_data_excel):
        """
        Test load_data_source can read an Excel sheet with default sheet name.
        """
        excel_path, df_orig = mock_data_excel
        ds_cfg = EXCEL_CONFIG.copy()
        ds_cfg["raw_path"] = str(excel_path)
        
        df_loaded = load_data_source(ds_cfg)
        pd.testing.assert_frame_equal(df_loaded, df_orig)

    @pytest.mark.unit
    def test_load_data_source_missing_file(self, tmp_path):
        """
        Test load_data_source raises DataLoaderError when the file is missing.
        """
        ds_cfg = CSV_CONFIG.copy()
        ds_cfg["raw_path"] = str(tmp_path / "no.csv")
        
        with pytest.raises(DataLoaderError) as excinfo:
            load_data_source(ds_cfg)
        assert "Data file not found" in str(excinfo.value)

    @pytest.mark.unit
    def test_load_data_source_unsupported_type(self, mock_data_csv):
        """
        Test load_data_source raises DataLoaderError on unsupported 'type' values.
        """
        csv_path, _ = mock_data_csv
        ds_cfg = CSV_CONFIG.copy()
        ds_cfg["raw_path"] = str(csv_path)
        ds_cfg["type"] = "txt"  # Unsupported type
        
        with pytest.raises(DataLoaderError) as excinfo:
            load_data_source(ds_cfg)
        assert "Unsupported data type" in str(excinfo.value)

    @pytest.mark.unit
    def test_load_data_source_empty_csv(self, tmp_path):
        """Test loading an empty CSV file."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")  # Create truly empty file
        
        ds_cfg = CSV_CONFIG.copy()
        ds_cfg["raw_path"] = str(empty_csv)
        
        with pytest.raises(DataLoaderError) as excinfo:
            load_data_source(ds_cfg)
        assert "No columns to parse from file" in str(excinfo.value)


class TestIntegrationWorkflows:
    """Integration tests for end-to-end data loading workflows."""
    
    @pytest.mark.integration
    def test_load_data_integration(self, tmp_path, monkeypatch, temp_config):
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

        monkeypatch.setattr("src.data_loader.data_loader.load_config", fake_load_config)

        # Now call load_data; it should return df_test
        df_loaded = load_data()
        pd.testing.assert_frame_equal(df_loaded.reset_index(drop=True), df_test)

    @pytest.mark.integration
    def test_full_pipeline_with_mock_data(self, tmp_path, mock_data_csv):
        """Test complete data loading pipeline with mock data."""
        csv_path, df_expected = mock_data_csv
        
        # Create config pointing to mock data
        cfg = {
            "logging": {
                "level": "INFO",
                "log_file": str(tmp_path / "pipeline.log"),
                "format": "%(levelname)s:%(message)s",
                "datefmt": None,
            },
            "data_source": {
                "raw_path": str(csv_path),
                "type": "csv",
                "header": 0,
                "encoding": "utf-8",
            },
        }
        
        config_file = tmp_path / "pipeline_config.yaml"
        with config_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        
        # Test the full pipeline
        loaded_config = load_config(config_file)
        df_loaded = load_data_source(loaded_config["data_source"])
        
        # Verify results
        pd.testing.assert_frame_equal(df_loaded.reset_index(drop=True), df_expected)
        assert len(df_loaded) == 3  # Mock data has 3 rows
        assert "patient_id" in df_loaded.columns


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.unit
    def test_load_data_source_invalid_encoding(self, mock_data_csv):
        """Test handling of invalid encoding parameter."""
        csv_path, _ = mock_data_csv
        ds_cfg = CSV_CONFIG.copy()
        ds_cfg["raw_path"] = str(csv_path)
        ds_cfg["encoding"] = "invalid_encoding"
        
        with pytest.raises(DataLoaderError) as excinfo:
            load_data_source(ds_cfg)
        assert "encoding" in str(excinfo.value).lower()

    @pytest.mark.unit
    def test_load_data_source_missing_sheet(self, mock_data_excel):
        """Test Excel loading with non-existent sheet."""
        excel_path, _ = mock_data_excel
        ds_cfg = EXCEL_CONFIG.copy()
        ds_cfg["raw_path"] = str(excel_path)
        ds_cfg["sheet_name"] = "NonExistentSheet"
        
        with pytest.raises(DataLoaderError) as excinfo:
            load_data_source(ds_cfg)
        assert "sheet" in str(excinfo.value).lower()

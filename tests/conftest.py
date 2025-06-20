"""
conftest.py

Shared test configuration and fixtures for the MLOps testing framework.

This file provides:
- Common test fixtures used across multiple test modules
- Test environment setup and teardown
- Mock data generation utilities
- Test isolation helpers
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest
import yaml


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_cancer_data():
    """
    Create mock cancer dataset for testing.
    
    Returns:
        pd.DataFrame: Mock cancer dataset with realistic features
    """
    return pd.DataFrame({
        "patient_id": range(1, 101),
        "age": [25 + i % 50 for i in range(100)],
        "gender": ["M" if i % 2 == 0 else "F" for i in range(100)],
        "diagnosis": ["positive" if i % 3 == 0 else "negative" for i in range(100)],
        "feature_1": [0.1 + i * 0.01 for i in range(100)],
        "feature_2": [0.2 + i * 0.02 for i in range(100)],
        "feature_3": [0.3 + i * 0.03 for i in range(100)],
        "feature_4": [0.4 + i * 0.04 for i in range(100)],
        "feature_5": [0.5 + i * 0.05 for i in range(100)]
    })


@pytest.fixture
def mock_config():
    """
    Create mock configuration for testing.
    
    Returns:
        Dict[str, Any]: Mock configuration dictionary
    """
    return {
        "logging": {
            "level": "INFO",
            "log_file": "test.log",
            "format": "%(levelname)s:%(message)s",
            "datefmt": None,
        },
        "data_source": {
            "raw_path": "data/raw/cancer.xlsx",
            "type": "excel",
            "header": 0,
            "sheet_name": "Sheet1",
            "encoding": None,
        },
        "preprocessing": {
            "rename_columns": {},
            "weight_col": None,
            "height_col": None,
            "icd10_chapter_flags": [],
            "interaction_columns": [],
            "outlier_columns": [],
            "z_threshold": 3.0,
            "datetime_column": None,
        },
        "features": {
            "continuous": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            "categorical": ["gender"],
            "feature_columns": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "gender"],
        },
        "model": {
            "type": "knn",
            "knn": {
                "params": {
                    "n_neighbors": 5,
                    "weights": "uniform",
                    "metric": "minkowski",
                },
                "save_path": "models/knn_model.pkl",
            },
            "save_path": "models/model.pkl",
        },
        "artifacts": {
            "splits_dir": "data/splits",
            "processed_dir": "data/processed",
            "preprocessing_pipeline": "models/pipeline.pkl",
            "model_path": "models/model.pkl",
            "metrics_dir": "metrics",
        },
    }


@pytest.fixture
def temp_config_file(tmp_path, mock_config):
    """
    Create a temporary config file for testing.
    
    Args:
        tmp_path: pytest temporary path fixture
        mock_config: Mock configuration dictionary
        
    Returns:
        Path: Path to temporary config file
    """
    config_file = tmp_path / "test_config.yaml"
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(mock_config, f)
    return config_file


@pytest.fixture
def mock_csv_data(tmp_path):
    """
    Create mock CSV data file for testing.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        tuple: (file_path, dataframe)
    """
    csv_path = tmp_path / "mock_data.csv"
    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 4, 5],
        "age": [25, 45, 35, 28, 52],
        "gender": ["M", "F", "M", "F", "M"],
        "diagnosis": ["positive", "negative", "positive", "negative", "positive"],
        "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "feature_2": [0.2, 0.3, 0.4, 0.5, 0.6]
    })
    df.to_csv(csv_path, index=False)
    return csv_path, df


@pytest.fixture
def mock_excel_data(tmp_path):
    """
    Create mock Excel data file for testing.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        tuple: (file_path, dataframe)
    """
    excel_path = tmp_path / "mock_data.xlsx"
    df = pd.DataFrame({
        "patient_id": [6, 7, 8, 9, 10],
        "age": [38, 42, 29, 55, 31],
        "gender": ["F", "M", "F", "M", "F"],
        "diagnosis": ["negative", "positive", "negative", "positive", "negative"],
        "feature_1": [0.6, 0.7, 0.8, 0.9, 1.0],
        "feature_2": [0.7, 0.8, 0.9, 1.0, 1.1]
    })
    df.to_excel(excel_path, sheet_name="Sheet1", index=False)
    return excel_path, df


@pytest.fixture
def mock_model_artifacts(tmp_path):
    """
    Create mock model artifacts directory structure.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Dict[str, Path]: Dictionary of artifact paths
    """
    artifacts = {
        "splits_dir": tmp_path / "splits",
        "processed_dir": tmp_path / "processed",
        "preprocessing_pipeline": tmp_path / "pipeline.pkl",
        "model_path": tmp_path / "model.pkl",
        "metrics_dir": tmp_path / "metrics",
    }
    
    # Create directories
    for path in artifacts.values():
        if path.suffix == "":  # Directory
            path.mkdir(parents=True, exist_ok=True)
        else:  # File
            path.parent.mkdir(parents=True, exist_ok=True)
    
    return artifacts


@pytest.fixture
def mock_metrics():
    """
    Create mock model metrics for testing.
    
    Returns:
        Dict[str, Any]: Mock metrics dictionary
    """
    return {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85,
        "roc_auc": 0.92,
        "total_samples": 100,
        "positive_samples": 35,
        "negative_samples": 65,
        "model_type": "knn",
        "dataset_name": "cancer_data",
        "version": "1.0.0"
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Setup test environment before each test.
    
    This fixture runs automatically for all tests.
    """
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup after test
    # Remove test environment variables
    os.environ.pop("TESTING", None)
    os.environ.pop("LOG_LEVEL", None)


@pytest.fixture
def mock_logger():
    """
    Create a mock logger for testing.
    
    Returns:
        logging.Logger: Mock logger instance
    """
    import logging
    
    # Create a logger that writes to a temporary file
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        log_file = f.name
    
    # Add file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    yield logger
    
    # Cleanup
    logger.handlers.clear()
    try:
        os.unlink(log_file)
    except OSError:
        pass


# Test markers for categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "data: mark test as data processing test"
    )
    config.addinivalue_line(
        "markers", "model: mark test as model training test"
    ) 
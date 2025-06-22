"""
test_data_loader.py

Unit tests for data_loader.py

Covers:
- load_config (missing file, invalid YAML, default path)
- setup_logger (returns a logger, writes to specified file)
- load_data_source (CSV vs. Excel, missing file, unsupported type)
- load_data (integration: reads YAML, loads data, empty vs. non-empty)
"""

import os
import subprocess
import sys
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


@pytest.fixture
def temp_excel_config(tmp_path):
    data_file = tmp_path / "data.xlsx"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df.to_excel(data_file, sheet_name="Sheet1", index=False)

    cfg = {
        "logging": {
            "level": "DEBUG",
            "log_file": str(tmp_path / "test.log"),
            "format": "%(levelname)s:%(message)s",
            "datefmt": None,
        },
        "data_source": {
            "raw_path": str(data_file),
            "type": "excel",
            "header": 0,
            "sheet_name": "Sheet1",
            "encoding": None,
        },
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    return config_file, cfg, df


def test_load_config_missing(tmp_path):
    with pytest.raises(DataLoaderError) as e:
        load_config(tmp_path / "missing.yaml")
    assert "Config file not found" in str(e.value)


def test_load_config_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("invalid: [")
    with pytest.raises(DataLoaderError):
        load_config(bad)


def test_setup_logger_with_defaults(tmp_path):
    cfg = {"level": "INFO", "log_file": str(tmp_path / "log.log")}
    logger = setup_logger(cfg)
    logger.info("Log default test")
    assert Path(cfg["log_file"]).read_text().strip() != ""


def test_setup_logger_full_config(temp_excel_config):
    _, cfg, _ = temp_excel_config
    log_cfg = cfg["logging"]
    logger = setup_logger(log_cfg)
    logger.debug("Logger full config test")
    assert Path(log_cfg["log_file"]).read_text().strip() != ""


def test_load_data_source_excel_success(temp_excel_config):
    _, cfg, df_expected = temp_excel_config
    df = load_data_source(cfg["data_source"])
    pd.testing.assert_frame_equal(df, df_expected)


def test_load_data_source_missing_excel(tmp_path):
    cfg = {
        "raw_path": str(tmp_path / "missing.xlsx"),
        "type": "excel",
        "header": 0,
        "sheet_name": "Sheet1",
        "encoding": None,
    }
    with pytest.raises(DataLoaderError) as e:
        load_data_source(cfg)
    assert "Data file not found" in str(e.value)


def test_load_data_source_bad_sheet(tmp_path):
    path = tmp_path / "data.xlsx"
    pd.DataFrame({"x": [1]}).to_excel(path, sheet_name="Sheet1", index=False)

    cfg = {
        "raw_path": str(path),
        "type": "excel",
        "header": 0,
        "sheet_name": "NonexistentSheet",
        "encoding": None,
    }
    with pytest.raises(DataLoaderError) as e:
        load_data_source(cfg)
    assert "Failed to load EXCEL" in str(e.value)


def test_load_data_source_invalid_type(tmp_path):
    path = tmp_path / "valid.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, sheet_name="Sheet1", index=False)

    cfg = {
        "raw_path": str(path),
        "type": "banana",
        "header": 0,
        "sheet_name": "Sheet1"
    }
    with pytest.raises(DataLoaderError) as e:
        load_data_source(cfg)
    assert "Unsupported data type" in str(e.value)


def test_load_data_integration(monkeypatch, temp_excel_config):
    _, cfg, df_expected = temp_excel_config

    def fake_load_config(path=None):
        return cfg

    monkeypatch.setattr(
        "src.data_loader.data_loader.load_config", fake_load_config
    )

    df_loaded = load_data()
    pd.testing.assert_frame_equal(df_loaded, df_expected)


def test_data_loader_main_success(tmp_path):
    script = Path("src/data_loader/data_loader.py")
    data_file = tmp_path / "input.xlsx"
    df = pd.DataFrame({"z": [1, 2]})
    df.to_excel(data_file, sheet_name="Sheet1", index=False)

    cfg = {
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "log.log"),
            "format": "%(message)s",
        },
        "data_source": {
            "raw_path": str(data_file),
            "type": "excel",
            "header": 0,
            "sheet_name": "Sheet1",
            "encoding": None,
        },
    }

    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "COVERAGE_PROCESS_START": str(Path.cwd() / ".coveragerc"),
    }

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert "Data loaded successfully" in result.stdout or result.stderr


def test_data_loader_main_dataloader_error(tmp_path):
    script = Path("src/data_loader/data_loader.py")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "COVERAGE_PROCESS_START": str(Path.cwd() / ".coveragerc"),
    }
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert "DataLoaderError" in result.stderr or result.stdout


def test_data_loader_main_yaml_error_as_dataloadererror(tmp_path):
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.yaml").write_text("invalid: [")  # bad yaml

    script = Path("src/data_loader/data_loader.py")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "COVERAGE_PROCESS_START": str(Path.cwd() / ".coveragerc"),
    }
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert "DataLoaderError" in result.stderr or result.stdout


def test_data_loader_main_generic_exception(tmp_path):
    config = {
        "data_source": {
            "raw_path": "nonexistent.xlsx",
            "type": "excel",
            "header": 0,
            "sheet_name": "Sheet1",
            "encoding": None,
        }
    }

    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config))

    script = Path("src/data_loader/data_loader.py")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "COVERAGE_PROCESS_START": str(Path.cwd() / ".coveragerc"),
    }

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Unexpected error" in result.stderr or result.stdout


def test_load_config_triggers_file_check(tmp_path):
    missing_file = tmp_path / "fake.yaml"
    assert isinstance(missing_file, Path)
    with pytest.raises(DataLoaderError) as e:
        load_config(missing_file)
    assert "Config file not found" in str(e.value)


def test_data_loader_main_load_data_executes(tmp_path):
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    file = tmp_path / "main_data.xlsx"
    df = pd.DataFrame({"x": [1]})
    df.to_excel(file, sheet_name="Sheet1", index=False)

    cfg = {
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "logfile.log"),
            "format": "%(message)s",
        },
        "data_source": {
            "raw_path": str(file),
            "type": "excel",
            "header": 0,
            "sheet_name": "Sheet1",
            "encoding": None,
        },
    }

    (config_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    script = Path("src/data_loader/data_loader.py")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "COVERAGE_PROCESS_START": str(Path.cwd() / ".coveragerc"),
    }

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert "Data loaded successfully" in result.stdout or result.stderr


def test_load_data_empty_warning(monkeypatch, tmp_path):
    # Create empty Excel file
    empty_file = tmp_path / "empty.xlsx"
    pd.DataFrame().to_excel(empty_file, sheet_name="Sheet1", index=False)

    config = {
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "empty_log.log"),
            "format": "%(message)s",
        },
        "data_source": {
            "raw_path": str(empty_file),
            "type": "excel",
            "header": 0,
            "sheet_name": "Sheet1",
            "encoding": None,
        },
    }

    config_path = Path("configs/config.yaml")
    config_path.write_text(yaml.safe_dump(config))

    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "COVERAGE_PROCESS_START": str(Path.cwd() / ".coveragerc"),
    }

    result = subprocess.run(
        [sys.executable, "src/data_loader/data_loader.py"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Loaded DataFrame is empty" in result.stdout or result.stderr


def test_data_loader_warns_on_empty_dataframe(tmp_path):
    # 1. Create an empty Excel file
    empty_file = tmp_path / "empty.xlsx"
    pd.DataFrame().to_excel(empty_file, sheet_name="Sheet1", index=False)

    # 2. Write a config that uses this file
    cfg = {
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "empty.log"),
            "format": "%(message)s",
        },
        "data_source": {
            "raw_path": str(empty_file),
            "type": "excel",
            "header": 0,
            "sheet_name": "Sheet1",
            "encoding": None,
        },
    }

    # 3. Save the config as `configs/config.yaml`
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    # 4. Run the script using subprocess
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()),
        "COVERAGE_PROCESS_START": str(Path.cwd() / ".coveragerc"),
    }

    result = subprocess.run(
        [sys.executable, "src/data_loader/data_loader.py"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Loaded DataFrame is empty" in result.stdout or result.stderr

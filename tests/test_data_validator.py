"""
test_data_validator.py

Unit tests for data_validator.py

Covers:
- load_config (missing file, invalid YAML)
- _is_dtype_compatible (various dtypes)
- _validate_column (missing required, type mismatch, missing values,
  out-of-range, allowed_values)
- validate_data (action_on_error="raise" vs "warn", report file creation)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data_validator.data_validator import (
    DataValidationError,
    _is_dtype_compatible,
    _validate_column,
    load_config,
    setup_logger,
    validate_data,
)


@pytest.fixture
def config_and_schema(tmp_path):
    config = {
        "logging": {
            "level": "DEBUG",
            "log_file": str(tmp_path / "val.log"),
            "format": "%(levelname)s:%(message)s"
        },
        "data_validation": {
            "enabled": True,
            "action_on_error": "raise",
            "report_path": str(tmp_path / "report.json"),
            "schema": {
                "columns": [
                    {"name": "id", "dtype": "int", "required": True, "min": 0, "max": 10},
                    {"name": "cat", "dtype": "int", "required": False, "allowed_values": [0, 1]},
                    {"name": "opt", "dtype": "float", "required": False},
                ]
            }
        }
    }
    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w") as f:
        yaml.safe_dump(config, f)
    return cfg_file, config


def test_load_config_file_not_found(tmp_path):
    path = tmp_path / "no.yaml"
    # pass a Path object to hit line 34
    with pytest.raises(DataValidationError):
        load_config(path)


def test_load_config_bad_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("bad: [")
    with pytest.raises(DataValidationError):
        load_config(path)


@pytest.mark.parametrize(
    "dtype, series, expected",
    [
        ("int", pd.Series([1], dtype="int"), True),
        ("float", pd.Series([1.1], dtype="float"), True),
        ("str", pd.Series(["a"], dtype="object"), True),
        ("bool", pd.Series([True], dtype="bool"), True),
        ("int", pd.Series([1.0], dtype="float"), False),
    ],
)
def test_is_dtype_compatible(dtype, series, expected):
    assert _is_dtype_compatible(series, dtype) is expected


def test_validate_column_all_checks(config_and_schema):
    _, config = config_and_schema
    errors, warnings, report = [], [], {}

    df = pd.DataFrame({
        "id": [-5, 5, 15],        # violates min/max
        "cat": [2, 0, 1],         # one invalid value
        # omit 'opt' entirely to hit optional-not-present
    })

    for col in config["data_validation"]["schema"]["columns"]:
        _validate_column(df, col, errors, warnings, report)

    assert any("below min" in e for e in errors)
    assert any("above max" in e for e in errors)
    assert any("invalid values" in e for e in errors)
    assert report["opt"]["status"] == "optional not present"


def test_validate_column_sample_fallback(config_and_schema, monkeypatch):
    _, config = config_and_schema
    df = pd.DataFrame({"id": [1]})
    errors, warnings, report = [], [], {}
    series = df["id"]
    monkeypatch.setattr(series, "dropna", lambda: (_ for _ in ()).throw(ValueError("fail")))
    _validate_column(df, config["data_validation"]["schema"]["columns"][0], errors, warnings, report)
    assert report["id"]["samples"] == []


def test_validate_data_early_exit_if_disabled(config_and_schema):
    _, config = config_and_schema
    config["data_validation"]["enabled"] = False
    df = pd.DataFrame({"id": [1]})
    validate_data(df, config)


def test_validate_data_early_exit_if_no_columns(config_and_schema):
    _, config = config_and_schema
    config["data_validation"]["schema"]["columns"] = []
    df = pd.DataFrame({"id": [1]})
    validate_data(df, config)


def test_validate_data_raises(tmp_path, config_and_schema):
    _, config = config_and_schema
    df = pd.DataFrame({"id": [None]})
    setup_logger(config)
    with pytest.raises(DataValidationError):
        validate_data(df, config)
    assert Path(config["data_validation"]["report_path"]).is_file()


def test_validate_data_warns(tmp_path, config_and_schema):
    _, config = config_and_schema
    config["data_validation"]["action_on_error"] = "warn"
    df = pd.DataFrame({"id": [None]})
    setup_logger(config)
    validate_data(df, config)


def test_main_usage_error():
    result = subprocess.run(
        ["coverage", "run", "-m", "src.data_validator.data_validator"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path.cwd())
        },
    )
    assert "Usage: python -m src.data.data_validator" in result.stdout


def test_main_sys_exit_on_bad_args():
    result = subprocess.run(
        ["coverage", "run", "-m", "src.data_validator.data_validator", "only-one.xlsx"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path.cwd())
        },
    )
    assert result.returncode == 1


def test_main_valid_run(tmp_path, config_and_schema):
    cfg_file, config = config_and_schema
    df = pd.DataFrame({"id": [5], "cat": [1], "opt": [3.1]})
    data_file = tmp_path / "data.xlsx"
    df.to_excel(data_file, index=False)

    result = subprocess.run(
        [
            "coverage",
            "run",
            "-m",
            "src.data_validator.data_validator",
            str(data_file),
            str(cfg_file)
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path.cwd())
        },
    )

    assert result.returncode == 0
    report_path = Path(config["data_validation"]["report_path"])
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["status"] == "pass"


def test_main_executes_directly(tmp_path, config_and_schema, monkeypatch):
    from src.data_validator.data_validator import main
    cfg_file, config = config_and_schema
    df = pd.DataFrame({"id": [5], "cat": [1], "opt": [3.1]})
    data_file = tmp_path / "data.xlsx"
    df.to_excel(data_file, index=False)

    monkeypatch.setattr(sys, "argv", ["script.py", str(data_file), str(cfg_file)])

    main()  # should complete without error


def test_main_argv_length_check(monkeypatch):
    from src.data_validator.data_validator import main
    monkeypatch.setattr(sys, "argv", ["script.py"])  # too few args
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

"""
test_main.py

Unit tests for main.py

Covers:
- _load_config: missing file, invalid YAML
- _setup_logging: creates log file, resets handlers
- main() with incorrect args (missing --stage), invalid stage, missing config
- main() 'data' stage: monkeypatch load_data and validate_data
- main() 'train' stage: monkeypatch load_data, validate_data,
  run_model_pipeline
- main() 'infer' stage: monkeypatch pd.read_excel, validate_data, run_inference
"""

import sys

import pandas as pd
import pytest
import yaml

# Import the module under test
from src.legacy_main import _load_config, _setup_logging, main


# Helper to capture sys.exit calls
class DummyExit(Exception):
    pass


@pytest.fixture(autouse=True)
def dummy_exit(monkeypatch):
    """
    Replace sys.exit with raising DummyExit for testing.
    """
    monkeypatch.setattr(
        sys,
        "exit",
        lambda code=0: (_ for _ in ()).throw(DummyExit(f"exit: {code}")),
    )


def test_load_config_missing(tmp_path):
    """
    _load_config should raise FileNotFoundError if config file is missing.
    """
    with pytest.raises(FileNotFoundError):
        _load_config(str(tmp_path / "no.yaml"))


def test_load_config_invalid_yaml(tmp_path):
    """
    _load_config should raise yaml.YAMLError if YAML is invalid.
    """
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: [closed_list", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        _load_config(str(bad))


def test_setup_logging_creates_and_resets(tmp_path):
    """
    _setup_logging should create log file directory and
    reset any existing handlers.
    """
    # Create a dummy config
    log_file = tmp_path / "logs" / "main.log"
    cfg = {
        "level": "INFO",
        "log_file": str(log_file),
        "format": "%(message)s",
        "datefmt": None,
    }
    _setup_logging(cfg)
    # Emit a log
    import logging

    logger = logging.getLogger(__name__)
    logger.info("hello")
    # File should be created
    assert log_file.is_file()


def test_main_missing_stage(tmp_path, monkeypatch, capsys):
    """
    Running main without required --stage should cause argparse to exit.
    """
    testargs = ["prog", "--config", "cfg.yaml"]
    monkeypatch.setattr(sys, "argv", testargs)
    with pytest.raises(DummyExit):
        main()


def test_main_invalid_config(tmp_path, monkeypatch):
    """
    Running main with a non-existent config file should cause exit.
    """
    testargs = ["prog", "--config", "no.yaml", "--stage", "data"]
    monkeypatch.setattr(sys, "argv", testargs)
    with pytest.raises(DummyExit):
        main()


def test_main_data_stage(monkeypatch, tmp_path):
    """
    Test 'data' stage by monkeypatching load_data and validate_data.
    """
    # Create a minimal config file
    cfg = {
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "m.log"),
            "format": "%(message)s",
            "datefmt": None,
        }
    }
    cfg_file = tmp_path / "config.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Monkeypatch functions
    dummy_df = pd.DataFrame({"a": [1]})
    monkeypatch.setattr("src.main.load_data", lambda: dummy_df)
    monkeypatch.setattr("src.main.validate_data", lambda df, cfg: None)

    testargs = ["prog", "--config", str(cfg_file), "--stage", "data"]
    monkeypatch.setattr(sys, "argv", testargs)
    # Should not raise
    main()


def test_main_train_stage(monkeypatch, tmp_path):
    """
    Test 'train' stage by monkeypatching load_data,
    validate_data, run_model_pipeline.
    """
    cfg = {
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "m2.log"),
            "format": "%(message)s",
            "datefmt": None,
        }
    }
    cfg_file = tmp_path / "config2.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    dummy_df = pd.DataFrame({"a": [1]})
    monkeypatch.setattr("src.main.load_data", lambda: dummy_df)
    monkeypatch.setattr("src.main.validate_data", lambda df, cfg: None)
    monkeypatch.setattr("src.main.run_model_pipeline", lambda df, cfg: None)

    testargs = ["prog", "--config", str(cfg_file), "--stage", "train"]
    monkeypatch.setattr(sys, "argv", testargs)
    main()


def test_main_infer_stage(monkeypatch, tmp_path):
    """
    Test 'infer' stage by monkeypatching pd.read_excel,
    validate_data, run_inference.
    """
    cfg = {
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "m3.log"),
            "format": "%(message)s",
            "datefmt": None,
        },
        "inference": {"return_proba": False},
    }
    cfg_file = tmp_path / "config3.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Create a dummy inference Excel
    df_inf = pd.DataFrame({"a": [1], "b": [2]})
    inf_file = tmp_path / "in.xlsx"
    df_inf.to_excel(inf_file, index=False)

    # Monkeypatch read_excel to return df_inf
    monkeypatch.setattr(pd, "read_excel", lambda path: df_inf)
    monkeypatch.setattr("src.main.validate_data", lambda df, cfg: None)
    monkeypatch.setattr("src.main.run_inference", lambda **kwargs: None)

    testargs = [
        "prog",
        "--config",
        str(cfg_file),
        "--stage",
        "infer",
        "--input_csv",
        str(inf_file),
        "--output_csv",
        str(tmp_path / "out.xlsx"),
    ]
    monkeypatch.setattr(sys, "argv", testargs)
    main()

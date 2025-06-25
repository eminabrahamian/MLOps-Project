"""
test_inference.py

Unit tests for inference.py

Covers:
- load_config (missing file, invalid YAML)
- setup_logger (creates log file, resets handlers)
- load_model/pipeline (missing file, invalid pickle)
- get_data (CSV, Excel, missing file, unsupported suffix)
- preprocess_inference_data (missing features,
  NumPy array input, DataFrame input)
- make_predictions (model with/without predict_proba,
  return_proba True/False)
- save_predictions (array only, tuple (preds,probs),
  directory creation)
- run_inference (happy-path with temp pipeline & model;
  error paths for missing config keys)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.inference.inference import (
    InferenceError,
    get_data,
    load_config,
    load_model,
    load_pipeline,
    make_predictions,
    preprocess_inference_data,
    run_inference,
    run_inference_df,
    save_predictions,
    setup_logger,
)


# Dummy pipeline: identity transform
class DummyPipeline:
    def __init__(self, expected_features):
        self.expected_features = expected_features
        self.fitted = False

    def fit_transform(self, X):
        self.fitted = True
        return X.values  # just return NumPy array

    def transform(self, X):
        self.fitted = True
        return X.values  # return array


# Dummy model: similar to DummyBinaryClassifier in evaluation tests


class DummyModel:
    def __init__(self, proba=False):
        self.proba = proba

    def predict(self, X):
        return np.zeros((X.shape[0],), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (X.shape[0], 1))


@pytest.fixture
def temp_inference_artifacts(tmp_path):
    """
    Create a temporary preprocessing pipeline pickle and model pickle.
    Also create a config.yaml pointing to those artifacts.
    """
    # Create dummy pipeline and model, serialize to disk
    pipeline = DummyPipeline(expected_features=["f1", "f2"])
    model = DummyModel(proba=True)
    pipe_file = tmp_path / "pipeline.pkl"
    model_file = tmp_path / "model.pkl"
    with pipe_file.open("wb") as f:
        pickle.dump(pipeline, f)
    with model_file.open("wb") as f:
        pickle.dump(model, f)

    # Create config dict and file
    cfg = {
        "logging": {
            "level": "INFO",
            "log_file": str(tmp_path / "inf.log"),
            "format": "%(levelname)s:%(message)s",
            "datefmt": None,
        },
        "artifacts": {
            "preprocessing_pipeline": str(pipe_file),
            "model_path": str(model_file),
        },
        "original_features": ["f1", "f2"],  # FIXED: added for correct shape
        "raw_features": ["f1", "f2"],
    }
    cfg_file = tmp_path / "config_inf.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return cfg_file, cfg, pipeline, model


def test_load_config_missing(tmp_path):
    """
    load_config should raise InferenceError if config file is missing.
    """
    fake = tmp_path / "noconf.yaml"
    with pytest.raises(InferenceError) as excinfo:
        load_config(fake)
    assert "Config file not found" in str(excinfo.value)


def test_load_config_invalid_yaml(tmp_path):
    """
    load_config should raise InferenceError on bad YAML syntax.
    """
    bad_yaml = tmp_path / "badinf.yaml"
    bad_yaml.write_text("bad: [not closed", encoding="utf-8")
    with pytest.raises(InferenceError) as excinfo:
        load_config(bad_yaml)
    assert "Invalid YAML" in str(excinfo.value)


def test_setup_logger_creates_file(tmp_path, temp_inference_artifacts):
    """
    setup_logger should create the specified log file and
    handle duplicate handlers.
    """
    cfg_file, cfg, _, _ = temp_inference_artifacts
    setup_logger(cfg)
    logger = logging.getLogger(__name__)
    # Emit a warning
    logger.warning("Test warning")
    # Ensure log file exists
    log_path = Path(cfg["logging"]["log_file"])
    assert log_path.is_file()


def test_load_model_missing(tmp_path):
    """
    load_model should raise InferenceError if model file is missing.
    """
    fake_model = tmp_path / "nofile.pkl"
    with pytest.raises(InferenceError) as excinfo:
        load_model(fake_model)
    assert "Model file not found" in str(excinfo.value)


def test_load_pipeline_missing(tmp_path):
    """
    load_pipeline should raise InferenceError if pipeline file is missing.
    """
    fake_pipe = tmp_path / "nopipe.pkl"
    with pytest.raises(InferenceError) as excinfo:
        load_pipeline(fake_pipe)
    assert "Pipeline file not found" in str(excinfo.value)


def test_get_data_csv_and_excel(tmp_path):
    """
    get_data should read CSV and Excel correctly; error on unsupported suffix.
    """
    # CSV
    df_csv = pd.DataFrame({"a": [1, 2]})
    csv_file = tmp_path / "data.csv"
    df_csv.to_csv(csv_file, index=False)
    out_csv = get_data(csv_file)
    pd.testing.assert_frame_equal(out_csv, df_csv)

    # Excel
    df_xlsx = pd.DataFrame({"b": [3, 4]})
    xlsx_file = tmp_path / "data.xlsx"
    df_xlsx.to_excel(xlsx_file, index=False)
    out_xlsx = get_data(xlsx_file)
    pd.testing.assert_frame_equal(out_xlsx, df_xlsx)

    # Unsupported
    fake = tmp_path / "data.txt"
    fake.write_text("hello", encoding="utf-8")
    with pytest.raises(InferenceError) as excinfo:
        get_data(fake)
    assert "Unsupported data format" in str(excinfo.value)


def test_preprocess_inference_data_missing_features(temp_inference_artifacts):
    """
    preprocess_inference_data should raise if
    required_features are missing in DataFrame.
    """
    cfg_file, cfg, pipeline, model = temp_inference_artifacts
    # DataFrame missing 'f2'
    df = pd.DataFrame({"f1": [1, 2]})
    with pytest.raises(InferenceError) as excinfo:
        preprocess_inference_data(df, pipeline, ["f1", "f2"])
    assert "Missing required features" in str(excinfo.value)


def test_preprocess_inference_data_ndarray_to_DF(temp_inference_artifacts):
    """
    preprocess_inference_data should accept a NumPy
    array by converting to DataFrame.
    """
    _, cfg, pipeline, model = temp_inference_artifacts
    # Construct a small DataFrame with f1,f2 and convert to NumPy
    df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    arr = df.values
    out = preprocess_inference_data(arr, pipeline, ["f1", "f2"])
    assert isinstance(out, np.ndarray)
    assert out.shape == arr.shape


def test_make_predictions_no_proba():
    """
    make_predictions should return only preds array if return_proba=False.
    """

    class SimpleModel:
        def predict(self, X):
            return np.ones(X.shape[0], dtype=int)

    mod = SimpleModel()
    X = np.zeros((3, 2))
    preds = make_predictions(mod, X, return_proba=False)
    assert isinstance(preds, np.ndarray) and preds.tolist() == [1, 1, 1]


def test_make_predictions_with_proba():
    """
    make_predictions should return (preds,probs) tuple if
    return_proba=True and model supports predict_proba.
    """

    class ProbaModel:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

        def predict_proba(self, X):
            return np.tile([0.2, 0.8], (X.shape[0], 1))

    mod = ProbaModel()
    X = np.zeros((2, 2))
    preds, probs = make_predictions(mod, X, return_proba=True)
    assert isinstance(preds, np.ndarray)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2, 2)


def test_save_predictions_array_only(tmp_path):
    """
    save_predictions should write an Excel with only 'prediction' column.
    """
    preds = np.array([0, 1, 0])
    out_file = tmp_path / "preds.xlsx"
    save_predictions(preds, out_file, data_index=[0, 1, 2])
    assert out_file.is_file()
    df = pd.read_excel(out_file, index_col=0)
    assert "prediction" in df.columns
    assert df["prediction"].tolist() == [0, 1, 0]


def test_save_predictions_with_probs(tmp_path):
    """
    save_predictions should write an Excel with class
    probability columns and 'prediction'.
    """
    preds = np.array([1, 0])
    probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    out_file = tmp_path / "preds2.xlsx"
    save_predictions((preds, probs), out_file, data_index=[10, 20])
    assert out_file.is_file()
    df = pd.read_excel(out_file, index_col=0)
    # Columns: class_0, class_1, prediction
    assert all(col in df.columns for col in ["class_0", "class_1", "prediction"])
    assert df["prediction"].tolist() == [1, 0]


def test_run_inference_happy_path(tmp_path, temp_inference_artifacts):
    """
    End-to-end run_inference:
      - Create a small CSV with f1,f2
      - Call run_inference and check that an output Excel is created
    """
    cfg_file, cfg, pipeline, model = temp_inference_artifacts
    df_in = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    in_csv = tmp_path / "in.csv"
    df_in.to_csv(in_csv, index=False)

    out_xlsx = tmp_path / "out_preds.xlsx"
    run_inference(str(in_csv), str(cfg_file), str(out_xlsx), return_proba=True)
    assert out_xlsx.is_file()
    df_out = pd.read_excel(out_xlsx)
    assert "prediction" in df_out.columns


def test_run_inference_missing_pipeline_key(tmp_path, temp_inference_artifacts):
    """
    run_inference should exit/raise if config missing
    'artifacts.preprocessing_pipeline'.
    """
    cfg_file, cfg, pipeline, model = temp_inference_artifacts
    # Remove preprocessing_pipeline key
    del cfg["artifacts"]["preprocessing_pipeline"]
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    # Create a valid CSV so get_data fails first on config
    in_csv = tmp_path / "in2.csv"
    pd.DataFrame({"f1": [0], "f2": [0]}).to_csv(in_csv, index=False)

    with pytest.raises(SystemExit):
        run_inference(
            str(in_csv),
            str(cfg_file),
            str(tmp_path / "o.xlsx"),
            return_proba=False,
        )


def test_run_inference_invalid_feature_names(tmp_path, temp_inference_artifacts):
    """
    Should trigger feature mismatch and sys.exit(1)
    """
    cfg_file, cfg, pipeline, model = temp_inference_artifacts
    # Invalidate expected features
    cfg["original_features"] = ["missing_col"]
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    df = pd.DataFrame({"f1": [1], "f2": [2]})
    in_csv = tmp_path / "bad_input.csv"
    df.to_csv(in_csv, index=False)

    with pytest.raises(SystemExit):
        run_inference(
            str(in_csv),
            str(cfg_file),
            str(tmp_path / "fail_preds.xlsx"),
            return_proba=False,
        )


def test_run_inference_df_basic(temp_inference_artifacts):
    """
    run_inference_df should return predictions for valid input.
    """
    _, cfg, _, _ = temp_inference_artifacts
    df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    result = run_inference_df(df, cfg, return_proba=True)

    assert "prediction" in result.columns
    assert "prediction_proba" in result.columns
    assert result.shape[0] == 1


def test_run_inference_df_missing_column(temp_inference_artifacts):
    """
    run_inference_df should raise ValueError if required columns are missing.
    """
    _, cfg, _, _ = temp_inference_artifacts
    df = pd.DataFrame({"f1": [1.0]})  # missing "f2"
    with pytest.raises(ValueError, match="Missing required input features"):
        run_inference_df(df, cfg)


def test_run_inference_df_model_failure(temp_inference_artifacts, monkeypatch):
    """
    run_inference_df should raise InferenceError if model.predict fails.
    """
    _, cfg, _, _ = temp_inference_artifacts

    class BadModel:
        def predict(self, X):
            raise RuntimeError("intentional failure")

    monkeypatch.setattr("src.inference.inference.load_model", lambda _: BadModel())

    df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    with pytest.raises(InferenceError, match="intentional failure"):
        run_inference_df(df, cfg)


def test_run_inference_df_pipeline_missing_feature(
    monkeypatch, temp_inference_artifacts
):
    """
    Test if error raised when pipeline transform fails due to shape mismatch.
    """
    _, cfg, _, _ = temp_inference_artifacts

    class BadPipeline:
        def transform(self, X):
            raise ValueError("bad input shape")

    monkeypatch.setattr(
        "src.inference.inference.load_pipeline", lambda _: BadPipeline()
    )

    df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    with pytest.raises(ValueError, match="bad input shape"):
        run_inference_df(df, cfg)

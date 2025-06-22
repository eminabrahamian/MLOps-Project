"""
Binary-classification evaluation utilities for our MLOps pipeline.

- Computes a configurable set of metrics for SK-learn compatible classifiers.
- Driven by config.yaml for metric selection and artifact locations.
- Can be invoked from model training pipeline or run standalone for reporting.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _specificity(tn: int, fp: int) -> float:
    """
    Compute specificity: TN / (TN + FP). Return NaN if denominator is zero.

    WHY:
        Specificity (true negative rate) is critical in imbalance scenarios.
        Handle zero-division to avoid runtime errors.
    """
    denom = tn + fp
    return tn / denom if denom else float("nan")


def _npv(tn: int, fn: int) -> float:
    """
    Compute negative predictive value: TN / (TN + FN).

    Return NaN if denominator is zero.

    WHY:
        NPV indicates reliability of negative predictions.
        Guard against zero-division.
    """
    denom = tn + fn
    return tn / denom if denom else float("nan")


def _round_dict_values(
    metrics: Dict[str, Any], digits: int = 3
) -> Dict[str, Any]:
    """
    Recursively round any float values in a nested metrics dict.

    WHY:
        Rounding improves readability in logs and saved
        JSON without altering structure.
    """
    rounded: Dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, dict):
            rounded[key] = _round_dict_values(val, digits)
        elif isinstance(val, float):
            rounded[key] = round(val, digits)
        else:
            rounded[key] = val
    return rounded


def evaluate_classification(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    config: Dict[str, Any],
    *,
    metrics: Optional[List[str]] = None,
    split_name: Optional[str] = None,
    log_results: bool = False,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics for a given split.

    Parameters
    ----------
    model : Any
        A fitted scikit-learn estimator with predict()
        (and optionally predict_proba()).
    X : np.ndarray
        Features matrix for this split.
    y_true : np.ndarray
        True labels for this split.
    config : Dict[str, Any]
        Full YAML configuration. Used to fetch default metric
        list if metrics is None.
    metrics : Optional[List[str]], default=None
        List of metric names (case-insensitive) to compute.
        If None, uses config["metrics"]["report"].
    split_name : Optional[str], default=None
        Name of the data split (e.g. "validation", "test")
        used for logging context.
    log_results : bool, default=False
        If True and split_name is provided,
        log all metric values at INFO level.
    save_path : Optional[str], default=None
        If provided, write the resulting metrics dict as JSON
        to this file.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping metric names to values. Includes
        nested confusion matrix.
    """
    # Define alias mapping to canonical names
    alias_map = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "roc auc": "ROC AUC",
        "specificity": "Specificity",
        "npv": "NPV",
        "negative predictive value": "NPV",
        "confusion matrix": "Confusion Matrix",
    }

    # Determine which metrics to compute
    if metrics is None:
        raw_metrics_cfg = config.get("metrics", {})
        if isinstance(raw_metrics_cfg, dict):
            metrics_list = raw_metrics_cfg.get("report", [])
        else:
            metrics_list = cast(List[str], raw_metrics_cfg)
    else:
        metrics_list = metrics

    # Normalize to canonical names (case-insensitive)
    canonical_metrics: List[str] = []
    for m in metrics_list:
        key = m.strip().lower()
        canonical_metrics.append(alias_map.get(key, m.strip()))

    # Always include confusion matrix
    if "Confusion Matrix" not in canonical_metrics:
        canonical_metrics.append("Confusion Matrix")

    # 1) Predict labels (and optionally probabilities inside ROC AUC)
    y_pred = model.predict(X)

    # 2) Build confusion matrix robustly for binary labels {0,1}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        tn = cm[0, 0]
        fp = fn = tp = 0
    else:
        # Cases with only one label present
        vals = np.unique(y_true)
        if len(vals) == 1 and vals[0] == 0:
            tn = int(cm.sum())  # all true negatives
            fp = fn = tp = 0
        elif len(vals) == 1 and vals[0] == 1:
            tp = int(cm.sum())
            tn = fp = fn = 0
        else:
            tn = fp = fn = tp = 0

    results: Dict[str, Any] = {}

    # 3) Compute each requested metric
    for name in canonical_metrics:
        if name == "Accuracy":
            results["Accuracy"] = float(accuracy_score(y_true, y_pred))

        elif name == "Precision":
            results["Precision"] = float(
                precision_score(y_true, y_pred, zero_division=0)
            )

        elif name == "Recall":
            results["Recall"] = float(
                recall_score(y_true, y_pred, zero_division=0)
            )

        elif name == "F1 Score":
            results["F1 Score"] = float(
                f1_score(y_true, y_pred, zero_division=0)
            )

        elif name == "ROC AUC":
            try:
                if (
                    hasattr(model, "predict_proba")
                    and len(np.unique(y_true)) == 2
                ):
                    proba = model.predict_proba(X)[:, 1]
                    results["ROC AUC"] = float(roc_auc_score(y_true, proba))
                else:
                    results["ROC AUC"] = float("nan")
            except Exception:
                results["ROC AUC"] = float("nan")

        elif name == "Specificity":
            results["Specificity"] = _specificity(tn, fp)

        elif name == "NPV":
            results["NPV"] = _npv(tn, fn)

        elif name == "Confusion Matrix":
            results["Confusion Matrix"] = {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }

    # 4) Optionally log the results
    if log_results and split_name:
        rounded = _round_dict_values(results)
        logger.info(
            "Metrics [%s]: %s", split_name, json.dumps(rounded, indent=2)
        )

    # 5) Optionally save to JSON
    if save_path:
        try:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            with save_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info("Saved metrics JSON to %s", save_path)
        except Exception as e:
            logger.error("Failed to save metrics to %s: %s", save_path, e)
            raise

    return results


def generate_split_report(
    config: Dict[str, Any],
    *,
    split: str = "validation",
    processed_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load processed split XLSX and trained model,
    compute metrics for that split.

    Parameters
    ----------
    config : Dict[str, Any]
        Full project configuration dict. Must contain:
          - "target": target column name
          - "artifacts": { "processed_dir": "...", "model_path": "...",
                        "metrics_dir": "..." }
          - "metrics": metric list under ["report"]
    split : str, default="validation"
        Name of the split, expects file "<split>_processed.xlsx" in
        processed_dir.
    processed_dir : Optional[str], default=None
        Directory containing processed XLSXs. If None,
        use config["artifacts"]["processed_dir"].
    model_path : Optional[str], default=None
        Path to pickled model file. If None,
        use config["artifacts"]["model_path"].
    save_path : Optional[str], default=None
        Path to write JSON report for this split. If None,
        save under config["artifacts"]["metrics_dir"]/split_metrics.json.

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary for this split. Empty dict if any error occurs.
    """
    cfg_art = config.get("artifacts", {})
    processed_dir = Path(PROJECT_ROOT) / processed_dir or cfg_art.get(
        "processed_dir", "data/processed"
    )
    model_path = Path(PROJECT_ROOT) / model_path or\
        cfg_art.get("model_path", "models/model.pkl")
    metrics_dir = Path(PROJECT_ROOT) / save_path or\
        cfg_art.get("metrics_dir", "models/")
    target_col = config.get("target")

    report: Dict[str, Any] = {}

    # 1) Load split DataFrame
    split_file = Path(processed_dir) / f"{split}_processed.xlsx"
    if not split_file.is_file():
        logger.warning("Processed file not found: %s", split_file)
        return report

    df_split = pd.read_excel(split_file)
    if target_col not in df_split.columns:
        logger.error(
            "Target column '%s' missing in %s", target_col, split_file
        )
        return report

    X = df_split.drop(columns=[target_col]).values
    y = df_split[target_col].values

    # 2) Load trained model
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model from %s", model_path)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return report

    # 3) Evaluate metrics
    metrics_cfg = config.get("metrics", {}).get("report", None)
    save_file = Path(metrics_dir) / f"{split}_metrics.json"
    metrics = evaluate_classification(
        model,
        X,
        y,
        config,
        metrics=metrics_cfg,
        split_name=split,
        log_results=True,
        save_path=str(save_file),
    )
    report = metrics
    return report

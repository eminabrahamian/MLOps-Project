import os
import logging
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

def evaluate_model(y_true, y_pred, y_proba):
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
        }
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["specificity"] = tn / (tn + fp)

        logging.info(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Error in evaluate_model: {e}")
        raise


def save_metrics(metrics, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame([metrics]).to_csv(output_path, index=False)
        logging.info(f"Saved metrics to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise


def evaluate_and_save(y_true, y_pred, y_proba, output_dir):
    metrics = evaluate_model(y_true, y_pred, y_proba)
    save_metrics(metrics, os.path.join(output_dir, "metrics.csv"))
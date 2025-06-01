#!/usr/bin/env python3
"""
Model module for MLOps pipeline.
Handles model building, training, saving, and loading functionality.
Implements a leakage-proof, end-to-end pipeline for model training and evaluation.
"""

import argparse
import json
import logging
import pickle
import yaml
from pathlib import Path
from typing import Dict, Any, List, Union


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)

# Import local modules
try:
    from src.data_load.data_loader import get_data
    from src.preprocess.preprocessing import build_preprocessing_pipeline, get_output_feature_names
    from src.evaluation.evaluator import evaluate_classification
except ImportError:
    logger.warning("Local modules not found, using fallback imports")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Model registry mapping
MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "logistic_regression": LogisticRegression
}

def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str, params: Dict[str, Any]) -> BaseEstimator:
    """
    Train a model based on specified type and parameters.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        model_type (str): Type of model to train
        params (Dict[str, Any]): Model parameters
        
    Returns:
        BaseEstimator: Trained model
        
    Raises:
        ValueError: If model type is not supported
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(**params)
    
    logger.info(f"Training {model_type} model with parameters: {params}")
    return model.fit(X_train, y_train)

def save_artifact(obj: Any, path: str) -> None:
    """
    Save an object to disk using pickle.
    
    Args:
        obj (Any): Object to save
        path (str): Path where object should be saved
        
    Raises:
        IOError: If object cannot be saved
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving artifact to {path}")
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        logger.error(f"Error saving artifact: {str(e)}")
        raise

def format_metrics(metrics: Dict[str, float], ndigits: int = 2) -> Dict[str, float]:
    """
    Format metrics by rounding numeric values.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metric names and values
        ndigits (int): Number of decimal places to round to
        
    Returns:
        Dict[str, float]: Formatted metrics
    """
    return {k: round(v, ndigits) if isinstance(v, (int, float)) else v 
            for k, v in metrics.items()}

def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Run the complete model training pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        config (Dict[str, Any]): Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Split data
    test_size = config['data_split']['test_size']
    valid_size = config['data_split']['valid_size']
    random_state = config['data_split']['random_state']
    
    # First split: separate test set
    train_valid, test = train_test_split(
        df, 
        test_size=test_size,
        random_state=random_state,
        stratify=df[config['target']]
    )
    
    # Second split: separate validation set
    valid_size_adjusted = valid_size / (1 - test_size)
    train, valid = train_test_split(
        train_valid,
        test_size=valid_size_adjusted,
        random_state=random_state,
        stratify=train_valid[config['target']]
    )
    
    # Save raw splits
    splits_dir = Path(config['artifacts']['splits_dir'])
    splits_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(splits_dir / 'train.csv', index=False)
    valid.to_csv(splits_dir / 'valid.csv', index=False)
    test.to_csv(splits_dir / 'test.csv', index=False)
    
    # Build and fit preprocessing pipeline
    preprocessing_pipeline = build_preprocessing_pipeline(config['raw_features'])
    X_train = train[config['raw_features']]
    y_train = train[config['target']]
    preprocessing_pipeline.fit(X_train)
    
    # Transform all splits
    X_train_processed = preprocessing_pipeline.transform(X_train)
    X_valid_processed = preprocessing_pipeline.transform(valid[config['raw_features']])
    X_test_processed = preprocessing_pipeline.transform(test[config['raw_features']])
    
    # Create processed DataFrames
    feature_names = get_output_feature_names(preprocessing_pipeline)
    train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    valid_processed = pd.DataFrame(X_valid_processed, columns=feature_names)
    test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Add target column
    train_processed[config['target']] = y_train.values
    valid_processed[config['target']] = valid[config['target']].values
    test_processed[config['target']] = test[config['target']].values
    
    # Save processed splits
    processed_dir = Path(config['artifacts']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_processed.to_csv(processed_dir / 'train.csv', index=False)
    valid_processed.to_csv(processed_dir / 'valid.csv', index=False)
    test_processed.to_csv(processed_dir / 'test.csv', index=False)
    
    # Save preprocessing pipeline
    save_artifact(preprocessing_pipeline, config['artifacts']['preprocessing_pipeline'])
    
    # Train model
    active_model = config['model']['active']
    model_params = config['model'][active_model]['params']
    model = train_model(X_train_processed, y_train, active_model, model_params)
    
    # Save model
    save_artifact(model, config['artifacts']['model_path'])
    if 'save_path' in config['model'][active_model]:
        save_artifact(model, config['model'][active_model]['save_path'])
    
    # Evaluate model
    valid_metrics = evaluate_classification(
        model, 
        X_valid_processed, 
        valid[config['target']].values
    )
    test_metrics = evaluate_classification(
        model, 
        X_test_processed, 
        test[config['target']].values
    )
    
    # Log metrics
    logger.info(f"Validation metrics: {json.dumps(format_metrics(valid_metrics))}")
    logger.info(f"Test metrics: {json.dumps(format_metrics(test_metrics))}")

def main():
    """Main entry point for model training script."""
    parser = argparse.ArgumentParser(description='Train and save ML model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load data
        try:
            df = get_data(args.config, data_stage="raw")
        except (ImportError, AttributeError):
            df = pd.read_csv(config["data_source"]["raw_path"])
        
        # Run pipeline
        run_model_pipeline(df, config)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main() 

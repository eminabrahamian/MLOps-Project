#!/usr/bin/env python3
"""
Model module for MLOps pipeline.
Handles model building, training, saving, and loading functionality.
"""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_model(config: Dict[str, Any]) -> BaseEstimator:
    """
    Build a model based on configuration settings.
    
    Args:
        config (Dict[str, Any]): Model configuration dictionary containing type and parameters
        
    Returns:
        BaseEstimator: Initialized scikit-learn model
        
    Raises:
        ValueError: If model type is not supported
    """
    model_type = config['model']['type']
    params = config['model']['params']
    
    logger.info(f"Building {model_type} model with parameters: {params}")
    
    if model_type == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model(model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
    """
    Train the model on provided data.
    
    Args:
        model (BaseEstimator): Model to train
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        
    Returns:
        BaseEstimator: Trained model
        
    Raises:
        ValueError: If input data is invalid
    """
    try:
        logger.info(f"Training model on data with shape: {X_train.shape}")
        return model.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def save_model(model: BaseEstimator, path: str) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model (BaseEstimator): Trained model to save
        path (str): Path where model should be saved
        
    Raises:
        IOError: If model cannot be saved
    """
    try:
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {path}")
        joblib.dump(model, path)
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(path: str) -> BaseEstimator:
    """
    Load a saved model from disk.
    
    Args:
        path (str): Path to saved model
        
    Returns:
        BaseEstimator: Loaded model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        IOError: If model cannot be loaded
    """
    try:
        logger.info(f"Loading model from {path}")
        return joblib.load(path)
    except FileNotFoundError:
        logger.error(f"Model file not found at {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def main():
    """Main entry point for model training script."""
    parser = argparse.ArgumentParser(description='Train and save ML model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Build model
        model = build_model(config)
        
        # Note: Training data loading and model training would happen here
        # This is just a placeholder for the script structure
        
        # Save model
        save_model(model, config['model']['output_path'])
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
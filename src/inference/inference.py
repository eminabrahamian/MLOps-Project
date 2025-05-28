#!/usr/bin/env python3
"""
Inference module for MLOps pipeline.
Handles model loading and prediction functionality for new data.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Union, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Add the parent directory to the path so we can import from sibling modules
sys.path.append(str(Path(__file__).parent.parent))
from models.model import load_model
from data.data_loader import load_data, load_config, get_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_inference_data(
    data: Union[pd.DataFrame, np.ndarray],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Preprocess the inference data according to the configuration.
    
    Args:
        data (Union[pd.DataFrame, np.ndarray]): Raw input data
        config (Dict[str, Any]): Configuration dictionary containing preprocessing steps
        
    Returns:
        np.ndarray: Preprocessed features ready for model inference
        
    Raises:
        ValueError: If data format is invalid or required features are missing
    """
    try:
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            feature_columns = config['features']['feature_columns']
            data = pd.DataFrame(data, columns=feature_columns)
        
        # Verify all required features are present
        required_features = config['features']['feature_columns']
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the required features in the correct order
        X = data[required_features]
        
        # Apply any necessary preprocessing steps
        # Add any specific preprocessing steps here based on your cancer data requirements
        
        return X.to_numpy()
    
    except Exception as e:
        logger.error(f"Error during inference preprocessing: {str(e)}")
        raise

def make_predictions(
    model: BaseEstimator,
    X: np.ndarray,
    return_proba: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Make predictions using the loaded model.
    
    Args:
        model (BaseEstimator): Loaded model
        X (np.ndarray): Preprocessed features
        return_proba (bool): Whether to return probability scores
        
    Returns:
        Union[np.ndarray, tuple]: Predictions and optionally probability scores
        
    Raises:
        ValueError: If model is not fitted or input data is invalid
    """
    try:
        # Make predictions
        predictions = model.predict(X)
        
        if return_proba and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            return predictions, probabilities
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def save_predictions(
    predictions: Union[np.ndarray, tuple],
    output_path: str,
    data_index: Union[pd.Index, List] = None
) -> None:
    """
    Save predictions to a CSV file.
    
    Args:
        predictions (Union[np.ndarray, tuple]): Model predictions
        output_path (str): Path to save predictions
        data_index (Union[pd.Index, List]): Index for the predictions
        
    Raises:
        IOError: If predictions cannot be saved
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare predictions DataFrame
        if isinstance(predictions, tuple):
            preds, probs = predictions
            df = pd.DataFrame(probs, columns=[f'class_{i}' for i in range(probs.shape[1])])
            df['prediction'] = preds
        else:
            df = pd.DataFrame({'prediction': predictions})
        
        # Add index if provided
        if data_index is not None:
            df.index = data_index
        
        # Save to CSV
        df.to_csv(output_path)
        logger.info(f"Saved predictions to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise

def run_inference(
    config_path: str,
    data_path: str,
    model_path: str,
    output_path: str,
    return_proba: bool = False
) -> None:
    """
    Main inference pipeline function.
    
    Args:
        config_path (str): Path to configuration file
        data_path (str): Path to inference data
        model_path (str): Path to saved model
        output_path (str): Path to save predictions
        return_proba (bool): Whether to return probability scores
        
    Raises:
        Exception: Any error in the inference pipeline
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)
        
        # Load and preprocess inference data
        logger.info(f"Loading inference data from {data_path}")
        raw_data = get_data(config_path)  # Use get_data to handle file type correctly
        X = preprocess_inference_data(raw_data, config)
        
        # Make predictions
        logger.info("Making predictions")
        predictions = make_predictions(model, X, return_proba)
        
        # Save predictions
        logger.info(f"Saving predictions to {output_path}")
        save_predictions(predictions, output_path, raw_data.index)
        
    except Exception as e:
        logger.error(f"Error in inference pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run model inference')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to inference data')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--output', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--proba', action='store_true', help='Return probability scores')
    
    args = parser.parse_args()
    
    run_inference(
        config_path=args.config,
        data_path=args.data,
        model_path=args.model,
        output_path=args.output,
        return_proba=args.proba
    ) 
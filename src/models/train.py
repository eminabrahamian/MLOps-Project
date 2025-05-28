#!/usr/bin/env python3
"""
Training script for the cancer classification model.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from sibling modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import get_data, load_config
from src.models.model import build_model, train_model, save_model
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    try:
        print("Starting model training...")
        
        # Load configuration
        print("Loading configuration...")
        config = load_config("configs/config.yaml")
        print("Configuration loaded successfully")
        
        # Load data using get_data which handles file type correctly
        print("Loading data...")
        data = get_data("configs/config.yaml")
        print(f"Data loaded successfully. Shape: {data.shape}")
        print("\nAvailable columns in the dataset:")
        print(data.columns.tolist())
        
        # Get features and target
        print("\nPreparing features and target...")
        X = data[config['features']['feature_columns']]
        y = data[config['features']['target_column']]
        print(f"Features shape: {X.shape}")
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config['data_split']['test_size'],
            random_state=config['data_split']['random_state']
        )
        print(f"Training set shape: {X_train.shape}")
        
        # Build and train model
        print("Building and training model...")
        model = build_model(config)
        trained_model = train_model(model, X_train, y_train)
        
        # Save model
        print("Saving model...")
        save_model(trained_model, config['model']['save_path'])
        print(f"Model saved to {config['model']['save_path']}")
        
        # Print basic evaluation
        train_score = trained_model.score(X_train, y_train)
        test_score = trained_model.score(X_test, y_test)
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
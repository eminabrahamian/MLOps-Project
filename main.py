import argparse
import logging
import os
import yaml

from src.data.data_loader import load_data
from src.data.data_validator import validate_data
from src.features.features import create_features       # we don't have a unifying feature engineering function
from src.data.preprocessing import run_preprocessing_pipeline
from src.models.model import train_model
from src.inference.inference import run_inference
from src.evaluation.evaluation import evaluate_and_save

def setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_pipeline(config, stage, input_csv=None, output_csv=None):
    data = None

    if stage in ['all', 'validate']:
        data = load_data()
        validate_data(data, config['validation'])

    if stage in ['all', 'train']:
        data = load_data()
        validate_data(data, config['validation'])

        X = create_features(data)
        pipeline = run_preprocessing_pipeline(X, config['preprocessing'])
        X_processed = pipeline.fit_transform(X.drop(columns=config['target_column']))
        y = X[config['target_column']]

        model, X_val, y_val, y_val_pred, y_val_proba = train_model(
            X_processed, y, config['model'], config['artifacts']
        )

        evaluate_and_save(y_val, y_val_pred, y_val_proba, config['artifacts'])

    if stage == 'infer':
        if not input_csv or not output_csv:
            raise ValueError("Inference stage requires input_csv and output_csv arguments.")
            model_path = config.get('model', {}).get('model_path', 'model.pkl')
        run_inference(config, input_csv, model_path, output_csv)


def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline Orchestrator")
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--stage', type=str, required=True, choices=['all', 'validate', 'train', 'infer'])
    parser.add_argument('--input_csv', type=str, help='Input file for inference stage')
    parser.add_argument('--output_csv', type=str, help='Output file for inference predictions')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config['log_file'])

    logging.info(f"Pipeline started with stage: {args.stage}")
    run_pipeline(config, args.stage, args.input_csv, args.output_csv)
    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
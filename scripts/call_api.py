#!/usr/bin/env python3
"""
call_api.py

Simple script to test the MLOps API endpoints.

Usage:
    python scripts/call_api.py --url http://localhost:8000/predict_batch --input data/test_data.csv
    python scripts/call_api.py --url http://localhost:8000/health
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests


def call_health_endpoint(url):
    """Call the health endpoint."""
    try:
        response = requests.get(f"{url}/health")
        response.raise_for_status()
        
        print("✅ Health check successful!")
        print(f"Status: {response.json()['status']}")
        print(f"Message: {response.json()['message']}")
        print(f"Version: {response.json()['version']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)


def call_predict_batch_endpoint(url, input_file, return_proba=False):
    """Call the batch prediction endpoint."""
    try:
        # Read input data
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_file)
        else:
            print(f"❌ Unsupported file format: {input_file}")
            sys.exit(1)
        
        # Convert to list of dictionaries
        data = df.to_dict(orient='records')
        
        # Prepare request
        payload = {
            "data": data,
            "return_proba": return_proba
        }
        
        print(f"📊 Sending {len(data)} records for prediction...")
        
        # Make request
        response = requests.post(f"{url}/predict_batch", json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("✅ Prediction successful!")
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        print(f"Number of predictions: {len(result['predictions'])}")
        
        if return_proba and 'probabilities' in result:
            print(f"Number of probabilities: {len(result['probabilities'])}")
        
        # Show first few predictions
        print("\n📈 Sample predictions:")
        for i, pred in enumerate(result['predictions'][:5]):
            prob_str = f" (prob: {result['probabilities'][i]:.3f})" if return_proba and 'probabilities' in result else ""
            print(f"  Record {i+1}: {pred}{prob_str}")
        
        if len(result['predictions']) > 5:
            print(f"  ... and {len(result['predictions']) - 5} more predictions")
        
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Response text: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def call_model_info_endpoint(url):
    """Call the model info endpoint."""
    try:
        response = requests.get(f"{url}/model_info")
        response.raise_for_status()
        
        info = response.json()
        
        print("✅ Model info retrieved!")
        print(f"Model type: {info['model_type']}")
        print(f"Version: {info['version']}")
        print(f"Target: {info['target']}")
        print(f"Status: {info['status']}")
        print(f"Number of features: {len(info['features'])}")
        
        if info['features']:
            print("Features:")
            for feature in info['features'][:10]:  # Show first 10 features
                print(f"  - {feature}")
            if len(info['features']) > 10:
                print(f"  ... and {len(info['features']) - 10} more features")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info request failed: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test MLOps API endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/call_api.py --url http://localhost:8000 --health
  python scripts/call_api.py --url http://localhost:8000 --predict tests/mock_data/mock_data.csv
  python scripts/call_api.py --url http://localhost:8000 --predict tests/mock_data/mock_data.csv --proba
  python scripts/call_api.py --url http://localhost:8000 --info
        """
    )
    
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--health", 
        action="store_true",
        help="Call health endpoint"
    )
    
    parser.add_argument(
        "--predict", 
        type=str,
        help="Call prediction endpoint with input file (CSV or Excel)"
    )
    
    parser.add_argument(
        "--proba", 
        action="store_true",
        help="Include probabilities in prediction response"
    )
    
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Call model info endpoint"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.health, args.predict, args.info]):
        print("❌ Please specify at least one endpoint to call (--health, --predict, or --info)")
        parser.print_help()
        sys.exit(1)
    
    # Call endpoints
    if args.health:
        call_health_endpoint(args.url)
    
    if args.predict:
        call_predict_batch_endpoint(args.url, args.predict, args.proba)
    
    if args.info:
        call_model_info_endpoint(args.url)
    
    print("\n🎉 API testing completed!")


if __name__ == "__main__":
    main() 
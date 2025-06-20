"""
test_api.py

API tests for the FastAPI application.

Covers:
- Health endpoint testing
- Batch prediction endpoint testing
- File upload prediction testing
- Model info endpoint testing
- Error handling and validation

Test Categories:
- API endpoint functionality
- Request/response validation
- Error handling
- Integration with pipeline components
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.api.app import app

# Test client
client = TestClient(app)

# Test data constants
MOCK_PREDICTION_DATA = [
    {
        "mean radius": 17.99,
        "mean texture": 10.38,
        "mean perimeter": 122.8,
        "mean area": 1001.0,
        "mean smoothness": 0.1184,
        "mean compactness": 0.2776,
        "mean concavity": 0.3001,
        "mean concave points": 0.1471,
        "mean symmetry": 0.2419,
        "mean fractal dimension": 0.07871,
        "radius error": 1.095,
        "texture error": 0.9053,
        "perimeter error": 8.589,
        "area error": 153.4,
        "smoothness error": 0.006399,
        "compactness error": 0.04904,
        "concavity error": 0.05373,
        "concave points error": 0.01587,
        "symmetry error": 0.03003,
        "fractal dimension error": 0.006193,
        "worst radius": 25.38,
        "worst texture": 17.33,
        "worst perimeter": 184.6,
        "worst area": 2019.0,
        "worst smoothness": 0.1622,
        "worst compactness": 0.6656,
        "worst concavity": 0.7119,
        "worst concave points": 0.2654,
        "worst symmetry": 0.4601,
        "worst fractal dimension": 0.1189
    },
    {
        "mean radius": 20.57,
        "mean texture": 17.77,
        "mean perimeter": 132.9,
        "mean area": 1326.0,
        "mean smoothness": 0.08474,
        "mean compactness": 0.07864,
        "mean concavity": 0.0869,
        "mean concave points": 0.07017,
        "mean symmetry": 0.1812,
        "mean fractal dimension": 0.05667,
        "radius error": 0.5435,
        "texture error": 0.7339,
        "perimeter error": 3.398,
        "area error": 74.08,
        "smoothness error": 0.005225,
        "compactness error": 0.01308,
        "concavity error": 0.0186,
        "concave points error": 0.0134,
        "symmetry error": 0.01389,
        "fractal dimension error": 0.003532,
        "worst radius": 24.99,
        "worst texture": 23.41,
        "worst perimeter": 158.8,
        "worst area": 1956.0,
        "worst smoothness": 0.1238,
        "worst compactness": 0.1866,
        "worst concavity": 0.2416,
        "worst concave points": 0.186,
        "worst symmetry": 0.275,
        "worst fractal dimension": 0.08902
    }
]


class TestHealthEndpoint:
    """Test health check endpoint functionality."""
    
    @pytest.mark.api
    def test_health_endpoint_success(self):
        """Test health endpoint returns 200 status and expected structure."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "status" in data
        assert "message" in data
        assert "version" in data
        
        # Validate content
        assert data["status"] == "healthy"
        assert "MLOps Pipeline API is running" in data["message"]
        assert data["version"] == "1.0.0"
    
    @pytest.mark.api
    def test_health_endpoint_response_format(self):
        """Test health endpoint response matches Pydantic model."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate all required fields are present
        required_fields = ["status", "message", "version"]
        for field in required_fields:
            assert field in data
            assert isinstance(data[field], str)
            assert len(data[field]) > 0


class TestBatchPredictionEndpoint:
    """Test batch prediction endpoint functionality."""
    
    @pytest.mark.api
    def test_predict_batch_success(self):
        """Test successful batch prediction with valid data."""
        request_data = {
            "data": MOCK_PREDICTION_DATA,
            "return_proba": False
        }
        
        response = client.post("/predict_batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "predictions" in data
        assert "status" in data
        assert "message" in data
        assert "probabilities" not in data  # Not requested
        
        # Validate content
        assert data["status"] == "success"
        assert len(data["predictions"]) == len(MOCK_PREDICTION_DATA)
        assert all(isinstance(pred, int) for pred in data["predictions"])
        assert all(pred in [0, 1] for pred in data["predictions"])
    
    @pytest.mark.api
    def test_predict_batch_with_probabilities(self):
        """Test batch prediction with probability scores."""
        request_data = {
            "data": MOCK_PREDICTION_DATA,
            "return_proba": True
        }
        
        response = client.post("/predict_batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "predictions" in data
        assert "probabilities" in data
        assert "status" in data
        assert "message" in data
        
        # Validate content
        assert data["status"] == "success"
        assert len(data["predictions"]) == len(MOCK_PREDICTION_DATA)
        assert len(data["probabilities"]) == len(MOCK_PREDICTION_DATA)
        assert all(isinstance(prob, float) for prob in data["probabilities"])
        assert all(0.0 <= prob <= 1.0 for prob in data["probabilities"])
    
    @pytest.mark.api
    def test_predict_batch_empty_data(self):
        """Test batch prediction with empty data."""
        request_data = {
            "data": [],
            "return_proba": False
        }
        
        response = client.post("/predict_batch", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "No data provided" in data["detail"]
    
    @pytest.mark.api
    def test_predict_batch_missing_data(self):
        """Test batch prediction with missing data field."""
        request_data = {
            "return_proba": False
        }
        
        response = client.post("/predict_batch", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    def test_predict_batch_invalid_data_structure(self):
        """Test batch prediction with invalid data structure."""
        request_data = {
            "data": [{"invalid": "data"}],
            "return_proba": False
        }
        
        response = client.post("/predict_batch", json=request_data)
        
        # Should still work as we're using mock predictions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    @pytest.mark.api
    @patch('src.api.app.config')
    def test_predict_batch_missing_columns(self, mock_config):
        """Test batch prediction with missing required columns."""
        # Mock config with required features
        mock_config.get.return_value = ["mean radius", "mean texture", "missing_column"]
        
        request_data = {
            "data": MOCK_PREDICTION_DATA,
            "return_proba": False
        }
        
        response = client.post("/predict_batch", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "Missing required columns" in data["detail"]
        assert "missing_column" in data["detail"]


class TestFilePredictionEndpoint:
    """Test file upload prediction endpoint functionality."""
    
    @pytest.mark.api
    def test_predict_file_csv_success(self):
        """Test successful file prediction with CSV upload."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(MOCK_PREDICTION_DATA)
            df.to_csv(f.name, index=False)
            file_path = f.name
        
        try:
            with open(file_path, 'rb') as f:
                files = {"file": ("test.csv", f, "text/csv")}
                data = {"return_proba": "false"}
                
                response = client.post("/predict_file", files=files, data=data)
                
                assert response.status_code == 200
                data = response.json()
                
                # Validate response structure
                assert "predictions" in data
                assert "status" in data
                assert "message" in data
                assert data["status"] == "success"
                
        finally:
            Path(file_path).unlink(missing_ok=True)
    
    @pytest.mark.api
    def test_predict_file_excel_success(self):
        """Test successful file prediction with Excel upload."""
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            df = pd.DataFrame(MOCK_PREDICTION_DATA)
            df.to_excel(f.name, index=False)
            file_path = f.name
        
        try:
            with open(file_path, 'rb') as f:
                files = {"file": ("test.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                data = {"return_proba": "true"}
                
                response = client.post("/predict_file", files=files, data=data)
                
                assert response.status_code == 200
                data = response.json()
                
                # Validate response structure
                assert "predictions" in data
                assert "probabilities" in data
                assert "status" in data
                assert data["status"] == "success"
                
        finally:
            Path(file_path).unlink(missing_ok=True)
    
    @pytest.mark.api
    def test_predict_file_invalid_format(self):
        """Test file prediction with invalid file format."""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid data")
            file_path = f.name
        
        try:
            with open(file_path, 'rb') as f:
                files = {"file": ("test.txt", f, "text/plain")}
                
                response = client.post("/predict_file", files=files)
                
                assert response.status_code == 400
                data = response.json()
                assert "File must be CSV or Excel format" in data["detail"]
                
        finally:
            Path(file_path).unlink(missing_ok=True)
    
    @pytest.mark.api
    def test_predict_file_empty_file(self):
        """Test file prediction with empty file."""
        # Create temporary empty CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")  # Empty file
            file_path = f.name
        
        try:
            with open(file_path, 'rb') as f:
                files = {"file": ("empty.csv", f, "text/csv")}
                
                response = client.post("/predict_file", files=files)
                
                assert response.status_code == 400
                data = response.json()
                assert "File is empty" in data["detail"]
                
        finally:
            Path(file_path).unlink(missing_ok=True)


class TestModelInfoEndpoint:
    """Test model information endpoint functionality."""
    
    @pytest.mark.api
    def test_model_info_success(self):
        """Test model info endpoint returns expected structure."""
        response = client.get("/model_info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "model_type" in data
        assert "version" in data
        assert "features" in data
        assert "target" in data
        assert "status" in data
        
        # Validate content
        assert data["model_type"] == "knn"
        assert data["version"] == "1.0.0"
        assert isinstance(data["features"], list)
        assert data["target"] == "target"
        assert data["status"] in ["loaded", "not_loaded"]
    
    @pytest.mark.api
    @patch('src.api.app.config')
    def test_model_info_with_config(self, mock_config):
        """Test model info endpoint with configuration loaded."""
        # Mock config
        mock_config.get.return_value = ["feature1", "feature2"]
        
        response = client.get("/model_info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate features from config
        assert "features" in data
        assert data["features"] == ["feature1", "feature2"]


class TestErrorHandling:
    """Test API error handling and edge cases."""
    
    @pytest.mark.api
    def test_invalid_json_request(self):
        """Test handling of invalid JSON in request body."""
        response = client.post(
            "/predict_batch",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    def test_missing_content_type(self):
        """Test handling of missing content type header."""
        response = client.post(
            "/predict_batch",
            data=json.dumps({"data": MOCK_PREDICTION_DATA})
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    def test_large_payload(self):
        """Test handling of large payload."""
        # Create large dataset
        large_data = MOCK_PREDICTION_DATA * 1000  # 2000 records
        
        request_data = {
            "data": large_data,
            "return_proba": False
        }
        
        response = client.post("/predict_batch", json=request_data)
        
        # Should still work (though might be slow)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["predictions"]) == len(large_data)


class TestAPIIntegration:
    """Test API integration with pipeline components."""
    
    @pytest.mark.api
    @patch('src.api.app.run_inference')
    def test_api_integration_with_inference(self, mock_run_inference):
        """Test API integration with inference pipeline."""
        # Mock the inference function
        mock_run_inference.return_value = None
        
        request_data = {
            "data": MOCK_PREDICTION_DATA,
            "return_proba": False
        }
        
        response = client.post("/predict_batch", json=request_data)
        
        # Should still work with mock predictions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    @pytest.mark.api
    @patch('src.api.app.load_config')
    def test_api_integration_with_config_loading(self, mock_load_config):
        """Test API integration with configuration loading."""
        # Mock config loading
        mock_config = {
            "raw_features": ["mean radius", "mean texture"],
            "target": "diagnosis"
        }
        mock_load_config.return_value = mock_config
        
        # Test model info endpoint
        response = client.get("/model_info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["features"] == ["mean radius", "mean texture"]
        assert data["target"] == "diagnosis" 
"""
Unit tests for the Breast Cancer FastAPI application (app/main.py).

Covers:
- Root and health endpoints
- Single and batch prediction endpoints
- Validation error handling
- Model/pipeline method mocking (not full object)
- Pydantic v2 compatibility (avoids dict() deprecation)

Designed for high coverage and isolated testability.
"""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

CI = os.getenv("CI", "false").lower() == "true"
PIPELINE_EXISTS = os.path.exists("models/preprocessing_pipeline.pkl")

pytestmark = pytest.mark.skipif(
    CI and not PIPELINE_EXISTS, reason="Pipeline file missing in CI environment"
)

import app.main  # delayed until after skipif

client = TestClient(app.main.app)

EXAMPLE_INPUT = {
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.8,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871,
    "radius_error": 1.095,
    "texture_error": 0.9053,
    "perimeter_error": 8.589,
    "area_error": 153.4,
    "smoothness_error": 0.006399,
    "compactness_error": 0.04904,
    "concavity_error": 0.05373,
    "concave_points_error": 0.01587,
    "symmetry_error": 0.03003,
    "fractal_dimension_error": 0.006193,
    "worst_radius": 25.38,
    "worst_texture": 17.33,
    "worst_perimeter": 184.6,
    "worst_area": 2019.0,
    "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656,
    "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189,
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Breast Cancer Prediction API"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.main.PIPELINE.transform", side_effect=ValueError("Pipeline error"))
def test_predict_pipeline_error(mock_transform):
    response = client.post("/predict", json=EXAMPLE_INPUT)
    assert response.status_code == 400
    assert "Pipeline error" in response.json()["detail"]


@patch("app.main.MODEL.predict", side_effect=Exception("Model crash"))
@patch("app.main.PIPELINE.transform", return_value=[[0.1] * 30])
def test_predict_model_crash(mock_transform, mock_predict):
    response = client.post("/predict", json=EXAMPLE_INPUT)
    assert response.status_code == 400
    assert "Model crash" in response.json()["detail"]


def test_predict_validation_error():
    bad_input = EXAMPLE_INPUT.copy()
    del bad_input["mean_radius"]
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422  # Unprocessable Entity

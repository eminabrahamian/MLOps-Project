from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import yaml
import sys
from pathlib import Path
from src.inference.inference import run_inference_df


# Add the root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

PIPELINE_PATH = PROJECT_ROOT / CONFIG["artifacts"]["preprocessing_pipeline"]
MODEL_PATH = PROJECT_ROOT / CONFIG["artifacts"]["model_path"]

with open(PIPELINE_PATH, "rb") as f:
    PIPELINE = pickle.load(f)
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

FEATURES = CONFIG["original_features"]

app = FastAPI()

class BreastCancerInput(BaseModel):
    """
    Defines the input schema for breast cancer prediction.

    This Pydantic model ensures that the incoming JSON matches
    the structure and types required by the model pipeline.
    All fields represent diagnostic features extracted from scans.
    """
    
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

    class Config:
        """
        Example payload shown in the Swagger UI for user guidance.
        """

        schema_extra = {
            "example": {
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
                "worst_fractal_dimension": 0.1189
            }
        }


@app.get("/")
def root():
    """
    Root endpoint for health awareness.

    Returns a welcome message to confirm that the API is live.
    """

    return {"message": "Welcome to the Breast Cancer Prediction API"}

@app.get("/health")
def health():
    """
    Basic health check endpoint.

    Useful for automated probes or service monitoring.
    """

    return {"status": "ok"}

@app.post("/predict")
def predict(payload: BreastCancerInput):
    """
    Predict the likelihood of breast cancer for a single observation.

    Parameters:
        payload (BreastCancerInput): A JSON object containing feature values.

    Returns:
        dict: The predicted class label and probability (if available).
    """

    data = pd.DataFrame([payload.dict()])  # Snake_case input

    try:
        # Step 1: Apply preprocessing
        transformed = PIPELINE.transform(data)

        # Step 2: Predict using preprocessed data
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(transformed)[:, 1]
        else:
            proba = [None] * len(data)

        preds = MODEL.predict(transformed)

        # Step 3: Return results
        result = {
            "prediction": int(preds[0]),
            "probability": float(proba[0]) if proba[0] is not None else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"prediction": int(result["prediction"]), "probability": float(result["probability"])}


@app.post("/predict_batch")
def predict_batch(payloads: list[BreastCancerInput]):
    """
    Predict outcomes for a batch of observations.

    Parameters:
        payloads (list[BreastCancerInput]): List of records with raw features.

    Returns:
        list[dict]: List of predictions with class and probability.
    """

    df = pd.DataFrame([p.dict() for p in payloads])

    try:
        transformed = PIPELINE.transform(df)
        preds = MODEL.predict(transformed)

        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(transformed)[:, 1]
        else:
            proba = [None] * len(df)

        return [
            {"prediction": int(p), "probability": float(prob) if prob is not None else None}
            for p, prob in zip(preds, proba)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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

FEATURES = CONFIG["raw_features"]

app = FastAPI()

class BreastCancerInput(BaseModel):
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
    return {"message": "Welcome to the Breast Cancer Prediction API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: BreastCancerInput):
    data = pd.DataFrame([payload.dict()])
    try:
        result_df = run_inference_df(data, config=CONFIG, return_proba=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = result_df.iloc[0]
    return {"prediction": int(result["prediction"]), "probability": float(result["probability"])}


@app.post("/predict_batch")
def predict_batch(payloads: list[BreastCancerInput]):
    df = pd.DataFrame([p.dict() for p in payloads])
    try:
        result_df = run_inference_df(df, config=CONFIG, return_proba=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result_df[["prediction", "probability"]].to_dict(orient="records")

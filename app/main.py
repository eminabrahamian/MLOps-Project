from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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

PIPELINE_PATH = PROJECT_ROOT / CONFIG["artifacts"]["preprocessing"]
MODEL_PATH = PROJECT_ROOT / CONFIG["artifacts"]["model"]

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
    # Add more fields if your model uses them

    class Config:
        schema_extra = {
            "example": {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184
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



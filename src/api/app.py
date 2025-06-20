"""
FastAPI application for the MLOps pipeline.

This module provides a REST API for the opioid abuse disorder prediction model,
including health checks and batch prediction endpoints.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np

# Import our pipeline components
from src.inference.inference import run_inference, InferenceError
from src.data_loader.data_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Pipeline API",
    description="API for opioid abuse disorder prediction model",
    version="1.0.0"
)

# Global variables for model and pipeline
model = None
pipeline = None
config = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")


class PredictionRequest(BaseModel):
    """Prediction request model."""
    data: List[Dict[str, Any]] = Field(..., description="Input data for prediction")
    return_proba: bool = Field(False, description="Whether to return probabilities")


class PredictionResponse(BaseModel):
    """Prediction response model."""
    predictions: List[int] = Field(..., description="Model predictions")
    probabilities: Optional[List[float]] = Field(None, description="Prediction probabilities")
    status: str = Field(..., description="Prediction status")
    message: str = Field(..., description="Status message")


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    global model, pipeline, config
    
    try:
        # Load configuration
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            config = load_config(config_path)
            logger.info("Configuration loaded successfully")
        else:
            logger.warning("Configuration file not found, using defaults")
            config = {}
        
        # Note: In a real implementation, you would load the model and pipeline here
        # For now, we'll use placeholder values for testing
        logger.info("API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Service status and information
    """
    try:
        return HealthResponse(
            status="healthy",
            message="MLOps Pipeline API is running",
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/predict_batch", response_model=PredictionResponse)
async def predict_batch(request: PredictionRequest):
    """
    Batch prediction endpoint.
    
    Args:
        request: PredictionRequest containing input data
        
    Returns:
        PredictionResponse: Model predictions and probabilities
    """
    try:
        if not request.data:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Validate required columns (using config if available)
        if config and "raw_features" in config:
            required_cols = config["raw_features"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required columns: {missing_cols}"
                )
        
        # In a real implementation, you would:
        # 1. Load the model and pipeline
        # 2. Preprocess the data
        # 3. Make predictions
        
        # For testing purposes, generate mock predictions
        n_samples = len(df)
        predictions = np.random.randint(0, 2, size=n_samples).tolist()
        
        response_data = {
            "predictions": predictions,
            "status": "success",
            "message": f"Generated {n_samples} predictions"
        }
        
        # Add probabilities if requested
        if request.return_proba:
            probabilities = np.random.random(size=n_samples).tolist()
            response_data["probabilities"] = probabilities
            response_data["message"] += " with probabilities"
        
        return PredictionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_file")
async def predict_file(
    file: UploadFile = File(...),
    return_proba: bool = False
):
    """
    File-based prediction endpoint.
    
    Args:
        file: Uploaded CSV or Excel file
        return_proba: Whether to return probabilities
        
    Returns:
        JSONResponse: Prediction results
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="File must be CSV or Excel format"
            )
        
        # Read file content
        content = await file.read()
        
        # Parse based on file type
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(pd.io.common.BytesIO(content))
        else:
            df = pd.read_excel(pd.io.common.BytesIO(content))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Convert to list of dictionaries for processing
        data = df.to_dict(orient='records')
        
        # Create prediction request
        request = PredictionRequest(data=data, return_proba=return_proba)
        
        # Use the batch prediction logic
        result = await predict_batch(request)
        
        return JSONResponse(content=result.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")


@app.get("/model_info")
async def model_info():
    """
    Get model information endpoint.
    
    Returns:
        Dict: Model metadata and configuration
    """
    try:
        info = {
            "model_type": "knn",
            "version": "1.0.0",
            "features": config.get("raw_features", []) if config else [],
            "target": config.get("target", "target") if config else "target",
            "status": "loaded" if model else "not_loaded"
        }
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model info")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
"""
FastAPI Prediction API for Credit Card Fraud Detection.
Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import json
import os

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud prediction using Cost-Sensitive Random Forest & XGBoost",
    version="1.0.0"
)

# --- Load pre-trained models ---
MODEL_DIR = "saved_models"

if not os.path.exists(MODEL_DIR):
    raise RuntimeError("No saved_models/ directory. Run `python train_and_save.py` first.")

models = {}
for filename in os.listdir(MODEL_DIR):
    if filename.endswith('.joblib') and filename != 'scaler.joblib':
        name = filename.replace('.joblib', '').replace('_', ' ').title()
        models[name] = joblib.load(os.path.join(MODEL_DIR, filename))

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
    feature_names = json.load(f)

print(f"Loaded {len(models)} models: {list(models.keys())}")

# --- Request/Response schemas ---
class TransactionRequest(BaseModel):
    """A single credit card transaction to evaluate."""
    features: list[float]
    model_name: str = "Xgboost"
    threshold: float = 0.50

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.0] * 30,
                "model_name": "Xgboost",
                "threshold": 0.50
            }
        }

class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: str
    threshold_used: float
    model_used: str

# --- Endpoints ---
@app.get("/")
def root():
    return {
        "service": "Credit Card Fraud Detection API",
        "models_available": list(models.keys()),
        "features_expected": len(feature_names),
        "docs": "/docs"
    }

@app.get("/models")
def list_models():
    return {"models": list(models.keys())}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    # Validate model name
    if request.model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' not found. Available: {list(models.keys())}"
        )

    # Validate feature count
    if len(request.features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features, got {len(request.features)}"
        )

    model = models[request.model_name]
    input_array = np.array(request.features).reshape(1, -1)
    
    prob = float(model.predict_proba(input_array)[0][1])
    prediction = "FRAUD" if prob >= request.threshold else "LEGITIMATE"

    return PredictionResponse(
        fraud_probability=round(prob, 4),
        prediction=prediction,
        threshold_used=request.threshold,
        model_used=request.model_name
    )

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(models)}

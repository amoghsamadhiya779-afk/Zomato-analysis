import logging
import json
import joblib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# State
ml_models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load available models
    model_files = list(MODEL_DIR.glob("*.pkl"))
    if not model_files:
        logger.warning("⚠️ No models found in /models. Please run src/train.py first.")
    
    for model_path in model_files:
        name = model_path.stem # e.g., "RandomForest"
        logger.info(f"Loading {name}...")
        ml_models[name] = joblib.load(model_path)
    
    # Load metrics
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            ml_models["metrics"] = json.load(f)

    yield
    ml_models.clear()

app = FastAPI(title="Zomato AI Backend", lifespan=lifespan)

# --- Schemas ---
class PredictionRequest(BaseModel):
    review: str
    model_name: str = "RandomForest" # Default

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    model_used: str

class ModelListResponse(BaseModel):
    available_models: List[str]
    metrics: Dict[str, float]

# --- Endpoints ---

@app.get("/models", response_model=ModelListResponse)
async def get_models():
    """Return available models and their accuracy for the UI dropdown."""
    # Filter out "metrics" key from the list of models
    models = [k for k in ml_models.keys() if k != "metrics"]
    metrics = ml_models.get("metrics", {})
    return {
        "available_models": models if models else ["RandomForest"],
        "metrics": metrics
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    model_name = request.model_name
    
    # Fallback logic
    if model_name not in ml_models:
        model_name = "RandomForest"
    
    if model_name not in ml_models:
         raise HTTPException(status_code=500, detail="Model not loaded. Run training script.")

    try:
        pipeline = ml_models[model_name]
        
        # Inference
        prediction = pipeline.predict([request.review])[0]
        proba = pipeline.predict_proba([request.review])[0]
        
        label = "Positive" if prediction == 1 else "Negative"
        confidence = proba[prediction]
        
        return {
            "sentiment": label,
            "confidence": round(float(confidence), 4),
            "model_used": model_name
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
import os
import json
import logging
from enum import Enum
from typing import Literal

import xgboost as xgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Logging setup
os.makedirs("app/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename="app/logs/app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Enums for validation
class Island(str, Enum):
    Biscoe = "Biscoe"
    Dream = "Dream"
    Torgersen = "Torgersen"

class Sex(str, Enum):
    male = "male"
    female = "female"

# Pydantic model
class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Load model and metadata
MODEL_PATH = "app/data/model.json"
FEATURES_PATH = "app/data/preprocess_meta.json"
CLASSES_PATH = "app/data/label_classes.json"

app = FastAPI()

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

with open(FEATURES_PATH) as f:
    meta = json.load(f)
    expected_columns = meta["feature_columns"]

with open(CLASSES_PATH) as f:
    class_labels = json.load(f)

logger.info("Model and metadata loaded successfully.")

def preprocess(features: PenguinFeatures):
    try:
        input_dict = features.model_dump()
        df = pd.DataFrame([input_dict])
        df = pd.get_dummies(df, columns=["sex", "island"])
        df = df.reindex(columns=expected_columns, fill_value=0)
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Welcome to Penguin Species Predictor!"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(features: PenguinFeatures):
    try:
        df_input = preprocess(features)
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]
        prediction_result = {
            "prediction": int(pred),
            "species": class_labels[pred],
            "probabilities": {class_labels[i]: round(float(p), 4) for i, p in enumerate(proba)}
        }
        logger.info(f"Prediction made: {prediction_result}")
        return prediction_result
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

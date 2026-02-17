import json
import os
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI
from src.schema import PredictRequest, PredictResponse

MODEL_PATH = "models/model.joblib"
META_PATH = "models/metadata.json"
LOG_PATH = "logs/predictions.jsonl"

app = FastAPI()

_model = None
_meta = None

def load_artifacts():
    global _model, _meta
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _meta is None:
        with open(META_PATH) as f:
            _meta = json.load(f)

@app.get("/health")
def health():
    load_artifacts()
    return {"status": "ok", "model_version": _meta["model_version"]}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    load_artifacts()

    incoming = dict(req.features)

    # Remove leakage features if passed
    for c in ["injury_claim", "property_claim", "vehicle_claim"]:
        incoming.pop(c, None)

    expected = set(_meta["numeric_features"] + _meta["categorical_features"])

    for c in expected:
        incoming.setdefault(c, None)

    incoming = {k: incoming[k] for k in expected}

    X = pd.DataFrame([incoming])
    pred = float(_model.predict(X)[0])

    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            "time": str(datetime.now()),
            "prediction": pred,
            "features": req.features
        }) + "\n")

    return PredictResponse(prediction=pred, model_version=_meta["model_version"])

import json
import os
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.schema import PredictRequest, PredictResponse

REG_MODEL_PATH = "models/model.joblib"
REG_META_PATH = "models/metadata.json"

FRAUD_MODEL_PATH = "models/fraud_model.joblib"
FRAUD_META_PATH = "models/fraud_metadata.json"

LOG_PATH = "logs/predictions.jsonl"

app = FastAPI(title="Claims ML API", version="1.0")

_reg_model = None
_reg_meta = None
_fraud_model = None
_fraud_meta = None


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_regression():
    global _reg_model, _reg_meta
    if _reg_model is None:
        if not os.path.exists(REG_MODEL_PATH):
            raise RuntimeError("Regression model not found. Train severity model first.")
        _reg_model = joblib.load(REG_MODEL_PATH)

    if _reg_meta is None:
        if not os.path.exists(REG_META_PATH):
            raise RuntimeError("Regression metadata not found.")
        _reg_meta = _load_json(REG_META_PATH)


def load_fraud():
    global _fraud_model, _fraud_meta
    if _fraud_model is None:
        if not os.path.exists(FRAUD_MODEL_PATH):
            raise RuntimeError("Fraud model not found. Run train_fraud.py first.")
        _fraud_model = joblib.load(FRAUD_MODEL_PATH)

    if _fraud_meta is None:
        if not os.path.exists(FRAUD_META_PATH):
            raise RuntimeError("Fraud metadata not found.")
        _fraud_meta = _load_json(FRAUD_META_PATH)


def log_event(kind: str, features: dict, output: dict):
    os.makedirs("logs", exist_ok=True)
    record = {
        "time": str(datetime.now()),
        "kind": kind,
        "output": output,
        "features": features,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    info = {"status": "ok"}
    if os.path.exists(REG_META_PATH):
        try:
            load_regression()
            info["regression_model_version"] = _reg_meta.get("model_version")
        except Exception:
            pass
    if os.path.exists(FRAUD_META_PATH):
        try:
            load_fraud()
            info["fraud_model_version"] = _fraud_meta.get("model_version")
        except Exception:
            pass
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Regression: total_claim_amount
    load_regression()

    incoming = dict(req.features)

    # Must match training: drop leakage components if present
    for c in ["injury_claim", "property_claim", "vehicle_claim"]:
        incoming.pop(c, None)

    expected = set(_reg_meta["numeric_features"] + _reg_meta["categorical_features"])
    for c in expected:
        incoming.setdefault(c, None)
    incoming = {k: incoming[k] for k in expected}

    X = pd.DataFrame([incoming])

    try:
        pred = float(_reg_model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Regression inference failed: {e}")

    out = {"prediction": pred, "model_version": _reg_meta["model_version"]}
    log_event("regression", req.features, out)
    return out


@app.post("/predict_fraud")
def predict_fraud(req: PredictRequest):
    # Classification: fraud_reported
    load_fraud()

    incoming = dict(req.features)

    # Must match training: drop leakage fields if present
    for c in ["total_claim_amount", "injury_claim", "property_claim", "vehicle_claim"]:
        incoming.pop(c, None)

    expected = set(_fraud_meta["numeric_features"] + _fraud_meta["categorical_features"])
    for c in expected:
        incoming.setdefault(c, None)
    incoming = {k: incoming[k] for k in expected}

    X = pd.DataFrame([incoming])

    try:
        proba = float(_fraud_model.predict_proba(X)[0][1])
        pred = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fraud inference failed: {e}")

    out = {
        "fraud_probability": proba,
        "fraud_predicted": pred,
        "model_version": _fraud_meta["model_version"],
        "threshold": 0.5,
    }
    log_event("fraud", req.features, out)
    return out


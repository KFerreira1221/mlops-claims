import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "models/model.joblib"
META_PATH = "models/metadata.json"

def build_pipeline(df: pd.DataFrame, target: str):
    leakage = ["injury_claim", "property_claim", "vehicle_claim"]
    drop_cols = [target] + [c for c in leakage if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])

    return pipe, X, y, num_cols, cat_cols

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(args.data)
    df = df.dropna(subset=[args.target])

    pipe, X, y, num_cols, cat_cols = build_pipeline(df, args.target)

    idx = np.arange(len(df))
    np.random.shuffle(idx)
    split = int(len(idx) * 0.8)

    X_train, X_test = X.iloc[idx[:split]], X.iloc[idx[split:]]
    y_train, y_test = y.iloc[idx[:split]], y.iloc[idx[split:]]

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(pipe, MODEL_PATH)

    baseline_stats = {}
    for c in num_cols:
        col = pd.to_numeric(X_train[c], errors="coerce")
        baseline_stats[c] = {
            "mean": float(np.nanmean(col)),
            "std": float(np.nanstd(col) + 1e-9),
        }

    meta = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "target": args.target,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "baseline_numeric_stats": baseline_stats,
        "metrics": {"mae": float(mae), "r2": float(r2)},
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("Training complete")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.3f}")

if __name__ == "__main__":
    main()

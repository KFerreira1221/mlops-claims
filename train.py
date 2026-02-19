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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


MODEL_PATH = "models/model.joblib"
META_PATH = "models/metadata.json"


def build_preprocessor(X: pd.DataFrame):
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
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return pre, num_cols, cat_cols


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if name == "num":
            feature_names.extend(list(cols))
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            feature_names.extend(list(ohe.get_feature_names_out(cols)))
    return feature_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns.")

    # Drop rows missing target
    df = df.dropna(subset=[args.target])

    # Drop columns that are entirely null (fixes _c39 / any similar)
    all_null_cols = [c for c in df.columns if df[c].isna().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)

    # Avoid leakage (these often sum to total_claim_amount)
    leakage = ["injury_claim", "property_claim", "vehicle_claim"]
    drop_cols = [args.target] + [c for c in leakage if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[args.target]

    # Train/test split
    idx = np.arange(len(df))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pre, num_cols, cat_cols = build_preprocessor(X_train)

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])

    # Fit full pipeline (no early stopping to avoid version mismatch)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    # Save artifacts
    joblib.dump(pipe, MODEL_PATH)

    baseline_stats = {}
    for c in num_cols:
        col = pd.to_numeric(X_train[c], errors="coerce")
        baseline_stats[c] = {
            "mean": float(np.nanmean(col)),
            "std": float(np.nanstd(col) + 1e-9),
            "missing_rate": float(np.mean(pd.isna(col))),
        }

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    meta = {
        "model_version": model_version,
        "target": args.target,
        "dropped_all_null_columns": all_null_cols,
        "dropped_leakage_features": [c for c in leakage if c in df.columns],
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "baseline_numeric_stats": baseline_stats,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"model_version": model_version, "mae": mae, "rmse": rmse, "r2": r2}, f, indent=2)

    # Feature importance
    feature_names = get_feature_names(pipe.named_steps["preprocess"])
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    fi.to_csv("reports/feature_importance.csv", index=False)

    print("✅ Trained XGBoost model & saved artifacts:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {META_PATH}")
    print("✅ Reports:")
    print("  - reports/metrics.json")
    print("  - reports/feature_importance.csv")
    print(f"✅ Metrics: MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f}")


if __name__ == "__main__":
    main()

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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


MODEL_PATH = "models/fraud_model.joblib"
META_PATH = "models/fraud_metadata.json"


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


def normalize_target(series: pd.Series) -> pd.Series:
    """
    fraud_reported is usually 'Y'/'N'. Convert to 1/0.
    Handles common variants.
    """
    s = series.astype(str).str.strip().str.upper()
    mapping = {"Y": 1, "YES": 1, "TRUE": 1, "1": 1,
               "N": 0, "NO": 0, "FALSE": 0, "0": 0}
    out = s.map(mapping)
    if out.isna().any():
        bad = sorted(set(s[out.isna()].tolist()))
        raise ValueError(f"Unexpected fraud_reported values: {bad[:20]}")
    return out.astype(int)


def metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float):
    pred = (proba >= thr).astype(int)
    acc = float(accuracy_score(y_true, pred))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    cm = confusion_matrix(y_true, pred).tolist()  # [[TN, FP],[FN, TP]]
    return {
        "threshold": float(thr),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV (e.g., data/insurance_claims.csv)")
    parser.add_argument("--target", default="fraud_reported", help="Target column name (default: fraud_reported)")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns.")

    # Drop rows missing target
    df = df.dropna(subset=[args.target])

    # Drop columns that are entirely null
    all_null_cols = [c for c in df.columns if df[c].isna().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)

    # Avoid leakage for fraud model (common “cheat” fields)
    leakage = ["total_claim_amount", "injury_claim", "property_claim", "vehicle_claim"]
    drop_cols = [args.target] + [c for c in leakage if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = normalize_target(df[args.target])

    # Train/test split
    idx = np.arange(len(df))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pre, num_cols, cat_cols = build_preprocessor(X_train)

    # Handle class imbalance (fraud usually rarer)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=900,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    # Probabilities for AUC + threshold tuning
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    # Default threshold 0.50 metrics
    m50 = metrics_at_threshold(np.array(y_test), proba, 0.50)

    # Threshold sweep (analysis)
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    sweep = [metrics_at_threshold(np.array(y_test), proba, t) for t in thresholds]

    # Save model
    joblib.dump(pipe, MODEL_PATH)

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    meta = {
        "model_version": model_version,
        "target": args.target,
        "dropped_all_null_columns": all_null_cols,
        "dropped_leakage_features": [c for c in leakage if c in df.columns],
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "class_balance_train": {"neg": neg, "pos": pos, "scale_pos_weight": float(scale_pos_weight)},
        "metrics_default_threshold_0_50": {
            "accuracy": m50["accuracy"],
            "precision": m50["precision"],
            "recall": m50["recall"],
            "f1": m50["f1"],
            "confusion_matrix": m50["confusion_matrix"],
            "auc": auc,
        },
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save detailed metrics report
    fraud_metrics = {
        "model_version": model_version,
        "auc": auc,
        "default_threshold": 0.50,
        "default_metrics": m50,
        "threshold_sweep": sweep,
    }

    with open("reports/fraud_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fraud_metrics, f, indent=2)

    # Feature importance
    feature_names = get_feature_names(pipe.named_steps["preprocess"])
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    fi.to_csv("reports/fraud_feature_importance.csv", index=False)

    # Print summary
    print("✅ Fraud model trained & saved:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {META_PATH}")
    print("✅ Reports:")
    print("  - reports/fraud_metrics.json")
    print("  - reports/fraud_feature_importance.csv")
    print(f"✅ AUC={auc:.3f}")
    print("✅ Default threshold=0.50 metrics:")
    print(f"   accuracy={m50['accuracy']:.3f} | precision={m50['precision']:.3f} | recall={m50['recall']:.3f} | f1={m50['f1']:.3f}")
    print(f"   confusion_matrix={m50['confusion_matrix']}")
    print("✅ Threshold sweep saved (0.30–0.70).")


if __name__ == "__main__":
    main()


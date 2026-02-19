# 🚀 Insurance Claims ML System (MLOps Project)

End-to-end machine learning system for **claim severity prediction** and **fraud detection**, packaged behind a FastAPI service and containerized with Docker for reproducible deployment.

---

## 📌 Project Overview

This project demonstrates a production-style ML workflow:

* 📊 Tabular feature preprocessing (mixed numeric + categorical)
* 🤖 XGBoost regression model for **claim severity**
* 🕵️ XGBoost classification model for **fraud detection**
* 🌐 FastAPI inference service
* 🐳 Docker containerization
* 📝 Prediction logging + model metadata
* 🎛️ Interactive web UI for testing

The goal is to simulate a realistic insurance analytics pipeline from training → serving → monitoring.

---

## 🧠 Models

### 1️⃣ Claim Severity (Regression)

Predicts:

* **target:** `total_claim_amount`

**Performance**

| Metric | Value  |
| ------ | ------ |
| R²     | 0.697  |
| MAE    | 11,560 |
| RMSE   | 15,446 |

---

### 2️⃣ Fraud Detection (Classification)

Predicts probability of:

* **target:** `fraud_reported`

**Performance**

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.850 |
| Precision | 0.696 |
| Recall    | 0.667 |
| F1        | 0.681 |

---

## 🏗️ System Architecture

```
Raw Data
   ↓
Feature Processing
   ↓
XGBoost Models
   ↓
FastAPI Service
   ↓
Docker Container
   ↓
Web UI / API Clients
```

---

## 📁 Repository Structure

```
mlops-claims/
├── data/
├── logs/
├── models/
│   ├── model.joblib
│   ├── metadata.json
│   ├── fraud_model.joblib
│   └── fraud_metadata.json
├── reports/
├── src/
│   ├── api.py
│   ├── train.py
│   ├── train_fraud.py
│   └── schema.py
├── templates/
│   └── index.html
├── static/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

### Option A — Python

```bash
pip install -r requirements.txt
python -m uvicorn src.api:app --reload
```

Open:

* http://127.0.0.1:8000/
* http://127.0.0.1:8000/docs

---

### Option B — Docker (Recommended)

Build image:

```bash
docker build -t mlops-claims-api .
```

Run container:

```bash
docker run -p 8000:8000 mlops-claims-api
```

Open:

```
http://127.0.0.1:8000/
```

---

## 🔌 API Endpoints

### Health Check

```
GET /health
```

Returns model versions and service status.

---

### Claim Severity Prediction

```
POST /predict
```

**Request**

```json
{
  "features": {
    "age": 40,
    "policy_state": "FL",
    "policy_annual_premium": 1200,
    "incident_type": "Single Vehicle Collision",
    "incident_severity": "Major Damage",
    "number_of_vehicles_involved": 1,
    "bodily_injuries": 1,
    "witnesses": 1,
    "police_report_available": "YES",
    "auto_year": 2016
  }
}
```

---

### Fraud Prediction

```
POST /predict_fraud
```

Returns fraud probability and binary prediction.

---

## 📊 Monitoring & Logging

The service logs every prediction to:

```
logs/predictions.jsonl
```

Each record includes:

* timestamp
* model type
* input features
* prediction output

This simulates production observability.

---

## 🎯 Key MLOps Features

* ✅ Reproducible Docker environment
* ✅ Model version tracking
* ✅ Separate train vs inference pipelines
* ✅ Structured prediction logging
* ✅ Threshold-based fraud classification
* ✅ FastAPI production service

---

## 🧪 Future Improvements

* Azure Container Apps deployment
* CI/CD pipeline (GitHub Actions)
* Feature store integration
* Model drift monitoring
* Authentication layer
* Batch inference pipeline

---

## 👤 Author

**Kevin Ferreira**
MS Artificial Intelligence — Florida Atlantic University
Statistics — Florida International University

* GitHub: https://github.com/KFerreira1221
* Portfolio: https://kferreira1221.github.io/Portfolio/

---

⭐ If you found this project interesting, feel free to star the repo!







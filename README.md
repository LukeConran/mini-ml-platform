# mini-ml-platform

A small end-to-end “mini ML platform” demo for **customer churn prediction**.

It includes:
- a data + training **pipeline** that logs runs to **MLflow** (local SQLite backend)
- a simple **model registry** workflow using MLflow *aliases* (e.g. `@production`)
- a **FastAPI** inference service that loads the `@production` model
- a **Streamlit** frontend that calls the API

## Architecture

```text
pipeline/ (dataset + training)  --->  MLflow tracking + registry (mlflow.db)
                                         |
                                         v
                                  FastAPI /predict
                                         ^
                                         |
                               Streamlit UI (calls API)
```

## Repo layout

```text
.
├─ api/
│  └─ app.py                  # FastAPI inference API (POST /predict)
├─ frontend/
│  └─ streamlit_app.py         # Streamlit UI
├─ pipeline/
│  ├─ dataset.py               # download + preprocess Kaggle Telco churn dataset
│  ├─ train.py                 # train & log multiple models to MLflow
│  ├─ retrain.py               # retrain + auto-promote best run to @production
│  ├─ register_model.py        # manual promotion tool (set model alias)
│  └─ simulate_batch.py        # simulate new batch and trigger retrain
├─ notebooks/                  # (optional) exploration
├─ run.sh                      # run API + Streamlit together
└─ .gitignore
```

## Prerequisites

- Python 3.10+ recommended
- You’ll need whatever Python deps are required by:
  - `fastapi`, `uvicorn`, `streamlit`, `requests`
  - `mlflow`, `pandas`, `scikit-learn`
  - `xgboost`, `lightgbm`, `imblearn`
  - `kagglehub`

## Quickstart (local)

### 1) Create & activate a venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install dependencies (example):

```bash
pip install fastapi uvicorn streamlit requests mlflow pandas scikit-learn xgboost lightgbm imbalanced-learn kagglehub
```

### 2) Download + preprocess the dataset

This downloads the Kaggle “Telco Customer Churn” dataset and writes:
- `data/raw.csv`
- `data/preprocessed.csv`

```bash
python pipeline/dataset.py
```

### 3) Train models & log runs to MLflow

This trains:
- RandomForest (RandomizedSearchCV)
- XGBoost (RandomizedSearchCV)
- LightGBM (RandomizedSearchCV)
- Soft-voting ensemble

and logs params/metrics/models to MLflow (tracking URI: local `mlflow.db`).

```bash
python pipeline/train.py
```

To view experiments:

```bash
mlflow ui
```

### 4) Register/promote a model to `@production`

The API loads: `models:/churn-model@production`.

You can promote a model in two ways:

**A) Manual promotion**
1. Find a run id in the MLflow UI
2. Register that run and set alias to production:

```bash
python pipeline/register_model.py <RUN_ID>
# or:
python pipeline/register_model.py <RUN_ID> --alias production
```

**B) Auto promotion via retraining**
`pipeline/retrain.py` compares the best new run’s `recall_churn` vs the current production model’s `recall_churn`.  
If the new one is better, it registers and sets `@production`.

```bash
python pipeline/retrain.py
```

Optionally merge new labeled raw data first:

```bash
python pipeline/retrain.py --new-data path/to/new_labeled_rows.csv
```

### 5) Run the API + Streamlit app

One command runs both:

```bash
./run.sh
```

- API: http://localhost:8000
- Streamlit: http://localhost:8501

## API

### `POST /predict`

Loads `models:/churn-model@production` (cached after first request) and returns a churn prediction + probability.

The request body must match the `CustomerData` schema in `api/app.py` (includes base fields + one-hot fields like `InternetService_Fiber_optic`, `Contract_Two_year`, etc.).

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"gender":true,"SeniorCitizen":false,"Partner":false,"Dependents":false,"tenure":12,"PhoneService":true,"PaperlessBilling":true,"MonthlyCharges":70.0,"TotalCharges":840.0,
       "MultipleLines_No":true,"MultipleLines_No_phone_service":false,"MultipleLines_Yes":false,
       "InternetService_DSL":false,"InternetService_Fiber_optic":true,"InternetService_No":false,
       "OnlineSecurity_No":true,"OnlineSecurity_No_internet_service":false,"OnlineSecurity_Yes":false,
       "OnlineBackup_No":true,"OnlineBackup_No_internet_service":false,"OnlineBackup_Yes":false,
       "DeviceProtection_No":true,"DeviceProtection_No_internet_service":false,"DeviceProtection_Yes":false,
       "TechSupport_No":true,"TechSupport_No_internet_service":false,"TechSupport_Yes":false,
       "StreamingTV_No":true,"StreamingTV_No_internet_service":false,"StreamingTV_Yes":false,
       "StreamingMovies_No":true,"StreamingMovies_No_internet_service":false,"StreamingMovies_Yes":false,
       "Contract_Month_to_month":true,"Contract_One_year":false,"Contract_Two_year":false,
       "PaymentMethod_Bank_transfer_automatic":false,"PaymentMethod_Credit_card_automatic":false,"PaymentMethod_Electronic_check":true,"PaymentMethod_Mailed_check":false}'
```

## Simulating a “new batch” retrain

This samples rows from `data/raw.csv` into a temp CSV and triggers retraining:

```bash
python pipeline/simulate_batch.py
# or:
python pipeline/simulate_batch.py --n 500
```

## Notes / gotchas

- The API will return **503** until there is a model registered under `churn-model@production`.
- `run.sh` kills processes on ports `8000` and `8501` before starting (helpful locally; be careful if you have other services on those ports).
- `pipeline/retrain.py` uses `from dataset import ...` and `from train import ...` expecting execution from within `pipeline/` (the scripts use `cwd=PIPELINE_DIR` in `simulate_batch.py`).

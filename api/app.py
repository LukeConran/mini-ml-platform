from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow.db")

model = None

app = FastAPI()

class CustomerData(BaseModel):
    gender: bool
    SeniorCitizen: bool
    Partner: bool
    Dependents: bool
    tenure: int
    PhoneService: bool
    PaperlessBilling: bool
    MonthlyCharges: float
    TotalCharges: float
    MultipleLines_No: bool
    MultipleLines_No_phone_service: bool
    MultipleLines_Yes: bool
    InternetService_DSL: bool
    InternetService_Fiber_optic: bool
    InternetService_No: bool
    OnlineSecurity_No: bool
    OnlineSecurity_No_internet_service: bool
    OnlineSecurity_Yes: bool
    OnlineBackup_No: bool
    OnlineBackup_No_internet_service: bool
    OnlineBackup_Yes: bool
    DeviceProtection_No: bool
    DeviceProtection_No_internet_service: bool
    DeviceProtection_Yes: bool
    TechSupport_No: bool
    TechSupport_No_internet_service: bool
    TechSupport_Yes: bool
    StreamingTV_No: bool
    StreamingTV_No_internet_service: bool
    StreamingTV_Yes: bool
    StreamingMovies_No: bool
    StreamingMovies_No_internet_service: bool
    StreamingMovies_Yes: bool
    Contract_Month_to_month: bool
    Contract_One_year: bool
    Contract_Two_year: bool
    PaymentMethod_Bank_transfer_automatic: bool
    PaymentMethod_Credit_card_automatic: bool
    PaymentMethod_Electronic_check: bool
    PaymentMethod_Mailed_check: bool

@app.post("/predict")
def predict(data: CustomerData):
    global model
    if model is None:
        try:
            model = mlflow.sklearn.load_model("models:/churn-model@production")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    return {"churn": bool(prediction[0]), "probability": float(probability[0])}
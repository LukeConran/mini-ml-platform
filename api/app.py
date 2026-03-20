import mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = mlflow.sklearn.load_model("models:/churn-model/Production")
    except Exception as e:
        print(f"WARNING: Could not load model on startup: {e}")
        print("The /predict endpoint will return 503 until the model is available.")
    yield

app = FastAPI(lifespan=lifespan)

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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check that MLflow is running and a Production model is registered.")
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    return {"churn": bool(prediction[0]), "probability": float(probability[0])}
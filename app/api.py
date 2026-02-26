from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Load model
BASE_DIR = Path(__file__).resolve().parent  # app folder
MODEL_PATH = BASE_DIR.parent / "models" / "churn_model_final_v1.joblib"
model_pipeline = joblib.load(MODEL_PATH)

# FastAPI app
app = FastAPI(title="Customer Churn Prediction API")

# --- Input schema ---
class CustomerFeatures(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    tenure_monthly_ratio: float
    services_count: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    contract_payment_interaction: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

# --- Health check ---
@app.get("/health")
def health():
    return {"status": "ok"}

# --- Predict endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    X = pd.DataFrame([customer.model_dump()])
    prob = model_pipeline.predict_proba(X)[:,1][0]
    pred = int(prob >= 0.5)
    return {"prediction": pred, "probability": round(prob,4)}

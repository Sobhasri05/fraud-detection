from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

# Step 1: Load the saved model and feature columns
model = joblib.load('fraud_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Step 2: Create the FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Predict if a transaction is fraudulent",
    version="1.0"
)

# Step 3: Define what a valid request looks like (Pydantic validation)
class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount (must be > 0)")
    location: int = Field(..., ge=0, le=2, description="Location: 0=NY, 1=CA, 2=TX")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")  # ✅ FIXED!

# Step 4: Homepage
@app.get("/")
def home():
    return {"message": "🚀 Fraud Detection API is running!", "status": "healthy"}

# Step 5: The prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    input_data = {
        'amount': transaction.amount,
        'hour': transaction.hour,
        'location': transaction.location
    }
    
    df = pd.DataFrame([input_data])
    df_processed = pd.get_dummies(df, columns=['location'], drop_first=True)
    
    for col in feature_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    df_processed = df_processed[feature_columns]
    
    prediction = model.predict(df_processed)[0]
    probability = model.predict_proba(df_processed)[0][1]
    
    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 4),
        "message": "🚨 FRAUD DETECTED!" if prediction == 1 else "✅ Transaction looks clean",
        "input_received": transaction.dict()
    }

# Step 6: Health check
@app.get("/health")
def health():
    return {"status": "ok"}
"""
Task 6: FastAPI Backend Server for Loan Default Risk Prediction
Serves REST API endpoints for Render Deployment & React/Web Frontend.
"""

import os
import math
import random
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, 'model.joblib')

app = FastAPI(
    title="LoanGuard AI API",
    description="FastAPI Machine Learning Inference Server for Loan Default Risk Prediction",
    version="1.0.0"
)

# Enable CORS for Vercel Frontend and Local Testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows Vercel frontend or local app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
model_source = "Unloaded"

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_source = "Trained Scikit-Learn Model (model.joblib)"
    except Exception:
        model = None

if model is None:
    model_source = "Pre-computed Calibrated Balanced Engine"

class ApplicantInput(BaseModel):
    Age: int = 42
    Income: float = 82500.0
    LoanAmount: float = 120000.0
    CreditScore: int = 650
    MonthsEmployed: int = 60
    NumCreditLines: int = 2
    InterestRate: float = 12.5
    LoanTerm: int = 36
    DTIRatio: float = 0.40
    Education: str = "Bachelor's"
    EmploymentType: str = "Full-time"
    MaritalStatus: str = "Married"
    HasMortgage: str = "Yes"
    HasDependents: str = "No"
    LoanPurpose: str = "Home"
    HasCoSigner: str = "Yes"

def fallback_predict(data: Dict[str, Any]) -> float:
    z = -1.25
    z += ((data['Age'] - 43.5) / 15.0) * (-0.60)
    z += ((data['Income'] - 82500) / 39000) * (-0.30)
    z += ((data['LoanAmount'] - 127500) / 70800) * (0.31)
    z += ((data['CreditScore'] - 574) / 159) * (-0.13)
    z += ((data['MonthsEmployed'] - 59.5) / 34.6) * (-0.37)
    z += ((data['NumCreditLines'] - 2.5) / 1.1) * (0.10)
    z += ((data['InterestRate'] - 13.5) / 6.6) * (0.49)
    z += ((data['LoanTerm'] - 36) / 17.0) * (-0.01)
    z += ((data['DTIRatio'] - 0.50) / 0.23) * (0.08)

    if data.get('Education') == 'High School': z += 0.04
    elif data.get('Education') == "Master's": z -= 0.07
    elif data.get('Education') == 'PhD': z -= 0.11

    if data.get('EmploymentType') == 'Part-time': z += 0.10
    elif data.get('EmploymentType') == 'Self-employed': z += 0.08
    elif data.get('EmploymentType') == 'Unemployed': z += 0.25

    if data.get('MaritalStatus') == 'Married': z -= 0.11
    if data.get('HasMortgage') == 'Yes': z -= 0.10
    if data.get('HasDependents') == 'Yes': z -= 0.13
    if data.get('HasCoSigner') == 'Yes': z -= 0.12

    prob = 1.0 / (1.0 + math.exp(-z))
    return max(0.01, min(0.99, prob))

@app.get("/")
def root():
    return {"message": "LoanGuard AI FastAPI Backend is Live", "docs": "/docs"}

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "Loan Default Prediction FastAPI Server",
        "model_loaded": model is not None,
        "model_source": model_source
    }

@app.get("/api/random-sample")
def random_sample(mode: str = "random"):
    if mode == "low_risk":
        sample = {
            "Age": 52, "Income": 135000, "LoanAmount": 45000, "CreditScore": 795,
            "MonthsEmployed": 108, "NumCreditLines": 2, "InterestRate": 6.8, "LoanTerm": 24,
            "DTIRatio": 0.18, "Education": "PhD", "EmploymentType": "Full-time",
            "MaritalStatus": "Married", "HasMortgage": "Yes", "HasDependents": "Yes",
            "LoanPurpose": "Home", "HasCoSigner": "Yes"
        }
    elif mode == "high_risk":
        sample = {
            "Age": 22, "Income": 24000, "LoanAmount": 185000, "CreditScore": 430,
            "MonthsEmployed": 6, "NumCreditLines": 4, "InterestRate": 22.8, "LoanTerm": 60,
            "DTIRatio": 0.78, "Education": "High School", "EmploymentType": "Unemployed",
            "MaritalStatus": "Single", "HasMortgage": "No", "HasDependents": "No",
            "LoanPurpose": "Business", "HasCoSigner": "No"
        }
    else:
        sample = {
            "Age": int(random.randint(21, 65)),
            "Income": float(random.randint(25000, 145000)),
            "LoanAmount": float(random.randint(15000, 240000)),
            "CreditScore": int(random.randint(350, 820)),
            "MonthsEmployed": int(random.randint(2, 115)),
            "NumCreditLines": int(random.randint(1, 4)),
            "InterestRate": round(random.uniform(4.5, 24.5), 2),
            "LoanTerm": int(random.choice([12, 24, 36, 48, 60])),
            "DTIRatio": round(random.uniform(0.12, 0.85), 2),
            "Education": random.choice(["Bachelor's", "High School", "Master's", "PhD"]),
            "EmploymentType": random.choice(["Full-time", "Part-time", "Self-employed", "Unemployed"]),
            "MaritalStatus": random.choice(["Married", "Single", "Divorced"]),
            "HasMortgage": random.choice(["No", "Yes"]),
            "HasDependents": random.choice(["No", "Yes"]),
            "LoanPurpose": random.choice(["Home", "Business", "Auto", "Education", "Other"]),
            "HasCoSigner": random.choice(["No", "Yes"])
        }
    return {"status": "success", "sample": sample}

@app.post("/api/predict")
def predict(applicant: ApplicantInput):
    data = applicant.dict()
    default_prob = None
    engine = ""

    if model is not None:
        try:
            df = pd.DataFrame([data])
            if hasattr(model, "predict_proba"):
                default_prob = float(model.predict_proba(df)[0][1])
                engine = "Scikit-Learn Pipeline predict_proba"
        except Exception:
            default_prob = None

    if default_prob is None:
        default_prob = fallback_predict(data)
        engine = "Calibrated Balanced Logistic Engine"

    prob_pct = round(default_prob * 100.0, 1)
    prediction = 1 if default_prob >= 0.50 else 0

    if prob_pct < 35.0:
        risk_tier = "Low Risk"
        decision = "Approval Recommended"
        subtext = "Applicant demonstrates strong financial indicators and low default probability."
    elif prob_pct < 60.0:
        risk_tier = "Moderate Risk"
        decision = "Conditional Approval / Manual Review"
        subtext = "Applicant presents mixed credit indicators. Collateral or co-signer strongly advised."
    else:
        risk_tier = "High Risk"
        decision = "High Default Risk / Rejection Advised"
        subtext = "Applicant profile exhibits elevated risk factors that breach safe default thresholds."

    # Financials
    r = (data['InterestRate'] / 100.0) / 12.0
    n = data['LoanTerm']
    P = data['LoanAmount']
    emi = P * (r * ((1 + r) ** n)) / (((1 + r) ** n) - 1) if r > 0 and n > 0 else P / max(1, n)
    total_repayment = emi * n
    total_interest = max(0, total_repayment - P)
    debt_burden_pct = (emi / max(1.0, data['Income'] / 12.0)) * 100.0

    try:
        import database
        database.log_prediction(data, default_prob, risk_tier, decision, engine)
    except Exception:
        pass

    return {
        "status": "success",
        "prediction": prediction,
        "default_probability": round(default_prob, 4),
        "default_probability_pct": prob_pct,
        "risk_tier": risk_tier,
        "decision": decision,
        "decision_subtext": subtext,
        "financial_summary": {
            "monthly_emi": round(emi, 2),
            "total_repayment": round(total_repayment, 2),
            "total_interest": round(total_interest, 2),
            "debt_burden_pct": round(debt_burden_pct, 1)
        },
        "inference_engine": engine
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

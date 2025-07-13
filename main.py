# FastAPI version of credit scoring service with Prometheus monitoring
# Save as main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict
import pickle
import pandas as pd
from prometheus_client import start_http_server, Summary, Counter
import time
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# Monitoring
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
PREDICTIONS = Counter('credit_scoring_requests_total', 'Total prediction requests')

app = FastAPI(title="Credit Scoring Service")

# Load model and scaler
model = pickle.load(open("credit_rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

amount_features = [
    "LIMIT_BAL", *[f"BILL_AMT{i}" for i in range(1, 7)], *[f"PAY_AMT{i}" for i in range(1, 7)]
]

class InputData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.post("/predict")
@REQUEST_TIME.time()
async def predict(data: InputData):
    start = time.time()
    PREDICTIONS.inc()

    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        df[amount_features] = scaler.transform(df[amount_features])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]

        return {
            "prediction": int(prediction),
            "probability": float(round(probability, 4)),
            "latency": round(time.time() - start, 4)
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# Start Prometheus server
start_http_server(8001)




app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("index.html")


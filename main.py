# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from prometheus_client import start_http_server, Summary, Counter
import time
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------
# 1) Monitoring metrics
# --------------------------------------------------
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
PREDICTIONS = Counter('credit_scoring_requests_total', 'Total prediction requests')

# --------------------------------------------------
# 2) FastAPI app + static mount
# --------------------------------------------------
app = FastAPI(title="Credit Scoring Service")
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=FileResponse)
def root():
    return FileResponse("index.html")


# --------------------------------------------------
# 3) Load model + scaler
# --------------------------------------------------
model = pickle.load(open("credit_rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

amount_features = [
    "LIMIT_BAL", *[f"BILL_AMT{i}" for i in range(1, 5)],
    *[f"PAY_AMT{i}" for i in range(1, 5)]
]

# --------------------------------------------------
# 4) Pydantic input schema
# --------------------------------------------------
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
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float

# --------------------------------------------------
# 5) Prediction endpoint (sync) with metrics
# --------------------------------------------------
@app.post("/predict")
@REQUEST_TIME.time()
def predict(data: InputData):
    start = time.time()
    PREDICTIONS.inc()
    try:
        payload = data.model_dump()

        # Ensure the DataFrame columns follow the original training order:
        df = pd.DataFrame([payload], columns=model.feature_names_in_)

        # Scale only the amount features (the rest remain untouched)
        df[amount_features] = scaler.transform(df[amount_features])

        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0,1])
        latency = round(time.time() - start, 4)

        return {"prediction": pred, "probability": prob, "latency": latency}

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# Deploying the trained model as a Flask scoring service with monitoring setup
# Assumes: credit_rf_model.pkl is trained and stored locally

from flask import Flask, request, jsonify
import pickle
import pandas as pd
from prometheus_client import start_http_server, Summary, Counter
import time

app = Flask(__name__)

# Load trained model
with open("D:\Workspace\PYTHON\Python Programs\CreditScoringModel\credit_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define monitoring metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
PREDICTIONS = Counter('credit_scoring_requests_total', 'Total prediction requests')

# Define API endpoint
@app.route("/predict", methods=["POST"])
@REQUEST_TIME.time()
def predict():
    start = time.time()
    PREDICTIONS.inc()
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # Preprocessing logic: Scale relevant features
        amount_features = [
            "LIMIT_BAL", *[f"BILL_AMT{i}" for i in range(1, 7)], *[f"PAY_AMT{i}" for i in range(1, 7)]
        ]
        scaler = pickle.load(open("scaler.pkl", "rb"))
        df[amount_features] = scaler.transform(df[amount_features])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(round(probability, 4)),
            "latency": round(time.time() - start, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Start Prometheus metrics server
start_http_server(8001)  # Prometheus scrapes from this port

if __name__ == "__main__":
    app.run(debug=True, port=8000)
# To run this service:
# 1. Save this script as app.py
# 2. Run `python app.py` in your terminal
# 3. Send a POST request to http://localhost:8000/predict with JSON data    

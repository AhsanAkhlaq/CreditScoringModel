# 💳 Credit Default Prediction API

## 🌐 Frontend Preview

<!-- Paste screenshots or gifs of the UI here -->

A production-ready machine learning web service that predicts the likelihood of a credit card client defaulting on their next payment. Built with a Random Forest model trained on the UCI "Default of Credit Card Clients" dataset, the system is served via a FastAPI backend and paired with a clean, interactive HTML/CSS frontend. The entire application is Dockerized for seamless local or cloud deployment.

---

## 🔗 Dataset

This project uses the [UCI Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/data) available on Kaggle. The dataset contains detailed financial data of 30,000 credit card holders, including:

* Demographic information (age, gender, education, marital status)
* Credit limit and bill amounts
* Payment history and repayment status
* Target variable: Whether the client defaulted in the next month (0 = No, 1 = Yes)

The dataset is clean and does not contain missing values, making it ideal for fast experimentation.

---

## 🌟 Key Features

* ✨ **Random Forest Classifier** using scikit-learn
* ⚖️ **Balanced class weights** to handle class imbalance
* 🚀 **FastAPI backend** for high-performance inference and REST API
* 🌐 **Interactive HTML/CSS frontend** with basic and advanced input controls
* 📄 **Dockerized** architecture for one-command setup
* 🌐 **Optional Feature Controls**: 8 required inputs with 15 toggleable advanced fields (defaulted to 0)

---

## 📁 Repository Structure

```
.
├── main.py                # FastAPI backend (serves API + index.html)
├── index.html             # Frontend UI (client-side form)
├── credit_rf_model.pkl    # Trained Random Forest model
├── scaler.pkl             # Fitted StandardScaler for numeric features
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── model.ipynb            # Model training and preprocessing notebook
├── UCI_Credit_Card.csv    # Raw dataset
└── README.md              # This documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/CreditScoringModel.git
cd CreditScoringModel
```

### 2. Build and Run with Docker

```bash
# Build the Docker image
docker build -t masterlord/creditscoringmodel .

# Run the container (exposes 8000 for frontend/API)
docker run -d -p 8000:8000 masterlord/creditscoringmodel
```

* 🔗 Frontend & API: [http://localhost:8000](http://localhost:8000)
* ⚖️ Prometheus Metrics (optional): [http://localhost:8001/metrics](http://localhost:8001/metrics)

> Docker Hub Image: [masterlord/creditscoringmodel](https://hub.docker.com/r/masterlord/creditscoringmodel)

### 3. Run Locally Without Docker (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then visit: [http://localhost:8000](http://localhost:8000)

---

## 🛠️ Usage

1. Open the web interface at `http://localhost:8000`
2. Fill out the **Credit Assessment Form** with basic details
3. Click **"Show Advanced Options"** to expand 15 additional optional inputs
4. Click **Predict Credit Risk**
5. View prediction result including:

   * Default risk category (Low Risk / Likely to Default)
   * Probability score
   * Latency (in seconds)

---

## 📦 Docker Hub

Official image published at:

> [https://hub.docker.com/r/masterlord/creditscoringmodel](https://hub.docker.com/r/masterlord/creditscoringmodel)

Pull the image manually:

```bash
docker pull masterlord/creditscoringmodel
```

---

## 🔍 Model Training & Preprocessing

* **Dataset**: UCI Credit Card Dataset (Kaggle)
* **Preprocessing**:

  * StandardScaler applied to numeric financial columns: `LIMIT_BAL`, `BILL_AMT1–6`, `PAY_AMT1–6`
  * No missing values
* **Model**: RandomForestClassifier with balanced class weights
* **Artifacts Saved**:

  * `credit_rf_model.pkl`: Trained model
  * `scaler.pkl`: Standard scaler for preprocessing

Training details and experiments can be found in `model.ipynb`

---

## ✉️ Contact

For feedback, issues, or contributions, feel free to open a pull request or issue on this repository.

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use and modify it for personal or commercial use.

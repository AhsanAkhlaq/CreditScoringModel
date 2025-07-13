# Dockerfile for FastAPI Credit Scoring Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY main.py . 
COPY credit_rf_model.pkl . 
COPY scaler.pkl .

# Install dependencies
RUN pip install fastapi uvicorn pandas scikit-learn prometheus-client pydantic

# Expose ports
EXPOSE 8000 8001

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

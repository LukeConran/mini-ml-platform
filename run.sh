#!/bin/bash
# Starts the MLflow tracking server, FastAPI backend, and Streamlit frontend.
# Usage: ./run.sh
# Stop everything: Ctrl+C

MLFLOW_PORT=5000
API_PORT=8000
STREAMLIT_PORT=8501

cleanup() {
    echo "\nShutting down..."
    kill $MLFLOW_PID $API_PID $STREAMLIT_PID 2>/dev/null
    exit 0
}
trap cleanup INT

# MLflow tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --port $MLFLOW_PORT &
MLFLOW_PID=$!
echo "MLflow running on http://localhost:$MLFLOW_PORT"

# Give MLflow a moment to start before the API tries to load from it
sleep 2

# FastAPI
MLFLOW_TRACKING_URI=http://localhost:$MLFLOW_PORT \
    uvicorn api.app:app --port $API_PORT --reload &
API_PID=$!
echo "API running on http://localhost:$API_PORT"

# Streamlit
streamlit run frontend/streamlit_app.py --server.port $STREAMLIT_PORT &
STREAMLIT_PID=$!
echo "Streamlit running on http://localhost:$STREAMLIT_PORT"

wait

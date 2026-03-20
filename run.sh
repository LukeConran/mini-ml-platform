#!/bin/bash
# Usage: ./run.sh   |   Stop: Ctrl+C

API_PORT=8000
STREAMLIT_PORT=8501

# Clear any leftover processes from previous runs
lsof -ti :$API_PORT :$STREAMLIT_PORT | xargs kill -9 2>/dev/null; true

cleanup() {
    echo "\nShutting down..."
    kill $API_PID $STREAMLIT_PID 2>/dev/null
    exit 0
}
trap cleanup INT

uvicorn api.app:app --port $API_PORT --reload &
API_PID=$!
echo "API running on http://localhost:$API_PORT"

streamlit run frontend/streamlit_app.py --server.port $STREAMLIT_PORT &
STREAMLIT_PID=$!
echo "Streamlit running on http://localhost:$STREAMLIT_PORT"

wait

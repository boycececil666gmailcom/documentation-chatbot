#!/bin/bash
set -e

# Start LangGraph dev server in background on port 2024
langgraph dev --host 0.0.0.0 --port 2024 &

# Start FastAPI backend engine in foreground on port 8000
exec python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

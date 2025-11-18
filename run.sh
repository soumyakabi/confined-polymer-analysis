#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Building and launching Confined Polymer Analysis Streamlit app..."
docker compose up --build -d

echo ""
echo "✅ App running at: http://localhost:8501"
echo "To follow logs: docker compose logs -f"
echo "To stop: docker compose down"

@echo off
echo 🚀 Building and launching Confined Polymer Analysis Streamlit app...
docker compose up --build -d

echo.
echo ✅ App running at: http://localhost:8501
echo Use "docker compose logs -f" to follow logs.
echo Use "docker compose down" to stop the container.
pause

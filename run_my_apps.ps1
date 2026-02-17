# run_my_apps.ps1
# Activate venv and run FastAPI + Streamlit automatically

# --- Activate virtual environment ---
.\venv\Scripts\activate

# --- Start FastAPI server in background ---
Start-Process uvicorn -ArgumentList "app.main:app --reload --host 127.0.0.1 --port 8000"

# Wait a few seconds to ensure FastAPI is up
Start-Sleep -Seconds 3

# --- Start Streamlit dashboard ---
streamlit run streamlit_app/dashboard.py
@echo off
echo ===================================================
echo   STARTING ML GUARD SERVICES
echo ===================================================

echo [1/2] Launching Backend API...
start "ML Guard Backend" cmd /k "cd backend && venv\Scripts\activate && uvicorn app.main:app --reload"

echo [2/2] Launching Frontend Dashboard...
start "ML Guard Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ===================================================
echo   All systems running!
echo   - API:        http://localhost:8000/docs
echo   - Dashboard:  http://localhost:5173
echo ===================================================
echo.

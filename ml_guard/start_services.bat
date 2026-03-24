@echo off
echo ===================================================
echo   STARTING ML GUARD SERVICES
echo ===================================================

echo [1/3] Launching Backend API...
start "ML Guard Backend" cmd /k "cd backend && venv\Scripts\activate && uvicorn app.main:app --host 127.0.0.1 --reload"

echo [2/3] Launching Celery Governance Worker...
start "ML Guard Celery" cmd /k "cd backend && venv\Scripts\activate && celery -A app.core.celery_app worker --loglevel=info -P solo"

echo [3/3] Launching Frontend Dashboard...
start "ML Guard Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ===================================================
echo   All systems running!
echo   - API:        http://localhost:8000/docs
echo   - Dashboard:  http://localhost:3000
echo   - Queue:      Redis (localhost:6379)
echo ===================================================
echo.

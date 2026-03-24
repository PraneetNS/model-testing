@echo off
echo ===================================================
echo   ML GUARD - PRODUCTION ENVIRONMENT SETUP
echo ===================================================

echo.
echo [1/2] Setting up Backend...
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
echo Backend dependencies installed.

echo.
echo [2/2] Setting up Frontend...
cd ../frontend
call npm install
echo Frontend dependencies installed.

echo.
echo ===================================================
echo   SETUP COMPLETE!
echo   Run 'start_services.bat' to launch the platform.
echo ===================================================
pause

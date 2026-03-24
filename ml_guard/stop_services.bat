@echo off
echo ===================================================
echo   STOPPING ML GUARD SERVICES
echo ===================================================

echo [1/2] Stopping Backend (Python)...
taskkill /F /IM python.exe
taskkill /F /IM uvicorn.exe

echo [2/2] Stopping Frontend (Node.js)...
taskkill /F /IM node.exe

echo.
echo ===================================================
echo   SERVICES STOPPED.
echo ===================================================
pause

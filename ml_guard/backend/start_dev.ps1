# start_dev.ps1
# Run this to start all ML Guard services

Write-Host "Starting ML Guard v7.2 Development Stack"

# Terminal 1 — FastAPI
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "cd '$PWD'; .\venv\Scripts\Activate.ps1; uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

# Terminal 2 — Celery Worker  
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "cd '$PWD'; .\venv\Scripts\Activate.ps1; celery -A app.core.celery_app worker --loglevel=info --pool=solo"

Write-Host "[*] FastAPI starting on http://127.0.0.1:8000"
Write-Host "[*] Celery worker starting"
Write-Host "[*] Swagger UI: http://127.0.0.1:8000/docs"

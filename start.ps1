# ============================================================
# start.ps1 - ML Guard One-Shot Startup Script (Windows)
# 
# Usage:
#   From repo root: .\start.ps1
#
# Starts (in separate windows):
#   1. Redis  (via wsl or local redis-server)
#   2. FastAPI backend (uvicorn)
#   3. Celery worker
#   4. Next.js frontend
# ============================================================

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "ml_guard\backend"
$Frontend = Join-Path $Root "ml_guard\frontend"
$Venv = Join-Path $Backend "venv\Scripts\python.exe"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   ML Guard - Platform Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# -- 1. Redis Check / Start --------------------------------
Write-Host "[1/4] Checking Redis..." -ForegroundColor Yellow

$redisRunning = $false
try {
    $result = & $Venv -c "import redis; r=redis.Redis.from_url('redis://localhost:6379/0'); r.ping(); print('ok')" 2>$null
    if ($result -eq "ok") { $redisRunning = $true }
} catch {}

if (-not $redisRunning) {
    Write-Host "      Redis not running. Attempting to start via WSL..." -ForegroundColor Yellow
    try {
        Start-Process "wsl" -ArgumentList "-e", "sudo", "service", "redis-server", "start" -WindowStyle Hidden
        Start-Sleep -Seconds 3
        $result = & $Venv -c "import redis; r=redis.Redis.from_url('redis://localhost:6379/0'); r.ping(); print('ok')" 2>$null
        if ($result -eq "ok") { $redisRunning = $true }
    } catch {}
    
    if (-not $redisRunning) {
        Write-Host ""
        Write-Host "  !  Redis could not be started automatically." -ForegroundColor Red
        Write-Host "     Please start Redis manually (e.g., via WSL: 'sudo service redis-server start')" -ForegroundColor Red
        Write-Host "     Then re-run this script." -ForegroundColor Red
        Write-Host ""
        Read-Host "Press Enter to continue anyway (Celery will fail without Redis)"
    }
} else {
    Write-Host "      Redis OK" -ForegroundColor Green
}

# -- 2. Backend - FastAPI ----------------------------------
Write-Host "[2/4] Starting FastAPI backend (port 8000)..." -ForegroundColor Yellow

$backendCmd = @"
cd '$Backend'; .\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
"@

Start-Process "powershell.exe" -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCmd `
    -WindowStyle Normal `
    -WorkingDirectory $Backend

Start-Sleep -Seconds 4
Write-Host "      FastAPI started - http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "      API Docs: http://127.0.0.1:8000/docs" -ForegroundColor DarkGray

# -- 3. Celery Worker -------------------------------------
Write-Host "[3/4] Starting Celery worker..." -ForegroundColor Yellow

$celeryCmd = @"
cd '$Backend'; .\venv\Scripts\celery.exe -A app.core.celery_app worker --loglevel=info --pool=solo
"@

Start-Process "powershell.exe" -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $celeryCmd `
    -WindowStyle Normal `
    -WorkingDirectory $Backend

Start-Sleep -Seconds 2
Write-Host "      Celery worker started" -ForegroundColor Green

# -- 4. Frontend - Next.js --------------------------------
Write-Host "[4/4] Starting Next.js frontend (port 3000)..." -ForegroundColor Yellow

$frontendCmd = @"
cd '$Frontend'; npm run dev
"@

Start-Process "powershell.exe" -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $frontendCmd `
    -WindowStyle Normal `
    -WorkingDirectory $Frontend

Start-Sleep -Seconds 3
Write-Host "      Next.js started - http://localhost:3000" -ForegroundColor Green

# -- Summary -----------------------------------------------
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   All services started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Dashboard  ->  http://localhost:3000" -ForegroundColor White
Write-Host "  API Docs   ->  http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "  Health     ->  http://127.0.0.1:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "  Logs are visible in each service window." -ForegroundColor DarkGray
Write-Host "  Close the windows to stop services." -ForegroundColor DarkGray
Write-Host ""

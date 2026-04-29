# Niyantrana Platform Startup Script (Windows PowerShell)
# Zero-manual-step deployment for Windows environments.

$ErrorActionPreference = "Stop"

Write-Host "Starting Niyantrana Platform..." -ForegroundColor Cyan

# 1. Check dependencies
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python 3.11+ required. Please install from python.org"
}
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "Node.js 20+ required. Please install from nodejs.org"
}

# 2. Check and create .env if missing
$backendPath = Join-Path $PSScriptRoot "ml_guard\backend"
$envPath = Join-Path $backendPath ".env"
$exampleEnv = Join-Path $backendPath ".env.example"

if (-not (Test-Path $envPath)) {
    if (Test-Path $exampleEnv) {
        Copy-Item $exampleEnv $envPath
        Write-Host "⚠️  Created .env from .env.example" -ForegroundColor Yellow
        Write-Host "⚠️  Please set DATABASE_URL, REDIS_URL, and SECRET_KEY in $envPath" -ForegroundColor Yellow
        Write-Host "⚠️  Then run .\run.ps1 again." -ForegroundColor Yellow
        exit
    } else {
        Write-Error ".env and .env.example missing in $backendPath"
    }
}

# 3. Setup Python environment
Push-Location $backendPath
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating Python virtual environment..." -ForegroundColor Green
    python -m venv venv
}

Write-Host "Installing backend dependencies..." -ForegroundColor Green
& ".\venv\Scripts\pip.exe" install -q -r requirements.txt

# 4. Run database migrations
Write-Host "Running database migrations..." -ForegroundColor Green
& ".\venv\Scripts\python.exe" -m alembic upgrade head

# 5. Seed initial data
Write-Host "Seeding database..." -ForegroundColor Green
& ".\venv\Scripts\python.exe" -c 'import asyncio; import sys; import os; sys.path.insert(0, os.getcwd()); from app.db.seed import seed_if_empty; asyncio.run(seed_if_empty())'

# 6. Start backend services
Write-Host "Starting FastAPI backend..." -ForegroundColor Green
$backendProcess = Start-Process -FilePath ".\venv\Scripts\uvicorn.exe" -ArgumentList "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload" -PassThru

Write-Host "Starting Celery worker..." -ForegroundColor Green
$workerProcess = Start-Process -FilePath ".\venv\Scripts\celery.exe" -ArgumentList "-A", "app.core.celery_app", "worker", "--loglevel=info", "-P", "solo" -PassThru

Write-Host "Starting Celery beat scheduler..." -ForegroundColor Green
$beatProcess = Start-Process -FilePath ".\venv\Scripts\celery.exe" -ArgumentList "-A", "app.core.celery_app", "beat", "--loglevel=info" -PassThru

# 7. Start frontend
$frontendPath = Join-Path $PSScriptRoot "ml_guard\frontend"
Push-Location $frontendPath

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Green
    npm install --legacy-peer-deps
}

Write-Host "Starting Next.js frontend..." -ForegroundColor Green
$frontendProcess = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev" -PassThru

Write-Host "`n----------------------------------------" -ForegroundColor Cyan
Write-Host '  Niyantrana is running' -ForegroundColor Green
Write-Host '----------------------------------------' -ForegroundColor Cyan
Write-Host '  Dashboard:  http://localhost:3000'
Write-Host '  API:        http://localhost:8000'
Write-Host '  API Docs:   http://localhost:8000/docs'
Write-Host '----------------------------------------' -ForegroundColor Cyan
Write-Host '  Default login: admin@niyantrana.ai'
Write-Host '  Default pass:  change-me-immediately'
Write-Host '----------------------------------------' -ForegroundColor Cyan
Write-Host '  Close this window to stop all services' -ForegroundColor Yellow
Write-Host ''
Pop-Location
Pop-Location

# Wait for processes
$processes = @($backendProcess, $workerProcess, $beatProcess, $frontendProcess) | Where-Object { $_ -ne $null -and $_.Id -ne $null }
if ($processes) {
    Wait-Process -Id ($processes.Id)
}

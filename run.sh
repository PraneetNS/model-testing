#!/bin/bash

# Exit on any error
set -e

echo "Starting Niyantrana Platform..."

# 1. Check dependencies
command -v python3 >/dev/null 2>&1 || { echo "Python 3.11+ required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js 20+ required"; exit 1; }
command -v redis-cli >/dev/null 2>&1 || { echo "Redis required. Install: brew install redis"; exit 1; }
command -v pg_isready >/dev/null 2>&1 || { echo "PostgreSQL required."; exit 1; }
python3 -c "import sys; assert sys.version_info >= (3,11)" || { echo "Python 3.11+ required"; exit 1; }
node -e "const v=process.versions.node.split('.'); if(v[0]<20) process.exit(1)" || { echo "Node 20+ required"; exit 1; }

# 2. Check and create .env if missing
if [ ! -f ml_guard/backend/.env ]; then
  cp ml_guard/backend/.env.example ml_guard/backend/.env 2>/dev/null || :
  echo "⚠️  Created .env from .env.example (if it existed)"
  echo "⚠️  Please set DATABASE_URL, REDIS_URL, and SECRET_KEY in ml_guard/backend/.env"
  echo "⚠️  Then run ./run.sh again."
  exit 1
fi

# Validate required vars are set
source ml_guard/backend/.env
[ -z "$DATABASE_URL" ] && echo "❌ DATABASE_URL not set in .env" && exit 1
[ -z "$REDIS_URL" ] && echo "❌ REDIS_URL not set in .env" && exit 1
[ -z "$SECRET_KEY" ] && echo "❌ SECRET_KEY not set in .env" && exit 1
[ "$SECRET_KEY" = "generate-with-openssl-rand-hex-32" ] && echo "❌ Replace SECRET_KEY with: openssl rand -hex 32" && exit 1

# 3. Start infrastructure services (if using Docker)
if command -v docker >/dev/null 2>&1; then
  echo "🐳 Starting PostgreSQL and Redis via Docker..."
  docker-compose -f docker-compose.dev.yml up -d postgres redis 2>/dev/null || :
  sleep 3
fi

# 4. Setup Python environment
cd ml_guard/backend
if [ ! -d venv ]; then
  python3 -m venv venv
  echo "✅ Created Python virtual environment"
fi
source venv/bin/activate
pip install -q -r requirements.txt
echo "✅ Python dependencies installed"

# 5. Run database migrations
echo "🗄️  Running database migrations..."
alembic upgrade head
if [ $? -ne 0 ]; then
  echo "❌ Migration failed. Check DATABASE_URL and PostgreSQL connection."
  exit 1
fi
echo "✅ Database migrations complete"

# 6. Seed initial data
python3 -c "
import asyncio
import sys
import os
sys.path.insert(0, os.getcwd())
from app.db.seed import seed_if_empty
asyncio.run(seed_if_empty())
"
echo "✅ Database seeded"

# 7. Start backend services
echo "🚀 Starting FastAPI backend..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "⚙️  Starting Celery worker..."
celery -A app.core.celery_app worker --loglevel=info -Q default,drift,audit,red_team &
WORKER_PID=$!

echo "⏰ Starting Celery beat scheduler..."
celery -A app.core.celery_app beat --loglevel=info &
BEAT_PID=$!

# 8. Wait for backend to be healthy
echo "⏳ Waiting for backend to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ Backend is ready"
    break
  fi
  sleep 1
  if [ $i -eq 30 ]; then
    echo "❌ Backend failed to start after 30 seconds. Check logs."
    kill $BACKEND_PID $WORKER_PID $BEAT_PID 2>/dev/null
    exit 1
  fi
done

# 9. Start frontend
cd ../frontend
if [ ! -d node_modules ]; then
  echo "📦 Installing frontend dependencies..."
  npm install --legacy-peer-deps
fi
echo "🌐 Starting Next.js frontend..."
npm run dev &
FRONTEND_PID=$!

# 10. Print status
echo ""
echo "════════════════════════════════════════"
echo "  ✅ Niyantrana is running"
echo "════════════════════════════════════════"
echo "  🌐 Dashboard:  http://localhost:3000"
echo "  ⚡ API:        http://localhost:8000"
echo "  📚 API Docs:   http://localhost:8000/docs"
echo "  🌸 Flower:     (run: celery flower)"
echo "════════════════════════════════════════"
echo "  Default login: admin@niyantrana.ai"
echo "  Default pass:  change-me-immediately"
echo "════════════════════════════════════════"
echo "  Press Ctrl+C to stop all services"
echo ""

# 11. Wait and handle Ctrl+C gracefully
trap "echo ''; echo 'Stopping all services...'; kill $BACKEND_PID $WORKER_PID $BEAT_PID $FRONTEND_PID 2>/dev/null; echo 'Done.'; exit 0" INT TERM
wait

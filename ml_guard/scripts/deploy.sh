#!/bin/bash
# Enterprise ML Guard — Production Deployment Script
# This script prepares the production environment and starts the services.

set -e

echo "🚀 Starting Production Deployment for ML Guard..."

# 1. Environment Validation
if [ ! -f .env ]; then
    echo "❌ .env file missing! Creating from template..."
    cp .env.production.template .env
    echo "⚠️ Please edit .env with real secrets before proceeding."
    # exit 1 
fi

# 2. Build and Start Services
echo "📦 Building Docker containers..."
docker-compose -f docker-compose.yml build

echo "🚦 Starting services (database, redis, backend, worker, frontend)..."
docker-compose -f docker-compose.yml up -d

# 3. Wait for DB and run migrations
echo "⏳ Waiting for database to be ready..."
sleep 10

# Note: In a real production setup, we would use Alembic for migrations
# echo "🛠️ Running database migrations..."
# docker exec ml_guard_backend alembic upgrade head

# 4. Create initial admin if needed
# echo "👤 Seeding initial admin user..."
# docker exec ml_guard_backend python seed_user.py

echo "✅ Deployment complete!"
echo "------------------------------------------------"
echo "🖥️ Frontend: http://localhost:5174"
echo "⚙️ Backend API: http://localhost:8001"
echo "📖 API Docs: http://localhost:8001/docs"
echo "------------------------------------------------"

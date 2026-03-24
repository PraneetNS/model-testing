FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy from project root context
COPY ml_guard/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project for access to core libs
COPY . .

# Set python path to include backend and root for imports
ENV PYTHONPATH=/app/ml_guard/backend:/app

# Use $PORT variable provided by cloud hosts
CMD ["sh", "-c", "uvicorn ml_guard.backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

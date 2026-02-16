#!/bin/bash
# ML Guard Startup Script
# Quick launcher for the ML Guard service

echo "🚀 Starting FireFlink ML Guard..."
echo "================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -r backend/requirements.txt

# Start the service
echo "🌟 Starting ML Guard service..."
echo ""
echo "📍 Service URLs:"
echo "   API:  http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo "   Health: http://localhost:8000/health"
echo ""
echo "🧪 Run demo: cd examples/customer_churn && python demo.py"
echo ""

cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
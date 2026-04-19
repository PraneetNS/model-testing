import os
import sys
import uvicorn

# Get the absolute path of the backend directory
# Root/ml_guard/backend
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml_guard", "backend"))

# Add backend directory to sys.path so 'app' package is discoverable
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Set environment variables if needed
os.environ["PYTHONPATH"] = f"{backend_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

if __name__ == "__main__":
    print(f"Starting ML Guard Backend from {backend_dir}...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

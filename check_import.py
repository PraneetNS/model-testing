import sys
import os

# Set current dir to backend
os.chdir("ml_guard/backend")
sys.path.insert(0, os.getcwd())

try:
    from app.main import app
    print("SUCCESS: app.main:app imported successfully")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()

import os
import sys
_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Backend root: {_backend_root}")
sys.path.insert(0, _backend_root)
print(f"Sys path: {sys.path[:3]}")

try:
    from app.core.config import settings
    print("Import SUCCESS")
except Exception as e:
    print(f"Import FAILED: {e}")
